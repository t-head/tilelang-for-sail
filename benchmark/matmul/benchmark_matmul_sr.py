import argparse
import itertools
import logging
import torch

import tilelang.language as T
from tilelang.autotuner import autotune
from tilelang import jit

from utils import print_benchmark_summary

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def ref_program(A, B):
    """
    A reference matrix multiplication program, used to compare performance.

    matmul_sr computes C = A @ B.T with trans_A=False (M-major A) and trans_B=True (N-major B),
    which means A has shape (M, K) and B has shape (N, K), result C is (M, N).
    Internally B is loaded into register (fragment) before MMA.

    Parameters
    ----------
    A : numpy.ndarray
        The matrix with shape (M, K).
    B : numpy.ndarray
        The matrix with shape (N, K).

    Returns
    -------
    np.ndarray
        The result of A @ B.T, shape (M, N).
    """
    return A @ B.T


def get_configs(M, N, K, with_roller, **kwargs):
    """
    Generate a list of configuration dictionaries that will be used for tuning.

    Parameters
    ----------
    with_roller : bool
        Whether to enable bitblas roller to deduce search spaces

    Returns
    -------
    list of dict
        Each configuration dict includes various block sizes, pipeline stages,
        thread numbers, and other parameters to explore during autotuning.
    """
    if with_roller:
        from tilelang.carver.template import MatmulTemplate
        from tilelang.carver.arch import CUDA
        from tilelang.carver.arch import CDNA
        from tilelang.carver.roller.rasterization import NoRasterization

        arch = CUDA("cuda") if torch.version.hip is None else CDNA("hip")
        topk = 10

        carve_template = MatmulTemplate(
            M=M,
            N=N,
            K=K,
            in_dtype=T.float16,
            out_dtype=T.float16,
            accum_dtype=T.float32,
        ).with_arch(arch)

        func = carve_template.equivalent_function()
        assert func is not None, "Function is None"

        roller_hints = carve_template.recommend_hints(topk=topk)

        if roller_hints is None:
            raise ValueError("No Roller Hints Found for TensorCore Scheduling")

        configs = []
        for hint in roller_hints:
            config = {}
            block_m, block_n = hint.block
            warp_m, warp_n = hint.warp
            # block_rows, block_cols represents warp partitioning
            block_rows, block_cols = block_m // warp_m, block_n // warp_n
            config["block_M"] = block_m
            config["block_N"] = block_n
            config["block_K"] = hint.rstep[0]
            config["num_stages"] = hint.pipeline_stage
            config["thread_num"] = block_rows * block_cols * 32
            configs.append(config)
    else:
        iter_params = dict(
            block_M=[64, 128, 256],
            block_N=[64, 128, 256],
            block_K=[32, 64],
            num_stages=[0, 1, 2, 3],
            thread_num=[128, 256],
        )
        return [{k: v for k, v in zip(iter_params, values)} for values in itertools.product(*iter_params.values())]
    return configs


@autotune(
    configs=get_configs,
    warmup=10,
    rep=100,
    ref_prog=ref_program,
)
@jit(
    out_idx=[2],
)
def matmul(
    M,
    N,
    K,
    with_roller,
    block_M=None,
    block_N=None,
    block_K=None,
    num_stages=None,
    thread_num=None,
):
    """
    Create an autotuned shared-register matrix multiplication kernel for matrices of shape:
      - A: (M, K)   -- trans_A=False, M-major, stays in shared memory
      - B: (N, K)   -- trans_B=True, N-major, loaded into register (fragment) before MMA
      - C: (M, N)

    A remains in shared memory ("shared source"),
    B is loaded via shared memory into register (fragment) ("register source").
    This is called "shared-register" (sr) for the A/B operand sources.

    Returns
    -------
    (best_latency, best_config, ref_latency)
        best_latency : float
            The best latency found among the tuned configurations.
        best_config : dict
            The parameter configuration that yielded best_latency.
        ref_latency : float
            The baseline latency of the reference program (for computing speedup).
    """
    # Provide concrete defaults so the kernel can be elaborated (e.g. for
    # cache-key / validation TIR generation by the autotuner) before the
    # actual tunable values are supplied. These are overridden by autotune.
    block_M = block_M or 64
    block_N = block_N or 64
    block_K = block_K or 32
    num_stages = num_stages if num_stages is not None else 1
    thread_num = thread_num or 128

    trans_A = False
    trans_B = True
    in_dtype = T.float16
    out_dtype = T.float16
    accum_dtype = T.float32

    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            B_local = T.alloc_fragment(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.copy(B_shared, B_local)
                T.gemm(A_shared, B_local, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autotuned Shared-Register MatMul Benchmark")
    parser.add_argument("--m", type=int, default=16384, help="Matrix dimension M")
    parser.add_argument("--n", type=int, default=16384, help="Matrix dimension N")
    parser.add_argument("--k", type=int, default=16384, help="Matrix dimension K")
    parser.add_argument("--with_roller", action="store_true", default=False, help="Whether to use roller to deduce search spaces")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "int8"], help="Input data type")
    args = parser.parse_args()

    M, N, K = args.m, args.n, args.k
    with_roller = args.with_roller

    # Compute total floating-point operations
    total_flops = 2 * M * N * K

    # Run autotuning
    best_result = matmul(M, N, K, with_roller)
    best_latency = best_result.latency
    best_config = best_result.config
    ref_latency = best_result.ref_latency

    # Print benchmark results
    tilelang_tflops = total_flops / best_latency * 1e-9
    ref_tflops = total_flops / ref_latency * 1e-9 if ref_latency is not None else 0.0
    print_benchmark_summary(
        "MatMul SR",
        f"M={M}, N={N}, K={K}",
        best_latency, tilelang_tflops,
        ref_latency if ref_latency is not None else 0.0, ref_tflops,
        "Reference", best_config,
    )



