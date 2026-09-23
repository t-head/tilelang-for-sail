"""DeepSeek DeepGEMM FP8 2xAcc benchmark with autotune.

Adapted from: examples/deepseek_deepgemm/example_deepgemm_fp8_2xAcc.py
Tunable parameters: block_N, num_stages, threads, enable_rasteration
"""

import argparse

import torch
import tilelang.language as T
from tilelang.autotuner import autotune
from tilelang import jit

from configs import get_deepgemm_configs
from utils import (
    gemm_flops,
    per_token_cast_to_fp8,
    per_block_cast_to_fp8,
    ref_deepgemm_fp8,
    print_benchmark_summary,
    bench_ref,
    inject_pass_configs_from_env,
)


def ref_program(A_fp8, B_fp8, C, scales_a, scales_b):
    """Reference for profiler validation."""
    out_dtype = C.dtype
    return ref_deepgemm_fp8(A_fp8, B_fp8, scales_a, scales_b, out_dtype)


@autotune(configs=get_deepgemm_configs(), warmup=5, rep=20, skip_check=True)
@jit(out_idx=[2])
def deepgemm_fp8(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    block_N=None,
    num_stages=None,
    threads=None,
    enable_rasteration=None,
):
    """FP8 2xAcc GEMM kernel (DeepGEMM style).

    Fixed: block_M=128, block_K=128, group_size=128.
    Tunable: block_N, num_stages, threads, enable_rasteration.
    """
    # Provide concrete defaults so the kernel can be elaborated (e.g. for
    # cache-key / validation TIR generation by the autotuner) before the
    # actual tunable values are supplied. These are overridden by autotune.
    block_N = block_N or 64
    num_stages = num_stages if num_stages is not None else 2
    threads = threads or 128
    enable_rasteration = enable_rasteration if enable_rasteration is not None else False

    group_size = 128
    block_M = 128
    block_K = 128

    A_shape = (M, K)
    Scales_A_shape = (M, T.ceildiv(K, group_size))
    B_shape = (N, K)
    Scales_B_shape = (T.ceildiv(N, group_size), T.ceildiv(K, group_size))
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K)
    C_shared_shape = (block_M, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
        scales_a: T.Tensor(Scales_A_shape, T.float32),
        scales_b: T.Tensor(Scales_B_shape, T.float32),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_shared = T.alloc_shared(C_shared_shape, out_dtype)
            Scale_C_shared = T.alloc_shared((block_M), T.float32)
            C_local = T.alloc_fragment(C_shared_shape, accum_dtype)
            C_local_accum = T.alloc_fragment(C_shared_shape, accum_dtype)

            T.use_swizzle(panel_size=10, enable=enable_rasteration)

            T.clear(C_local)
            T.clear(C_local_accum)
            K_iters = T.ceildiv(K, block_K)
            for k in T.Pipelined(K_iters, num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                Scale_B = scales_b[bx * block_N // group_size, k]
                for i in T.Parallel(block_M):
                    Scale_C_shared[i] = scales_a[by * block_M + i, k] * Scale_B

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                # Promote to enable 2xAcc
                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j] * Scale_C_shared[i]
                T.clear(C_local)
            # Store
            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


def run_profile(M, N, K, in_dtype_str, out_dtype_str, block_N, num_stages, threads, enable_rasteration):
    """Run the kernel once with an explicit config for ncu/acu profiling."""
    import torch

    in_dtype = getattr(T, in_dtype_str)
    out_dtype = getattr(T, out_dtype_str)
    accum_dtype = T.float32
    out_torch_dtype = getattr(torch, out_dtype_str)

    A = torch.randn(M, K, dtype=out_torch_dtype, device="cuda")
    B = torch.randn(N, K, dtype=out_torch_dtype, device="cuda")
    A_fp8, scales_a = per_token_cast_to_fp8(A)
    B_fp8, scales_b = per_block_cast_to_fp8(B)

    inject_pass_configs_from_env(deepgemm_fp8)
    kernel = deepgemm_fp8(
        M,
        N,
        K,
        in_dtype,
        out_dtype,
        accum_dtype,
        block_N=block_N,
        num_stages=num_stages,
        threads=threads,
        enable_rasteration=enable_rasteration,
    )
    kernel(A_fp8, B_fp8, scales_a, scales_b)


def run_profile_ref(M, N, K, in_dtype_str, out_dtype_str):
    """Run the reference once for ncu/acu profiling."""
    import torch

    out_torch_dtype = getattr(torch, out_dtype_str)
    A_ref = torch.randn(M, K, dtype=out_torch_dtype, device="cuda")
    B_ref = torch.randn(K, N, dtype=out_torch_dtype, device="cuda")

    torch.matmul(A_ref, B_ref)


def main(M=1024, N=1024, K=8192, in_dtype_str="float8_e4m3fn", out_dtype_str="bfloat16"):
    """Run autotune and print results."""

    in_dtype = getattr(T, in_dtype_str)
    out_dtype = getattr(T, out_dtype_str)
    accum_dtype = T.float32

    total_flops = gemm_flops(M, N, K)

    best_result = deepgemm_fp8(M, N, K, in_dtype, out_dtype, accum_dtype)
    best_latency = best_result.latency
    best_config = best_result.config
    tflops = total_flops / best_latency * 1e-9

    # Reference: torch.matmul with bfloat16 inputs
    out_torch_dtype = getattr(torch, out_dtype_str)
    A_ref = torch.randn(M, K, dtype=out_torch_dtype, device="cuda")
    B_ref = torch.randn(K, N, dtype=out_torch_dtype, device="cuda")

    def ref_gemm():
        return torch.matmul(A_ref, B_ref)

    ref_latency = bench_ref(ref_gemm)
    ref_tflops = total_flops / ref_latency * 1e-9 if ref_latency > 0 else 0

    print_benchmark_summary(
        "DeepGEMM FP8",
        f"M={M}, N={N}, K={K}",
        best_latency,
        tflops,
        ref_latency,
        ref_tflops,
        "Reference",
        best_config,
    )

    return best_latency, tflops, best_config, ref_latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepGEMM FP8 Autotune Benchmark")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--in_dtype", type=str, default="float8_e4m3fn")
    parser.add_argument("--out_dtype", type=str, default="bfloat16")
    parser.add_argument("--profile", action="store_true", help="Run kernel once with given config for ncu/acu profiling")
    parser.add_argument("--profile-ref", action="store_true", help="Run reference once for ncu/acu profiling")
    parser.add_argument("--block_N", type=int, default=None)
    parser.add_argument("--num_stages", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--enable_rasteration", type=lambda v: v.lower() in ("true", "1", "yes"), default=None, help="Enable rasteration (true/false)"
    )
    args = parser.parse_args()

    if args.profile:
        run_profile(
            args.m, args.n, args.k, args.in_dtype, args.out_dtype, args.block_N, args.num_stages, args.threads, args.enable_rasteration
        )
    elif args.profile_ref:
        run_profile_ref(args.m, args.n, args.k, args.in_dtype, args.out_dtype)
    else:
        main(args.m, args.n, args.k, args.in_dtype, args.out_dtype)
