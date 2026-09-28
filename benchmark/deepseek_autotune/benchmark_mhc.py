"""DeepSeek mHC (Multi-Head Contribution) pre benchmark with autotune.

Adapted from: examples/deepseek_mhc/example_mhc_pre.py
The mHC pre block consists of two sub-kernels:
  1. gemm_sqrsum: fused GEMM + square-sum
  2. big_fuse: RMS norm + split mixes + sinkhorn + apply pre mix

Tunable parameters: token_block, hidden_block, num_stages
"""

import argparse

import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune
from tilelang.autotuner.capture import set_autotune_inputs
from tilelang import jit

from configs import get_mhc_pre_configs
from utils import print_benchmark_summary, bench_ref, inject_pass_configs_from_env


@autotune(configs=get_mhc_pre_configs(), warmup=5, rep=20, skip_check=True)
@jit(
    out_idx=[2, 3],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def mhc_pre_gemm_sqrsum(
    hc_mult3: int,
    hc_hidden_size: int,
    token_block=None,
    hidden_block=None,
    num_stages=None,
):
    """Fused GEMM + sqrsum sub-kernel of mHC pre block.

    Tunable: token_block, hidden_block, num_stages.
    """
    # Provide concrete defaults so the kernel can be elaborated (e.g. for
    # cache-key / validation TIR generation by the autotuner) before the
    # actual tunable values are supplied. These are overridden by autotune.
    token_block = token_block or 16
    hidden_block = hidden_block or 128
    num_stages = num_stages if num_stages is not None else 1

    assert hc_mult3 <= 32
    num_tokens = T.dynamic("num_tokens")
    assert hc_hidden_size % hidden_block == 0

    @T.prim_func
    def main(
        x: T.Tensor((num_tokens, hc_hidden_size), T.bfloat16),
        fn: T.Tensor((hc_mult3, hc_hidden_size), T.float32),
        out: T.Tensor((num_tokens, hc_mult3), T.float32),
        sqrsum: T.Tensor((num_tokens,), T.float32),
    ):
        with T.Kernel(T.ceildiv(num_tokens, token_block)) as px:
            out_frag = T.alloc_fragment((token_block, 32), T.float32)
            sqrsum_part = T.alloc_fragment((token_block, 4), T.float32)
            T.clear(out_frag)
            T.clear(sqrsum_part)
            for pz in T.Pipelined(hc_hidden_size // hidden_block, num_stages=num_stages):
                x_smem_16 = T.alloc_shared((token_block, hidden_block), T.bfloat16)
                fn_smem = T.alloc_shared((32, hidden_block), T.float32)

                T.annotate_layout({x_smem_16: tilelang.layout.make_swizzled_layout(x_smem_16)})

                T.copy(x[px * token_block, pz * hidden_block], x_smem_16)
                T.copy(fn[0, pz * hidden_block], fn_smem)

                x_frag_16 = T.alloc_fragment((token_block, hidden_block), T.bfloat16)
                T.copy(x_smem_16, x_frag_16)
                x_frag = T.alloc_fragment((token_block, hidden_block), T.float32)
                T.copy(x_frag_16, x_frag)

                for jj in T.serial(hidden_block // 4):
                    for i, j in T.Parallel(token_block, 4):
                        sqrsum_part[i, j] += x_frag[i, jj * 4 + j] * x_frag[i, jj * 4 + j]

                T.gemm(
                    x_frag,
                    fn_smem,
                    out_frag,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=False,
                )
            sqrsum_l = T.alloc_fragment(token_block, T.float32)
            T.reduce_sum(sqrsum_part, sqrsum_l)
            for i in T.Parallel(token_block):
                sqrsum[px * token_block + i] = sqrsum_l[i]
            for i, j in T.Parallel(token_block, 32):
                if j < hc_mult3:
                    out[px * token_block + i, j] = out_frag[i, j]

    return main


def sinkhorn_normalize_ref(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    """Reference sinkhorn normalization."""
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def mhc_pre_ref(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat):
    """Reference mHC pre forward."""
    hc_mult = residual.shape[-2]
    residual_flat = residual.flatten(-2, -1).float()
    sqrsum = residual_flat.square().sum(-1)
    mixes = residual_flat @ fn.T * (sqrsum.unsqueeze(-1) / fn.shape[-1] + rms_eps).rsqrt()

    hc_scale_expanded = torch.cat(
        [
            hc_scale[0].expand(hc_mult),
            hc_scale[1].expand(hc_mult),
            hc_scale[2].expand(hc_mult * hc_mult),
        ]
    )
    mixes = mixes * hc_scale_expanded + hc_base

    pre_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(-1) + hc_pre_eps
    post_mix = (mixes[:, hc_mult : 2 * hc_mult].sigmoid() * hc_post_mult_value).unsqueeze(-1)
    res_mix = mixes[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult)
    res_mix = sinkhorn_normalize_ref(res_mix, repeat=sinkhorn_repeat, eps=hc_sinkhorn_eps)
    layer_input = (residual * pre_mix).sum(-2).bfloat16()

    return post_mix, res_mix, layer_input


def run_profile(n, hidden_size, hc_mult, token_block, hidden_block, num_stages):
    """Run the kernel once with an explicit config for ncu/acu profiling."""
    import torch

    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size

    x = torch.randn(n, hc_hidden_size, dtype=torch.bfloat16, device="cuda")
    fn = torch.randn(hc_mult3, hc_hidden_size, dtype=torch.float32, device="cuda")

    inject_pass_configs_from_env(mhc_pre_gemm_sqrsum)
    with set_autotune_inputs(x, fn):
        kernel = mhc_pre_gemm_sqrsum(
            hc_mult3,
            hc_hidden_size,
            token_block=token_block,
            hidden_block=hidden_block,
            num_stages=num_stages,
        )
    kernel(x, fn)


def run_profile_ref(n, hidden_size, hc_mult):
    """Run the reference once for ncu/acu profiling."""
    import torch

    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size

    x = torch.randn(n, hc_hidden_size, dtype=torch.bfloat16, device="cuda")
    fn = torch.randn(hc_mult3, hc_hidden_size, dtype=torch.float32, device="cuda")
    x_ref = x.float()

    x_ref @ fn.T
    (x_ref * x_ref).sum(dim=-1)


def main(n=1024, hidden_size=2560, hc_mult=4):
    """Run autotune for gemm_sqrsum and print results."""

    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult
    hc_hidden_size = hc_mult * hidden_size

    # Prepare concrete tensors for dynamic-shape kernel autotune
    x = torch.randn(n, hc_hidden_size, dtype=torch.bfloat16, device="cuda")
    fn = torch.randn(hc_mult3, hc_hidden_size, dtype=torch.float32, device="cuda")

    with set_autotune_inputs(x, fn):
        best_result = mhc_pre_gemm_sqrsum(hc_mult3, hc_hidden_size)

    best_latency = best_result.latency
    best_config = best_result.config

    # Approximate FLOPs: GEMM (2*n*hc_mult3*hc_hidden_size) + sqrsum (2*n*hc_hidden_size)
    total_flops = 2 * n * hc_mult3 * hc_hidden_size + 2 * n * hc_hidden_size

    # Reference: torch matmul + square-sum
    x_ref = x.float()

    def ref_mhc():
        gemm_out = x_ref @ fn.T
        sq = (x_ref * x_ref).sum(dim=-1)
        return gemm_out, sq

    ref_latency = bench_ref(ref_mhc)
    ref_tflops = total_flops / ref_latency * 1e-9 if ref_latency > 0 else 0

    print_benchmark_summary(
        "mHC Pre",
        f"n={n}, hidden_size={hidden_size}, hc_mult={hc_mult}",
        best_latency,
        total_flops / best_latency * 1e-9,
        ref_latency,
        ref_tflops,
        "Reference",
        best_config,
    )

    return best_latency, total_flops / best_latency * 1e-9, best_config, ref_latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mHC Pre Autotune Benchmark")
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--hidden_size", type=int, default=2560)
    parser.add_argument("--hc_mult", type=int, default=4)
    parser.add_argument("--profile", action="store_true", help="Run kernel once with given config for ncu/acu profiling")
    parser.add_argument("--profile-ref", action="store_true", help="Run reference once for ncu/acu profiling")
    parser.add_argument("--token_block", type=int, default=None)
    parser.add_argument("--hidden_block", type=int, default=None)
    parser.add_argument("--num_stages", type=int, default=None)
    args = parser.parse_args()

    if args.profile:
        run_profile(args.n, args.hidden_size, args.hc_mult, args.token_block, args.hidden_block, args.num_stages)
    elif args.profile_ref:
        run_profile_ref(args.n, args.hidden_size, args.hc_mult)
    else:
        main(args.n, args.hidden_size, args.hc_mult)
