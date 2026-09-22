"""DeepSeek mHC pre big_fuse benchmark with autotune.

Adapted from: examples/deepseek_mhc/example_mhc_pre.py
The big_fuse kernel performs: RMS norm + split mixes + sinkhorn + apply pre mix.
It runs independently given random gemm_out_mul/gemm_out_sqrsum tensors.

Tunable parameters: threads, num_stages
"""

import argparse
import math

import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune
from tilelang.autotuner.capture import set_autotune_inputs
from tilelang import jit

from configs import get_mhc_big_fuse_configs
from utils import print_benchmark_summary, bench_ref, inject_pass_configs_from_env


@autotune(configs=get_mhc_big_fuse_configs(), warmup=5, rep=20, skip_check=True)
@jit(out_idx=[5, 6, 7], pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
})
def mhc_big_fuse(
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    hc_mult: int = 4,
    threads=None,
    num_stages=None,
):
    """Big fuse kernel: RMS norm + split mixes + sinkhorn + apply pre mix.

    Tunable: threads, num_stages.
    """
    # Provide concrete defaults so the kernel can be elaborated (e.g. for
    # cache-key / validation TIR generation by the autotuner) before the
    # actual tunable values are supplied. These are overridden by autotune.
    threads = threads or 64
    num_stages = num_stages if num_stages is not None else 1

    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    hidden_block = math.gcd(512, hidden_size)

    @T.prim_func
    def main(
        gemm_out_mul: T.Tensor((n_splits, num_tokens, hc_mult3), T.float32),
        gemm_out_sqrsum: T.Tensor((n_splits, num_tokens), T.float32),
        hc_scale: T.Tensor((3,), T.float32),
        hc_base: T.Tensor((hc_mult3,), T.float32),
        residual: T.Tensor((num_tokens, hc_mult, hidden_size), T.bfloat16),
        post_mix: T.Tensor((num_tokens, hc_mult), T.float32),
        comb_mix: T.Tensor((num_tokens, hc_mult * hc_mult), T.float32),
        layer_input: T.Tensor((num_tokens, hidden_size), T.bfloat16),
    ):
        with T.Kernel(num_tokens, threads=threads) as i:
            # _pre_norm_fn_fwd_norm
            rms = T.alloc_fragment(1, T.float32)
            mixes = T.alloc_fragment(hc_mult3, T.float32)
            T.clear(mixes)
            rms[0] = 0
            for i_split in T.serial(n_splits):
                rms[0] += gemm_out_sqrsum[i_split, i]
            rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)
            for j in T.Parallel(hc_mult3):
                mixes[j] = 0
                for i_split in T.serial(n_splits):
                    mixes[j] += gemm_out_mul[i_split, i, j]
                mixes[j] *= rms[0]
            mixes_shared = T.alloc_shared(hc_mult3, T.float32)
            T.copy(mixes, mixes_shared)

            if T.get_thread_binding() < 32:
                # _pre_split_mixes_fwd (post & comb)
                cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
                for j in T.Parallel(hc_mult):
                    post_mix[i, j] = T.sigmoid(mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]) * hc_post_mult_value
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2] + hc_base[j * hc_mult + k + hc_mult * 2]

                # _sinkhorn_fwd
                row_sum = T.alloc_fragment(hc_mult, T.float32)
                col_sum = T.alloc_fragment(hc_mult, T.float32)

                row_max = T.alloc_fragment(hc_mult, T.float32)
                T.reduce_max(cm, row_max, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = T.exp(cm[j, k] - row_max[j])
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

                for _ in T.serial(sinkhorn_repeat - 1):
                    T.reduce_sum(cm, row_sum, dim=1)
                    for j, k in T.Parallel(hc_mult, hc_mult):
                        cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                    T.reduce_sum(cm, col_sum, dim=0)
                    for j, k in T.Parallel(hc_mult, hc_mult):
                        cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

                for j, k in T.Parallel(hc_mult, hc_mult):
                    comb_mix[i, j * hc_mult + k] = cm[j, k]
            else:
                # _pre_split_mixes_fwd (pre)
                pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
                for j in T.Parallel(hc_mult):
                    pre_mix_shared[j] = (
                        T.sigmoid(
                            mixes_shared[j] * hc_scale[0] + hc_base[j],
                        )
                        + hc_pre_eps
                    )
                # _pre_apply_mix_fwd
                for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=num_stages):
                    xs = T.alloc_shared((hc_mult, hidden_block), T.float32)
                    xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                    T.copy(residual[i, 0, i0_h * hidden_block], xs)
                    T.copy(xs, xl)

                    ol = T.alloc_fragment(hidden_block, T.float32)
                    T.clear(ol)

                    for i_hc in T.serial(hc_mult):
                        pre = pre_mix_shared[i_hc]
                        for i1_h in T.Parallel(hidden_block):
                            ol[i1_h] += pre * xl[i_hc, i1_h]

                    T.copy(ol, layer_input[i, i0_h * hidden_block])

    return main


def sinkhorn_normalize_ref(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    """Reference sinkhorn normalization."""
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def ref_big_fuse(residual, gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base,
                 rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat):
    """Reference big_fuse: RMS norm + split mixes + sinkhorn + apply pre mix."""
    hc_mult = residual.shape[1]
    hidden_size = residual.shape[2]
    n_splits = gemm_out_mul.shape[0]

    # RMS norm + aggregate splits
    sqrsum = gemm_out_sqrsum.sum(dim=0)  # [num_tokens]
    rms = torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)
    mixes = gemm_out_mul.sum(dim=0)  # [num_tokens, hc_mult3]
    mixes = mixes * rms.unsqueeze(-1)

    # Split mixes
    pre_mix = (mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]).sigmoid() + hc_pre_eps
    post_mix = (mixes[:, hc_mult:2*hc_mult] * hc_scale[1] + hc_base[hc_mult:2*hc_mult]).sigmoid() * hc_post_mult_value
    comb_raw = mixes[:, 2*hc_mult:] * hc_scale[2] + hc_base[2*hc_mult:]
    comb_raw = comb_raw.view(-1, hc_mult, hc_mult)

    # Sinkhorn
    comb_mix = sinkhorn_normalize_ref(comb_raw, repeat=sinkhorn_repeat, eps=hc_sinkhorn_eps)

    # Apply pre mix
    layer_input = (residual.float() * pre_mix.unsqueeze(-1)).sum(dim=1).bfloat16()

    return post_mix, comb_mix.view(-1, hc_mult * hc_mult), layer_input


def run_profile(n, hidden_size, hc_mult, threads, num_stages,
                 rms_eps=1e-6, hc_pre_eps=1e-6,
                 hc_sinkhorn_eps=1e-6, hc_post_mult_value=1.0, sinkhorn_repeat=10):
    """Run the kernel once with an explicit config for ncu/acu profiling."""
    import torch

    n_splits = 1
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult

    gemm_out_mul = torch.randn(n_splits, n, hc_mult3, dtype=torch.float32, device="cuda")
    gemm_out_sqrsum = torch.randn(n_splits, n, dtype=torch.float32, device="cuda").abs()
    hc_scale = torch.randn(3, dtype=torch.float32, device="cuda") * 0.1
    hc_base = torch.randn(hc_mult3, dtype=torch.float32, device="cuda") * 0.1
    residual = torch.randn(n, hc_mult, hidden_size, dtype=torch.bfloat16, device="cuda")

    inject_pass_configs_from_env(mhc_big_fuse)
    with set_autotune_inputs(gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual):
        kernel = mhc_big_fuse(
            hidden_size, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
            hc_post_mult_value, sinkhorn_repeat, n_splits, hc_mult,
            threads=threads, num_stages=num_stages,
        )
    kernel(gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual)


def run_profile_ref(n, hidden_size, hc_mult,
                     rms_eps=1e-6, hc_pre_eps=1e-6,
                     hc_sinkhorn_eps=1e-6, hc_post_mult_value=1.0, sinkhorn_repeat=10):
    """Run the reference once for ncu/acu profiling."""
    import torch

    n_splits = 1
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult

    gemm_out_mul = torch.randn(n_splits, n, hc_mult3, dtype=torch.float32, device="cuda")
    gemm_out_sqrsum = torch.randn(n_splits, n, dtype=torch.float32, device="cuda").abs()
    hc_scale = torch.randn(3, dtype=torch.float32, device="cuda") * 0.1
    hc_base = torch.randn(hc_mult3, dtype=torch.float32, device="cuda") * 0.1
    residual = torch.randn(n, hc_mult, hidden_size, dtype=torch.bfloat16, device="cuda")

    ref_big_fuse(residual, gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base,
                 rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat)


def main(n=1024, hidden_size=2560, hc_mult=4, rms_eps=1e-6, hc_pre_eps=1e-6,
         hc_sinkhorn_eps=1e-6, hc_post_mult_value=1.0, sinkhorn_repeat=10):
    """Run autotune for big_fuse and print results."""
    from tilelang.profiler import do_bench

    n_splits = 1
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult

    # Generate inputs
    gemm_out_mul = torch.randn(n_splits, n, hc_mult3, dtype=torch.float32, device="cuda")
    gemm_out_sqrsum = torch.randn(n_splits, n, dtype=torch.float32, device="cuda").abs()
    hc_scale = torch.randn(3, dtype=torch.float32, device="cuda") * 0.1
    hc_base = torch.randn(hc_mult3, dtype=torch.float32, device="cuda") * 0.1
    residual = torch.randn(n, hc_mult, hidden_size, dtype=torch.bfloat16, device="cuda")

    with set_autotune_inputs(gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual):
        best_result = mhc_big_fuse(
            hidden_size, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
            hc_post_mult_value, sinkhorn_repeat, n_splits, hc_mult,
        )

    best_latency = best_result.latency
    best_config = best_result.config

    # FLOPs estimate: mainly the weighted sum (pre_apply_mix) + sinkhorn iterations
    # pre_apply_mix: n * hc_mult * hidden_size * 2 (mul + add)
    # sinkhorn: n * hc_mult^2 * sinkhorn_repeat * ~6 ops
    total_flops = (2 * n * hc_mult * hidden_size +
                   n * hc_mult * hc_mult * sinkhorn_repeat * 6)

    # Reference
    def ref_fn():
        return ref_big_fuse(residual, gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base,
                            rms_eps, hc_pre_eps, hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat)

    ref_latency = bench_ref(ref_fn)
    ref_tflops = total_flops / ref_latency * 1e-9 if ref_latency > 0 else 0

    print_benchmark_summary(
        "mHC BigFuse",
        f"n={n}, hidden_size={hidden_size}, hc_mult={hc_mult}",
        best_latency, total_flops / best_latency * 1e-9,
        ref_latency, ref_tflops,
        "Reference", best_config,
    )

    return best_latency, total_flops / best_latency * 1e-9, best_config, ref_latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mHC BigFuse Autotune Benchmark")
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--hidden_size", type=int, default=2560)
    parser.add_argument("--hc_mult", type=int, default=4)
    parser.add_argument("--profile", action="store_true",
                        help="Run kernel once with given config for ncu/acu profiling")
    parser.add_argument("--profile-ref", action="store_true",
                        help="Run reference once for ncu/acu profiling")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--num_stages", type=int, default=None)
    args = parser.parse_args()

    if args.profile:
        run_profile(args.n, args.hidden_size, args.hc_mult,
                    args.threads, args.num_stages)
    elif args.profile_ref:
        run_profile_ref(args.n, args.hidden_size, args.hc_mult)
    else:
        main(args.n, args.hidden_size, args.hc_mult)
