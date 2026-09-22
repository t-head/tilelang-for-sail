"""DeepSeek mHC post benchmark with autotune.

Adapted from: examples/deepseek_mhc/example_mhc_post.py
The mHC post kernel applies: x_out = comb_mix^T @ residual + post_mix * x

Tunable parameters: n_thr (threads), h_blk (hidden tile size)
"""

import argparse
import math

import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune
from tilelang.autotuner.capture import set_autotune_inputs
from tilelang import jit

from configs import get_mhc_post_configs
from utils import print_benchmark_summary, bench_ref, inject_pass_configs_from_env


@autotune(configs=get_mhc_post_configs(), warmup=5, rep=20, skip_check=True)
@jit(out_idx=[4], pass_configs={
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
})
def mhc_post(
    hc: int,
    hidden: int,
    n_thr=None,
    h_blk=None,
):
    """mHC post kernel: x_out[i] = comb_mix^T[i] @ residual[i] + post_mix[i] * x[i]

    Tunable: n_thr (threads), h_blk (hidden block size).
    """
    # Provide concrete defaults so the kernel can be elaborated (e.g. for
    # cache-key / validation TIR generation by the autotuner) before the
    # actual tunable values are supplied. These are overridden by autotune.
    n_thr = n_thr or 64
    h_blk = h_blk or 256

    n = T.dynamic("num_tokens")
    h = hidden
    h_blk_actual = math.gcd(hidden, h_blk)

    @T.prim_func
    def main(
        a: T.Tensor((n, hc, hc), T.float32),
        b: T.Tensor((n, hc, h), T.bfloat16),
        c: T.Tensor((n, hc), T.float32),
        d: T.Tensor((n, h), T.bfloat16),
        x: T.Tensor((n, hc, h), T.bfloat16),
    ):
        with T.Kernel(n, threads=n_thr) as i_n:
            x_shared = T.alloc_shared((hc, h_blk_actual), T.bfloat16)
            b_shared = T.alloc_shared((hc, h_blk_actual), T.bfloat16)
            d_shared = T.alloc_shared(h_blk_actual, T.bfloat16)

            x_local = T.alloc_fragment((hc, h_blk_actual), T.float32)
            b_local = T.alloc_fragment((hc, h_blk_actual), T.float32)
            d_local = T.alloc_fragment(h_blk_actual, T.float32)

            a_local = T.alloc_fragment((hc, hc), T.float32)
            c_local = T.alloc_fragment(hc, T.float32)
            T.copy(a[i_n, 0, 0], a_local)
            T.copy(c[i_n, 0], c_local)

            for i0_h in T.Pipelined(T.ceildiv(h, h_blk_actual), num_stages=2):
                T.copy(b[i_n, 0, i0_h * h_blk_actual], b_shared)
                T.copy(d[i_n, i0_h * h_blk_actual], d_shared)

                T.copy(b_shared, b_local)
                T.copy(d_shared, d_local)
                for i_hco, i1_h in T.Parallel(hc, h_blk_actual):
                    x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
                    for i_hci in T.serial(hc):
                        x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]
                T.copy(x_local, x_shared)

                T.copy(x_shared, x[i_n, 0, i0_h * h_blk_actual])

    return main


def ref_mhc_post(x, residual, post_layer_mix, comb_res_mix):
    """Reference: x_out = comb_mix^T @ residual + post_mix * x."""
    term2 = torch.bmm(comb_res_mix.mT, residual.float())
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()


def run_profile(n, hidden_size, hc_mult, n_thr, h_blk):
    """Run the kernel once with an explicit config for ncu/acu profiling."""
    import torch

    comb_res_mix = torch.randn(n, hc_mult, hc_mult, dtype=torch.float32, device="cuda")
    residual = torch.randn(n, hc_mult, hidden_size, dtype=torch.bfloat16, device="cuda")
    post_layer_mix = torch.randn(n, hc_mult, dtype=torch.float32, device="cuda")
    x = torch.randn(n, hidden_size, dtype=torch.bfloat16, device="cuda")

    inject_pass_configs_from_env(mhc_post)
    with set_autotune_inputs(comb_res_mix, residual, post_layer_mix, x):
        kernel = mhc_post(hc_mult, hidden_size, n_thr=n_thr, h_blk=h_blk)
    kernel(comb_res_mix, residual, post_layer_mix, x)


def run_profile_ref(n, hidden_size, hc_mult):
    """Run the reference once for ncu/acu profiling."""
    import torch

    comb_res_mix = torch.randn(n, hc_mult, hc_mult, dtype=torch.float32, device="cuda")
    residual = torch.randn(n, hc_mult, hidden_size, dtype=torch.bfloat16, device="cuda")
    post_layer_mix = torch.randn(n, hc_mult, dtype=torch.float32, device="cuda")
    x = torch.randn(n, hidden_size, dtype=torch.bfloat16, device="cuda")
    post_layer_mix_3d = post_layer_mix.unsqueeze(-1)

    ref_mhc_post(x, residual, post_layer_mix_3d, comb_res_mix)


def main(n=4096, hidden_size=2560, hc_mult=4):
    """Run autotune for mhc_post and print results."""
    from tilelang.profiler import do_bench

    # Generate inputs matching the mhc_post signature:
    # a = comb_res_mix: [n, hc, hc]
    # b = residual: [n, hc, h]
    # c = post_layer_mix (squeezed): [n, hc]
    # d = x: [n, h]
    # output: x_out: [n, hc, h]
    comb_res_mix = torch.randn(n, hc_mult, hc_mult, dtype=torch.float32, device="cuda")
    residual = torch.randn(n, hc_mult, hidden_size, dtype=torch.bfloat16, device="cuda")
    post_layer_mix = torch.randn(n, hc_mult, dtype=torch.float32, device="cuda")
    x = torch.randn(n, hidden_size, dtype=torch.bfloat16, device="cuda")

    with set_autotune_inputs(comb_res_mix, residual, post_layer_mix, x):
        best_result = mhc_post(hc_mult, hidden_size)

    best_latency = best_result.latency
    best_config = best_result.config

    # FLOPs: bmm (2*n*hc*hc*h) + element-wise (2*n*hc*h)
    total_flops = 2 * n * hc_mult * hc_mult * hidden_size + 2 * n * hc_mult * hidden_size

    # Reference
    post_layer_mix_3d = post_layer_mix.unsqueeze(-1)  # [n, hc, 1]

    def ref_fn():
        return ref_mhc_post(x, residual, post_layer_mix_3d, comb_res_mix)

    ref_latency = bench_ref(ref_fn)
    ref_tflops = total_flops / ref_latency * 1e-9 if ref_latency > 0 else 0

    print_benchmark_summary(
        "mHC Post",
        f"n={n}, hidden_size={hidden_size}, hc_mult={hc_mult}",
        best_latency, total_flops / best_latency * 1e-9,
        ref_latency, ref_tflops,
        "Reference", best_config,
    )

    return best_latency, total_flops / best_latency * 1e-9, best_config, ref_latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mHC Post Autotune Benchmark")
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--hidden_size", type=int, default=2560)
    parser.add_argument("--hc_mult", type=int, default=4)
    parser.add_argument("--profile", action="store_true",
                        help="Run kernel once with given config for ncu/acu profiling")
    parser.add_argument("--profile-ref", action="store_true",
                        help="Run reference once for ncu/acu profiling")
    parser.add_argument("--n_thr", type=int, default=None)
    parser.add_argument("--h_blk", type=int, default=None)
    args = parser.parse_args()

    if args.profile:
        run_profile(args.n, args.hidden_size, args.hc_mult,
                    args.n_thr, args.h_blk)
    elif args.profile_ref:
        run_profile_ref(args.n, args.hidden_size, args.hc_mult)
    else:
        main(args.n, args.hidden_size, args.hc_mult)
