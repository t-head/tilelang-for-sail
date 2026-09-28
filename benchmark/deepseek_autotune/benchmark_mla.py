"""DeepSeek MLA Decode benchmark with autotune.

Adapted from: examples/deepseek_mla/example_mla_decode.py
Tunable parameters: block_N, block_H, num_stages, threads
"""

import argparse

import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune
from tilelang import jit

from configs import get_mla_configs
from utils import mla_decode_flops, print_benchmark_summary, bench_ref, inject_pass_configs_from_env


@autotune(configs=get_mla_configs(), warmup=5, rep=20, skip_check=True)
@jit(
    out_idx=[6],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def mla_decode(
    batch,
    heads,
    kv_head_num,
    seqlen_kv,
    dim,
    pe_dim,
    num_split,
    softmax_scale,
    block_N=None,
    block_H=None,
    num_stages=None,
    threads=None,
):
    """MLA decode kernel with online softmax.

    Tunable: block_N, block_H, num_stages, threads.
    """
    # Provide concrete defaults so the kernel can be elaborated (e.g. for
    # cache-key / validation TIR generation by the autotuner) before the
    # actual tunable values are supplied. These are overridden by autotune.
    block_N = block_N or 32
    block_H = block_H or 16
    num_stages = num_stages if num_stages is not None else 1
    threads = threads or 128

    scale = float(softmax_scale * 1.44269504)  # log2(e)
    dtype = T.float16
    accum_dtype = T.float32
    kv_group_num = heads // kv_head_num
    VALID_BLOCK_H = min(block_H, kv_group_num)
    assert kv_head_num == 1, "kv_head_num must be 1"

    @T.prim_func
    def main(
        Q: T.Tensor([batch, heads, dim], dtype),
        Q_pe: T.Tensor([batch, heads, pe_dim], dtype),
        KV: T.Tensor([batch, seqlen_kv, kv_head_num, dim], dtype),
        K_pe: T.Tensor([batch, seqlen_kv, kv_head_num, pe_dim], dtype),
        glse: T.Tensor([batch, heads, num_split], dtype),
        Output_partial: T.Tensor([batch, heads, num_split, dim], dtype),
        Output: T.Tensor([batch, heads, dim], dtype),
    ):
        with T.Kernel(heads // min(block_H, kv_group_num), batch, threads=threads) as (hid, bid):
            Q_shared = T.alloc_shared([block_H, dim], dtype)
            S_shared = T.alloc_shared([block_H, block_N], dtype)
            Q_pe_shared = T.alloc_shared([block_H, pe_dim], dtype)
            KV_shared = T.alloc_shared([block_N, dim], dtype)
            K_pe_shared = T.alloc_shared([block_N, pe_dim], dtype)
            O_shared = T.alloc_shared([block_H, dim], dtype)
            acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
            acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_H], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
            scores_scale = T.alloc_fragment([block_H], accum_dtype)
            scores_sum = T.alloc_fragment([block_H], accum_dtype)
            logsum = T.alloc_fragment([block_H], accum_dtype)

            cur_kv_head = hid // (kv_group_num // block_H)

            T.copy(Q[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, :], Q_shared)
            T.copy(Q_pe[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, :], Q_pe_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = T.ceildiv(seqlen_kv, block_N)
            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(KV[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], KV_shared)
                T.copy(K_pe[bid, k * block_N : (k + 1) * block_N, cur_kv_head, :], K_pe_shared)
                T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True)
                T.gemm(Q_pe_shared, K_pe_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)
                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_H):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_H):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_H, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                T.copy(acc_s, S_shared)
                for i in T.Parallel(block_H):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] *= scores_scale[i]
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)
            for i, j in T.Parallel(block_H, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bid, hid * VALID_BLOCK_H : (hid + 1) * VALID_BLOCK_H, :])

    return main


def run_profile(batch, heads, kv_heads, kv_ctx, dim, pe_dim, block_N, block_H, num_stages, threads):
    """Run the kernel once with an explicit config for ncu/acu profiling.

    When all tunable params are provided the @autotune decorator skips the
    search and the @jit decorator compiles a single kernel, allowing the
    profiler to capture exactly one launch.
    """
    import torch

    num_split = 1
    softmax_scale = (dim + pe_dim) ** -0.5
    dtype = torch.float16

    Q = torch.randn(batch, heads, dim, dtype=dtype, device="cuda")
    Q_pe = torch.randn(batch, heads, pe_dim, dtype=dtype, device="cuda")
    KV = torch.randn(batch, kv_ctx, kv_heads, dim, dtype=dtype, device="cuda")
    K_pe = torch.randn(batch, kv_ctx, kv_heads, pe_dim, dtype=dtype, device="cuda")
    glse = torch.zeros(batch, heads, num_split, dtype=dtype, device="cuda")
    Output_partial = torch.zeros(batch, heads, num_split, dim, dtype=dtype, device="cuda")

    inject_pass_configs_from_env(mla_decode)
    kernel = mla_decode(
        batch,
        heads,
        kv_heads,
        kv_ctx,
        dim,
        pe_dim,
        num_split,
        softmax_scale,
        block_N=block_N,
        block_H=block_H,
        num_stages=num_stages,
        threads=threads,
    )
    kernel(Q, Q_pe, KV, K_pe, glse, Output_partial)


def run_profile_ref(batch, heads, kv_heads, kv_ctx, dim, pe_dim):
    """Run the reference once for ncu/acu profiling."""
    import torch
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata

    dtype = torch.float16
    d = dim + pe_dim
    block_size = 64
    max_seqlen_pad = (kv_ctx + 255) // 256 * 256
    q_fmla = torch.randn(batch, 1, heads, d, dtype=dtype, device="cuda")
    block_table = torch.arange(batch * max_seqlen_pad // block_size, dtype=torch.int32, device="cuda").view(
        batch, max_seqlen_pad // block_size
    )
    blocked_k = torch.randn(block_table.numel(), block_size, kv_heads, d, dtype=dtype, device="cuda")
    cache_seqlens = torch.full((batch,), kv_ctx, dtype=torch.int32, device="cuda")
    tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, 1 * heads // kv_heads, kv_heads)

    flash_mla_with_kvcache(
        q_fmla,
        blocked_k,
        block_table,
        cache_seqlens,
        dim,
        tile_scheduler_metadata,
        num_splits,
        causal=True,
    )


def main(batch=132, heads=128, kv_heads=1, kv_ctx=8192, dim=512, pe_dim=64):
    """Run autotune and print results."""
    import torch

    num_split = 1
    softmax_scale = (dim + pe_dim) ** -0.5
    total_flops = mla_decode_flops(batch, heads, kv_ctx, dim, pe_dim)

    best_result = mla_decode(batch, heads, kv_heads, kv_ctx, dim, pe_dim, num_split, softmax_scale)
    best_latency = best_result.latency
    best_config = best_result.config

    # Reference: DeepSeek FlashMLA (flash_mla_with_kvcache), invoked the same way
    # as examples/deepseek_mla/benchmark_mla.py::run_flash_mla. It expects a paged
    # KV cache, so we build a block_table + blocked_k from the same problem size.
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata

    dtype = torch.float16
    d = dim + pe_dim  # full MLA head dim (nope + rope), must be 576 for flash_mla
    block_size = 64
    max_seqlen_pad = (kv_ctx + 255) // 256 * 256
    q_fmla = torch.randn(batch, 1, heads, d, dtype=dtype, device="cuda")
    block_table = torch.arange(batch * max_seqlen_pad // block_size, dtype=torch.int32, device="cuda").view(
        batch, max_seqlen_pad // block_size
    )
    blocked_k = torch.randn(block_table.numel(), block_size, kv_heads, d, dtype=dtype, device="cuda")
    cache_seqlens = torch.full((batch,), kv_ctx, dtype=torch.int32, device="cuda")
    tile_scheduler_metadata, num_splits = get_mla_metadata(cache_seqlens, 1 * heads // kv_heads, kv_heads)

    def ref_flash_mla():
        return flash_mla_with_kvcache(
            q_fmla,
            blocked_k,
            block_table,
            cache_seqlens,
            dim,
            tile_scheduler_metadata,
            num_splits,
            causal=True,
        )

    ref_latency = bench_ref(ref_flash_mla)
    ref_tflops = total_flops / ref_latency * 1e-9 if ref_latency > 0 else 0

    print_benchmark_summary(
        "MLA Decode",
        f"batch={batch}, heads={heads}, kv_ctx={kv_ctx}, dim={dim}, pe_dim={pe_dim}",
        best_latency,
        total_flops / best_latency * 1e-9,
        ref_latency,
        ref_tflops,
        "FlashMLA",
        best_config,
    )

    return best_latency, total_flops / best_latency * 1e-9, best_config, ref_latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLA Decode Autotune Benchmark")
    parser.add_argument("--batch", type=int, default=132)
    parser.add_argument("--heads", type=int, default=128)
    parser.add_argument("--kv_heads", type=int, default=1)
    parser.add_argument("--kv_ctx", type=int, default=8192)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--pe_dim", type=int, default=64)
    parser.add_argument("--profile", action="store_true", help="Run kernel once with given config for ncu/acu profiling")
    parser.add_argument("--profile-ref", action="store_true", help="Run reference once for ncu/acu profiling")
    parser.add_argument("--block_N", type=int, default=None)
    parser.add_argument("--block_H", type=int, default=None)
    parser.add_argument("--num_stages", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    if args.profile:
        run_profile(
            args.batch,
            args.heads,
            args.kv_heads,
            args.kv_ctx,
            args.dim,
            args.pe_dim,
            args.block_N,
            args.block_H,
            args.num_stages,
            args.threads,
        )
    elif args.profile_ref:
        run_profile_ref(args.batch, args.heads, args.kv_heads, args.kv_ctx, args.dim, args.pe_dim)
    else:
        main(args.batch, args.heads, args.kv_heads, args.kv_ctx, args.dim, args.pe_dim)
