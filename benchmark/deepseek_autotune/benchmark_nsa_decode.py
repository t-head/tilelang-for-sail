"""DeepSeek NSA (Native Sparse Attention) decode benchmark with autotune.

Adapted from: examples/deepseek_nsa/example_tilelang_nsa_decode.py
The NSA decode kernel performs sparse attention with seq_len_q=1 (inference mode).

Tunable parameters: num_stages, threads
"""

import argparse

import torch
import tilelang
from tilelang import language as T
from tilelang.autotuner import autotune
from tilelang import jit

from configs import get_nsa_decode_configs
from utils import nsa_flops, print_benchmark_summary, bench_ref, inject_pass_configs_from_env


@autotune(configs=get_nsa_decode_configs(), warmup=5, rep=20, skip_check=True)
@jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def nsa_decode(
    batch,
    heads,
    seq_len,  # K/V context length
    dim,
    scale=None,
    block_size=64,
    groups=1,
    selected_blocks=16,
    num_stages=None,
    threads=None,
):
    """NSA decode kernel (seq_len_q=1 inference).

    Tunable: num_stages, threads.
    """
    # Provide concrete defaults so the kernel can be elaborated (e.g. for
    # cache-key / validation TIR generation by the autotuner) before the
    # actual tunable values are supplied. These are overridden by autotune.
    num_stages = num_stages if num_stages is not None else 0
    threads = threads or 32

    if scale is None:
        scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    else:
        scale = scale * 1.44269504

    head_kv = heads // groups
    q_shape = [batch, 1, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    block_indices_shape = [batch, 1, head_kv, selected_blocks]
    block_indices_dtype = T.int32
    dtype = T.float16
    accum_dtype = T.float32
    block_S = block_size
    block_T = min(128, tilelang.math.next_power_of_2(dim))

    NK = tilelang.cdiv(dim, block_T)
    NV = tilelang.cdiv(dim, block_T)
    assert NK == 1, "The key dimension can not be larger than 256"

    S = selected_blocks
    G = groups
    BS = block_S
    BK = BV = block_T

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        BlockIndices: T.Tensor(block_indices_shape, block_indices_dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(1, NV, batch * head_kv, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([G, BK], dtype)
            K_shared = T.alloc_shared([BS, BK], dtype)
            V_shared = T.alloc_shared([BS, BV], dtype)
            O_shared = T.alloc_shared([G, BV], dtype)

            acc_s = T.alloc_fragment([G, BS], accum_dtype)
            acc_s_cast = T.alloc_fragment([G, BS], dtype)
            acc_o = T.alloc_fragment([G, BV], accum_dtype)
            scores_max = T.alloc_fragment([G], accum_dtype)
            scores_max_prev = T.alloc_fragment([G], accum_dtype)
            scores_scale = T.alloc_fragment([G], accum_dtype)
            scores_sum = T.alloc_fragment([G], accum_dtype)
            logsum = T.alloc_fragment([G], accum_dtype)

            i_v, i_bh = by, bz
            i_b, i_h = i_bh // head_kv, i_bh % head_kv

            NS = S
            T.copy(Q[i_b, 0, i_h * G : (i_h + 1) * G, :], Q_shared)

            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            for i in T.Pipelined(NS, num_stages=num_stages):
                i_s = BlockIndices[i_b, 0, i_h, i] * BS
                if i_s >= 0:
                    T.copy(K[i_b, i_s : i_s + BS, i_h, :], K_shared)

                    T.clear(acc_s)
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=True)

                    for i in T.Parallel(G):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(G, BS):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(G):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)

                    for i, j in T.Parallel(G, BV):
                        acc_o[i, j] *= scores_scale[i]

                    T.copy(V[i_b, i_s : i_s + BS, i_h, i_v * BV : (i_v + 1) * BV], V_shared)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(G, BV):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[i_b, 0, i_h * G : (i_h + 1) * G, i_v * BV : (i_v + 1) * BV])

    return main


def run_profile(batch, heads, seq_len, dim, selected_blocks, block_size, num_stages, threads):
    """Run the kernel once with an explicit config for ncu/acu profiling."""
    import torch

    groups = heads
    scale = 0.1
    dtype = torch.float16
    head_kv = heads // groups

    Q = torch.randn(batch, 1, heads, dim, dtype=dtype, device="cuda")
    K = torch.randn(batch, seq_len, head_kv, dim, dtype=dtype, device="cuda")
    V = torch.randn(batch, seq_len, head_kv, dim, dtype=dtype, device="cuda")
    BlockIndices = torch.randint(0, max(1, seq_len // block_size), (batch, 1, head_kv, selected_blocks), dtype=torch.int32, device="cuda")

    inject_pass_configs_from_env(nsa_decode)
    kernel = nsa_decode(
        batch,
        heads,
        seq_len,
        dim,
        scale=scale,
        block_size=block_size,
        groups=groups,
        selected_blocks=selected_blocks,
        num_stages=num_stages,
        threads=threads,
    )
    kernel(Q, K, V, BlockIndices)


def run_profile_ref(batch, heads, seq_len, dim, selected_blocks, block_size):
    """Run the reference once for ncu/acu profiling."""
    import torch

    scale = 0.1
    total_kv = selected_blocks * block_size
    Q = torch.randn(batch, 1, heads, dim, dtype=torch.float16, device="cuda")
    K_flat = torch.randn(batch, heads, total_kv, dim, dtype=torch.float16, device="cuda")
    V_flat = torch.randn(batch, heads, total_kv, dim, dtype=torch.float16, device="cuda")

    q = Q.permute(0, 2, 1, 3).float()
    scores = torch.einsum("bhqd,bhtd->bhqt", q, K_flat.float()) * scale
    attn = scores.softmax(dim=-1)
    torch.einsum("bhqt,bhtd->bhqd", attn, V_flat.float()).half()


def main(batch=2, heads=16, seq_len=64, dim=32, selected_blocks=1, block_size=32):
    """Run autotune and print results."""

    groups = heads  # 1 kv head -> groups = heads
    scale = 0.1
    # FLOPs for decode: seq_len_q=1
    total_flops = nsa_flops(batch, heads, 1, dim, selected_blocks, block_size)

    best_result = nsa_decode(
        batch,
        heads,
        seq_len,
        dim,
        scale=scale,
        block_size=block_size,
        groups=groups,
        selected_blocks=selected_blocks,
    )
    best_latency = best_result.latency
    best_config = best_result.config

    # Reference: naive attention over selected blocks (seq_len_q=1)
    total_kv = selected_blocks * block_size
    Q = torch.randn(batch, 1, heads, dim, dtype=torch.float16, device="cuda")
    K_flat = torch.randn(batch, heads, total_kv, dim, dtype=torch.float16, device="cuda")
    V_flat = torch.randn(batch, heads, total_kv, dim, dtype=torch.float16, device="cuda")

    def ref_nsa_decode():
        # Q: [B, 1, H, D] -> [B, H, 1, D]
        q = Q.permute(0, 2, 1, 3).float()
        scores = torch.einsum("bhqd,bhtd->bhqt", q, K_flat.float()) * scale
        attn = scores.softmax(dim=-1)
        return torch.einsum("bhqt,bhtd->bhqd", attn, V_flat.float()).half()

    ref_latency = bench_ref(ref_nsa_decode)
    ref_tflops = total_flops / ref_latency * 1e-9 if ref_latency > 0 else 0

    print_benchmark_summary(
        "NSA Decode",
        f"batch={batch}, heads={heads}, seq_len={seq_len}, dim={dim}",
        best_latency,
        total_flops / best_latency * 1e-9,
        ref_latency,
        ref_tflops,
        "Reference",
        best_config,
    )

    return best_latency, total_flops / best_latency * 1e-9, best_config, ref_latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSA Decode Autotune Benchmark")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--selected_blocks", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--profile", action="store_true", help="Run kernel once with given config for ncu/acu profiling")
    parser.add_argument("--profile-ref", action="store_true", help="Run reference once for ncu/acu profiling")
    parser.add_argument("--num_stages", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    if args.profile:
        run_profile(args.batch, args.heads, args.seq_len, args.dim, args.selected_blocks, args.block_size, args.num_stages, args.threads)
    elif args.profile_ref:
        run_profile_ref(args.batch, args.heads, args.seq_len, args.dim, args.selected_blocks, args.block_size)
    else:
        main(args.batch, args.heads, args.seq_len, args.dim, args.selected_blocks, args.block_size)
