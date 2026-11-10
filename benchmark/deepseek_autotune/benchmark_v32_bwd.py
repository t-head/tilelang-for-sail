"""DeepSeek V32 sparse MLA backward benchmark with autotune.

Adapted from: examples/deepseek_v32/sparse_mla_bwd.py
Tunable parameters: block_size, num_stages, threads

Only the main `bwd` kernel is tuned; the lightweight `preprocess` (Delta) and
`postprocess` (fp32->bf16 cast) wrappers from the example are not benchmarked.
Unlike the DSA variant (varlen flattened layout, exp domain), the V32 kernel
uses a batched 4D layout and base-2 exponentials (T.exp2 / log2-domain Lse).
"""

import argparse
import math

import torch
import tilelang
from tilelang import language as T
from tilelang.autotuner import autotune
from tilelang.autotuner.capture import set_autotune_inputs
from tilelang import jit

from configs import get_v32_bwd_configs
from utils import print_benchmark_summary, bench_ref, inject_pass_configs_from_env


@autotune(configs=get_v32_bwd_configs(), warmup=5, rep=20, skip_check=True)
@jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True,
    },
)
def sparse_mla_bwd(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_size=None,
    num_stages=None,
    threads=None,
):
    """Sparse MLA backward kernel.

    Tunable: block_size, num_stages, threads.
    """
    # Provide concrete defaults so the kernel can be elaborated (e.g. for
    # cache-key / validation TIR generation by the autotuner) before the
    # actual tunable values are supplied. These are overridden by autotune.
    block_size = block_size or 32
    num_stages = num_stages if num_stages is not None else 0

    assert is_causal is True, "non-casual is not supported now"
    assert topk % block_size == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"

    if sm_scale is None:
        sm_scale = (dim + tail_dim) ** (-0.5)
    sm_scale_mul_reciprocal_log2 = sm_scale * 1.44269504  # log2(e)

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    k_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    delta_shape = [batch, seq_len, heads]
    lse_shape = [batch, seq_len, heads]
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    block_H = min(64, padded_H)
    # adaptive: this kernel is memory-pipe bound. When the head count is not split across
    # blocks (block_H>=64), 256 threads (8 warps) issue many more cp.async copies at once
    # than 128 (4 warps), greatly raising in-flight bytes -- the main perf lever here.
    # Smaller head blocks lack the GEMM warp-tiling work to fill 256 threads, so fall back
    # to 128 and still build.
    if threads is None:
        threads = 256 if block_H >= 64 else 128
    assert padded_H % block_H == 0
    NH = padded_H // block_H
    BS = block_size
    NS = tilelang.cdiv(topk, block_size)

    split_store = 2

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(k_shape, dtype),
        dO: T.Tensor(o_shape, dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
        Delta: T.Tensor(delta_shape, accum_dtype),
        dQ: T.Tensor(q_shape, dtype),
        dKV: T.Tensor(k_shape, accum_dtype),
    ):
        with T.Kernel(seq_len, batch, kv_group * NH, threads=threads) as (s_i, by, bz):
            Q_shared = T.alloc_shared([block_H, dim + tail_dim], dtype)
            KV_shared = T.alloc_shared([BS, dim], dtype)
            KV_tail_shared = T.alloc_shared([BS, tail_dim], dtype)
            dO_shared = T.alloc_shared([block_H, dim], dtype)
            mask = T.alloc_fragment([BS], "bool")

            P_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dP_shared_cast = T.alloc_shared([block_H, BS], dtype)
            dQ_shared = T.alloc_shared([block_H, dim], dtype)
            dQ_tail_shared = T.alloc_shared([block_H, tail_dim], dtype)

            acc_p = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([block_H, BS], accum_dtype)
            acc_dq = T.alloc_fragment([block_H, dim], accum_dtype)
            acc_dq_tail = T.alloc_fragment([block_H, tail_dim], accum_dtype)
            acc_dkv = T.alloc_fragment([BS, dim], accum_dtype)
            acc_dkv_tail = T.alloc_fragment([BS, tail_dim], accum_dtype)
            acc_dkv_shared = T.alloc_shared([BS // split_store, dim], accum_dtype)
            acc_dkv_tail_shared = T.alloc_shared([BS // split_store, tail_dim], accum_dtype)

            max_kv_i = s_i

            # TODO: merge the statements when the compiler has better support for non-power-of-2 extents.
            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, :dim], Q_shared[:, :dim])
            T.copy(Q[by, s_i, bz * block_H : (bz + 1) * block_H, dim:], Q_shared[:, dim:])
            T.copy(dO[by, s_i, bz * block_H : (bz + 1) * block_H, :dim], dO_shared)

            T.clear(acc_dq)
            T.clear(acc_dq_tail)

            # Process each block of indices
            for i_i in T.Pipelined(NS, num_stages=num_stages):
                # Check which indices are valid
                for bi_i in T.Parallel(BS):
                    mask[bi_i] = Indices[by, s_i, bz // NH, i_i * BS + bi_i] <= max_kv_i

                # Compute attention scores
                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_p.dtype))

                # Load KV, V for this block of indices
                for bi_i, d_i in T.Parallel(BS, dim):
                    KV_shared[bi_i, d_i] = KV[by, Indices[by, s_i, bz // NH, i_i * BS + bi_i], bz // NH, d_i]

                T.gemm(Q_shared[:, :dim], KV_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                for bi_i, d_i in T.Parallel(BS, tail_dim):
                    KV_tail_shared[bi_i, d_i] = KV[by, Indices[by, s_i, bz // NH, i_i * BS + bi_i], bz // NH, dim + d_i]
                T.gemm(Q_shared[:, dim:], KV_tail_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_p[h_i, bi_i] = T.exp2(acc_p[h_i, bi_i] * sm_scale_mul_reciprocal_log2 - Lse[by, s_i, bz * block_H + h_i])

                T.copy(acc_p, P_shared_cast)

                T.gemm(dO_shared, KV_shared, acc_dp, transpose_B=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True)

                for h_i, bi_i in T.Parallel(block_H, BS):
                    acc_dp[h_i, bi_i] = acc_p[h_i, bi_i] * (acc_dp[h_i, bi_i] - Delta[by, s_i, bz * block_H + h_i]) * sm_scale

                T.copy(acc_dp, dP_shared_cast)
                T.gemm(dP_shared_cast, KV_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(dP_shared_cast, KV_tail_shared, acc_dq_tail, policy=T.GemmWarpPolicy.FullCol)

                T.gemm(dP_shared_cast, Q_shared[:, :dim], acc_dkv, transpose_A=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True)
                T.gemm(P_shared_cast, dO_shared, acc_dkv, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)

                T.clear(acc_dkv_tail)
                T.gemm(dP_shared_cast, Q_shared[:, dim:], acc_dkv_tail, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)

                for s in range(split_store):
                    for bi_i, d_i in T.Parallel(BS, dim):
                        if bi_i < BS // split_store:
                            acc_dkv_shared[bi_i, d_i] = acc_dkv[bi_i + s * (BS // split_store), d_i]

                    for bi_i, d_i in T.Parallel(BS, tail_dim):
                        if bi_i < BS // split_store:
                            acc_dkv_tail_shared[bi_i, d_i] = acc_dkv_tail[bi_i + s * (BS // split_store), d_i]

                    for bi_i, d_i in T.Parallel(BS // split_store, dim // 4):
                        T.atomic_addx4(
                            dKV[by, Indices[by, s_i, bz // NH, i_i * BS + bi_i + s * (BS // split_store)], bz // NH, d_i * 4],
                            acc_dkv_shared[bi_i, d_i * 4],
                        )

                    # Atomically update dKV, dKV_tail tensors
                    for bi_i, d_i in T.Parallel(BS // split_store, tail_dim // 4):
                        T.atomic_addx4(
                            dKV[by, Indices[by, s_i, bz // NH, i_i * BS + bi_i + s * (BS // split_store)], bz // NH, dim + d_i * 4],
                            acc_dkv_tail_shared[bi_i, d_i * 4],
                        )

            # Store the accumulated dQ
            T.copy(acc_dq, dQ_shared)
            T.copy(acc_dq_tail, dQ_tail_shared)

            T.copy(dQ_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, :dim])
            T.copy(dQ_tail_shared, dQ[by, s_i, bz * block_H : (bz + 1) * block_H, dim:])

    return main


# ---------------------------------------------------------------------------
# Input construction (mirrors examples/deepseek_v32/sparse_mla_bwd.py::test_sparse_mla_bwd)
# ---------------------------------------------------------------------------


def _build_bench_inputs(batch, seq_len, seq_len_kv, heads, kv_group, topk, dim, tail_dim):
    """Build benchmark inputs for the V32 sparse MLA bwd kernel.

    The example derives Lse/Delta from a full forward pass; here they are
    synthesized with bounded values instead because the bwd kernel's latency
    does not depend on them and bounded values keep exp2() free of inf/NaN.
    The V32 kernel works in the log2 domain (T.exp2 / log2-domain Lse), so the
    synthetic Lse is centered around log2(topk).
    """
    device = "cuda"
    dtype = torch.bfloat16
    dim_qk = dim + tail_dim

    q = torch.randn(batch, seq_len, heads, dim_qk, dtype=dtype, device=device)
    kv = torch.randn(batch, seq_len_kv, kv_group, dim_qk, dtype=dtype, device=device)
    do = torch.randn(batch, seq_len, heads, dim, dtype=dtype, device=device)

    # Per token, topk distinct key positions sampled over the whole KV range so
    # every gather is in-bounds; the causal mask (idx <= position) inside the
    # kernel selects the valid ones, matching real topk+causal behavior.
    indices = torch.zeros(batch, seq_len, kv_group, topk, dtype=torch.int32, device=device)
    for b in range(batch):
        for g in range(kv_group):
            perm = torch.argsort(torch.rand(seq_len, seq_len_kv, device=device), dim=1)[:, :topk].to(torch.int32)
            indices[b, :, g, :] = perm

    lse = torch.rand(batch, seq_len, heads, dtype=torch.float32, device=device) * 0.5 + math.log2(topk)
    delta = torch.rand(batch, seq_len, heads, dtype=torch.float32, device=device) * 0.1
    return q, kv, do, indices, lse, delta


# ---------------------------------------------------------------------------
# Reference (torch autograd, mirrors ref_sparse_mla_bwd_interface)
# ---------------------------------------------------------------------------


def ref_sparse_mla_bwd(q, kv, do, indices, dim, sm_scale=None):
    """Reference sparse MLA backward via torch autograd on a chunked fwd.

    Differentiating the softmax attention forward reproduces exactly the
    dP = p * (dO.V - Delta) * scale identity used by the TileLang kernel.
    """
    B, S, H, dim_qk = q.shape
    S_kv = kv.shape[1]
    kv_group = kv.shape[2]
    topk = indices.shape[-1]
    if sm_scale is None:
        sm_scale = dim_qk**-0.5

    q_f = q.detach().float().clone().requires_grad_(True)
    kv_f = kv.detach().float().clone().requires_grad_(True)
    do_f = do.detach().float()

    # Keep the gathered [chunk, G, topk, dim_qk] buffer bounded (~150MB).
    chunk = max(1, 65536 // topk)

    o_parts = []
    for b in range(B):
        kv_seq = kv_f[b]  # [S_kv, G, dim_qk]
        for start in range(0, S, chunk):
            end = min(start + chunk, S)
            c = end - start
            idx = indices[b, start:end]  # [c, G, topk]
            kv_g = torch.stack([kv_seq[:, g, :][idx[:, g, :].clamp(max=S_kv - 1)] for g in range(kv_group)], dim=1)  # [c, G, topk, dim_qk]

            q_c = q_f[b, start:end].view(c, kv_group, H // kv_group, dim_qk)
            scores = torch.einsum("cghd,cgnd->cghn", q_c, kv_g)  # [c, G, h_per_g, topk]

            pos = torch.arange(start, end, device=q.device).view(-1, 1, 1)
            valid = idx <= pos  # [c, G, topk]; broadcast over the head dim below
            scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf")) * sm_scale

            p = scores.softmax(dim=-1)
            v_g = kv_g[..., :dim]
            o = torch.einsum("cghn,cgnd->cghd", p, v_g).reshape(c, H, dim)
            o_parts.append(o)

    o = torch.cat(o_parts).view(B, S, H, dim)
    o.backward(do_f)
    return q_f.grad, kv_f.grad


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def v32_bwd_flops(batch, seq_len, heads, topk, dim, tail_dim):
    """FLOPs for the sparse MLA bwd (same gemm count as the example)."""
    per_token_flop = 2 * sum(
        [
            heads * dim * topk,
            heads * (dim + tail_dim) * topk,
            heads * (dim + tail_dim) * topk,
            heads * (dim + tail_dim) * topk,
            heads * dim * topk,
        ]
    )
    return batch * seq_len * per_token_flop


def run_profile(batch, seq_len, seq_len_kv, heads, kv_group, topk, dim, tail_dim, block_size, num_stages, threads):
    """Run the kernel once with an explicit config for ncu/acu profiling."""
    q, kv, do, indices, lse, delta = _build_bench_inputs(batch, seq_len, seq_len_kv, heads, kv_group, topk, dim, tail_dim)
    inject_pass_configs_from_env(sparse_mla_bwd)
    with set_autotune_inputs(q, kv, do, indices, lse, delta):
        kernel = sparse_mla_bwd(
            heads,
            dim,
            tail_dim,
            topk,
            kv_group,
            is_causal=True,
            block_size=block_size,
            num_stages=num_stages,
            threads=threads,
        )
    kernel(q, kv, do, indices, lse, delta)


def run_profile_ref(batch, seq_len, seq_len_kv, heads, kv_group, topk, dim, tail_dim):
    """Run the reference once for ncu/acu profiling."""
    q, kv, do, indices, lse, delta = _build_bench_inputs(batch, seq_len, seq_len_kv, heads, kv_group, topk, dim, tail_dim)
    ref_sparse_mla_bwd(q, kv, do, indices, dim)


def main(batch=1, seq_len=4096, seq_len_kv=8192, heads=64, kv_group=1, topk=2048, dim=512, tail_dim=64):
    """Run autotune and print results."""
    q, kv, do, indices, lse, delta = _build_bench_inputs(batch, seq_len, seq_len_kv, heads, kv_group, topk, dim, tail_dim)

    with set_autotune_inputs(q, kv, do, indices, lse, delta):
        best_result = sparse_mla_bwd(heads, dim, tail_dim, topk, kv_group, is_causal=True)

    best_latency = best_result.latency
    best_config = best_result.config

    total_flops = v32_bwd_flops(batch, seq_len, heads, topk, dim, tail_dim)

    def ref_fn():
        return ref_sparse_mla_bwd(q, kv, do, indices, dim)

    ref_latency = bench_ref(ref_fn)
    ref_tflops = total_flops / ref_latency * 1e-9 if ref_latency > 0 else 0

    print_benchmark_summary(
        "V32 Sparse MLA Bwd",
        f"batch={batch}, seq={seq_len}, seq_kv={seq_len_kv}, heads={heads}, topk={topk}",
        best_latency,
        total_flops / best_latency * 1e-9,
        ref_latency,
        ref_tflops,
        "Reference",
        best_config,
    )

    return best_latency, total_flops / best_latency * 1e-9, best_config, ref_latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V32 Sparse MLA Bwd Autotune Benchmark")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--seq_len_kv", type=int, default=8192)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--kv_group", type=int, default=1)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--tail_dim", type=int, default=64)
    parser.add_argument("--profile", action="store_true", help="Run kernel once with given config for ncu/acu profiling")
    parser.add_argument("--profile-ref", action="store_true", help="Run reference once for ncu/acu profiling")
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--num_stages", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()

    if args.profile:
        run_profile(
            args.batch,
            args.seq_len,
            args.seq_len_kv,
            args.heads,
            args.kv_group,
            args.topk,
            args.dim,
            args.tail_dim,
            args.block_size,
            args.num_stages,
            args.threads,
        )
    elif args.profile_ref:
        run_profile_ref(args.batch, args.seq_len, args.seq_len_kv, args.heads, args.kv_group, args.topk, args.dim, args.tail_dim)
    else:
        main(args.batch, args.seq_len, args.seq_len_kv, args.heads, args.kv_group, args.topk, args.dim, args.tail_dim)
