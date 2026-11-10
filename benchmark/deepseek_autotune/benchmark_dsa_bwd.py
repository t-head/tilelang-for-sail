"""DSA sparse MLA backward benchmark with autotune.

Adapted from: examples/dsa_sparse_finetune/sparse_mla_bwd_autotune.py (the eager
autotune version whose kernel structure was validated as the PPU baseline by the
2026-11-04 five-variant investigation; acc_dkv relay buffers are independently
allocated shared memory, not views over KV_shared).
Tunable parameters: block_size, num_stages, threads

Only the main `bwd` computation is tuned; the lightweight `preprocess` (Delta) and
`postprocess` (fp32->bf16 cast) wrappers from the example are not benchmarked.
The prim_func embeds a small dKV zero-init launch (same as the example) so the
jit kernel is self-contained and its outputs are always fully defined.

Every tuned config is validated against the chunked torch-autograd reference
through the autotuner's ref_prog / manual_check_prog hooks; to make that check
meaningful, Lse/Delta inputs carry the true forward statistics (derived once
from the same chunked forward) instead of synthesized values.
"""

import argparse

import torch
import tilelang
from tilelang import language as T
from tilelang.autotuner import autotune
from tilelang.autotuner.capture import set_autotune_inputs
from tilelang import jit

from configs import get_dsa_bwd_configs
from utils import print_benchmark_summary, bench_ref, inject_pass_configs_from_env, calc_diff


# ---------------------------------------------------------------------------
# Autotune correctness check (ref_prog / manual_check_prog)
# ---------------------------------------------------------------------------


def ref_bwd_for_autotune(q, kv, do, indices, lse, delta, offsets, token_indices):
    """Reference program for the autotuner's per-config correctness check.

    Receives exactly the tensors bound via set_autotune_inputs. lse/delta are
    recomputed inside the autograd reference; the comparison is consistent
    because _build_bench_inputs derives them from the same chunked forward.
    """
    return ref_sparse_mla_bwd(q, kv, do, indices, offsets, do.shape[-1])


def check_bwd_outputs(lib_outs, ref_outs):
    """manual_check_prog: global cosine-distance comparison of (dQ, dKV)."""
    dq_diff = calc_diff(lib_outs[0], ref_outs[0])
    dkv_diff = calc_diff(lib_outs[1], ref_outs[1])
    assert dq_diff < 1e-4, f"autotune dQ check failed: diff={dq_diff:.3e}"
    assert dkv_diff < 1e-4, f"autotune dKV check failed: diff={dkv_diff:.3e}"


@autotune(
    configs=get_dsa_bwd_configs(),
    warmup=5,
    rep=20,
    ref_prog=ref_bwd_for_autotune,
    manual_check_prog=check_bwd_outputs,
    # Note: cache_input_tensors is intentionally NOT set here.  The benchmark
    # regenerates supply inputs per trial (unlike the example which builds one
    # input set and reuses it), so caching triggers "Incompatible input tensor
    # properties" errors that silently skip ~1/3 of the configs.
    early_stop=True,
    early_stop_factor=1.5,
)
@jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
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
    threads = threads or 128

    assert is_causal is True, "non-casual is not supported now"
    assert topk % block_size == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"

    if sm_scale is None:
        sm_scale = (dim + tail_dim) ** (-0.5)

    S = T.dynamic("S")
    B_plus_one = T.dynamic("B_plus_one")

    H_kv = heads // kv_group
    q_shape = [S, heads, dim + tail_dim]
    k_shape = [S, kv_group, dim + tail_dim]
    o_shape = [S, heads, dim]
    indices_shape = [S, kv_group, topk]
    delta_shape = [S, heads]
    lse_shape = [S, heads]
    offsets_shape = [B_plus_one]
    token_indices_shape = [S, 2]
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    padded_H = max(tilelang.math.next_power_of_2(H_kv), 16)
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
        Offsets: T.Tensor(offsets_shape, indices_dtype),
        TokenIndices: T.Tensor(token_indices_shape, indices_dtype),
        dQ: T.Tensor(q_shape, dtype),
        dKV: T.Tensor(k_shape, accum_dtype),
    ):
        # Zero-init dKV: it is only written back via atomic_add on the rows
        # sampled by some token's topk indices, so rows never covered by any
        # token would otherwise keep uninitialized memory.
        with T.Kernel(T.ceildiv(S, BS), kv_group, threads=threads) as (b_s_i, bz):
            for s_i, d_i in T.Parallel(BS, dim + tail_dim):
                if b_s_i * BS + s_i < S:
                    dKV[b_s_i * BS + s_i, bz, d_i] = 0

        with T.Kernel(S, kv_group, threads=threads) as (b_s_i, bz):
            Q_shared = T.alloc_shared([padded_H, dim], dtype)
            Q_tail_shared = T.alloc_shared([padded_H, tail_dim], dtype)
            KV_shared = T.alloc_shared([BS, dim], dtype)
            KV_tail_shared = T.alloc_shared([BS, tail_dim], dtype)
            dO_shared = T.alloc_shared([padded_H, dim], dtype)
            mask = T.alloc_fragment([BS], "bool")

            P_shared_cast = T.alloc_shared([padded_H, BS], dtype)
            dP_shared_cast = T.alloc_shared([padded_H, BS], dtype)
            dQ_shared = T.alloc_shared([padded_H, dim], dtype)
            dQ_tail_shared = T.alloc_shared([padded_H, tail_dim], dtype)

            acc_p = T.alloc_fragment([padded_H, BS], accum_dtype)
            acc_dp = T.alloc_fragment([padded_H, BS], accum_dtype)
            acc_dq = T.alloc_fragment([padded_H, dim], accum_dtype)
            acc_dq_tail = T.alloc_fragment([padded_H, tail_dim], accum_dtype)
            acc_dkv = T.alloc_fragment([BS, dim], accum_dtype)
            acc_dkv_tail = T.alloc_fragment([BS, tail_dim], accum_dtype)
            acc_dkv_shared = T.alloc_shared([BS // split_store, dim], accum_dtype)
            acc_dkv_tail_shared = T.alloc_shared([BS // split_store, tail_dim], accum_dtype)

            b_i, s_i = TokenIndices[b_s_i, 0], TokenIndices[b_s_i, 1]
            bos, eos = Offsets[b_i], Offsets[b_i + 1]

            max_kv_i = s_i

            T.copy(Q[bos + s_i, bz * padded_H : (bz + 1) * padded_H, :dim], Q_shared)
            T.copy(Q[bos + s_i, bz * padded_H : (bz + 1) * padded_H, dim:], Q_tail_shared)
            T.copy(dO[bos + s_i, bz * padded_H : (bz + 1) * padded_H, :dim], dO_shared)

            T.clear(acc_dq)
            T.clear(acc_dq_tail)

            # Process each block of indices
            for i_i in T.Pipelined(NS, num_stages=num_stages):
                # Check which indices are valid
                for bi_i in T.Parallel(BS):
                    mask[bi_i] = (Indices[bos + s_i, bz, i_i * BS + bi_i] <= max_kv_i) & (Indices[bos + s_i, bz, i_i * BS + bi_i] != -1)

                # Compute attention scores
                for h_i, bi_i in T.Parallel(padded_H, BS):
                    acc_p[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_p.dtype))

                # Load KV, V for this block of indices
                for bi_i, d_i in T.Parallel(BS, dim):
                    KV_shared[bi_i, d_i] = KV[bos + Indices[bos + s_i, bz, i_i * BS + bi_i], bz, d_i]

                T.gemm(Q_shared, KV_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                for bi_i, d_i in T.Parallel(BS, tail_dim):
                    KV_tail_shared[bi_i, d_i] = KV[bos + Indices[bos + s_i, bz, i_i * BS + bi_i], bz, dim + d_i]
                T.gemm(Q_tail_shared, KV_tail_shared, acc_p, transpose_B=True, policy=T.GemmWarpPolicy.FullCol)

                for h_i, bi_i in T.Parallel(padded_H, BS):
                    acc_p[h_i, bi_i] = T.exp(acc_p[h_i, bi_i] * sm_scale - Lse[bos + s_i, bz * padded_H + h_i])

                T.copy(acc_p, P_shared_cast)

                T.gemm(dO_shared, KV_shared, acc_dp, transpose_B=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True)

                for h_i, bi_i in T.Parallel(padded_H, BS):
                    acc_dp[h_i, bi_i] = acc_p[h_i, bi_i] * (acc_dp[h_i, bi_i] - Delta[bos + s_i, bz * padded_H + h_i]) * sm_scale

                T.copy(acc_dp, dP_shared_cast)
                T.gemm(dP_shared_cast, KV_shared, acc_dq, policy=T.GemmWarpPolicy.FullCol)
                T.gemm(dP_shared_cast, KV_tail_shared, acc_dq_tail, policy=T.GemmWarpPolicy.FullCol)

                T.gemm(dP_shared_cast, Q_shared, acc_dkv, transpose_A=True, policy=T.GemmWarpPolicy.FullCol, clear_accum=True)
                T.gemm(P_shared_cast, dO_shared, acc_dkv, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)

                T.clear(acc_dkv_tail)
                T.gemm(dP_shared_cast, Q_tail_shared, acc_dkv_tail, transpose_A=True, policy=T.GemmWarpPolicy.FullCol)

                for s in range(split_store):
                    for bi_i, d_i in T.Parallel(BS, dim):
                        if bi_i < BS // split_store:
                            acc_dkv_shared[bi_i, d_i] = acc_dkv[bi_i + s * (BS // split_store), d_i]

                    for bi_i, d_i in T.Parallel(BS, tail_dim):
                        if bi_i < BS // split_store:
                            acc_dkv_tail_shared[bi_i, d_i] = acc_dkv_tail[bi_i + s * (BS // split_store), d_i]

                    for bi_i, d_i in T.Parallel(BS // split_store, dim):
                        T.atomic_add(
                            dKV[bos + Indices[bos + s_i, bz, i_i * BS + bi_i + s * (BS // split_store)], bz, d_i],
                            acc_dkv_shared[bi_i, d_i],
                        )

                    # Atomically update dKV, dKV_tail tensors
                    for bi_i, d_i in T.Parallel(BS // split_store, tail_dim):
                        T.atomic_add(
                            dKV[bos + Indices[bos + s_i, bz, i_i * BS + bi_i + s * (BS // split_store)], bz, dim + d_i],
                            acc_dkv_tail_shared[bi_i, d_i],
                        )

            # Store the accumulated dQ
            T.copy(acc_dq, dQ_shared)
            T.copy(acc_dq_tail, dQ_tail_shared)

            T.copy(dQ_shared, dQ[bos + s_i, bz * padded_H : (bz + 1) * padded_H, :dim])
            T.copy(dQ_tail_shared, dQ[bos + s_i, bz * padded_H : (bz + 1) * padded_H, dim:])

    return main


# ---------------------------------------------------------------------------
# Input construction (mirrors examples/dsa_sparse_finetune/sparse_mla_bwd.py::test_sparse_mla_bwd)
# ---------------------------------------------------------------------------


def _prepare_token_indices(offsets):
    """Same as examples/dsa_sparse_finetune/index.py::prepare_token_indices."""
    lens = torch.diff(offsets.long())
    position_ids = torch.cat([torch.arange(int(n), dtype=torch.int32, device=offsets.device) for n in lens.unbind()])
    sequence_ids = (position_ids == 0).cumsum(0, dtype=torch.int32) - 1
    return torch.stack([sequence_ids, position_ids], dim=1).to(torch.int32)


def _build_bench_inputs(batch, seq_len, heads, kv_group, topk, dim, tail_dim):
    """Build benchmark inputs for the sparse MLA bwd kernel.

    Lse/Delta carry the true forward statistics (computed once with the chunked
    fp32 forward shared with the reference) so the autotuner's per-config
    correctness check is meaningful: the kernel's dP = p * (dO.V - Delta) * scale
    identity holds only for the real values.

    Indices are sampled per token from the causally visible keys [0, pos], so
    every token keeps at least one valid key and lse stays finite; prefixes
    shorter than topk are padded with in-bounds values that the kernel's causal
    mask (idx <= pos) ignores.
    """
    device = "cuda"
    dtype = torch.bfloat16
    dim_qk = dim + tail_dim
    S = batch * seq_len
    assert seq_len >= topk, "each sequence must be long enough to select topk keys"

    q = torch.randn(S, heads, dim_qk, dtype=dtype, device=device)
    kv = torch.randn(S, kv_group, dim_qk, dtype=dtype, device=device)
    do = torch.randn(S, heads, dim, dtype=dtype, device=device)
    offsets = torch.arange(0, S + 1, seq_len, dtype=torch.int32, device=device)

    # Causal-consistent topk indices: sort a random key per row with future
    # columns masked to +inf, so the first topk entries are a random subset of
    # the visible keys (vectorized equivalent of the example's per-token
    # randperm). All values stay in [0, seq_len) so the gather is in-bounds.
    future = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    indices = torch.zeros(S, kv_group, topk, dtype=torch.int32, device=device)
    for b in range(batch):
        bos = b * seq_len
        for g in range(kv_group):
            key = torch.rand(seq_len, seq_len, device=device).masked_fill(future, float("inf"))
            indices[bos : bos + seq_len, g, :] = torch.argsort(key, dim=1)[:, :topk].to(torch.int32)

    token_indices = _prepare_token_indices(offsets)

    # True forward statistics for the reference-consistent Lse/Delta inputs.
    with torch.no_grad():
        o, lse = _chunked_fwd_o_lse(q.float(), kv.float(), indices, offsets, dim, dim_qk**-0.5)
    delta = torch.einsum("shd,shd->sh", do.float(), o)

    return q, kv, do, indices, lse, delta, offsets, token_indices


# ---------------------------------------------------------------------------
# Reference (torch autograd, mirrors ref_sparse_mla_bwd_interface)
# ---------------------------------------------------------------------------


def _chunked_fwd_o_lse(q_f, kv_f, indices, offsets, dim, sm_scale):
    """Chunked fp32 sparse forward shared by the reference and input builder.

    Returns (o, lse): ``o`` keeps the autograd graph when ``q_f``/``kv_f``
    require grad; ``lse`` is the per-(token, head) log-sum-exp of the scaled
    scores, matching what the bwd kernel consumes.
    """
    _, H, dim_qk = q_f.shape
    kv_group = kv_f.shape[1]
    topk = indices.shape[-1]

    # Keep the gathered [chunk, G, topk, dim_qk] buffer bounded (~150MB).
    chunk = max(1, 65536 // topk)

    o_parts = []
    lse_parts = []
    for bos, eos in zip(offsets.tolist()[:-1], offsets.tolist()[1:]):
        seq_len = eos - bos
        kv_seq = kv_f[bos:eos]  # [L, G, dim_qk]
        for start in range(0, seq_len, chunk):
            end = min(start + chunk, seq_len)
            c = end - start
            idx = indices[bos + start : bos + end]  # [c, G, topk], sequence-relative
            kv_g = torch.stack([kv_seq[:, g, :][idx[:, g, :].clamp(max=seq_len - 1)] for g in range(kv_group)], dim=1)  # [c, G, topk, dim_qk]

            q_c = q_f[bos + start : bos + end].view(c, kv_group, H // kv_group, dim_qk)
            scores = torch.einsum("cghd,cgnd->cghn", q_c, kv_g)  # [c, G, h_per_g, topk]

            pos = torch.arange(start, end, device=q_f.device).view(-1, 1, 1)
            valid = (idx <= pos) & (idx >= 0)  # [c, G, topk]; broadcast over the head dim below
            scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf")) * sm_scale

            p = scores.softmax(dim=-1)
            v_g = kv_g[..., :dim]
            o_parts.append(torch.einsum("cghn,cgnd->cghd", p, v_g).reshape(c, H, dim))
            lse_parts.append(torch.logsumexp(scores, dim=-1).reshape(c, H))

    return torch.cat(o_parts), torch.cat(lse_parts)


def ref_sparse_mla_bwd(q, kv, do, indices, offsets, dim, sm_scale=None):
    """Reference sparse MLA backward via torch autograd on a chunked fwd.

    Differentiating the softmax attention forward reproduces exactly the
    dP = p * (dO.V - Delta) * scale identity used by the TileLang kernel.
    """
    dim_qk = q.shape[-1]
    if sm_scale is None:
        sm_scale = dim_qk**-0.5

    q_f = q.detach().float().clone().requires_grad_(True)
    kv_f = kv.detach().float().clone().requires_grad_(True)
    do_f = do.detach().float()

    o, _ = _chunked_fwd_o_lse(q_f, kv_f, indices, offsets, dim, sm_scale)
    o.backward(do_f)
    return q_f.grad, kv_f.grad


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def dsa_bwd_flops(batch, seq_len, heads, topk, dim, tail_dim):
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


def run_profile(batch, seq_len, heads, kv_group, topk, dim, tail_dim, block_size, num_stages, threads):
    """Run the kernel once with an explicit config for ncu/acu profiling."""
    q, kv, do, indices, lse, delta, offsets, token_indices = _build_bench_inputs(batch, seq_len, heads, kv_group, topk, dim, tail_dim)
    inject_pass_configs_from_env(sparse_mla_bwd)
    with set_autotune_inputs(q, kv, do, indices, lse, delta, offsets, token_indices):
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
    kernel(q, kv, do, indices, lse, delta, offsets, token_indices)


def run_profile_ref(batch, seq_len, heads, kv_group, topk, dim, tail_dim):
    """Run the reference once for ncu/acu profiling."""
    q, kv, do, indices, lse, delta, offsets, token_indices = _build_bench_inputs(batch, seq_len, heads, kv_group, topk, dim, tail_dim)
    ref_sparse_mla_bwd(q, kv, do, indices, offsets, dim)


def main(batch=1, seq_len=2048, heads=64, kv_group=1, topk=512, dim=512, tail_dim=64):
    """Run autotune and print results."""
    q, kv, do, indices, lse, delta, offsets, token_indices = _build_bench_inputs(batch, seq_len, heads, kv_group, topk, dim, tail_dim)

    with set_autotune_inputs(q, kv, do, indices, lse, delta, offsets, token_indices):
        best_result = sparse_mla_bwd(heads, dim, tail_dim, topk, kv_group, is_causal=True)

    best_latency = best_result.latency
    best_config = best_result.config

    total_flops = dsa_bwd_flops(batch, seq_len, heads, topk, dim, tail_dim)

    def ref_fn():
        return ref_sparse_mla_bwd(q, kv, do, indices, offsets, dim)

    ref_latency = bench_ref(ref_fn)
    ref_tflops = total_flops / ref_latency * 1e-9 if ref_latency > 0 else 0

    print_benchmark_summary(
        "DSA Sparse MLA Bwd",
        f"batch={batch}, seq_len={seq_len}, heads={heads}, topk={topk}",
        best_latency,
        total_flops / best_latency * 1e-9,
        ref_latency,
        ref_tflops,
        "Reference",
        best_config,
    )

    return best_latency, total_flops / best_latency * 1e-9, best_config, ref_latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSA Sparse MLA Bwd Autotune Benchmark")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--kv_group", type=int, default=1)
    parser.add_argument("--topk", type=int, default=512)
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
        run_profile_ref(args.batch, args.seq_len, args.heads, args.kv_group, args.topk, args.dim, args.tail_dim)
    else:
        main(args.batch, args.seq_len, args.heads, args.kv_group, args.topk, args.dim, args.tail_dim)
