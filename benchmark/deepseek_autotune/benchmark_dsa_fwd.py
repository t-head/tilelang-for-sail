"""DSA sparse MLA forward benchmark with autotune.

Adapted from: examples/dsa_sparse_finetune/sparse_mla_fwd.py (kernel body
kept line-for-line; only the eager @tilelang.jit wrapper is converted to the
lazy @autotune + @jit(out_idx=[-2, -1]) prim_func form used by this module).
Tunable parameters: block_I, num_stages, threads

Known PPU limitation (2026-11-03, PPU0015): the LayoutInference pass fails
on this kernel with "Layout infer conflict between m_i and alpha" at the
online-softmax rescale (`alpha[h_i] = T.exp((m_i_prev - m_i) * sm_scale)`);
the example's fwd hit the same error. Runs fine on NVIDIA targets. Fixing it
requires either reusing m_i_prev instead of the separate alpha fragment or
unifying same-source fragment layout inference in the pass.

The reference is the dense masked-softmax attention from the example; only
the output `o` is compared (eps 1e-2, bf16 online softmax rescaling),
matching the example's check.
"""

import argparse

import torch
import tilelang
from tilelang import language as T
from tilelang.autotuner import autotune
from tilelang.autotuner.capture import set_autotune_inputs
from tilelang import jit

from configs import get_dsa_fwd_configs
from utils import print_benchmark_summary, bench_ref, inject_pass_configs_from_env, calc_diff


# ---------------------------------------------------------------------------
# Autotune correctness check (ref_prog / manual_check_prog)
# ---------------------------------------------------------------------------


# The ref_prog hook only receives the tensors bound via set_autotune_inputs,
# so the kernel's `dim` (output width) is derived from q's width minus this
# constant. All benchmark tiers pin dim=512 / tail_dim=64 (q width 576).
_REF_TAIL_DIM = 64


def ref_fwd_for_autotune(q, kv, indices, offsets, token_indices):
    """Reference program for the autotuner's per-config correctness check.

    Receives exactly the tensors bound via set_autotune_inputs. The jit
    kernel returns two outputs (o, Lse) via out_idx=[-2, -1] and the profiler's
    manual_assert_close asserts equal output counts, so the ref pads a
    placeholder Lse; only ``o`` is actually compared (check_fwd_outputs),
    matching the example's correctness gate which also only checks o.
    """
    o = ref_sparse_mla_fwd(q, kv, indices, offsets, q.shape[-1] - _REF_TAIL_DIM)
    lse_placeholder = torch.zeros(q.shape[0], q.shape[1], dtype=o.dtype, device=o.device)
    return o, lse_placeholder


def check_fwd_outputs(lib_outs, ref_outs):
    """manual_check_prog: cosine-distance comparison of the output o."""
    o_diff = calc_diff(lib_outs[0], ref_outs[0])
    assert o_diff < 1e-2, f"autotune fwd o check failed: diff={o_diff:.3e}"


@autotune(
    configs=get_dsa_fwd_configs(),
    warmup=5,
    rep=20,
    ref_prog=ref_fwd_for_autotune,
    manual_check_prog=check_fwd_outputs,
    # Note: cache_input_tensors is intentionally NOT set here (same reason as
    # benchmark_dsa_bwd.py: benchmark supply inputs are regenerated per trial).
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
def sparse_mla_fwd(
    heads,
    dim,
    tail_dim,
    topk,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_I=None,
    num_stages=None,
    threads=None,
):
    """Sparse MLA forward kernel (online softmax, varlen flattened layout).

    Tunable: block_I, num_stages, threads.
    """
    # Provide concrete defaults so the kernel can be elaborated (e.g. for
    # cache-key / validation TIR generation by the autotuner) before the
    # actual tunable values are supplied. These are overridden by autotune.
    block_I = block_I or 32
    num_stages = num_stages if num_stages is not None else 2
    threads = threads or 128

    assert dim == tilelang.math.next_power_of_2(dim), f"haven't check padding correctness yet, dim={dim}"
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim), f"haven't check padding correctness yet, dim={tail_dim}"
    assert is_causal is True, "non-casual is not supported"
    assert topk % block_I == 0, "otherwise will load some index=0 thus causing wrong kv to be loaded"
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5

    seq_len = T.dynamic("seq_len")
    batch_plus_one = T.dynamic("batch_plus_one")

    head_kv = heads // kv_group
    q_shape = [seq_len, heads, dim + tail_dim]
    kv_shape = [seq_len, kv_group, dim + tail_dim]
    o_shape = [seq_len, heads, dim]
    lse_shape = [seq_len, heads]
    indices_shape = [seq_len, kv_group, topk]
    offsets_shape = [batch_plus_one]
    token_indices_shape = [seq_len, 2]
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    G = kv_group
    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1, (
            "here we solve the H padding automatically, other wise you should handle Q copy and Output copy with your mask (when kv_group == 1, use g_i * padded_H:(g_i+1) * padded_H would be handled automatically)"
        )
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Offsets: T.Tensor(offsets_shape, indices_dtype),
        TokenIndices: T.Tensor(token_indices_shape, indices_dtype),
        Output: T.Tensor(o_shape, dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
    ):
        with T.Kernel(seq_len * REPLICATE_H, kv_group, threads=threads) as (
            bx,
            by,
        ):
            Q_shared = T.alloc_shared([H_per_block, D], dtype)
            Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
            mask = T.alloc_fragment([BI], "bool")

            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan

            b_s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            b_i, s_i = TokenIndices[b_s_i, 0], TokenIndices[b_s_i, 1]
            bos, eos = Offsets[b_i], Offsets[b_i + 1]
            g_i = by
            q_i = s_i
            max_kv_i = q_i

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            T.copy(Q[bos + s_i, H0:H1, :D], Q_shared)
            T.copy(Q[bos + s_i, H0:H1, D:], Q_tail_shared)

            for i_i in T.Pipelined(NI, num_stages=num_stages):
                for bi_i in T.Parallel(BI):
                    mask[bi_i] = (Indices[bos + s_i, g_i, i_i * BI + bi_i] <= max_kv_i) & (Indices[bos + s_i, g_i, i_i * BI + bi_i] != -1)

                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[bos + Indices[bos + s_i, g_i, i_i * BI + bi_i], g_i, d_i]
                for bi_i, d_i in T.Parallel(BI, D_tail):
                    K_tail_shared[bi_i, d_i] = KV[bos + Indices[bos + s_i, g_i, i_i * BI + bi_i], g_i, D + d_i]

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.gemm(
                    Q_tail_shared,
                    K_tail_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                T.reduce_sum(acc_s, sumexp_i, dim=1)  # is this a accumulate operator?
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            # Rescale
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] /= sumexp[h_i]
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log(sumexp[h_i]) + m_i[h_i] * sm_scale

            T.copy(acc_o, Output[bos + s_i, H0:H1, :])
            T.copy(sumexp, Lse[bos + s_i, H0:H1])

    return main


# ---------------------------------------------------------------------------
# Input construction (mirrors benchmark_dsa_bwd.py::_build_bench_inputs)
# ---------------------------------------------------------------------------


def _prepare_token_indices(offsets):
    """Same as examples/dsa_sparse_finetune/index.py::prepare_token_indices."""
    lens = torch.diff(offsets.long())
    position_ids = torch.cat([torch.arange(int(n), dtype=torch.int32, device=offsets.device) for n in lens.unbind()])
    sequence_ids = (position_ids == 0).cumsum(0, dtype=torch.int32) - 1
    return torch.stack([sequence_ids, position_ids], dim=1).to(torch.int32)


def _build_bench_inputs(batch, seq_len, heads, kv_group, topk, dim, tail_dim):
    """Build benchmark inputs for the sparse MLA fwd kernel.

    Indices are sampled per token from the causally visible keys [0, pos] via
    the vectorized masked-argsort used by benchmark_dsa_bwd.py: sort a random
    key per row with future columns masked to +inf, so the first topk entries
    are a random subset of the visible keys. All values stay in [0, seq_len)
    so the gather is in-bounds, and every token keeps at least one valid key.
    """
    device = "cuda"
    dtype = torch.bfloat16
    dim_qk = dim + tail_dim
    S = batch * seq_len
    assert seq_len >= topk, "each sequence must be long enough to select topk keys"

    q = torch.randn(S, heads, dim_qk, dtype=dtype, device=device)
    kv = torch.randn(S, kv_group, dim_qk, dtype=dtype, device=device)
    offsets = torch.arange(0, S + 1, seq_len, dtype=torch.int32, device=device)

    future = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    indices = torch.zeros(S, kv_group, topk, dtype=torch.int32, device=device)
    for b in range(batch):
        bos = b * seq_len
        for g in range(kv_group):
            key = torch.rand(seq_len, seq_len, device=device).masked_fill(future, float("inf"))
            indices[bos : bos + seq_len, g, :] = torch.argsort(key, dim=1)[:, :topk].to(torch.int32)

    token_indices = _prepare_token_indices(offsets)

    return q, kv, indices, offsets, token_indices


# ---------------------------------------------------------------------------
# Reference (dense masked-softmax attention, mirrors the example's
# ref_sparse_mla_fwd_interface with dim parameterized instead of hardcoded)
# ---------------------------------------------------------------------------


def ref_sparse_mla_fwd(q, kv, indices, offsets, dim, sm_scale=None):
    """Dense reference for the sparse MLA forward.

    Builds the full [seq, seq] selection mask from the topk indices, applies
    the causal mask, and runs one masked-softmax attention per sequence.
    Memory is O(seq_len^2) per sequence; fine for the benchmark shapes here.
    """
    Q = q.float()
    KV = kv.float()
    all_o = []
    for i in range(offsets.shape[0] - 1):
        q_ = Q[None, offsets[i] : offsets[i + 1]]
        kv_ = KV[None, offsets[i] : offsets[i + 1]]
        indices_ = indices[None, offsets[i] : offsets[i + 1]].clone()

        indices_ = indices_.transpose(1, 2)
        b, sq, h, dim_q = q_.shape
        b, sk, g, _ = kv_.shape

        k = kv_
        v = kv_[..., :dim]

        b, _, _, dim_v = v.shape
        g_index = g
        h_index = h // g
        compressed_casual_mask = torch.arange(0, sq, dtype=torch.int32, device="cuda").view(-1, 1) >= torch.arange(
            1 - 1, sk * 1, 1, dtype=torch.int32, device="cuda"
        ).view(1, -1)

        indices_[indices_ > sk] = sk
        mask = q_.new_zeros(b, g_index, sq, sk + 1, dtype=torch.bool).scatter(3, indices_.long(), 1)
        mask = mask[..., :-1]
        mask = mask & compressed_casual_mask.view(1, 1, sq, sk)
        mask[:, :, : 1 - 1, 0] = True
        mask = mask.view(b, g_index, 1, sq, sk)

        q_ = q_.view(b, sq, g, -1, dim_q)
        score = torch.einsum("bmghd,bngd->bghmn", q_, k)
        sm_scale = dim_q**-0.5 if sm_scale is None else sm_scale
        score = score.masked_fill(~mask, float("-inf")).mul(sm_scale)
        p = score.softmax(dim=-1)
        p = p.view(b, g_index, h_index, -1, sq, sk)
        p = p.view(b, g, -1, sq, sk)
        o = torch.einsum("bghmn,bngd->bmghd", p.type(v.dtype), v)
        o = o.reshape(b, sq, h, dim_v)
        all_o.append(o.squeeze(0))
    o = torch.cat(all_o, dim=0)
    return o.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def dsa_fwd_flops(batch, seq_len, heads, topk, dim, tail_dim):
    """FLOPs for the sparse MLA fwd (same gemm count as the example)."""
    per_token_flop = 2 * heads * topk * ((dim + tail_dim) + dim)
    return batch * seq_len * per_token_flop


def run_profile(batch, seq_len, heads, kv_group, topk, dim, tail_dim, block_I, num_stages, threads):
    """Run the kernel once with an explicit config for ncu/acu profiling."""
    q, kv, indices, offsets, token_indices = _build_bench_inputs(batch, seq_len, heads, kv_group, topk, dim, tail_dim)
    inject_pass_configs_from_env(sparse_mla_fwd)
    with set_autotune_inputs(q, kv, indices, offsets, token_indices):
        kernel = sparse_mla_fwd(
            heads,
            dim,
            tail_dim,
            topk,
            kv_group,
            is_causal=True,
            block_I=block_I,
            num_stages=num_stages,
            threads=threads,
        )
    kernel(q, kv, indices, offsets, token_indices)


def run_profile_ref(batch, seq_len, heads, kv_group, topk, dim, tail_dim):
    """Run the reference once for ncu/acu profiling."""
    q, kv, indices, offsets, token_indices = _build_bench_inputs(batch, seq_len, heads, kv_group, topk, dim, tail_dim)
    ref_sparse_mla_fwd(q, kv, indices, offsets, dim)


def main(batch=1, seq_len=2048, heads=64, kv_group=1, topk=512, dim=512, tail_dim=64):
    """Run autotune and print results."""
    q, kv, indices, offsets, token_indices = _build_bench_inputs(batch, seq_len, heads, kv_group, topk, dim, tail_dim)

    with set_autotune_inputs(q, kv, indices, offsets, token_indices):
        best_result = sparse_mla_fwd(heads, dim, tail_dim, topk, kv_group, is_causal=True)

    best_latency = best_result.latency
    best_config = best_result.config

    total_flops = dsa_fwd_flops(batch, seq_len, heads, topk, dim, tail_dim)

    def ref_fn():
        return ref_sparse_mla_fwd(q, kv, indices, offsets, dim)

    ref_latency = bench_ref(ref_fn)
    ref_tflops = total_flops / ref_latency * 1e-9 if ref_latency > 0 else 0

    print_benchmark_summary(
        "DSA Sparse MLA Fwd",
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
    parser = argparse.ArgumentParser(description="DSA Sparse MLA Fwd Autotune Benchmark")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=64)
    parser.add_argument("--kv_group", type=int, default=1)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--tail_dim", type=int, default=64)
    parser.add_argument("--profile", action="store_true", help="Run kernel once with given config for ncu/acu profiling")
    parser.add_argument("--profile-ref", action="store_true", help="Run reference once for ncu/acu profiling")
    parser.add_argument("--block_I", type=int, default=None)
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
            args.block_I,
            args.num_stages,
            args.threads,
        )
    elif args.profile_ref:
        run_profile_ref(args.batch, args.seq_len, args.heads, args.kv_group, args.topk, args.dim, args.tail_dim)
    else:
        main(args.batch, args.seq_len, args.heads, args.kv_group, args.topk, args.dim, args.tail_dim)
