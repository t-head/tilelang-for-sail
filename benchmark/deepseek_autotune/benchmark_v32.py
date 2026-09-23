"""DeepSeek V32 sparse MLA fwd benchmark with autotune.

Adapted from: examples/deepseek_v32/sparse_mla_fwd.py
Tunable parameters: block_I, num_stages, threads
"""

import argparse

import torch
import tilelang
from tilelang import language as T
from tilelang.autotuner import autotune
from tilelang.autotuner.capture import set_autotune_inputs
from tilelang import jit

from configs import get_v32_configs
from utils import print_benchmark_summary, bench_ref, inject_pass_configs_from_env


@autotune(configs=get_v32_configs(), warmup=5, rep=20, skip_check=True)
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
    """Sparse MLA forward kernel with online softmax.

    Tunable: block_I, num_stages, threads.
    """
    # Provide concrete defaults so the kernel can be elaborated (e.g. for
    # cache-key / validation TIR generation by the autotuner) before the
    # actual tunable values are supplied. These are overridden by autotune.
    block_I = block_I or 32
    num_stages = num_stages if num_stages is not None else 1
    threads = threads or 128

    assert dim == tilelang.math.next_power_of_2(dim)
    assert tail_dim == tilelang.math.next_power_of_2(tail_dim)
    assert is_causal is True, "non-causal not supported"
    assert topk % block_I == 0

    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504
    else:
        sm_scale = sm_scale * 1.44269504

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    head_kv = heads // kv_group
    q_shape = [batch, seq_len, heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    lse_shape = [batch, seq_len, heads]
    indices_dtype = T.int32
    dtype = T.bfloat16
    accum_dtype = T.float32

    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    if head_kv > 64:
        assert head_kv % 64 == 0
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        KV: T.Tensor(kv_shape, dtype),
        Indices: T.Tensor(indices_shape, indices_dtype),
        Output: T.Tensor(o_shape, dtype),
        Lse: T.Tensor(lse_shape, accum_dtype),
    ):
        with T.Kernel(seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (bx, by, bz):
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
            T.fill(m_i, -(2**30))

            b_i, g_i = by, bz
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            max_kv_i = s_i

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
            T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

            for i_i in T.Pipelined(NI, num_stages=num_stages):
                for bi_i in T.Parallel(BI):
                    mask[bi_i] = Indices[b_i, s_i, g_i, i_i * BI + bi_i] <= max_kv_i

                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i]
                for bi_i, d_i in T.Parallel(BI, D_tail):
                    K_tail_shared[bi_i, d_i] = KV[b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, D + d_i]

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(mask[bi_i], 0, -T.infinity(acc_s.dtype))
                T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                T.gemm(Q_tail_shared, K_tail_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    m_i[h_i] = T.max(m_i[h_i], m_i_prev[h_i])
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale)
                T.reduce_sum(acc_s, sumexp_i, dim=1)
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] /= sumexp[h_i]
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])
            T.copy(sumexp, Lse[b_i, s_i, H0:H1])

    return main


def run_profile(batch, seq_len, seq_len_kv, heads, kv_group, topk, dim, tail_dim, block_I, num_stages, threads):
    """Run the kernel once with an explicit config for ncu/acu profiling."""
    import torch

    dtype = torch.bfloat16
    dim_qk = dim + tail_dim

    Q = torch.randn(batch, seq_len, heads, dim_qk, dtype=dtype, device="cuda")
    KV = torch.randn(batch, seq_len_kv, kv_group, dim_qk, dtype=dtype, device="cuda")
    Indices = torch.full((batch, seq_len, kv_group, topk), 0, dtype=torch.int32, device="cuda")
    for b in range(batch):
        for t in range(seq_len):
            for h in range(kv_group):
                valid = max(1, t)
                idx = torch.randperm(valid, device="cuda")[: min(topk, valid)]
                Indices[b, t, h, : len(idx)] = idx
    inject_pass_configs_from_env(sparse_mla_fwd)
    with set_autotune_inputs(Q, KV, Indices):
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
    kernel(Q, KV, Indices)


def run_profile_ref(batch, seq_len, seq_len_kv, heads, kv_group, topk, dim, tail_dim):
    """Run the reference once for ncu/acu profiling."""
    import torch
    from flash_mla import flash_mla_sparse_fwd

    dtype = torch.bfloat16
    dim_qk = dim + tail_dim
    sm_scale = (dim + tail_dim) ** -0.5

    Q = torch.randn(batch, seq_len, heads, dim_qk, dtype=dtype, device="cuda")
    KV = torch.randn(batch, seq_len_kv, kv_group, dim_qk, dtype=dtype, device="cuda")
    Indices_ref = torch.full((batch, seq_len, kv_group, topk), -1, dtype=torch.int32, device="cuda")
    for b in range(batch):
        for t in range(seq_len):
            for h in range(kv_group):
                valid = max(1, t)
                idx = torch.randperm(valid, device="cuda")[: min(topk, valid)]
                Indices_ref[b, t, h, : len(idx)] = idx

    for bi in range(batch):
        flash_mla_sparse_fwd(Q[bi], KV[bi], Indices_ref[bi], sm_scale, dim)


def main(batch=1, seq_len=4096, seq_len_kv=4096, heads=128, kv_group=1, topk=2048, dim=512, tail_dim=64):
    """Run autotune and print results."""

    dtype = torch.bfloat16
    dim_qk = dim + tail_dim

    # Prepare concrete tensors for dynamic-shape kernel autotune
    Q = torch.randn(batch, seq_len, heads, dim_qk, dtype=dtype, device="cuda")
    KV = torch.randn(batch, seq_len_kv, kv_group, dim_qk, dtype=dtype, device="cuda")
    # Build indices: for each query position, pick random kv positions.
    #   Indices     -> for the TileLang kernel: padding = 0 (in-bounds gather, no OOB)
    #   Indices_ref -> for flash_mla_sparse_fwd: padding = -1 (its documented invalid marker)
    # Real selections are identical, only the padding value differs, so the
    # per-query workload (topk gather) is the same for a fair latency comparison.
    Indices = torch.full((batch, seq_len, kv_group, topk), 0, dtype=torch.int32, device="cuda")
    Indices_ref = torch.full((batch, seq_len, kv_group, topk), -1, dtype=torch.int32, device="cuda")
    for b in range(batch):
        for t in range(seq_len):
            for h in range(kv_group):
                valid = max(1, t)
                idx = torch.randperm(valid, device="cuda")[: min(topk, valid)]
                Indices[b, t, h, : len(idx)] = idx
                Indices_ref[b, t, h, : len(idx)] = idx
    with set_autotune_inputs(Q, KV, Indices):
        best_result = sparse_mla_fwd(heads, dim, tail_dim, topk, kv_group, is_causal=True)

    best_latency = best_result.latency
    best_config = best_result.config

    total_flops = batch * seq_len * (dim + tail_dim + dim) * topk * 2 * heads

    # Reference: DeepSeek FlashMLA sparse prefill kernel (flash_mla_sparse_fwd),
    # invoked the same way as examples/deepseek_mla/benchmark_mla.py::run_flash_mla.
    from flash_mla import flash_mla_sparse_fwd

    sm_scale = (dim + tail_dim) ** -0.5

    def ref_flash_mla():
        # flash_mla_sparse_fwd expects q:[s_q, h_q, d_qk], kv:[s_kv, h_kv, d_qk],
        # indices:[s_q, h_kv, topk]; there is no batch dim, so loop over batch.
        for bi in range(batch):
            flash_mla_sparse_fwd(Q[bi], KV[bi], Indices_ref[bi], sm_scale, dim)

    ref_latency = bench_ref(ref_flash_mla)
    ref_tflops = total_flops / ref_latency * 1e-9 if ref_latency > 0 else 0

    print_benchmark_summary(
        "V32 Sparse MLA Fwd",
        f"batch={batch}, seq={seq_len}, heads={heads}, topk={topk}",
        best_latency,
        total_flops / best_latency * 1e-9,
        ref_latency,
        ref_tflops,
        "FlashMLA",
        best_config,
    )

    return best_latency, total_flops / best_latency * 1e-9, best_config, ref_latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V32 Sparse MLA Fwd Autotune Benchmark")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--seq_len_kv", type=int, default=4096)
    parser.add_argument("--heads", type=int, default=128)
    parser.add_argument("--kv_group", type=int, default=1)
    parser.add_argument("--topk", type=int, default=2048)
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
            args.seq_len_kv,
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
        run_profile_ref(args.batch, args.seq_len, args.seq_len_kv, args.heads, args.kv_group, args.topk, args.dim, args.tail_dim)
    else:
        main(args.batch, args.seq_len, args.seq_len_kv, args.heads, args.kv_group, args.topk, args.dim, args.tail_dim)
