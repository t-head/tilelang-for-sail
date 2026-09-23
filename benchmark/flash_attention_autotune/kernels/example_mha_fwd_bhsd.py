import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
import tilelang.language as T
import argparse
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from fa_configs import get_configs
from utils import inject_pass_configs_from_env


@autotune(configs=get_configs(), warmup=3, rep=10, early_stop=True)
@tilelang.jit(
    out_idx=[3],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    },
)
def flashattn(batch, heads, seq_q, seq_kv, dim, is_causal, block_M=64, block_N=64, num_stages=1, threads=128):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    q_shape = [batch, heads, seq_q, dim]
    kv_shape = [batch, heads, seq_kv, dim]
    dtype = T.float16
    accum_dtype = T.float32

    past_len = seq_kv - seq_q
    assert past_len >= 0, "seq_kv must be greater than or equal to seq_q"

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(T.ceildiv(seq_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            loop_range = (
                T.min(T.ceildiv(seq_kv, block_N), T.ceildiv((bx + 1) * block_M + past_len, block_N))
                if is_causal
                else T.ceildiv(seq_kv, block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=num_stages):
                T.copy(K[bz, by, k * block_N : (k + 1) * block_N, :], K_shared)
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        q_idx = bx * block_M + i + past_len
                        k_idx = k * block_N + j
                        acc_s[i, j] = T.if_then_else(q_idx >= k_idx, 0, -T.infinity(acc_s.dtype))
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(k * block_N + j >= seq_kv, -T.infinity(acc_s.dtype), 0)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(V[bz, by, k * block_N : (k + 1) * block_N, :], V_shared)
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output[bz, by, bx * block_M : (bx + 1) * block_M, :])

    return main


def ref_program(Q, K, V, is_causal):
    dim = Q.size(-1)
    scores = torch.einsum("bhqd,bhkd->bhqk", Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    if is_causal:
        seq_q = Q.size(2)
        seq_kv = K.size(2)
        mask = torch.tril(torch.ones(seq_q, seq_kv, device=scores.device), seq_kv - seq_q)
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bhkd->bhqd", attention_weights, V)
    return output


def main(
    batch: int = 1,
    heads: int = 1,
    seq_q: int = 256,
    seq_kv: int = 256,
    dim: int = 64,
    is_causal: bool = False,
    tune: bool = False,
    Q=None,
    K=None,
    V=None,
    block_M=None,
    block_N=None,
    num_stages=None,
    threads=None,
):
    flops_per_matmul = 2.0 * batch * heads * seq_q * seq_kv * dim
    total_flops = 2 * flops_per_matmul
    if is_causal:
        total_flops *= 0.5

    if not tune:
        inject_pass_configs_from_env(flashattn)
        kernel = flashattn(
            batch, heads, seq_q, seq_kv, dim, is_causal, block_M=block_M, block_N=block_N, num_stages=num_stages, threads=threads
        )
        Q = torch.empty(batch, heads, seq_q, dim, dtype=torch.half, device="cuda").normal_().requires_grad_()
        K = torch.empty_like(Q).normal_().requires_grad_()
        V = torch.empty_like(Q).normal_().requires_grad_()
        kernel(Q, K, V)
    else:
        kernel = flashattn(batch, heads, seq_q, seq_kv, dim, is_causal)
        best_latency = kernel.latency
        best_config = kernel.config
        current_file = os.path.basename(__file__)
        output_lines = [
            f"Current file: {current_file}",
            f"\tBest latency: {best_latency}",
            f"\tBest TFlops: {total_flops / best_latency * 1e-9}",
            f"\tBest config: {best_config}",
        ]

        for line in output_lines:
            print(line)

        best_kernel = flashattn(
            batch,
            heads,
            seq_q,
            seq_kv,
            dim,
            is_causal,
            best_config["block_M"],
            best_config["block_N"],
            best_config["num_stages"],
            best_config["threads"],
        )
        Q = Q.transpose(1, 2).contiguous()
        K = K.transpose(1, 2).contiguous()
        V = V.transpose(1, 2).contiguous()
        return best_kernel(Q, K, V), best_latency, total_flops / best_latency * 1e-9, best_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--heads", type=int, default=1, help="heads")
    parser.add_argument("--seq_q", type=int, default=256, help="query sequence length")
    parser.add_argument("--seq_kv", type=int, default=256, help="key/value sequence length")
    parser.add_argument("--dim", type=int, default=64, help="dim")
    parser.add_argument("--causal", type=bool, default=False, help="Causal flag")
    parser.add_argument("--tune", action="store_true", help="tune configs")
    # 1. block_M: 通常对应行的分块大小，常见值为 16, 32, 64, 128
    parser.add_argument("--block_M", type=int, default=64, help="Block size for dimension M (rows). Common: 16, 32, 64, 128")

    # 2. block_N: 通常对应列的分块大小，常见值为 16, 32, 64, 128
    parser.add_argument("--block_N", type=int, default=64, help="Block size for dimension N (cols). Common: 16, 32, 64, 128")

    # 3. num_stages: 流水线阶段数，用于隐藏内存延迟。常见值为 2, 3, 4
    # 注意：阶段数越多可能占用更多寄存器/共享内存，需权衡
    parser.add_argument("--num_stages", type=int, default=1, help="Number of pipeline stages for latency hiding. Common: 2, 3, 4")

    # 4. threads: 每个块的线程数 (num_warps * 32)。常见值为 128, 256, 512, 1024
    parser.add_argument(
        "--threads", type=int, default=128, help="Number of threads per block. Must be multiple of 32. Common: 128, 256, 512"
    )
    args = parser.parse_args()

    main(
        args.batch,
        args.heads,
        args.seq_q,
        args.seq_kv,
        args.dim,
        args.causal,
        False,
        block_M=args.block_M,
        block_N=args.block_N,
        num_stages=args.num_stages,
        threads=args.threads,
    )
