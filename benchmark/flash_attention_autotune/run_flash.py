import argparse
import torch

try:
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func
    from flash_attn.flash_attn_interface import flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False
    raise ValueError("No flash-2 found")

def main(batch, heads, seq_len, head_dim, groups, causal, algo, mode, device="cuda"):
    """Run flash-2 benchmark and return latency and TFlops"""
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16

    if algo == "mha":
        qkv = torch.randn((batch, seq_len, 3, heads, head_dim), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_qkvpacked_func(qkv, causal=causal)
    elif algo == "gqa":
        head_kv = heads // groups
        q = torch.randn((batch, seq_len, heads, head_dim), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((batch, seq_len, head_kv, head_dim), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((batch, seq_len, head_kv, head_dim), dtype=dtype, device=device, requires_grad=True)
        fn = lambda: flash_attn_func(q, k, v, causal=causal)

    o = fn()
    if mode == "bwd":
        dO = torch.randn_like(o)
        o.backward(dO, retain_graph=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--seq_len", type=int, required=True)
    parser.add_argument("--head_dim", type=int, required=True)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument('--causal', type=bool, default=False, help='Causal flag')
    parser.add_argument("--algo", type=str, default="mha", choices=["mha", "gqa"])
    parser.add_argument("--mode", type=str, default="fwd")

    args = parser.parse_args()

    main(args.batch, args.heads, args.seq_len, args.head_dim, args.groups, args.causal, args.algo, args.mode)