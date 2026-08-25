"""External (non-tunable) parameter configurations for DeepSeek autotune benchmarks.

Each kernel type has SINGLE / DAILY / FULL tiers.
Select via environment variable: BENCHMARK_CONFIG=SINGLE|DAILY|FULL (default: SINGLE).
"""

# ---------------------------------------------------------------------------
# MLA Decode
# ---------------------------------------------------------------------------

MLA_SINGLE_CONFIG = {
    "batch": [1],
    "heads": [128],
    "kv_heads": [1],
    "kv_ctx": [8192],
    "dim": [512],
    "pe_dim": [64],
}

MLA_DAILY_CONFIG = {
    "batch": [1, 32, 132],
    "heads": [128],
    "kv_heads": [1],
    "kv_ctx": [4096, 8192, 16384],
    "dim": [512],
    "pe_dim": [64],
}

MLA_FULL_CONFIG = {
    "batch": [1, 16, 32, 64, 132],
    "heads": [128],
    "kv_heads": [1],
    "kv_ctx": [2048, 4096, 8192, 16384, 32768],
    "dim": [512],
    "pe_dim": [64],
}

# ---------------------------------------------------------------------------
# Paged MLA Decode (PagedAttention KV cache)
# ---------------------------------------------------------------------------

MLA_PAGED_SINGLE_CONFIG = {
    "batch": [128],
    "h_q": [128],
    "h_kv": [1],
    "cache_seqlen": [8192],
    "d": [576],
    "dv": [512],
}

MLA_PAGED_DAILY_CONFIG = {
    "batch": [32, 128],
    "h_q": [128],
    "h_kv": [1],
    "cache_seqlen": [4096, 8192, 16384],
    "d": [576],
    "dv": [512],
}

MLA_PAGED_FULL_CONFIG = {
    "batch": [16, 32, 64, 128],
    "h_q": [128],
    "h_kv": [1],
    "cache_seqlen": [2048, 4096, 8192, 16384, 32768],
    "d": [576],
    "dv": [512],
}

# ---------------------------------------------------------------------------
# DeepGEMM FP8
# ---------------------------------------------------------------------------

DEEPGEMM_SINGLE_CONFIG = {
    "M": [1024],
    "N": [1024],
    "K": [8192],
    "in_dtype": ["float8_e4m3fn"],
    "out_dtype": ["bfloat16"],
}

DEEPGEMM_DAILY_CONFIG = {
    "M": [1024, 2048, 4096],
    "N": [1024, 2048, 4096],
    "K": [4096, 8192],
    "in_dtype": ["float8_e4m3fn"],
    "out_dtype": ["bfloat16"],
}

DEEPGEMM_FULL_CONFIG = {
    "M": [1024, 2048, 4096, 8192],
    "N": [1024, 2048, 4096, 8192],
    "K": [4096, 8192, 16384],
    "in_dtype": ["float8_e4m3fn"],
    "out_dtype": ["bfloat16", "float32"],
}

# ---------------------------------------------------------------------------
# NSA (Native Sparse Attention)
# ---------------------------------------------------------------------------

NSA_SINGLE_CONFIG = {
    "batch": [2],
    "heads": [16],
    "seq_len": [64],
    "dim": [32],
    "selected_blocks": [1],
    "block_size": [32],
    "is_causal": [True],
}

NSA_DAILY_CONFIG = {
    "batch": [2, 4],
    "heads": [16, 32],
    "seq_len": [64, 256, 1024],
    "dim": [32, 64],
    "selected_blocks": [1, 4],
    "block_size": [32, 64],
    "is_causal": [True],
}

NSA_FULL_CONFIG = {
    "batch": [2, 4, 8],
    "heads": [16, 32, 64],
    "seq_len": [64, 256, 1024, 4096],
    "dim": [32, 64, 128],
    "selected_blocks": [1, 4, 8, 16],
    "block_size": [32, 64],
    "is_causal": [True, False],
}

# ---------------------------------------------------------------------------
# mHC (Multi-Head Contribution) Pre
# ---------------------------------------------------------------------------

MHC_SINGLE_CONFIG = {
    "n": [1024],
    "hidden_size": [2560],
    "hc_mult": [4],
}

MHC_DAILY_CONFIG = {
    "n": [512, 1024, 2048],
    "hidden_size": [1280, 2560],
    "hc_mult": [4],
}

MHC_FULL_CONFIG = {
    "n": [512, 1024, 2048, 8192],
    "hidden_size": [1280, 2560, 4096],
    "hc_mult": [4],
}

# ---------------------------------------------------------------------------
# mHC Big Fuse
# ---------------------------------------------------------------------------

MHC_BIG_FUSE_SINGLE_CONFIG = {
    "n": [1024],
    "hidden_size": [2560],
    "hc_mult": [4],
}

MHC_BIG_FUSE_DAILY_CONFIG = {
    "n": [512, 1024, 2048],
    "hidden_size": [1280, 2560],
    "hc_mult": [4],
}

MHC_BIG_FUSE_FULL_CONFIG = {
    "n": [512, 1024, 2048, 8192],
    "hidden_size": [1280, 2560, 4096],
    "hc_mult": [4],
}

# ---------------------------------------------------------------------------
# mHC Post
# ---------------------------------------------------------------------------

MHC_POST_SINGLE_CONFIG = {
    "n": [4096],
    "hidden_size": [2560],
    "hc_mult": [4],
}

MHC_POST_DAILY_CONFIG = {
    "n": [1024, 4096],
    "hidden_size": [1280, 2560, 7168],
    "hc_mult": [4],
}

MHC_POST_FULL_CONFIG = {
    "n": [1024, 4096, 8192],
    "hidden_size": [1280, 2560, 4096, 7168],
    "hc_mult": [4],
}

# ---------------------------------------------------------------------------
# NSA Decode
# ---------------------------------------------------------------------------

NSA_DECODE_SINGLE_CONFIG = {
    "batch": [2],
    "heads": [16],
    "seq_len": [64],
    "dim": [32],
    "selected_blocks": [1],
    "block_size": [32],
}

NSA_DECODE_DAILY_CONFIG = {
    "batch": [2, 4],
    "heads": [16, 32],
    "seq_len": [64, 256],
    "dim": [32, 64],
    "selected_blocks": [1, 4],
    "block_size": [32, 64],
}

NSA_DECODE_FULL_CONFIG = {
    "batch": [2, 4, 8],
    "heads": [16, 32, 64],
    "seq_len": [64, 256, 1024],
    "dim": [32, 64, 128],
    "selected_blocks": [1, 4, 8],
    "block_size": [32, 64],
}

# ---------------------------------------------------------------------------
# V32 Sparse MLA Fwd
# ---------------------------------------------------------------------------

V32_SINGLE_CONFIG = {
    "batch": [1],
    "seq_len": [4096],
    "seq_len_kv": [4096],
    "heads": [128],
    "kv_group": [1],
    "topk": [2048],
    "dim": [512],
    "tail_dim": [64],
}

V32_DAILY_CONFIG = {
    "batch": [1],
    "seq_len": [2048, 4096, 8192],
    "seq_len_kv": [4096, 8192],
    "heads": [128],
    "kv_group": [1],
    "topk": [1024, 2048],
    "dim": [512],
    "tail_dim": [64],
}

V32_FULL_CONFIG = {
    "batch": [1, 2, 4],
    "seq_len": [2048, 4096, 8192],
    "seq_len_kv": [4096, 8192, 16384],
    "heads": [128],
    "kv_group": [1],
    "topk": [512, 1024, 2048, 4096],
    "dim": [512],
    "tail_dim": [64],
}


# ---------------------------------------------------------------------------
# Helper to select config tier by name
# ---------------------------------------------------------------------------

_CONFIG_TIERS = {
    "mla": {"SINGLE": MLA_SINGLE_CONFIG, "DAILY": MLA_DAILY_CONFIG, "FULL": MLA_FULL_CONFIG},
    "mla_paged": {"SINGLE": MLA_PAGED_SINGLE_CONFIG, "DAILY": MLA_PAGED_DAILY_CONFIG, "FULL": MLA_PAGED_FULL_CONFIG},
    "deepgemm": {"SINGLE": DEEPGEMM_SINGLE_CONFIG, "DAILY": DEEPGEMM_DAILY_CONFIG, "FULL": DEEPGEMM_FULL_CONFIG},
    "nsa": {"SINGLE": NSA_SINGLE_CONFIG, "DAILY": NSA_DAILY_CONFIG, "FULL": NSA_FULL_CONFIG},
    "nsa_decode": {"SINGLE": NSA_DECODE_SINGLE_CONFIG, "DAILY": NSA_DECODE_DAILY_CONFIG, "FULL": NSA_DECODE_FULL_CONFIG},
    "mhc": {"SINGLE": MHC_SINGLE_CONFIG, "DAILY": MHC_DAILY_CONFIG, "FULL": MHC_FULL_CONFIG},
    "mhc_big_fuse": {"SINGLE": MHC_BIG_FUSE_SINGLE_CONFIG, "DAILY": MHC_BIG_FUSE_DAILY_CONFIG, "FULL": MHC_BIG_FUSE_FULL_CONFIG},
    "mhc_post": {"SINGLE": MHC_POST_SINGLE_CONFIG, "DAILY": MHC_POST_DAILY_CONFIG, "FULL": MHC_POST_FULL_CONFIG},
    "v32": {"SINGLE": V32_SINGLE_CONFIG, "DAILY": V32_DAILY_CONFIG, "FULL": V32_FULL_CONFIG},
}


def get_bench_config(kernel_name: str, tier: str = "SINGLE") -> dict:
    """Get the benchmark config for a given kernel and tier.

    Args:
        kernel_name: one of "mla", "deepgemm", "nsa", "mhc", "v32"
        tier: one of "SINGLE", "DAILY", "FULL"
    """
    tier = tier.upper()
    if kernel_name not in _CONFIG_TIERS:
        raise ValueError(f"Unknown kernel: {kernel_name}. Available: {list(_CONFIG_TIERS.keys())}")
    if tier not in _CONFIG_TIERS[kernel_name]:
        raise ValueError(f"Unknown tier: {tier}. Available: SINGLE, DAILY, FULL")
    return _CONFIG_TIERS[kernel_name][tier]
