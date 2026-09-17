"""Pytest entry for DeepSeek autotune benchmarks.

Usage:
    # Run with default DAILY config
    pytest test_deepseek_bench.py -v

    # Run with SINGLE or FULL config
    BENCHMARK_CONFIG=SINGLE pytest test_deepseek_bench.py -v

    # Run specific kernel
    pytest test_deepseek_bench.py -v -k "mla"

    # Disable cycle/tensor-core profiling
    TILELANG_PROFILE_CYCLES=0 pytest test_deepseek_bench.py -v

    # Run reference benchmarks too (default: skip reference)
    TILELANG_SKIP_REF=0 pytest test_deepseek_bench.py -v

    # Standalone
    python test_deepseek_bench.py
"""

import gc
import itertools
import json
import os
import sys

import torch

# Ensure the benchmark package directory is on sys.path so that
# sibling modules (bench_configs, benchmark_*) can be imported
# regardless of the working directory pytest is invoked from.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import pytest

from bench_configs import get_bench_config
from utils import (
    print_benchmark_summary,
    run_cycle_on_device,
    get_device_type,
    setup_ppu_env,
)

# ---------------------------------------------------------------------------
# Config tier selection
# ---------------------------------------------------------------------------

_TIER = os.environ.get("BENCHMARK_CONFIG", "DAILY").upper()

# Set to '1' to also collect acu/ncu cycle and tensor-core utilisation numbers.
PROFILE_CYCLES = os.environ.get("TILELANG_PROFILE_CYCLES", "1") == "1"
# Skip reference benchmarks (both latency do_bench and ncu/acu cycle profiling).
SKIP_REF = os.environ.get("TILELANG_SKIP_REF", "1") == "1"
# Device for cycle profiling: auto-detected if None.
PROFILE_DEV = os.environ.get("TILELANG_PROFILE_DEV", None)


# ---------------------------------------------------------------------------
# Per-kernel metadata for cycle profiling
# ---------------------------------------------------------------------------

_KERNEL_META = {
    "mla": {
        "name": "MLA Decode",
        "script": "benchmark_mla.py",
        "ref_name": "FlashMLA",
        "config_keys": ["block_N", "block_H", "num_stages", "threads"],
        "ref_filters": ["flash_fwd"],
    },
    "mla_paged": {
        "name": "MLA Decode Paged",
        "script": "benchmark_mla_paged.py",
        "ref_name": "Reference",
        "config_keys": ["block_N", "block_H", "num_stages", "threads"],
        "ref_filters": None,
    },
    "deepgemm": {
        "name": "DeepGEMM FP8",
        "script": "benchmark_deepgemm.py",
        "ref_name": "Reference",
        "config_keys": ["block_N", "num_stages", "threads", "enable_rasteration"],
        "ref_filters": None,
    },
    "nsa": {
        "name": "NSA Fwd",
        "script": "benchmark_nsa.py",
        "ref_name": "Reference",
        "config_keys": ["num_stages", "threads"],
        "ref_filters": None,
    },
    "nsa_decode": {
        "name": "NSA Decode",
        "script": "benchmark_nsa_decode.py",
        "ref_name": "Reference",
        "config_keys": ["num_stages", "threads"],
        "ref_filters": None,
    },
    "mhc": {
        "name": "mHC Pre",
        "script": "benchmark_mhc.py",
        "ref_name": "Reference",
        "config_keys": ["token_block", "hidden_block", "num_stages"],
        "ref_filters": None,
    },
    "mhc_big_fuse": {
        "name": "mHC BigFuse",
        "script": "benchmark_mhc_big_fuse.py",
        "ref_name": "Reference",
        "config_keys": ["threads", "num_stages"],
        "ref_filters": None,
    },
    "mhc_post": {
        "name": "mHC Post",
        "script": "benchmark_mhc_post.py",
        "ref_name": "Reference",
        "config_keys": ["n_thr", "h_blk"],
        "ref_filters": None,
    },
    "v32": {
        "name": "V32 Sparse MLA Fwd",
        "script": "benchmark_v32.py",
        "ref_name": "FlashMLA",
        "config_keys": ["block_I", "num_stages", "threads"],
        "ref_filters": ["flash"],
    },
}


def _maybe_profile(kernel_key, problem_args, best_config):
    """Run acu/ncu profiling if TILELANG_PROFILE_CYCLES=1.

    Returns (tl_cycle, tl_tc, ref_cycle, ref_tc), or (None, None, None, None)
    when profiling is disabled.
    """
    if not PROFILE_CYCLES:
        return None, None, None, None

    dev = PROFILE_DEV or get_device_type()
    if dev == "ppu":
        setup_ppu_env()

    meta = _KERNEL_META[kernel_key]
    script_path = os.path.join(_THIS_DIR, meta["script"])
    log_file = os.path.join(_THIS_DIR, "gpu_cycles_single_case.log")

    # Profile TileLang kernel
    gc.collect()
    torch.cuda.empty_cache()
    profile_args = list(problem_args) + ["--profile"]
    for key in meta["config_keys"]:
        profile_args.extend([f"--{key}", str(best_config[key])])

    # Inject pass_configs from best_config into env for acu/ncu profiling.
    # The autotune skip path loses per-config pass_configs; benchmark scripts
    # read TILELANG_PASS_CONFIGS and set jit_impl.pass_configs before kernel call.
    from enum import Enum
    pass_configs = best_config.get("pass_configs")
    if pass_configs:
        safe_pc = {k.value if isinstance(k, Enum) else k: v for k, v in pass_configs.items()}
        os.environ["TILELANG_PASS_CONFIGS"] = json.dumps(safe_pc)
    else:
        os.environ.pop("TILELANG_PASS_CONFIGS", None)

    tl_cycle, tl_tc = run_cycle_on_device(
        script_path, profile_args, dev=dev, log_file=log_file, framework="tilelang"
    )

    # Profile reference (skip when TILELANG_SKIP_REF=1)
    if SKIP_REF:
        ref_cycle, ref_tc = 0, 0
    else:
        gc.collect()
        torch.cuda.empty_cache()
        ref_profile_args = list(problem_args) + ["--profile-ref"]
        ref_cycle, ref_tc = run_cycle_on_device(
            script_path, ref_profile_args, dev=dev, log_file=log_file,
            framework="ref", kernel_filters=meta.get("ref_filters")
        )

    gc.collect()
    torch.cuda.empty_cache()

    return tl_cycle, tl_tc, ref_cycle, ref_tc


# ---------------------------------------------------------------------------
# Utility: expand config dict into pytest.param list
# ---------------------------------------------------------------------------


def generate_configs(config_dict: dict):
    """Expand a config dict (key -> list[values]) into pytest.param instances.

    Returns a list of pytest.param(...) with ids built from param values.
    """
    keys = list(config_dict.keys())
    value_lists = [config_dict[k] for k in keys]
    params = []
    for combo in itertools.product(*value_lists):
        id_str = "-".join(f"{k}={v}" for k, v in zip(keys, combo))
        params.append(pytest.param(*combo, id=id_str))
    return params


# ---------------------------------------------------------------------------
# MLA Decode
# ---------------------------------------------------------------------------

_mla_config = get_bench_config("mla", _TIER)
_mla_keys = list(_mla_config.keys())


@pytest.mark.parametrize(",".join(_mla_keys), generate_configs(_mla_config))
def test_bench_mla(batch, heads, kv_heads, kv_ctx, dim, pe_dim):
    from benchmark_mla import main

    latency, tflops, best_config, ref_latency = main(
        batch=batch, heads=heads, kv_heads=kv_heads, kv_ctx=kv_ctx, dim=dim, pe_dim=pe_dim
    )
    assert latency > 0, f"Invalid latency: {latency}"
    assert tflops > 0, f"Invalid TFlops: {tflops}"

    config_str = (f"batch={batch}, heads={heads}, kv_heads={kv_heads}, "
                  f"kv_ctx={kv_ctx}, dim={dim}, pe_dim={pe_dim}")
    ref_tflops = tflops * latency / ref_latency if ref_latency > 0 else 0
    problem_args = [
        "--batch", str(batch), "--heads", str(heads),
        "--kv_heads", str(kv_heads), "--kv_ctx", str(kv_ctx),
        "--dim", str(dim), "--pe_dim", str(pe_dim),
    ]
    tl_cycles, tl_tc, ref_cycles, ref_tc = _maybe_profile("mla", problem_args, best_config)

    print_benchmark_summary(
        "MLA Decode", config_str,
        latency, tflops, ref_latency, ref_tflops,
        "FlashMLA", best_config,
        tilelang_cycles=tl_cycles, tilelang_tc=tl_tc,
        ref_cycles=ref_cycles, ref_tc=ref_tc,
    )


# ---------------------------------------------------------------------------
# Paged MLA Decode (PagedAttention KV cache)
# ---------------------------------------------------------------------------

_mla_paged_config = get_bench_config("mla_paged", _TIER)
_mla_paged_keys = list(_mla_paged_config.keys())


@pytest.mark.parametrize(",".join(_mla_paged_keys), generate_configs(_mla_paged_config))
def test_bench_mla_paged(batch, h_q, h_kv, cache_seqlen, d, dv):
    from benchmark_mla_paged import main

    latency, tflops, best_config, ref_latency = main(
        batch=batch, h_q=h_q, h_kv=h_kv, cache_seqlen=cache_seqlen, d=d, dv=dv
    )
    assert latency > 0, f"Invalid latency: {latency}"
    assert tflops > 0, f"Invalid TFlops: {tflops}"

    config_str = (f"batch={batch}, h_q={h_q}, h_kv={h_kv}, "
                  f"cache_seqlen={cache_seqlen}, d={d}, dv={dv}")
    ref_tflops = tflops * latency / ref_latency if ref_latency > 0 else 0
    problem_args = [
        "--batch", str(batch), "--h_q", str(h_q),
        "--h_kv", str(h_kv), "--cache_seqlen", str(cache_seqlen),
        "--d", str(d), "--dv", str(dv),
    ]
    tl_cycles, tl_tc, ref_cycles, ref_tc = _maybe_profile(
        "mla_paged", problem_args, best_config
    )

    print_benchmark_summary(
        "MLA Decode Paged", config_str,
        latency, tflops, ref_latency, ref_tflops,
        "Reference", best_config,
        tilelang_cycles=tl_cycles, tilelang_tc=tl_tc,
        ref_cycles=ref_cycles, ref_tc=ref_tc,
    )


# ---------------------------------------------------------------------------
# DeepGEMM FP8 (skipped: PPU 1.0 does not support FP8)
# ---------------------------------------------------------------------------

_deepgemm_config = get_bench_config("deepgemm", _TIER)
_deepgemm_keys = list(_deepgemm_config.keys())

from tilelang.contrib import hgcc
try:
    arch = hgcc.get_target_compute_version()
    compute_version = hgcc.parse_compute_version(arch)
except Exception:
    arch = "unknown"
    compute_version = (0, 0)

@pytest.mark.skipif(
    compute_version != (1, 5),
    reason=f"Requires PPU 1.5, but have {arch}",
)
@pytest.mark.parametrize(",".join(_deepgemm_keys), generate_configs(_deepgemm_config))
def test_bench_deepgemm(M, N, K, in_dtype, out_dtype):
    from benchmark_deepgemm import main

    latency, tflops, best_config, ref_latency = main(
        M=M, N=N, K=K, in_dtype_str=in_dtype, out_dtype_str=out_dtype
    )
    assert latency > 0, f"Invalid latency: {latency}"
    assert tflops > 0, f"Invalid TFlops: {tflops}"

    config_str = f"M={M}, N={N}, K={K}, in_dtype={in_dtype}, out_dtype={out_dtype}"
    ref_tflops = tflops * latency / ref_latency if ref_latency > 0 else 0
    problem_args = [
        "--m", str(M), "--n", str(N), "--k", str(K),
        "--in_dtype", str(in_dtype), "--out_dtype", str(out_dtype),
    ]
    tl_cycles, tl_tc, ref_cycles, ref_tc = _maybe_profile(
        "deepgemm", problem_args, best_config
    )

    print_benchmark_summary(
        "DeepGEMM FP8", config_str,
        latency, tflops, ref_latency, ref_tflops,
        "Reference", best_config,
        tilelang_cycles=tl_cycles, tilelang_tc=tl_tc,
        ref_cycles=ref_cycles, ref_tc=ref_tc,
    )


# ---------------------------------------------------------------------------
# NSA (Native Sparse Attention) Fwd
# ---------------------------------------------------------------------------

_nsa_config = get_bench_config("nsa", _TIER)
_nsa_keys = list(_nsa_config.keys())


@pytest.mark.parametrize(",".join(_nsa_keys), generate_configs(_nsa_config))
def test_bench_nsa(batch, heads, seq_len, dim, selected_blocks, block_size, is_causal):
    from benchmark_nsa import main

    latency, tflops, best_config, ref_latency = main(
        batch=batch, heads=heads, seq_len=seq_len, dim=dim,
        selected_blocks=selected_blocks, block_size=block_size, is_causal=is_causal
    )
    assert latency > 0, f"Invalid latency: {latency}"
    assert tflops > 0, f"Invalid TFlops: {tflops}"

    config_str = (f"batch={batch}, heads={heads}, seq_len={seq_len}, dim={dim}, "
                  f"selected_blocks={selected_blocks}, block_size={block_size}, "
                  f"is_causal={is_causal}")
    ref_tflops = tflops * latency / ref_latency if ref_latency > 0 else 0
    problem_args = [
        "--batch", str(batch), "--heads", str(heads),
        "--seq_len", str(seq_len), "--dim", str(dim),
        "--selected_blocks", str(selected_blocks), "--block_size", str(block_size),
    ]
    if is_causal:
        problem_args.append("--causal")
    tl_cycles, tl_tc, ref_cycles, ref_tc = _maybe_profile(
        "nsa", problem_args, best_config
    )

    print_benchmark_summary(
        "NSA Fwd", config_str,
        latency, tflops, ref_latency, ref_tflops,
        "Reference", best_config,
        tilelang_cycles=tl_cycles, tilelang_tc=tl_tc,
        ref_cycles=ref_cycles, ref_tc=ref_tc,
    )


# ---------------------------------------------------------------------------
# NSA Decode
# ---------------------------------------------------------------------------

_nsa_decode_config = get_bench_config("nsa_decode", _TIER)
_nsa_decode_keys = list(_nsa_decode_config.keys())


@pytest.mark.parametrize(",".join(_nsa_decode_keys), generate_configs(_nsa_decode_config))
def test_bench_nsa_decode(batch, heads, seq_len, dim, selected_blocks, block_size):
    from benchmark_nsa_decode import main

    latency, tflops, best_config, ref_latency = main(
        batch=batch, heads=heads, seq_len=seq_len, dim=dim,
        selected_blocks=selected_blocks, block_size=block_size
    )
    assert latency > 0, f"Invalid latency: {latency}"
    assert tflops > 0, f"Invalid TFlops: {tflops}"

    config_str = (f"batch={batch}, heads={heads}, seq_len={seq_len}, dim={dim}, "
                  f"selected_blocks={selected_blocks}, block_size={block_size}")
    ref_tflops = tflops * latency / ref_latency if ref_latency > 0 else 0
    problem_args = [
        "--batch", str(batch), "--heads", str(heads),
        "--seq_len", str(seq_len), "--dim", str(dim),
        "--selected_blocks", str(selected_blocks), "--block_size", str(block_size),
    ]
    tl_cycles, tl_tc, ref_cycles, ref_tc = _maybe_profile(
        "nsa_decode", problem_args, best_config
    )

    print_benchmark_summary(
        "NSA Decode", config_str,
        latency, tflops, ref_latency, ref_tflops,
        "Reference", best_config,
        tilelang_cycles=tl_cycles, tilelang_tc=tl_tc,
        ref_cycles=ref_cycles, ref_tc=ref_tc,
    )


# ---------------------------------------------------------------------------
# mHC Pre (Multi-Head Contribution)
# ---------------------------------------------------------------------------

_mhc_config = get_bench_config("mhc", _TIER)
_mhc_keys = list(_mhc_config.keys())


@pytest.mark.parametrize(",".join(_mhc_keys), generate_configs(_mhc_config))
def test_bench_mhc(n, hidden_size, hc_mult):
    from benchmark_mhc import main

    latency, tflops, best_config, ref_latency = main(
        n=n, hidden_size=hidden_size, hc_mult=hc_mult
    )
    assert latency > 0, f"Invalid latency: {latency}"
    assert tflops > 0, f"Invalid TFlops: {tflops}"

    config_str = f"n={n}, hidden_size={hidden_size}, hc_mult={hc_mult}"
    ref_tflops = tflops * latency / ref_latency if ref_latency > 0 else 0
    problem_args = [
        "--n", str(n), "--hidden_size", str(hidden_size), "--hc_mult", str(hc_mult),
    ]
    tl_cycles, tl_tc, ref_cycles, ref_tc = _maybe_profile(
        "mhc", problem_args, best_config
    )

    print_benchmark_summary(
        "mHC Pre", config_str,
        latency, tflops, ref_latency, ref_tflops,
        "Reference", best_config,
        tilelang_cycles=tl_cycles, tilelang_tc=tl_tc,
        ref_cycles=ref_cycles, ref_tc=ref_tc,
    )


# ---------------------------------------------------------------------------
# mHC Big Fuse
# ---------------------------------------------------------------------------

_mhc_big_fuse_config = get_bench_config("mhc_big_fuse", _TIER)
_mhc_big_fuse_keys = list(_mhc_big_fuse_config.keys())


@pytest.mark.parametrize(",".join(_mhc_big_fuse_keys), generate_configs(_mhc_big_fuse_config))
def test_bench_mhc_big_fuse(n, hidden_size, hc_mult):
    from benchmark_mhc_big_fuse import main

    latency, tflops, best_config, ref_latency = main(
        n=n, hidden_size=hidden_size, hc_mult=hc_mult
    )
    assert latency > 0, f"Invalid latency: {latency}"
    assert tflops > 0, f"Invalid TFlops: {tflops}"

    config_str = f"n={n}, hidden_size={hidden_size}, hc_mult={hc_mult}"
    ref_tflops = tflops * latency / ref_latency if ref_latency > 0 else 0
    problem_args = [
        "--n", str(n), "--hidden_size", str(hidden_size), "--hc_mult", str(hc_mult),
    ]
    tl_cycles, tl_tc, ref_cycles, ref_tc = _maybe_profile(
        "mhc_big_fuse", problem_args, best_config
    )

    print_benchmark_summary(
        "mHC BigFuse", config_str,
        latency, tflops, ref_latency, ref_tflops,
        "Reference", best_config,
        tilelang_cycles=tl_cycles, tilelang_tc=tl_tc,
        ref_cycles=ref_cycles, ref_tc=ref_tc,
    )


# ---------------------------------------------------------------------------
# mHC Post
# ---------------------------------------------------------------------------

_mhc_post_config = get_bench_config("mhc_post", _TIER)
_mhc_post_keys = list(_mhc_post_config.keys())


@pytest.mark.parametrize(",".join(_mhc_post_keys), generate_configs(_mhc_post_config))
def test_bench_mhc_post(n, hidden_size, hc_mult):
    from benchmark_mhc_post import main

    latency, tflops, best_config, ref_latency = main(
        n=n, hidden_size=hidden_size, hc_mult=hc_mult
    )
    assert latency > 0, f"Invalid latency: {latency}"
    assert tflops > 0, f"Invalid TFlops: {tflops}"

    config_str = f"n={n}, hidden_size={hidden_size}, hc_mult={hc_mult}"
    ref_tflops = tflops * latency / ref_latency if ref_latency > 0 else 0
    problem_args = [
        "--n", str(n), "--hidden_size", str(hidden_size), "--hc_mult", str(hc_mult),
    ]
    tl_cycles, tl_tc, ref_cycles, ref_tc = _maybe_profile(
        "mhc_post", problem_args, best_config
    )

    print_benchmark_summary(
        "mHC Post", config_str,
        latency, tflops, ref_latency, ref_tflops,
        "Reference", best_config,
        tilelang_cycles=tl_cycles, tilelang_tc=tl_tc,
        ref_cycles=ref_cycles, ref_tc=ref_tc,
    )


# ---------------------------------------------------------------------------
# V32 Sparse MLA Fwd
# ---------------------------------------------------------------------------

_v32_config = get_bench_config("v32", _TIER)
_v32_keys = list(_v32_config.keys())


@pytest.mark.parametrize(",".join(_v32_keys), generate_configs(_v32_config))
def test_bench_v32(batch, seq_len, seq_len_kv, heads, kv_group, topk, dim, tail_dim):
    from benchmark_v32 import main

    latency, tflops, best_config, ref_latency = main(
        batch=batch, seq_len=seq_len, seq_len_kv=seq_len_kv,
        heads=heads, kv_group=kv_group, topk=topk, dim=dim, tail_dim=tail_dim
    )
    assert latency > 0, f"Invalid latency: {latency}"
    assert tflops > 0, f"Invalid TFlops: {tflops}"

    config_str = (f"batch={batch}, seq_len={seq_len}, seq_len_kv={seq_len_kv}, "
                  f"heads={heads}, kv_group={kv_group}, topk={topk}, "
                  f"dim={dim}, tail_dim={tail_dim}")
    ref_tflops = tflops * latency / ref_latency if ref_latency > 0 else 0
    problem_args = [
        "--batch", str(batch), "--seq_len", str(seq_len),
        "--seq_len_kv", str(seq_len_kv), "--heads", str(heads),
        "--kv_group", str(kv_group), "--topk", str(topk),
        "--dim", str(dim), "--tail_dim", str(tail_dim),
    ]
    tl_cycles, tl_tc, ref_cycles, ref_tc = _maybe_profile(
        "v32", problem_args, best_config
    )

    print_benchmark_summary(
        "V32 Sparse MLA Fwd", config_str,
        latency, tflops, ref_latency, ref_tflops,
        "FlashMLA", best_config,
        tilelang_cycles=tl_cycles, tilelang_tc=tl_tc,
        ref_cycles=ref_cycles, ref_tc=ref_tc,
    )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tilelang.testing
    tilelang.testing.main()
