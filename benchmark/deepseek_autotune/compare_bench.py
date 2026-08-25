"""DeepSeek autotune benchmark with ncu/acu cycle and tensor-core profiling.

For each kernel configuration:
1. Run autotune (main()) to find best config and measure latency / TFlops
   for both TileLang and the reference.  main() prints its own table.
2. Run the TileLang kernel under ncu/acu (--profile) to collect cycles / tc.
3. Run the reference under ncu/acu (--profile-ref) to collect cycles / tc.
4. Print a final summary table comparing TileLang vs reference:
   Latency, TFlops, Cycles, TC Efficiency.

Usage:
    # Run with default DAILY config
    python compare_bench.py

    # Run with SINGLE or FULL config
    BENCHMARK_CONFIG=SINGLE python compare_bench.py

    # Run specific kernel only
    python compare_bench.py --kernel mla

    # Run one pytest case, aligned with flash_attention_autotune
    python -m pytest 'benchmark/deepseek_autotune/compare_bench.py::test_bench_comparison[MLA_B1_H128_KVH1_KVCTX8192_D512_PE64]' -s -v
"""

import argparse
import gc
import itertools
import os
import sys

import torch
import pytest
from tabulate import tabulate

# ---------------------------------------------------------------------------
# Path setup – ensure sibling modules are importable regardless of CWD
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from bench_configs import get_bench_config
from utils import (
    run_cycle_on_device,
    get_device_type,
    setup_ppu_env,
)

_TIER = os.environ.get("BENCHMARK_CONFIG", "DAILY").upper()


# ---------------------------------------------------------------------------
# Device detection (mirrors flash_attention_autotune/compare_bench.py)
# ---------------------------------------------------------------------------

def detect_device():
    """Detect GPU vs PPU and set PPU env vars if needed."""
    dev = get_device_type()
    if dev == "ppu":
        setup_ppu_env()
    return dev


def _ref_kernel_filters(kernel_name):
    """Return reference kernel-name filters for profiler log parsing."""
    filters = {
        # FlashMLA references launch setup kernels plus flash_fwd kernels;
        # only flash_fwd kernels correspond to the benchmarked reference path.
        "MLA Decode": ["flash_fwd"],
        "V32 Sparse MLA Fwd": ["flash"],
    }
    return filters.get(kernel_name)


# ---------------------------------------------------------------------------
# Per-kernel profiling helpers
# ---------------------------------------------------------------------------
#
# Each helper:
#   1. Calls the benchmark's main() to obtain best_config, latency, tflops,
#      and ref_latency.
#   2. Builds CLI args for the --profile mode of the same script.
#   3. Runs the script under ncu/acu via run_cycle_on_device().
#   4. Returns a result dict.


def _bench_with_cycles(kernel_name, script_name, main_func, main_kwargs,
                       config_keys, problem_args, dev):
    """Generic helper: run main() then profile both kernel and reference.

    1. Calls main() to obtain best_config, latency, tflops, ref_latency.
    2. Runs the TileLang kernel under ncu/acu (--profile) to get TL cycles/tc.
    3. Runs the reference under ncu/acu (--profile-ref) to get ref cycles/tc.

    Returns a result dict with all metrics, or None on failure.
    """
    print(f"\n{'=' * 70}")
    print(f"Running: {kernel_name} with {main_kwargs}")
    print(f"{'=' * 70}")

    try:
        latency, tflops, best_config, ref_latency = main_func(**main_kwargs)
    except Exception as e:
        print(f"[{kernel_name}] main() failed: {e}")
        return None

    # ref_tflops from the same total_flops: tflops * latency / ref_latency
    ref_tflops = tflops * latency / ref_latency if ref_latency > 0 else 0

    config_str = ", ".join(f"{k}={v}" for k, v in main_kwargs.items())
    script_path = os.path.join(_THIS_DIR, script_name)
    log_file = os.path.join(_THIS_DIR, "gpu_cycles_single_case.log")

    # --- Profile TileLang kernel ---
    profile_args = list(problem_args)
    profile_args.append("--profile")
    for key in config_keys:
        profile_args.extend([f"--{key}", str(best_config[key])])

    gc.collect()
    torch.cuda.empty_cache()
    tl_cycle, tl_tc = run_cycle_on_device(
        script_path, profile_args, dev=dev, log_file=log_file,
        framework="tilelang",
    )

    # --- Profile reference ---
    ref_profile_args = list(problem_args)
    ref_profile_args.append("--profile-ref")

    gc.collect()
    torch.cuda.empty_cache()
    ref_cycle, ref_tc = run_cycle_on_device(
        script_path, ref_profile_args, dev=dev, log_file=log_file,
        framework="ref", kernel_filters=_ref_kernel_filters(kernel_name),
    )

    gc.collect()
    torch.cuda.empty_cache()

    # --- Per-config comparison table (aligned with flash_attention format) ---
    print(f"\nResults for {kernel_name} [{config_str}]:")
    print("-" * 60)
    print(f"tilelang_best_config: {best_config}")

    if (tl_cycle > 0 and ref_cycle > 0 and tl_tc > 0 and ref_tc > 0
            and ref_latency > 0 and ref_tflops > 0):
        table_data = [
            ["Metric", "TileLang", "Reference", "Ratio (T/Ref)"],
            ["Latency (ms)", f"{latency:.4f}", f"{ref_latency:.4f}",
             f"{latency / ref_latency:.3f}x"],
            ["TFlops", f"{tflops:.2f}", f"{ref_tflops:.2f}",
             f"{tflops / ref_tflops:.3f}x"],
            ["Cycles", f"{tl_cycle:,.0f}", f"{ref_cycle:,.0f}",
             f"{tl_cycle / ref_cycle:.3f}x"],
            ["TC Eff", f"{tl_tc:.2f}", f"{ref_tc:.2f}",
             f"{tl_tc / ref_tc:.3f}x"],
        ]
        print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
    else:
        print(f"  TileLang:  {latency:.4f} ms, {tflops:.2f} TFlops, {tl_cycle:,.0f} cycles, TC={tl_tc:.2f}")
        print(f"  Reference: {ref_latency:.4f} ms, {ref_tflops:.2f} TFlops, {ref_cycle:,.0f} cycles, TC={ref_tc:.2f}")

    return {
        "kernel": kernel_name,
        "config": config_str,
        "latency": latency,
        "tflops": tflops,
        "cycle": tl_cycle,
        "tc": tl_tc,
        "ref_latency": ref_latency,
        "ref_tflops": ref_tflops,
        "ref_cycle": ref_cycle,
        "ref_tc": ref_tc,
        "speedup": ref_latency / latency if latency > 0 else 0,
        "best_config": best_config,
    }


def _get_ref_name(kernel_name):
    """Return the reference implementation name for a given kernel."""
    ref_map = {
        "MLA Decode": "FlashMLA",
        "MLA Decode Paged": "Reference",
        "DeepGEMM FP8": "Reference",
        "NSA Fwd": "Reference",
        "NSA Decode": "Reference",
        "mHC Pre": "Reference",
        "mHC BigFuse": "Reference",
        "mHC Post": "Reference",
        "V32 Sparse MLA Fwd": "FlashMLA",
    }
    return ref_map.get(kernel_name, "Reference")


def _case_id(kernel_key, params):
    """Build pytest case ids similar to flash_attention_autotune."""
    if kernel_key == "mla":
        return (f"MLA_B{params['batch']}_H{params['heads']}_KVH{params['kv_heads']}"
                f"_KVCTX{params['kv_ctx']}_D{params['dim']}_PE{params['pe_dim']}")
    if kernel_key == "mla_paged":
        return (f"MLA_PAGED_B{params['batch']}_HQ{params['h_q']}_HKV{params['h_kv']}"
                f"_L{params['cache_seqlen']}_D{params['d']}_DV{params['dv']}")
    if kernel_key == "deepgemm":
        return (f"DEEPGEMM_M{params['M']}_N{params['N']}_K{params['K']}"
                f"_{params['in_dtype']}_{params['out_dtype']}")
    if kernel_key == "nsa":
        causal_str = "Causal" if params.get("is_causal", True) else "NonCausal"
        return (f"NSA_B{params['batch']}_H{params['heads']}_L{params['seq_len']}"
                f"_D{params['dim']}_SB{params['selected_blocks']}"
                f"_BS{params['block_size']}_{causal_str}")
    if kernel_key == "nsa_decode":
        return (f"NSA_DECODE_B{params['batch']}_H{params['heads']}_L{params['seq_len']}"
                f"_D{params['dim']}_SB{params['selected_blocks']}_BS{params['block_size']}")
    if kernel_key == "mhc":
        return f"MHC_N{params['n']}_H{params['hidden_size']}_M{params['hc_mult']}"
    if kernel_key == "mhc_big_fuse":
        return f"MHC_BIG_FUSE_N{params['n']}_H{params['hidden_size']}_M{params['hc_mult']}"
    if kernel_key == "mhc_post":
        return f"MHC_POST_N{params['n']}_H{params['hidden_size']}_M{params['hc_mult']}"
    if kernel_key == "v32":
        return (f"V32_B{params['batch']}_L{params['seq_len']}_LKV{params['seq_len_kv']}"
                f"_H{params['heads']}_KVG{params['kv_group']}_TOPK{params['topk']}"
                f"_D{params['dim']}_TAIL{params['tail_dim']}")
    return f"{kernel_key.upper()}_" + "_".join(f"{k}{v}" for k, v in params.items())


def generate_benchmark_cases():
    """Generate pytest.param cases for all DeepSeek profiling benchmarks."""
    cases = []
    kernel_keys = [
        "mla", "mla_paged", "deepgemm", "nsa", "nsa_decode",
        "mhc", "mhc_big_fuse", "mhc_post", "v32",
    ]
    for kernel_key in kernel_keys:
        cfg = get_bench_config(kernel_key, _TIER)
        for combo in itertools.product(*[cfg[k] for k in cfg]):
            params = dict(zip(cfg.keys(), combo))
            cases.append(pytest.param(kernel_key, params, id=_case_id(kernel_key, params)))
    return cases


def run_benchmark_case(kernel_key, params, dev):
    """Run exactly one benchmark config, used by pytest node-id selection."""
    if kernel_key == "deepgemm" and dev == "ppu":
        pytest.skip("DeepGEMM FP8 is skipped on PPU 1.0 (no FP8 support).")

    if kernel_key == "mla":
        from benchmark_mla import main as mla_main
        problem_args = [
            "--batch", str(params["batch"]), "--heads", str(params["heads"]),
            "--kv_heads", str(params["kv_heads"]), "--kv_ctx", str(params["kv_ctx"]),
            "--dim", str(params["dim"]), "--pe_dim", str(params["pe_dim"]),
        ]
        return _bench_with_cycles("MLA Decode", "benchmark_mla.py", mla_main, params,
                                  ["block_N", "block_H", "num_stages", "threads"], problem_args, dev)

    if kernel_key == "mla_paged":
        from benchmark_mla_paged import main as mla_paged_main
        problem_args = [
            "--batch", str(params["batch"]), "--h_q", str(params["h_q"]),
            "--h_kv", str(params["h_kv"]), "--cache_seqlen", str(params["cache_seqlen"]),
            "--d", str(params["d"]), "--dv", str(params["dv"]),
        ]
        return _bench_with_cycles("MLA Decode Paged", "benchmark_mla_paged.py", mla_paged_main, params,
                                  ["block_N", "block_H", "num_stages", "threads"], problem_args, dev)

    if kernel_key == "deepgemm":
        from benchmark_deepgemm import main as deepgemm_main
        problem_args = [
            "--m", str(params["M"]), "--n", str(params["N"]), "--k", str(params["K"]),
            "--in_dtype", str(params["in_dtype"]), "--out_dtype", str(params["out_dtype"]),
        ]
        main_kwargs = {
            "M": params["M"], "N": params["N"], "K": params["K"],
            "in_dtype_str": params["in_dtype"], "out_dtype_str": params["out_dtype"],
        }
        return _bench_with_cycles("DeepGEMM FP8", "benchmark_deepgemm.py", deepgemm_main, main_kwargs,
                                  ["block_N", "num_stages", "threads", "enable_rasteration"], problem_args, dev)

    if kernel_key == "nsa":
        from benchmark_nsa import main as nsa_main
        problem_args = [
            "--batch", str(params["batch"]), "--heads", str(params["heads"]),
            "--seq_len", str(params["seq_len"]), "--dim", str(params["dim"]),
            "--selected_blocks", str(params["selected_blocks"]), "--block_size", str(params["block_size"]),
        ]
        if params.get("is_causal", True):
            problem_args.append("--causal")
        main_kwargs = {
            "batch": params["batch"], "heads": params["heads"], "seq_len": params["seq_len"],
            "dim": params["dim"], "selected_blocks": params["selected_blocks"],
            "block_size": params["block_size"], "is_causal": params.get("is_causal", True),
        }
        return _bench_with_cycles("NSA Fwd", "benchmark_nsa.py", nsa_main, main_kwargs,
                                  ["num_stages", "threads"], problem_args, dev)

    if kernel_key == "nsa_decode":
        from benchmark_nsa_decode import main as nsa_decode_main
        problem_args = [
            "--batch", str(params["batch"]), "--heads", str(params["heads"]),
            "--seq_len", str(params["seq_len"]), "--dim", str(params["dim"]),
            "--selected_blocks", str(params["selected_blocks"]), "--block_size", str(params["block_size"]),
        ]
        return _bench_with_cycles("NSA Decode", "benchmark_nsa_decode.py", nsa_decode_main, params,
                                  ["num_stages", "threads"], problem_args, dev)

    if kernel_key == "mhc":
        from benchmark_mhc import main as mhc_main
        problem_args = ["--n", str(params["n"]), "--hidden_size", str(params["hidden_size"]), "--hc_mult", str(params["hc_mult"])]
        return _bench_with_cycles("mHC Pre", "benchmark_mhc.py", mhc_main, params,
                                  ["token_block", "hidden_block", "num_stages"], problem_args, dev)

    if kernel_key == "mhc_big_fuse":
        from benchmark_mhc_big_fuse import main as mhc_bf_main
        problem_args = ["--n", str(params["n"]), "--hidden_size", str(params["hidden_size"]), "--hc_mult", str(params["hc_mult"])]
        return _bench_with_cycles("mHC BigFuse", "benchmark_mhc_big_fuse.py", mhc_bf_main, params,
                                  ["threads", "num_stages"], problem_args, dev)

    if kernel_key == "mhc_post":
        from benchmark_mhc_post import main as mhc_post_main
        problem_args = ["--n", str(params["n"]), "--hidden_size", str(params["hidden_size"]), "--hc_mult", str(params["hc_mult"])]
        return _bench_with_cycles("mHC Post", "benchmark_mhc_post.py", mhc_post_main, params,
                                  ["n_thr", "h_blk"], problem_args, dev)

    if kernel_key == "v32":
        from benchmark_v32 import main as v32_main
        problem_args = [
            "--batch", str(params["batch"]), "--seq_len", str(params["seq_len"]),
            "--seq_len_kv", str(params["seq_len_kv"]), "--heads", str(params["heads"]),
            "--kv_group", str(params["kv_group"]), "--topk", str(params["topk"]),
            "--dim", str(params["dim"]), "--tail_dim", str(params["tail_dim"]),
        ]
        return _bench_with_cycles("V32 Sparse MLA Fwd", "benchmark_v32.py", v32_main, params,
                                  ["block_I", "num_stages", "threads"], problem_args, dev)

    raise ValueError(f"Unknown kernel: {kernel_key}")


PYTEST_BENCHMARK_CASES = generate_benchmark_cases()


@pytest.mark.parametrize("kernel_key, params", PYTEST_BENCHMARK_CASES)
def test_bench_comparison(kernel_key, params):
    """Pytest entry aligned with flash_attention_autotune compare_bench.py."""
    dev = detect_device()
    result = run_benchmark_case(kernel_key, params, dev)
    assert result is not None, f"Benchmark failed for {kernel_key}: {params}"
    assert result["latency"] > 0, f"Invalid TileLang latency for {kernel_key}: {params}"
    assert result["tflops"] > 0, f"Invalid TileLang TFlops for {kernel_key}: {params}"
    assert result["ref_latency"] > 0, f"Invalid reference latency for {kernel_key}: {params}"


# ---------------------------------------------------------------------------
# Per-kernel benchmark runners
# ---------------------------------------------------------------------------

def bench_mla(dev):
    from benchmark_mla import main as mla_main
    cfg = get_bench_config("mla", _TIER)
    results = []
    for combo in itertools.product(*[cfg[k] for k in cfg]):
        params = dict(zip(cfg.keys(), combo))
        problem_args = [
            "--batch", str(params["batch"]),
            "--heads", str(params["heads"]),
            "--kv_heads", str(params["kv_heads"]),
            "--kv_ctx", str(params["kv_ctx"]),
            "--dim", str(params["dim"]),
            "--pe_dim", str(params["pe_dim"]),
        ]
        r = _bench_with_cycles(
            "MLA Decode", "benchmark_mla.py", mla_main, params,
            ["block_N", "block_H", "num_stages", "threads"],
            problem_args, dev,
        )
        if r:
            results.append(r)
    return results


def bench_mla_paged(dev):
    from benchmark_mla_paged import main as mla_paged_main
    cfg = get_bench_config("mla_paged", _TIER)
    results = []
    for combo in itertools.product(*[cfg[k] for k in cfg]):
        params = dict(zip(cfg.keys(), combo))
        problem_args = [
            "--batch", str(params["batch"]),
            "--h_q", str(params["h_q"]),
            "--h_kv", str(params["h_kv"]),
            "--cache_seqlen", str(params["cache_seqlen"]),
            "--d", str(params["d"]),
            "--dv", str(params["dv"]),
        ]
        r = _bench_with_cycles(
            "MLA Decode Paged", "benchmark_mla_paged.py", mla_paged_main, params,
            ["block_N", "block_H", "num_stages", "threads"],
            problem_args, dev,
        )
        if r:
            results.append(r)
    return results


def bench_deepgemm(dev):
    from benchmark_deepgemm import main as deepgemm_main
    cfg = get_bench_config("deepgemm", _TIER)
    results = []
    for combo in itertools.product(*[cfg[k] for k in cfg]):
        params = dict(zip(cfg.keys(), combo))
        problem_args = [
            "--m", str(params["M"]),
            "--n", str(params["N"]),
            "--k", str(params["K"]),
            "--in_dtype", str(params["in_dtype"]),
            "--out_dtype", str(params["out_dtype"]),
        ]
        main_kwargs = {
            "M": params["M"], "N": params["N"], "K": params["K"],
            "in_dtype_str": params["in_dtype"], "out_dtype_str": params["out_dtype"],
        }
        r = _bench_with_cycles(
            "DeepGEMM FP8", "benchmark_deepgemm.py", deepgemm_main, main_kwargs,
            ["block_N", "num_stages", "threads", "enable_rasteration"],
            problem_args, dev,
        )
        if r:
            results.append(r)
    return results


def bench_nsa(dev):
    from benchmark_nsa import main as nsa_main
    cfg = get_bench_config("nsa", _TIER)
    results = []
    for combo in itertools.product(*[cfg[k] for k in cfg]):
        params = dict(zip(cfg.keys(), combo))
        problem_args = [
            "--batch", str(params["batch"]),
            "--heads", str(params["heads"]),
            "--seq_len", str(params["seq_len"]),
            "--dim", str(params["dim"]),
            "--selected_blocks", str(params["selected_blocks"]),
            "--block_size", str(params["block_size"]),
        ]
        if params.get("is_causal", True):
            problem_args.append("--causal")
        main_kwargs = {
            "batch": params["batch"], "heads": params["heads"],
            "seq_len": params["seq_len"], "dim": params["dim"],
            "selected_blocks": params["selected_blocks"],
            "block_size": params["block_size"],
            "is_causal": params.get("is_causal", True),
        }
        r = _bench_with_cycles(
            "NSA Fwd", "benchmark_nsa.py", nsa_main, main_kwargs,
            ["num_stages", "threads"],
            problem_args, dev,
        )
        if r:
            results.append(r)
    return results


def bench_nsa_decode(dev):
    from benchmark_nsa_decode import main as nsa_decode_main
    cfg = get_bench_config("nsa_decode", _TIER)
    results = []
    for combo in itertools.product(*[cfg[k] for k in cfg]):
        params = dict(zip(cfg.keys(), combo))
        problem_args = [
            "--batch", str(params["batch"]),
            "--heads", str(params["heads"]),
            "--seq_len", str(params["seq_len"]),
            "--dim", str(params["dim"]),
            "--selected_blocks", str(params["selected_blocks"]),
            "--block_size", str(params["block_size"]),
        ]
        r = _bench_with_cycles(
            "NSA Decode", "benchmark_nsa_decode.py", nsa_decode_main, params,
            ["num_stages", "threads"],
            problem_args, dev,
        )
        if r:
            results.append(r)
    return results


def bench_mhc(dev):
    from benchmark_mhc import main as mhc_main
    cfg = get_bench_config("mhc", _TIER)
    results = []
    for combo in itertools.product(*[cfg[k] for k in cfg]):
        params = dict(zip(cfg.keys(), combo))
        problem_args = [
            "--n", str(params["n"]),
            "--hidden_size", str(params["hidden_size"]),
            "--hc_mult", str(params["hc_mult"]),
        ]
        r = _bench_with_cycles(
            "mHC Pre", "benchmark_mhc.py", mhc_main, params,
            ["token_block", "hidden_block", "num_stages"],
            problem_args, dev,
        )
        if r:
            results.append(r)
    return results


def bench_mhc_big_fuse(dev):
    from benchmark_mhc_big_fuse import main as mhc_bf_main
    cfg = get_bench_config("mhc_big_fuse", _TIER)
    results = []
    for combo in itertools.product(*[cfg[k] for k in cfg]):
        params = dict(zip(cfg.keys(), combo))
        problem_args = [
            "--n", str(params["n"]),
            "--hidden_size", str(params["hidden_size"]),
            "--hc_mult", str(params["hc_mult"]),
        ]
        r = _bench_with_cycles(
            "mHC BigFuse", "benchmark_mhc_big_fuse.py", mhc_bf_main, params,
            ["threads", "num_stages"],
            problem_args, dev,
        )
        if r:
            results.append(r)
    return results


def bench_mhc_post(dev):
    from benchmark_mhc_post import main as mhc_post_main
    cfg = get_bench_config("mhc_post", _TIER)
    results = []
    for combo in itertools.product(*[cfg[k] for k in cfg]):
        params = dict(zip(cfg.keys(), combo))
        problem_args = [
            "--n", str(params["n"]),
            "--hidden_size", str(params["hidden_size"]),
            "--hc_mult", str(params["hc_mult"]),
        ]
        r = _bench_with_cycles(
            "mHC Post", "benchmark_mhc_post.py", mhc_post_main, params,
            ["n_thr", "h_blk"],
            problem_args, dev,
        )
        if r:
            results.append(r)
    return results


def bench_v32(dev):
    from benchmark_v32 import main as v32_main
    cfg = get_bench_config("v32", _TIER)
    results = []
    for combo in itertools.product(*[cfg[k] for k in cfg]):
        params = dict(zip(cfg.keys(), combo))
        problem_args = [
            "--batch", str(params["batch"]),
            "--seq_len", str(params["seq_len"]),
            "--seq_len_kv", str(params["seq_len_kv"]),
            "--heads", str(params["heads"]),
            "--kv_group", str(params["kv_group"]),
            "--topk", str(params["topk"]),
            "--dim", str(params["dim"]),
            "--tail_dim", str(params["tail_dim"]),
        ]
        r = _bench_with_cycles(
            "V32 Sparse MLA Fwd", "benchmark_v32.py", v32_main, params,
            ["block_I", "num_stages", "threads"],
            problem_args, dev,
        )
        if r:
            results.append(r)
    return results


# ---------------------------------------------------------------------------
# Kernel registry
# ---------------------------------------------------------------------------

KERNEL_RUNNERS = {
    "mla": bench_mla,
    "mla_paged": bench_mla_paged,
    "deepgemm": bench_deepgemm,
    "nsa": bench_nsa,
    "nsa_decode": bench_nsa_decode,
    "mhc": bench_mhc,
    "mhc_big_fuse": bench_mhc_big_fuse,
    "mhc_post": bench_mhc_post,
    "v32": bench_v32,
}


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------

def print_summary(all_results):
    """Print a final summary table with TL and ref: latency, TFlops, cycles, tc."""
    print(f"\n{'=' * 90}")
    print("SUMMARY (with ncu/acu profiling)")
    print(f"{'=' * 90}\n")

    summary_data = []
    success_count = 0
    for r in all_results:
        if isinstance(r.get("latency"), (int, float)) and isinstance(r.get("ref_latency"), (int, float)):
            summary_data.append([
                r["kernel"],
                r["config"][:40],
                f"{r['latency']:.4f}",
                f"{r['ref_latency']:.4f}",
                f"{r['tflops']:.2f}",
                f"{r['ref_tflops']:.2f}",
                f"{r['cycle']:,.0f}",
                f"{r['ref_cycle']:,.0f}",
                f"{r['tc']:.2f}",
                f"{r['ref_tc']:.2f}",
                f"{r['speedup']:.3f}x",
            ])
            success_count += 1

    print(f"Successfully completed: {success_count}/{len(all_results)} configurations\n")

    print(tabulate(
        summary_data,
        headers=["Kernel", "Config",
                 "TL (ms)", "Ref (ms)",
                 "TL TF", "Ref TF",
                 "TL Cycles", "Ref Cycles",
                 "TL TC", "Ref TC",
                 "Speedup"],
        tablefmt="grid",
    ))

    # Save to file (compact format, aligned with flash_attention)
    output_file = os.path.join(_THIS_DIR, "comparison_results.txt")
    with open(output_file, "w") as f:
        f.write("DeepSeek Autotune Benchmark Results (with ncu/acu profiling)\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"Config tier: {_TIER}\n")
        f.write(f"Total configs: {len(all_results)}, Successful: {success_count}\n\n")
        for r in all_results:
            f.write(f"Kernel: {r['kernel']}  Config: {r['config']}\n")
            f.write(f"  TileLang:  {r['latency']:.4f} ms, {r['tflops']:.2f} TFlops, {r['cycle']:,.0f} cycles, TC={r['tc']:.2f}\n")
            f.write(f"  Reference: {r['ref_latency']:.4f} ms, {r['ref_tflops']:.2f} TFlops, {r['ref_cycle']:,.0f} cycles, TC={r['ref_tc']:.2f}\n")
            f.write(f"  Speedup: {r['speedup']:.3f}x, Best Config: {r['best_config']}\n")
            f.write("\n")

    print(f"\nDetailed results saved to: {output_file}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DeepSeek autotune benchmark with ncu/acu profiling"
    )
    parser.add_argument(
        "--kernel", type=str, default=None,
        help="Run only a specific kernel (e.g. mla, nsa, v32). "
             "Default: run all kernels.",
    )
    args = parser.parse_args()

    print(f"=== DeepSeek Autotune Benchmark with Profiling (tier: {_TIER}) ===\n")

    dev = detect_device()
    print(f"Detected device: {dev} (profiler: {'acu' if dev == 'ppu' else 'ncu'})\n")

    # Select which kernels to run
    if args.kernel:
        kernel_keys = [args.kernel]
    else:
        kernel_keys = list(KERNEL_RUNNERS.keys())

    all_results = []
    for kname in kernel_keys:
        if kname not in KERNEL_RUNNERS:
            print(f"Unknown kernel: {kname}. Available: {list(KERNEL_RUNNERS.keys())}")
            continue

        if kname == "deepgemm":
            print("\n[NOTE] DeepGEMM FP8 is skipped on PPU 1.0 (no FP8 support).")
            if dev == "ppu":
                continue

        print(f"\n--- {kname} ---")
        try:
            results = KERNEL_RUNNERS[kname](dev)
            all_results.extend(results)
        except Exception as e:
            print(f"[{kname}] failed: {e}")
            import traceback
            traceback.print_exc()

    if all_results:
        print_summary(all_results)
    else:
        print("\nNo results to summarize.")
