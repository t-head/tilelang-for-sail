"""Shared utilities for matmul autotune benchmarks."""

import os
import re
import subprocess
import tempfile

import tilelang


def with_aiu_lower_tuning(configs):
    """Expand each config with TL_DISABLE_AIU_LOWER True/False variants.

    Only applied on PPU 1.5; on PPU 1.0 or non-PPU targets the original
    configs are returned unchanged.

    PPU 1.5 compute_version is (1, 5).
    """
    try:
        from tilelang.contrib import hgcc
        _arch = hgcc.get_target_compute_version()
        _compute_version = hgcc.parse_compute_version(_arch)
        if _compute_version != (1, 5):
            return configs
    except Exception:
        return configs

    expanded = []
    for cfg in configs:
        for val in (True, False):
            new_cfg = dict(cfg)
            new_cfg["pass_configs"] = {
                tilelang.PassConfigKey.TL_DISABLE_AIU_LOWER: val
            }
            expanded.append(new_cfg)
    return expanded


def print_benchmark_summary(name, config_str, tilelang_latency, tilelang_tflops,
                            ref_latency, ref_tflops, ref_name, best_config,
                            tilelang_cycles=None,
                            ref_cycles=None):
    """Print a tabulate grid summary for a single benchmark configuration.

    When cycle profiling data is provided, that metric is appended to the
    same table instead of being printed separately.
    """
    from tabulate import tabulate

    print(f"\n[{name}] {config_str}")
    print("-" * 60)
    print(f"tilelang_best_config: {best_config}")

    ratio_latency = tilelang_latency / ref_latency if ref_latency > 0 else float("nan")
    ratio_tflops = tilelang_tflops / ref_tflops if ref_tflops > 0 else float("nan")

    table_data = [
        ["Metric", "TileLang", ref_name, "Ratio (TL/Ref)"],
        ["Latency (ms)", f"{tilelang_latency * 1000:.4f}", f"{ref_latency * 1000:.4f}", f"{ratio_latency:.3f}x"],
        ["TFlops", f"{tilelang_tflops:.2f}", f"{ref_tflops:.2f}", f"{ratio_tflops:.3f}x"],
    ]
    if tilelang_cycles is not None and ref_cycles is not None:
        ratio_cycles = tilelang_cycles / ref_cycles if ref_cycles > 0 else float("nan")
        table_data.append(
            ["Cycles", str(tilelang_cycles), str(ref_cycles), f"{ratio_cycles:.3f}x"]
        )
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))


# ---------------------------------------------------------------------------
# Cycle profiling helpers (acu/ncu)
# ---------------------------------------------------------------------------

GPU_METRICS = "sm__cycles_active.max"
PPU_METRICS = "ce__cycles_active.max"


def _detect_dev():
    """Return 'ppu' if acu is available, otherwise 'gpu' if ncu is available."""
    if subprocess.run("command -v acu", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
        return "ppu"
    if subprocess.run("command -v ncu", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0:
        return "gpu"
    return None


def _profiler_binary(dev):
    if dev == "ppu":
        return "ASIGHT_FEATURE_INSTRUCTION_COUNT=0 acu"
    if dev == "gpu":
        return "ncu"
    raise ValueError(f"Unsupported dev: {dev}")


def _metrics_string(dev):
    if dev == "ppu":
        return PPU_METRICS
    if dev == "gpu":
        return GPU_METRICS
    raise ValueError(f"Unsupported dev: {dev}")


def _run_cmd(cmd, timeout=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE):
    print(f"Run command: {cmd}, timeout: {timeout}")
    ret = subprocess.run(
        args=cmd, timeout=timeout, shell=True, stdout=stdout, stderr=stderr, encoding="utf-8"
    )
    if stdout:
        for line in (ret.stdout or "").splitlines() + (ret.stderr or "").splitlines():
            print(line)
    if ret.returncode != 0:
        print("Run command failed!")
    else:
        print("Run command succeed!")
    return ret


def _read_cycle_from_nculog(filename):
    """Parse an acu/ncu details log and return a list of (kernel_name, cycles)."""
    # Match kernel summary lines such as "  main_kernel (...), Device 0"
    # (acu/PPU) or "  main_kernel (...), Context 1, Stream 7, Device 0, CC 9.0"
    # (ncu/NVIDIA).
    kernel_pattern = r"\s+\S+\s+\(.*\),.*Device\s+\d+"
    cycles_pattern = "__cycles_active.max"

    kernel_list = []
    cycles_list = []
    with open(filename, newline="") as log_file:
        for line in log_file.read().split("\n"):
            if re.search(kernel_pattern, line):
                kernel_list.append(line.strip())
            elif re.search(cycles_pattern, line):
                cycles_list.append(int(line.strip().split()[-1]))

    if len(kernel_list) != len(cycles_list):
        print("Not valid cycle log file!")
        return []

    return [(kernel_list[i], cycles_list[i]) for i in range(len(kernel_list))]


def _dominant_kernel(filename, target_substring=None):
    """Pick the dominant kernel from an acu/ncu log.

    If target_substring is provided, only consider kernels whose name contains it.
    Otherwise identify the compute kernel by matching known kernel name patterns:
      - "main_kernel"  (TileLang generated kernel)
      - "gemm"          (cuBLAS gemm_ktype0_* on PPU)
      - "nvjet"         (cuDNN nvjet_sm90_* on NVIDIA GPUs)
    Raises RuntimeError if no known compute kernel is found.
    """
    entries = _read_cycle_from_nculog(filename)
    if not entries:
        raise RuntimeError(f"No valid cycle entries parsed from {filename}")

    if target_substring:
        matched = [e for e in entries if target_substring in e[0]]
        if matched:
            entries = matched

    # Identify the actual compute kernel by known name patterns.
    for substring in ("main_kernel", "gemm", "nvjet"):
        matched = [e for e in entries if substring in e[0]]
        if matched:
            entries = matched
            break
    else:
        kernel_names = [e[0][:80] for e in entries]
        raise RuntimeError(
            f"No known compute kernel found in {filename}. "
            f"Found kernels: {kernel_names}"
        )

    # Prefer higher cycle count.
    entries.sort(key=lambda e: e[1], reverse=True)
    return entries[0][1]


def profile_python_script(script_code, dev=None, log_file="./gpu_cycles_single_case.log", timeout=None):
    """Profile a Python code snippet with acu/ncu and return cycles."""
    if dev is None:
        dev = _detect_dev()
    if dev is None:
        raise RuntimeError("Neither acu nor ncu found; cannot profile cycles.")

    profiler = _profiler_binary(dev)
    metrics = _metrics_string(dev)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(script_code)
        script_path = f.name

    _run_cmd(f"rm -f {log_file}")
    cmd = (
        f'{profiler} --clock-control none --metrics="{metrics}" '
        f'--page=details python {script_path} 2>&1 | tee -a {log_file}'
    )
    _run_cmd(cmd, timeout=timeout, stdout=None, stderr=None)

    return _dominant_kernel(log_file)


def print_cycle_summary(name, config_str, tilelang_cycles,
                        ref_cycles, ref_name="Reference"):
    """Print a tabulate grid summary of cycle profiling results."""
    from tabulate import tabulate

    ratio_cycles = tilelang_cycles / ref_cycles if ref_cycles > 0 else float("nan")

    print(f"\n[{name}] {config_str} — Cycle profile")
    print("-" * 60)
    table_data = [
        ["Metric", "TileLang", ref_name, "Ratio (TL/Ref)"],
        ["Cycles", str(tilelang_cycles), str(ref_cycles), f"{ratio_cycles:.3f}x"],
    ]
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
