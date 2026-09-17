"""Shared utilities for DeepSeek autotune benchmarks.

Includes reference implementations, input tensor constructors, FLOPs helpers,
and ncu/acu profiling utilities for collecting cycle and tensor-core metrics.
"""

import os
import re
import subprocess
from typing import Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# FLOPs helpers
# ---------------------------------------------------------------------------

def gemm_flops(M: int, N: int, K: int) -> int:
    """Total FLOPs for a single GEMM (2*M*N*K)."""
    return 2 * M * N * K


def mla_decode_flops(batch: int, heads: int, kv_ctx: int, dim: int, pe_dim: int) -> int:
    """FLOPs for MLA decode: QK + QK_pe + PV."""
    qk_flops = 2 * batch * heads * kv_ctx * (dim + pe_dim)
    pv_flops = 2 * batch * heads * kv_ctx * dim
    return qk_flops + pv_flops


def nsa_flops(batch: int, heads: int, seq_len: int, dim: int, selected_blocks: int, block_size: int) -> int:
    """FLOPs for NSA fwd: QK + PV over selected blocks."""
    total_kv = selected_blocks * block_size
    qk_flops = 2 * batch * heads * seq_len * total_kv * dim
    pv_flops = 2 * batch * heads * seq_len * total_kv * dim
    return qk_flops + pv_flops


# ---------------------------------------------------------------------------
# DeepGEMM FP8 helpers
# ---------------------------------------------------------------------------

def ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token FP8 quantization (group_size=128)."""
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (
        (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n),
        (x_amax / 448.0).view(m, -1),
    )


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-block FP8 quantization (128x128 blocks)."""
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(ceildiv(m, 128) * 128, ceildiv(n, 128) * 128, dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))


def ref_deepgemm_fp8(A_fp8, B_fp8, A_scale, B_scale, out_dtype):
    """Reference FP8 GEMM with per-token/per-block scaling (DeepGEMM style)."""
    M, N, K = A_fp8.shape[0], B_fp8.shape[0], A_fp8.shape[1]
    A_scales = A_scale.view(M // 128, 128, K // 128).permute(0, 2, 1)
    B_scales = B_scale.repeat_interleave(128, dim=1).view(N // 128, K // 128, 128)
    C = torch.zeros(M, N, device="cuda", dtype=out_dtype)
    c_acc = torch.zeros(128, 128, device="cuda", dtype=torch.float32)
    for i in range(ceildiv(M, 128)):
        for j in range(ceildiv(N, 128)):
            c_acc.zero_()
            for k in range(ceildiv(K, 128)):
                c = torch._scaled_mm(
                    A_fp8[i * 128:(i + 1) * 128, k * 128:(k + 1) * 128],
                    B_fp8[j * 128:(j + 1) * 128, k * 128:(k + 1) * 128].T,
                    scale_a=A_scales[i, k].view(128, 1).contiguous(),
                    scale_b=B_scales[j, k].view(1, 128).contiguous(),
                    out_dtype=torch.bfloat16,
                )
                c_acc += c.to(torch.float32)
            C[i * 128:(i + 1) * 128, j * 128:(j + 1) * 128] = c_acc.to(out_dtype)
    return C


# ---------------------------------------------------------------------------
# MLA decode reference
# ---------------------------------------------------------------------------

def ref_mla_decode(q, q_pe, kv, k_pe, softmax_scale=None):
    """Reference MLA decode attention using broadcasting.

    Matches examples/deepseek_mla/example_mla_decode.py::ref_program logic
    but without the einops dependency. Uses implicit broadcasting in einsum
    to avoid materializing expanded tensors.

    Args:
        q: [batch, heads, dim]
        q_pe: [batch, heads, pe_dim]
        kv: [batch, seqlen_kv, kv_head_num, dim]
        k_pe: [batch, seqlen_kv, kv_head_num, pe_dim]
    Returns:
        output: [batch, heads, dim]
    """
    batch, heads, dim = q.shape
    pe_dim = q_pe.shape[-1]
    kv_head_num = kv.shape[2]
    num_head_groups = heads // kv_head_num
    scale = softmax_scale if softmax_scale is not None else (dim + pe_dim) ** -0.5

    # Reshape Q into GQA groups: [B, kv_head, groups, dim]
    # "b (h g) d -> b g h d" with g=num_head_groups means h=kv_head_num
    query = torch.cat([q, q_pe], dim=-1)  # [B, heads, dim+pe]
    query = query.view(batch, kv_head_num, num_head_groups, dim + pe_dim)  # [B, kv_h, groups, D]

    # Key: [B, seqlen, kv_head, dim+pe] -> [B, kv_head, seqlen, dim+pe]
    key = torch.cat([kv, k_pe], dim=-1).permute(0, 2, 1, 3)  # [B, kv_h, S, D]

    # Broadcasting einsum: query[B, kv_h, groups, D] x key[B, kv_h, S, D] -> [B, kv_h, groups, S]
    scores = torch.einsum("bgnd,bgsd->bgns", query.float(), key.float())
    attention = F.softmax(scores * scale, dim=-1)

    # Value: [B, kv_head, seqlen, dim]
    val = kv.permute(0, 2, 1, 3)  # [B, kv_h, S, dim]

    # Output: [B, kv_h, groups, S] x [B, kv_h, S, dim] -> [B, kv_h, groups, dim]
    out = torch.einsum("bgns,bgsd->bgnd", attention, val.float())

    # Reshape back: [B, kv_h, groups, dim] -> [B, heads, dim]
    out = out.reshape(batch, heads, dim)
    return out.to(q.dtype)


# ---------------------------------------------------------------------------
# Diff utility
# ---------------------------------------------------------------------------

def calc_diff(x, y):
    """Cosine-distance based diff metric."""
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


# ---------------------------------------------------------------------------
# Pretty-print helpers (matches benchmark/flash_attention_autotune style)
# ---------------------------------------------------------------------------


def bench_ref(fn, warmup=5, rep=20):
    """Run reference benchmark under do_bench, or skip if TILELANG_SKIP_REF=1.

    Returns 0.0 when skipped so callers should guard against division by zero.
    """
    if os.environ.get("TILELANG_SKIP_REF"):
        return 0.0
    from tilelang.profiler import do_bench
    return do_bench(fn, warmup=warmup, rep=rep)


def _format_ratio(numerator, denominator):
    """Format a TileLang/Ref ratio string, e.g. ``"0.854x"``.

    Profiling bugs (e.g. acu reporting 0 cycles/TC) turn the ratio into a
    0/0 division.  Return ``"Invalid"`` in that case instead of crashing or
    printing a misleading number.
    """
    if denominator:
        return f"{numerator / denominator:.3f}x"
    return "Invalid"


def print_benchmark_summary(name, config_str, tilelang_latency, tilelang_tflops,
                            ref_latency, ref_tflops, ref_name, best_config,
                            tilelang_cycles=None, tilelang_tc=None,
                            ref_cycles=None, ref_tc=None):
    """Print a tabulate grid summary for a single benchmark configuration.

    When cycle/tensor-core profiling data is provided, those metrics are
    appended to the same table instead of being printed separately.
    Mirrors the per-result table format used in
    benchmark/flash_attention_autotune/compare_bench.py.
    """
    from tabulate import tabulate

    print(f"\n[{name}] {config_str}")
    print("-" * 60)
    print(f"tilelang_best_config: {best_config}")

    table_data = [
        ["Metric", "TileLang", ref_name, "Ratio (TL/Ref)"],
        ["Latency (ms)", f"{tilelang_latency:.4f}", f"{ref_latency:.4f}",
         _format_ratio(tilelang_latency, ref_latency)],
        ["TFlops", f"{tilelang_tflops:.2f}", f"{ref_tflops:.2f}",
         _format_ratio(tilelang_tflops, ref_tflops)],
    ]
    if tilelang_cycles is not None and ref_cycles is not None:
        table_data.append(
            ["Cycles", f"{tilelang_cycles:,.0f}", f"{ref_cycles:,.0f}",
             _format_ratio(tilelang_cycles, ref_cycles)]
        )
    if tilelang_tc is not None and ref_tc is not None:
        table_data.append(
            ["TC Efficiency", f"{tilelang_tc:.2f}", f"{ref_tc:.2f}",
             _format_ratio(tilelang_tc, ref_tc)]
        )
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))


# ---------------------------------------------------------------------------
# ncu / acu profiling utilities
# ---------------------------------------------------------------------------
#
# Mirrors benchmark/flash_attention_autotune/utils.py: builds ncu (GPU) or
# acu (PPU) commands, runs the target script as a subprocess, and parses the
# resulting log to extract SM/CE cycle counts and tensor-core efficiency.


def inject_pass_configs_from_env(kernel_func):
    """Inject pass_configs from TILELANG_PASS_CONFIGS env var onto jit_impl.

    The autotune skip path (all tunable params provided) calls jit_compile()
    with no arguments, so per-config pass_configs is lost.  This injects the
    best config's pass_configs directly onto jit_impl before the kernel is
    called in acu/ncu profiling scripts.
    """
    import json
    pc_str = os.environ.get("TILELANG_PASS_CONFIGS", "")
    if pc_str:
        pc = json.loads(pc_str)
        kernel_func.jit_impl.pass_configs = pc


def run_cmd(cmd: str, timeout=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE):
    """Run a shell command and stream its output."""
    print(f"Run command: {cmd}, timeout: {timeout}")
    ret = subprocess.run(
        args=cmd, timeout=timeout, shell=True,
        stdout=stdout, stderr=stderr, encoding="utf-8",
    )
    if stdout:
        for line in ret.stdout.splitlines() + ret.stderr.splitlines():
            print(line)
    if ret.returncode != 0:
        print("Run command failed!")
    else:
        print("Run command succeed!")
    return ret


def _match_any(text: str, patterns) -> bool:
    """Return True if any lowercase pattern is contained in text."""
    if not patterns:
        return False
    text = text.lower()
    return any(pattern.lower() in text for pattern in patterns)


def read_cycle_from_nculog(filename, framework="tilelang", kernel_filters=None,
                           exclude_kernel_filters=None, verbose=True):
    """Parse an ncu/acu ``--page=details`` log file.

    Extracts per-kernel cycle counts (``__cycles_active.max``) and tensor-core
    efficiency (``pct_of_peak_sustained_active``).

    Args:
        filename: Path to the profiler log.
        framework: ``"tilelang"`` uses TileLang main-kernel filters by default;
            ``"ref"`` uses explicit filters if provided and otherwise excludes
            common setup/random/fill kernels.
        kernel_filters: Optional list of substrings to include.
        exclude_kernel_filters: Optional list of substrings to exclude.
        verbose: Print matched kernels for debugging filter correctness.

    Returns:
        ``(cycle_sum, tc_avg)`` – total matched cycles and cycle-weighted
        tensor-core efficiency.
    """
    # Match kernel name lines in ncu/acu --page=details logs.
    # Both ncu and acu emit lines like:
    #   <kernel_name> (<grid>)x(<block>), Device <n>
    # Some PPU kernels (e.g. gemm_ktype0_...) do not contain the word
    # "kernel", so we match on the ", Device <n>" suffix instead.
    kernel_pattern = r",\s*Device\s+\d+"
    cycles_pattern = "__cycles_active.max"
    tc_pattern = "pct_of_peak_sustained_active"
    kernel_list = []
    cycles_list = []
    tc_list = []
    with open(filename, newline="") as log_file:
        for line in log_file.read().split("\n"):
            if re.search(kernel_pattern, line):
                kernel_list.append(line.strip())
            if re.search(cycles_pattern, line):
                cycles_list.append(int(line.strip().split()[-1]))
            if re.search(tc_pattern, line):
                tc_list.append(float(line.strip().split()[-1]))

    assert len(kernel_list) == len(cycles_list), (
        f"kernel/cycle mismatch: {len(kernel_list)} vs {len(cycles_list)}"
    )
    assert len(kernel_list) == len(tc_list), (
        f"kernel/tc mismatch: {len(kernel_list)} vs {len(tc_list)}"
    )

    if kernel_filters is None and framework == "tilelang":
        kernel_filters = ["main_kernel", "main"]
    if exclude_kernel_filters is None and framework == "ref":
        exclude_kernel_filters = [
            "normal", "rand", "randperm", "arange", "fill",
            "copy", "distribution", "vectorized_elementwise",
        ]

    matched = []
    cycle_sum = 0
    tc_weighted_sum = 0.0
    for op, cycle, tc in zip(kernel_list, cycles_list, tc_list):
        include = True
        if kernel_filters:
            include = _match_any(op, kernel_filters)
        if include and exclude_kernel_filters and _match_any(op, exclude_kernel_filters):
            include = False
        if include:
            matched.append((op, cycle, tc))
            cycle_sum += cycle
            tc_weighted_sum += tc * cycle

    if verbose:
        print(f"Matched {framework} kernels from {os.path.basename(filename)}:")
        for op, cycle, tc in matched:
            print(f"  cycles={cycle:>12,}, tc={tc:>8.2f}, kernel={op[:180]}")

    if cycle_sum == 0:
        print("Not valid profiling log or no matching kernels found!")
        return 0, 0

    tc_avg = tc_weighted_sum / cycle_sum
    return cycle_sum, tc_avg


def get_device_type() -> str:
    """Detect whether the current GPU is a PPU or a standard NVIDIA GPU."""
    device_name = torch.cuda.get_device_name().lower()
    if "ppu" in device_name or "zw" in device_name:
        return "ppu"
    return "gpu"


def get_metrics_string(dev: str = "gpu") -> str:
    """Return the profiler metrics string for the given device type."""
    if dev == "gpu":
        return (
            "sm__cycles_active.max,"
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
        )
    else:
        return (
            "ce__cycles_active.max,"
            "cu__inst_executed_pipe_tensor_fp16.avg.pct_of_peak_sustained_active"
        )


def setup_ppu_env():
    """Set PPU-specific environment variables for profiling."""
    os.environ["HGGC_RESET_CACHE"] = "1"
    os.environ["ALIPPU_RESET_CE_MASK"] = "1"


def run_cycle_on_device(
    script_path: str,
    script_args: list,
    dev: str = "gpu",
    log_file: str = "./gpu_cycles_single_case.log",
    framework: str = "tilelang",
    kernel_filters=None,
    exclude_kernel_filters=None,
):
    """Run a Python script under ncu/acu and return ``(cycle, tc)``.

    Args:
        script_path: Path to the benchmark script to profile.
        script_args: List of CLI argument strings to pass to the script.
        dev: ``"gpu"`` (uses ``ncu``) or ``"ppu"`` (uses ``acu``).
        log_file: Where to save the profiler output.
        framework: Passed to :func:`read_cycle_from_nculog` for kernel filtering.
        kernel_filters: Optional substrings of kernel names to include.
        exclude_kernel_filters: Optional substrings of kernel names to exclude.

    Returns:
        ``(cycle, tc)`` – total cycles and cycle-weighted tensor-core efficiency.
    """
    run_cmd(f"rm -f {log_file}")

    profiler = "ncu" if dev == "gpu" else "acu"
    metrics = get_metrics_string(dev)
    args_str = " ".join(str(a) for a in script_args)

    cmd = (
        f'{profiler} --clock-control none --metrics="{metrics}" '
        f'--page=details python {script_path} {args_str} '
        f"2>&1 | tee -a {log_file}"
    )
    run_cmd(cmd)

    cycle, tc = read_cycle_from_nculog(
        log_file, framework=framework,
        kernel_filters=kernel_filters,
        exclude_kernel_filters=exclude_kernel_filters,
    )
    return cycle, tc


def print_cycle_summary(name, config_str, tilelang_latency, tilelang_tflops,
                         tilelang_cycle, tilelang_tc, ref_latency, ref_tflops,
                         ref_name, best_config, ref_cycle=0, ref_tc=0):
    """Print a summary table that includes cycles and tensor-core efficiency
    for both TileLang and the reference."""
    from tabulate import tabulate

    print(f"\n[{name}] {config_str}")
    print("-" * 70)
    print(f"tilelang_best_config: {best_config}")

    table_data = [
        ["Metric", "TileLang", ref_name, "Ratio (TL/Ref)"],
        ["Latency (ms)", f"{tilelang_latency:.4f}", f"{ref_latency:.4f}",
         _format_ratio(tilelang_latency, ref_latency)],
        ["TFlops", f"{tilelang_tflops:.2f}", f"{ref_tflops:.2f}",
         _format_ratio(tilelang_tflops, ref_tflops)],
        ["Cycles", f"{tilelang_cycle:,.0f}", f"{ref_cycle:,.0f}",
         _format_ratio(tilelang_cycle, ref_cycle)],
        ["TC Efficiency", f"{tilelang_tc:.2f}", f"{ref_tc:.2f}",
         _format_ratio(tilelang_tc, ref_tc)],
    ]
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
