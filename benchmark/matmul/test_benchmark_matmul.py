"""Pytest wrapper for matmul benchmark kernels with configurable parameters."""

import sys
import os
import textwrap

import tilelang.language as T

# Ensure benchmark/matmul is in the path so we can import the benchmark scripts
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

import benchmark_matmul
import benchmark_matmul_sp
import benchmark_matmul_intrinsic
import benchmark_matmul_sr
import benchmark_matmul_rs
from utils import print_benchmark_summary, profile_python_script

# Detect compute capability for platform-specific skip logic
from tilelang.contrib import hgcc

_arch = hgcc.get_target_compute_version()
_compute_version = hgcc.parse_compute_version(_arch)


# Set to '1' to also collect acu/ncu cycle and tensor-core utilisation numbers.
PROFILE_CYCLES = os.environ.get("TILELANG_PROFILE_CYCLES", "1") == "1"
# Device for cycle profiling: auto-detected if None.
PROFILE_DEV = os.environ.get("TILELANG_PROFILE_DEV", None)


# ---------------------------------------------------------------------------
# Preset test cases
# ---------------------------------------------------------------------------

DEFAULT_CASES = [
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
    (512, 1024, 768),
    (8192, 4096, 4096),
    (4096, 8192, 8192),
    (2048, 2048, 512),
    (4096, 14336, 4096),
    # (1, 4096, 4096),
    # (1, 11008, 4096),
    # (1, 4096, 11008),
    # (1, 8192, 8192),
    # (8192, 1, 8192),
    # (8192, 8192, 1),
]


def _fmt(m, n, k):
    return f"M{m}_N{n}_K{k}"


def _gemm_flops(m, n, k):
    return 2 * m * n * k


def _record_result(results, name, m, n, k, latency, ref_latency, config):
    flops = _gemm_flops(m, n, k)
    tflops = flops / latency * 1e-9 if latency > 0 else 0.0
    ref_tflops = flops / ref_latency * 1e-9 if ref_latency is not None and ref_latency > 0 else 0.0
    results.append(
        {
            "name": name,
            "m": m,
            "n": n,
            "k": k,
            "latency": latency,
            "ref_latency": ref_latency,
            "tflops": tflops,
            "ref_tflops": ref_tflops,
            "config": config,
        }
    )


def _safe_config_repr(config):
    """Return a repr of config that is valid Python source.

    IntEnum values (e.g. GemmWarpPolicy) are converted to int to avoid
    ``<GemmWarpPolicy.Square: 0>`` syntax errors in generated profiling scripts.
    PassConfigKey enum keys inside nested dicts (e.g. pass_configs) are
    converted to their string values for the same reason.
    """
    from enum import Enum

    safe = {}
    for k, v in config.items():
        if isinstance(v, int) and not isinstance(v, bool) and hasattr(v, "name"):
            safe[k] = int(v)
        elif isinstance(v, dict):
            safe[k] = {kk.value if isinstance(kk, Enum) else kk: vv for kk, vv in v.items()}
        else:
            safe[k] = v
    return repr(safe)


def _pass_configs_inject(config, obj_path):
    """Extract pass_configs from config for direct jit_impl injection.

    The autotune skip path (all tunable params provided) calls jit_compile()
    with no arguments, so per-config pass_configs is lost.  To work around
    this without modifying the autotuner, we set jit_impl.pass_configs
    directly in the profiling script before calling the kernel.

    Returns (inject_line, config_without_pass_configs).
    """
    from enum import Enum

    config = dict(config)
    pass_configs = config.pop("pass_configs", None)

    if pass_configs:
        safe_pc = {k.value if isinstance(k, Enum) else k: v for k, v in pass_configs.items()}
        inject = f"{obj_path}.jit_impl.pass_configs = {safe_pc!r}"
    else:
        inject = ""

    return inject, config


def _maybe_profile(kernel_script, ref_script):
    """Run acu/ncu profiling if TILELANG_PROFILE_CYCLES=1 and return (cycles, tc) tuples.

    Returns (tilelang_cycles, tilelang_tc, ref_cycles, ref_tc). When profiling
    is disabled all four values are None so callers can omit the metrics.
    """
    if not PROFILE_CYCLES:
        return None, None, None, None
    tilelang_cycles, tilelang_tc = profile_python_script(kernel_script, dev=PROFILE_DEV, timeout=300)
    ref_cycles, ref_tc = profile_python_script(ref_script, dev=PROFILE_DEV, timeout=300)
    return tilelang_cycles, tilelang_tc, ref_cycles, ref_tc


# ---------------------------------------------------------------------------
# benchmark_matmul
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "m,n,k",
    DEFAULT_CASES,
    ids=[_fmt(*c) for c in DEFAULT_CASES],
)
def test_matmul(m, n, k, with_roller=False):
    """Run benchmark_matmul and verify results.

    Usage:
        pytest test_benchmark_matmul.py::test_matmul[M4096_N4096_K4096] -v -s
        pytest test_benchmark_matmul.py -v -s
    """
    result = benchmark_matmul.matmul(m, n, k, with_roller)
    assert result.latency > 0, "Autotune returned zero/negative latency"
    assert result.config is not None, "Autotune returned no config"

    flops = _gemm_flops(m, n, k)
    tilelang_tflops = flops / result.latency * 1e-9
    ref_tflops = flops / result.ref_latency * 1e-9 if result.ref_latency is not None else 0.0
    _pc_inject, _config_no_pc = _pass_configs_inject(result.config, "benchmark_matmul.matmul")
    kernel_script = textwrap.dedent(f"""\
        import sys; sys.path.insert(0, '.')
        import torch, benchmark_matmul
        {_pc_inject}
        A = torch.randn({m}, {k}, device='cuda', dtype=torch.float16)
        B = torch.randn({n}, {k}, device='cuda', dtype=torch.float16)
        kernel = benchmark_matmul.matmul({m}, {n}, {k}, False, **{_safe_config_repr(_config_no_pc)})
        def run():
            kernel(A, B)
            torch.cuda.synchronize()
        run()
    """)
    ref_script = textwrap.dedent(f"""\
        import torch
        A = torch.randn({m}, {k}, device='cuda', dtype=torch.float16)
        B = torch.randn({n}, {k}, device='cuda', dtype=torch.float16)
        def run():
            _ = A @ B.T
            torch.cuda.synchronize()
        run()
    """)
    tl_cycles, tl_tc, ref_cycles, ref_tc = _maybe_profile(kernel_script, ref_script)

    print_benchmark_summary(
        "MatMul",
        f"M={m}, N={n}, K={k}",
        result.latency,
        tilelang_tflops,
        result.ref_latency if result.ref_latency is not None else 0.0,
        ref_tflops,
        "Reference",
        result.config,
        tilelang_cycles=tl_cycles,
        tilelang_tc=tl_tc,
        ref_cycles=ref_cycles,
        ref_tc=ref_tc,
    )


# ---------------------------------------------------------------------------
# benchmark_matmul_intrinsic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "m,n,k",
    DEFAULT_CASES,
    ids=[_fmt(*c) for c in DEFAULT_CASES],
)
@pytest.mark.skip(reason="Skipping intrinsic")
def test_matmul_intrinsic(m, n, k, with_roller=False):
    """Run benchmark_matmul_intrinsic and verify results.

    Usage:
        pytest test_benchmark_matmul.py::test_matmul_intrinsic[M4096_N4096_K4096] -v -s
    """
    in_dtype = T.float16
    out_dtype = T.float16
    accum_dtype = T.float32
    result = benchmark_matmul_intrinsic.matmul(m, n, k, in_dtype, out_dtype, accum_dtype, with_roller)
    assert result.latency > 0, "Autotune returned zero/negative latency"
    assert result.config is not None, "Autotune returned no config"

    flops = _gemm_flops(m, n, k)
    tilelang_tflops = flops / result.latency * 1e-9
    ref_tflops = flops / result.ref_latency * 1e-9 if result.ref_latency is not None else 0.0
    _pc_inject, _config_no_pc = _pass_configs_inject(result.config, "benchmark_matmul_intrinsic.matmul")
    kernel_script = textwrap.dedent(f"""\
        import sys; sys.path.insert(0, '.')
        import torch, tilelang.language as T, benchmark_matmul_intrinsic
        {_pc_inject}
        A = torch.randn({m}, {k}, device='cuda', dtype=torch.float16)
        B = torch.randn({n}, {k}, device='cuda', dtype=torch.float16)
        kernel = benchmark_matmul_intrinsic.matmul({m}, {n}, {k}, T.float16, T.float16, T.float32, False, **{_safe_config_repr(_config_no_pc)})
        def run():
            kernel(A, B)
            torch.cuda.synchronize()
        run()
    """)
    ref_script = textwrap.dedent(f"""\
        import torch
        A = torch.randn({m}, {k}, device='cuda', dtype=torch.float16)
        B = torch.randn({n}, {k}, device='cuda', dtype=torch.float16)
        def run():
            _ = A @ B.T
            torch.cuda.synchronize()
        run()
    """)
    tl_cycles, tl_tc, ref_cycles, ref_tc = _maybe_profile(kernel_script, ref_script)

    print_benchmark_summary(
        "MatMul Intrinsic",
        f"M={m}, N={n}, K={k}",
        result.latency,
        tilelang_tflops,
        result.ref_latency if result.ref_latency is not None else 0.0,
        ref_tflops,
        "Reference",
        result.config,
        tilelang_cycles=tl_cycles,
        tilelang_tc=tl_tc,
        ref_cycles=ref_cycles,
        ref_tc=ref_tc,
    )


# ---------------------------------------------------------------------------
# benchmark_matmul_sp (sparsity-aware gemm_sp_v2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "m,n,k",
    DEFAULT_CASES,
    ids=[_fmt(*c) for c in DEFAULT_CASES],
)
@pytest.mark.skipif(
    _compute_version == (1, 5),
    reason="matmul_sp is only supported on PPU sm80",
)
def test_matmul_sp(m, n, k, accum_dtype="float"):
    """Run benchmark_matmul_sp and verify results.

    Usage:
        pytest test_benchmark_matmul.py::test_matmul_sp[M4096_N4096_K4096] -v -s
    """
    result = benchmark_matmul_sp.matmul_sp(m, n, k, T.float16, accum_dtype, "int16")
    assert result.latency > 0, "Autotune returned zero/negative latency"
    assert result.config is not None, "Autotune returned no config"

    flops = _gemm_flops(m, n, k)
    tilelang_tflops = flops / result.latency * 1e-9
    ref_tflops = flops / result.ref_latency * 1e-9 if result.ref_latency is not None else 0.0
    _pc_inject, _config_no_pc = _pass_configs_inject(result.config, "benchmark_matmul_sp.matmul_sp")
    kernel_script = textwrap.dedent(f"""\
        import sys; sys.path.insert(0, '.')
        import torch, tilelang.language as T, benchmark_matmul_sp
        from tilelang.utils.sparse import get_e_factor
        {_pc_inject}
        e_dtype_str = 'int16'
        ef = get_e_factor(T.float16, e_dtype_str)
        edtype = {{'int16': torch.int16, 'uint8': torch.uint8, 'int8': torch.int8, 'int32': torch.int32}}[e_dtype_str]
        A_sparse = torch.randn({m}, {k} // 2, device='cuda', dtype=torch.float16)
        E = torch.randint(0, 10, ({m}, {k} // ef), device='cuda', dtype=edtype)
        B = torch.randn({k}, {n}, device='cuda', dtype=torch.float16)
        config = {_safe_config_repr(_config_no_pc)}
        config['policy'] = T.GemmWarpPolicy.Square
        kernel = benchmark_matmul_sp.matmul_sp({m}, {n}, {k}, T.float16, {accum_dtype!r}, e_dtype_str, config=config)
        def run():
            kernel(A_sparse, E, B)
            torch.cuda.synchronize()
        run()
    """)
    ref_script = textwrap.dedent(f"""\
        import torch
        A = torch.randn({m}, {k}, device='cuda', dtype=torch.float16)
        B = torch.randn({n}, {k}, device='cuda', dtype=torch.float16)
        def run():
            _ = A @ B.T
            torch.cuda.synchronize()
        run()
    """)
    tl_cycles, tl_tc, ref_cycles, ref_tc = _maybe_profile(kernel_script, ref_script)

    print_benchmark_summary(
        "MatMul SP",
        f"M={m}, N={n}, K={k}",
        result.latency,
        tilelang_tflops,
        result.ref_latency if result.ref_latency is not None else 0.0,
        ref_tflops,
        "Reference",
        result.config,
        tilelang_cycles=tl_cycles,
        tilelang_tc=tl_tc,
        ref_cycles=ref_cycles,
        ref_tc=ref_tc,
    )


# ---------------------------------------------------------------------------
# benchmark_matmul_rs (Register Source: A shared→register, B shared)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "m,n,k",
    DEFAULT_CASES,
    ids=[_fmt(*c) for c in DEFAULT_CASES],
)
def test_matmul_rs(m, n, k, with_roller=False):
    """Run benchmark_matmul_rs and verify results.

    matmul_rs loads A into register (fragment) before MMA (register source),
    while B stays in shared memory (shared source). Requires trans_A=True
    for K-major A register layout. dtype: f16/f16/f32.

    Usage:
        pytest test_benchmark_matmul.py::test_matmul_rs[M4096_N4096_K4096] -v -s
    """
    result = benchmark_matmul_rs.matmul(m, n, k, with_roller)
    assert result.latency > 0, "Autotune returned zero/negative latency"
    assert result.config is not None, "Autotune returned no config"

    flops = _gemm_flops(m, n, k)
    tilelang_tflops = flops / result.latency * 1e-9
    ref_tflops = flops / result.ref_latency * 1e-9 if result.ref_latency is not None else 0.0
    _pc_inject, _config_no_pc = _pass_configs_inject(result.config, "benchmark_matmul_rs.matmul")
    kernel_script = textwrap.dedent(f"""\
        import sys; sys.path.insert(0, '.')
        import torch, tilelang.language as T, benchmark_matmul_rs
        {_pc_inject}
        A = torch.randn({k}, {m}, device='cuda', dtype=torch.float16)
        B = torch.randn({k}, {n}, device='cuda', dtype=torch.float16)
        kernel = benchmark_matmul_rs.matmul({m}, {n}, {k}, False, T.float16, T.float16, T.float32, **{_safe_config_repr(_config_no_pc)})
        def run():
            kernel(A, B)
            torch.cuda.synchronize()
        run()
    """)
    ref_script = textwrap.dedent(f"""\
        import torch
        A = torch.randn({k}, {m}, device='cuda', dtype=torch.float16)
        B = torch.randn({k}, {n}, device='cuda', dtype=torch.float16)
        def run():
            _ = A.T @ B
            torch.cuda.synchronize()
        run()
    """)
    tl_cycles, tl_tc, ref_cycles, ref_tc = _maybe_profile(kernel_script, ref_script)

    print_benchmark_summary(
        "MatMul RS",
        f"M={m}, N={n}, K={k}",
        result.latency,
        tilelang_tflops,
        result.ref_latency if result.ref_latency is not None else 0.0,
        ref_tflops,
        "Reference",
        result.config,
        tilelang_cycles=tl_cycles,
        tilelang_tc=tl_tc,
        ref_cycles=ref_cycles,
        ref_tc=ref_tc,
    )


# ---------------------------------------------------------------------------
# benchmark_matmul_sr (Shared-Register: A shared, B shared→register)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "m,n,k",
    DEFAULT_CASES,
    ids=[_fmt(*c) for c in DEFAULT_CASES],
)
def test_matmul_sr(m, n, k, with_roller=False):
    """Run benchmark_matmul_sr and verify results.

    matmul_sr loads B into register (fragment) before MMA (register source),
    while A stays in shared memory (shared source). Requires trans_B=True
    for N-major B register layout. dtype: f16/f16/f32.

    Usage:
        pytest test_benchmark_matmul.py::test_matmul_sr[M4096_N4096_K4096] -v -s
    """
    result = benchmark_matmul_sr.matmul(m, n, k, with_roller)
    assert result.latency > 0, "Autotune returned zero/negative latency"
    assert result.config is not None, "Autotune returned no config"

    flops = _gemm_flops(m, n, k)
    tilelang_tflops = flops / result.latency * 1e-9
    ref_tflops = flops / result.ref_latency * 1e-9 if result.ref_latency is not None else 0.0
    _pc_inject, _config_no_pc = _pass_configs_inject(result.config, "benchmark_matmul_sr.matmul")
    kernel_script = textwrap.dedent(f"""\
        import sys; sys.path.insert(0, '.')
        import torch, tilelang.language as T, benchmark_matmul_sr
        {_pc_inject}
        A = torch.randn({m}, {k}, device='cuda', dtype=torch.float16)
        B = torch.randn({n}, {k}, device='cuda', dtype=torch.float16)
        kernel = benchmark_matmul_sr.matmul({m}, {n}, {k}, False, **{_safe_config_repr(_config_no_pc)})
        def run():
            kernel(A, B)
            torch.cuda.synchronize()
        run()
    """)
    ref_script = textwrap.dedent(f"""\
        import torch
        A = torch.randn({m}, {k}, device='cuda', dtype=torch.float16)
        B = torch.randn({n}, {k}, device='cuda', dtype=torch.float16)
        def run():
            _ = A @ B.T
            torch.cuda.synchronize()
        run()
    """)
    tl_cycles, tl_tc, ref_cycles, ref_tc = _maybe_profile(kernel_script, ref_script)

    print_benchmark_summary(
        "MatMul SR",
        f"M={m}, N={n}, K={k}",
        result.latency,
        tilelang_tflops,
        result.ref_latency if result.ref_latency is not None else 0.0,
        ref_tflops,
        "Reference",
        result.config,
        tilelang_cycles=tl_cycles,
        tilelang_tc=tl_tc,
        ref_cycles=ref_cycles,
        ref_tc=ref_tc,
    )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tilelang.testing

    tilelang.testing.main()
