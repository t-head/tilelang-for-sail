"""Tests for per-config pass_configs auto-tuning support."""

import importlib
import itertools
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
import tilelang.testing
import tilelang
import tilelang.language as T
from tilelang.autotuner import AutoTuner, autotune
from tilelang.autotuner.tuner import AutoTuneImpl, _PASS_CONFIGS_KEY
from tilelang.transform import PassConfigKey


def _get_test_configs():
    """Generate test configs using Cartesian product, including pass_configs variations."""
    block_M = [64, 128]
    block_N = [64, 128]
    block_K = [64, 128]
    warp_spec = [True, False]
    _configs = list(itertools.product(block_M, block_N, block_K, warp_spec))
    return [
        {
            "block_M": c[0],
            "block_N": c[1],
            "block_K": c[2],
            "pass_configs": {PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: c[3]},
        }
        for c in _configs
    ]


def _get_test_configs_mixed():
    """Generate configs where some have pass_configs and others don't."""
    block_M = [64, 128]
    block_N = [64, 128]
    block_K = [64, 128]
    configs = []
    for bm, bn, bk in itertools.product(block_M, block_N, block_K):
        # Some configs with pass_configs
        configs.append(
            {
                "block_M": bm,
                "block_N": bn,
                "block_K": bk,
                "pass_configs": {PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
            }
        )
        # Some configs without pass_configs (use global default)
        configs.append(
            {
                "block_M": bm,
                "block_N": bn,
                "block_K": bk,
            }
        )
    return configs


def _make_matmul_kernel(M, N, K):
    """Helper to create a simple matmul kernel factory."""

    def kernel(block_M=None, block_N=None, block_K=None):

        @T.prim_func
        def main(
            A: T.Tensor((M, K), T.float16),
            B: T.Tensor((N, K), T.float16),
            C: T.Tensor((M, N), T.float16),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), T.float16)
                B_shared = T.alloc_shared((block_N, block_K), T.float16)
                C_local = T.alloc_fragment((block_M, block_N), T.float32)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                T.copy(C_local, C[by * block_M, bx * block_N])

        return main

    return kernel


@tilelang.testing.requires_cuda
def test_autotune_with_per_config_pass_configs():
    """Test that per-config pass_configs are correctly applied during autotuning."""
    M, N, K = 128, 128, 128
    kernel = _make_matmul_kernel(M, N, K)

    configs = _get_test_configs()

    autotuner = (
        AutoTuner.from_kernel(kernel=kernel, configs=configs)
        .set_compile_args(
            out_idx=[-1],
            target="auto",
        )
        .set_profile_args(
            supply_type=tilelang.TensorSupplyType.Integer,
            skip_check=True,
        )
    )
    result = autotuner.run(warmup=1, rep=3)
    assert result is not None
    assert result.config is not None
    assert result.kernel is not None


@tilelang.testing.requires_cuda
def test_autotune_mixed_pass_configs():
    """Test configs where some have pass_configs and others don't."""
    M, N, K = 128, 128, 128
    kernel = _make_matmul_kernel(M, N, K)

    configs = _get_test_configs_mixed()

    autotuner = (
        AutoTuner.from_kernel(kernel=kernel, configs=configs)
        .set_compile_args(
            out_idx=[-1],
            target="auto",
        )
        .set_profile_args(
            supply_type=tilelang.TensorSupplyType.Integer,
            skip_check=True,
        )
    )
    result = autotuner.run(warmup=1, rep=3)
    assert result is not None
    assert result.config is not None


@tilelang.testing.requires_cuda
def test_autotune_pass_configs_merge_with_global():
    """Test that per-config pass_configs merge correctly over global pass_configs."""
    M, N, K = 128, 128, 128
    kernel = _make_matmul_kernel(M, N, K)

    # Global pass_configs set via set_compile_args
    global_pass_configs = {PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False}

    configs = _get_test_configs()

    autotuner = (
        AutoTuner.from_kernel(kernel=kernel, configs=configs)
        .set_compile_args(
            out_idx=[-1],
            target="auto",
            pass_configs=global_pass_configs,
        )
        .set_profile_args(
            supply_type=tilelang.TensorSupplyType.Integer,
            skip_check=True,
        )
    )
    result = autotuner.run(warmup=1, rep=3)
    assert result is not None
    assert result.config is not None


@tilelang.testing.requires_cuda
def test_autotune_decorator_with_per_config_pass_configs():
    """Test @autotune + @tilelang.jit decorator pattern with per-config pass_configs."""
    M, N, K = 128, 128, 128

    configs = _get_test_configs()

    @autotune(configs=configs, warmup=1, rep=3, skip_check=True, supply_type=tilelang.TensorSupplyType.Integer)
    @tilelang.jit(out_idx=[-1])
    def matmul_kernel(M, N, K, block_M=64, block_N=64, block_K=64):

        @T.prim_func
        def main(
            A: T.Tensor((M, K), T.float16),
            B: T.Tensor((N, K), T.float16),
            C: T.Tensor((M, N), T.float16),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), T.float16)
                B_shared = T.alloc_shared((block_N, block_K), T.float16)
                C_local = T.alloc_fragment((block_M, block_N), T.float32)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                T.copy(C_local, C[by * block_M, bx * block_N])

        return main

    kernel = matmul_kernel(M, N, K, __return_kernel=True)
    assert kernel is not None


@tilelang.testing.requires_cuda
def test_autotune_decorator_pass_configs_override_jit_global():
    """Test that per-config pass_configs in @autotune override @tilelang.jit's global pass_configs."""
    M, N, K = 128, 128, 128

    configs = _get_test_configs_mixed()

    @autotune(configs=configs, warmup=1, rep=3, skip_check=True, supply_type=tilelang.TensorSupplyType.Integer)
    @tilelang.jit(out_idx=[-1], pass_configs={PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
    def matmul_kernel(M, N, K, block_M=64, block_N=64, block_K=64):

        @T.prim_func
        def main(
            A: T.Tensor((M, K), T.float16),
            B: T.Tensor((N, K), T.float16),
            C: T.Tensor((M, N), T.float16),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                A_shared = T.alloc_shared((block_M, block_K), T.float16)
                B_shared = T.alloc_shared((block_N, block_K), T.float16)
                C_local = T.alloc_fragment((block_M, block_N), T.float32)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                T.copy(C_local, C[by * block_M, bx * block_N])

        return main

    kernel = matmul_kernel(M, N, K, __return_kernel=True)
    assert kernel is not None


@pytest.fixture
def compile_only_jit():
    @tilelang.jit(pass_configs={PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
    def kernel(size, block=16):
        @T.prim_func
        def main(A: T.Tensor((size,), T.float32)):
            T.evaluate(block)

        return main

    kernel.initialize_jit_mode(128, block=16)
    return kernel


@pytest.mark.parametrize("mode", ["lazy", "eager"])
@pytest.mark.parametrize("override", [False, True])
def test_autotune_independent_phase1_keys_overlap(monkeypatch, compile_only_jit, mode, override):
    jit = compile_only_jit
    jit.mode = mode
    jit.func.set_mode(mode)
    original = jit.pass_configs
    rendezvous = threading.Barrier(2)

    def compile_stub(self, *args, **kwargs):
        assert args == (128,)
        block = kwargs["block"]
        expected = dict(original)
        if override:
            expected[PassConfigKey.TL_ENABLE_FAST_MATH] = block == 16
        assert (self is jit) is (not override)
        assert self.pass_configs == expected
        rendezvous.wait(timeout=10)
        assert jit.pass_configs is original
        assert self.pass_configs == expected
        return block

    jit_module = importlib.import_module("tilelang.jit")
    monkeypatch.setattr(jit_module.JITImpl, "compile", compile_stub)
    compile_config = AutoTuneImpl(jit)._make_jit_compile_func(mode, (128,), {"block": 8})
    configs = [{"block": block} for block in (16, 32)]
    if override:
        for config in configs:
            config[_PASS_CONFIGS_KEY] = {PassConfigKey.TL_ENABLE_FAST_MATH: config["block"] == 16}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(compile_config, **config) for config in configs]
        assert [future.result(timeout=15) for future in futures] == [16, 32]


@pytest.mark.parametrize("override", [False, True])
def test_autotune_equivalent_phase1_keys_serialize_and_reuse_kernel(monkeypatch, compile_only_jit, override):
    jit = compile_only_jit
    entered = threading.Event()
    contended = threading.Event()
    compiled = []
    kernel = object()

    class ObservedLock:
        def __init__(self):
            self.lock = threading.Lock()

        def __enter__(self):
            if not self.lock.acquire(blocking=False):
                contended.set()
                assert self.lock.acquire(timeout=10), "phase1 lock was not released"
            return self

        def __exit__(self, *exc):
            self.lock.release()

    def compile_stub(self, *args, **kwargs):
        assert (self is jit) is (not override)
        assert kwargs == {"block": 16}
        compiled.append(self.pass_configs)
        entered.set()
        assert contended.wait(10), "equivalent configuration did not wait for the cold compile"
        return kernel

    tuner_module = importlib.import_module("tilelang.autotuner.tuner")
    jit_module = importlib.import_module("tilelang.jit")
    # Replace only the tuner's lock factory, not threading.Lock process-wide.
    monkeypatch.setattr(tuner_module, "threading", SimpleNamespace(Lock=ObservedLock))
    monkeypatch.setattr(jit_module.JITImpl, "compile", compile_stub)
    compile_config = AutoTuneImpl(jit)._make_jit_compile_func("lazy", (128,), {"block": 16})
    fast_math = PassConfigKey.TL_ENABLE_FAST_MATH
    first_config = {_PASS_CONFIGS_KEY: {fast_math: True}} if override else {}
    second_config = {_PASS_CONFIGS_KEY: {fast_math: False}} if override else {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(compile_config, **first_config)
        assert entered.wait(10)
        # Different raw configs (and pass configs), but the same phase1 key.
        second = pool.submit(compile_config, block=16, **second_config)
        assert first.result(timeout=15) is kernel
        assert second.result(timeout=15) is kernel
    if override:
        assert compiled == [{**jit.pass_configs, fast_math: True}, {**jit.pass_configs, fast_math: False}]
        assert not jit._kernel_cache
    else:
        assert compiled == [jit.pass_configs]
        assert compile_config(block=16) is kernel
        assert len(compiled) == len(jit._kernel_cache) == 1


@pytest.mark.parametrize(
    "global_configs", [None, {PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True, PassConfigKey.TL_ENABLE_FAST_MATH: False}]
)
def test_autotune_pass_config_copies_survive_compile_exception(monkeypatch, compile_only_jit, global_configs):
    jit = replace(compile_only_jit, pass_configs=global_configs)
    sibling = replace(jit)
    original = dict(global_configs or {})
    copies = []
    fast_math = PassConfigKey.TL_ENABLE_FAST_MATH

    def compile_stub(self, *args, **kwargs):
        # Check during compilation as well as afterwards: temporary mutation is unsafe.
        assert jit.pass_configs is global_configs
        assert sibling.pass_configs is global_configs
        assert dict(jit.pass_configs or {}) == original
        assert self is not jit and self is not sibling
        assert self.pass_configs is not global_configs
        assert self.func is jit.func
        copies.append(self)
        if len(copies) == 2:
            assert kwargs["block"] == 32
            raise RuntimeError("injected compile failure")
        return self

    jit_module = importlib.import_module("tilelang.jit")
    monkeypatch.setattr(jit_module.JITImpl, "compile", compile_stub)
    compile_config = AutoTuneImpl(jit)._make_jit_compile_func("lazy", (128,), {})
    override = {fast_math: True}
    first = compile_config(block=16, **{_PASS_CONFIGS_KEY: override})
    with pytest.raises(RuntimeError, match="injected compile failure"):
        compile_config(block=32, **{_PASS_CONFIGS_KEY: {fast_math: False}})
    # An empty override still copies the JIT; retrying also checks exception unlock.
    retry = compile_config(block=32, **{_PASS_CONFIGS_KEY: {}})
    assert first.pass_configs == {**original, fast_math: True}
    assert copies[1].pass_configs == {**original, fast_math: False}
    assert retry.pass_configs == original
    assert len({id(copy) for copy in copies}) == 3
    assert override == {fast_math: True}
    assert jit.pass_configs is sibling.pass_configs is global_configs
    assert dict(jit.pass_configs or {}) == original


@pytest.mark.parametrize("override", [None, {}])
def test_autotune_eager_dispatch_only_compiles(monkeypatch, compile_only_jit, override):
    jit = compile_only_jit
    jit.mode = "eager"
    jit.func.set_mode("eager")
    kernel = object()
    calls = []

    def forbidden_call(*args, **kwargs):
        pytest.fail("eager autotuning must not invoke JITImpl.__call__ or execute a kernel")

    def compile_stub(self, *args, **kwargs):
        assert (self is jit) is (override is None)
        calls.append((args, kwargs))
        return kernel

    jit_module = importlib.import_module("tilelang.jit")
    monkeypatch.setattr(jit_module.JITImpl, "__call__", forbidden_call)
    monkeypatch.setattr(jit_module.JITImpl, "compile", compile_stub)
    compile_config = AutoTuneImpl(jit)._make_jit_compile_func("eager", (128,), {"block": 8})
    config = {"block": 32}
    if override is not None:
        config[_PASS_CONFIGS_KEY] = override
    assert compile_config(**config) is kernel
    assert calls == [((128,), {"block": 32})]


if __name__ == "__main__":
    tilelang.testing.main()
