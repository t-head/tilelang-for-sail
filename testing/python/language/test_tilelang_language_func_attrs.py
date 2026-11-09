"""Test T.annotate_compile_flags, T.annotate_pass_configs, and out_idx via PrimFunc attrs."""

import importlib
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import tilelang
from tilelang import language as T
from tilelang.transform import PassConfigKey


def test_out_idx_via_attr_lazy():
    """out_idx should be stored as PrimFunc attr when using T.empty + return."""

    @T.prim_func
    def kernel(A):
        A: T.Tensor[[128, 128], T.float32]
        B = T.empty([128, 128], T.float32)
        with T.Kernel(1):
            for i in T.serial(128):
                for j in T.serial(128):
                    B[i, j] = A[i, j] + 1.0
        return B

    assert "tilelang_out_idx" in kernel.attrs
    assert list(kernel.attrs["tilelang_out_idx"]) == [-1]

    compiled = tilelang.compile(kernel)
    a = torch.randn(128, 128, device="cuda")
    b = compiled(a)
    torch.testing.assert_close(b, a + 1.0)


def test_all_attrs_together_lazy():
    """annotate_pass_configs, annotate_compile_flags, and out_idx should all work together."""

    @T.prim_func
    def kernel(A):
        A: T.Tensor[[64, 64], T.float32]
        T.annotate_pass_configs({PassConfigKey.TL_ENABLE_FAST_MATH: True})
        T.annotate_compile_flags(["--use_fast_math"])
        B = T.empty([64, 64], T.float32)
        with T.Kernel(1):
            for i in T.serial(64):
                for j in T.serial(64):
                    B[i, j] = A[i, j] * 2.0
        return B

    attrs = kernel.attrs
    assert "tilelang_out_idx" in attrs
    assert "tilelang_pass_configs" in attrs
    assert "tilelang_compile_flags" in attrs

    compiled = tilelang.compile(kernel)
    a = torch.randn(64, 64, device="cuda")
    b = compiled(a)
    torch.testing.assert_close(b, a * 2.0)


def test_eager_mode_attrs():
    """Eager mode should support annotate_pass_configs and out_idx via T.empty."""

    @tilelang.jit
    def kernel(A):
        M, N = T.const("M N")
        A: T.Tensor[[M, N], T.float32]
        B = T.empty([M, N], T.float32)
        T.annotate_pass_configs({PassConfigKey.TL_ENABLE_FAST_MATH: True})
        with T.Kernel(1):
            for i in T.serial(M):
                for j in T.serial(N):
                    B[i, j] = A[i, j] + 1.0
        return B

    a = torch.randn(32, 32, device="cuda")
    result = kernel(a)
    torch.testing.assert_close(result, a + 1.0)


def test_out_idx_conflict_detection():
    """Specifying both T.empty return and external out_idx should raise ValueError."""

    @T.prim_func
    def kernel(A):
        A: T.Tensor[[32, 32], T.float32]
        B = T.empty([32, 32], T.float32)
        with T.Kernel(1):
            for i in T.serial(32):
                for j in T.serial(32):
                    B[i, j] = A[i, j]
        return B

    with pytest.raises(ValueError, match="Out index conflict"):
        tilelang.compile(kernel, out_idx=[-1])


def test_no_out_idx_when_not_using_empty():
    """When T.empty is not used, tilelang_out_idx attr should not be present."""

    @T.prim_func
    def kernel(A, B):
        A: T.Tensor[[32, 32], T.float32]
        B: T.Tensor[[32, 32], T.float32]
        with T.Kernel(1):
            for i in T.serial(32):
                for j in T.serial(32):
                    B[i, j] = A[i, j]

    assert kernel.attrs is None or "tilelang_out_idx" not in kernel.attrs

    compiled = tilelang.compile(kernel, out_idx=[-1])
    a = torch.randn(32, 32, device="cuda")
    b = compiled(a)
    torch.testing.assert_close(b, a)


def test_pass_configs_only_lazy():
    """annotate_pass_configs should work without T.empty or annotate_compile_flags."""

    @T.prim_func
    def kernel(A, B):
        A: T.Tensor[[32, 32], T.float32]
        B: T.Tensor[[32, 32], T.float32]
        T.annotate_pass_configs({PassConfigKey.TL_ENABLE_FAST_MATH: True})
        with T.Kernel(1):
            for i in T.serial(32):
                for j in T.serial(32):
                    B[i, j] = A[i, j] + 1.0

    assert "tilelang_pass_configs" in kernel.attrs
    assert kernel.attrs is None or "tilelang_out_idx" not in kernel.attrs

    compiled = tilelang.compile(kernel, out_idx=[-1])
    a = torch.randn(32, 32, device="cuda")
    b = compiled(a)
    torch.testing.assert_close(b, a + 1.0)


def test_compile_flags_only_lazy():
    """annotate_compile_flags should work standalone."""

    @T.prim_func
    def kernel(A, B):
        A: T.Tensor[[32, 32], T.float32]
        B: T.Tensor[[32, 32], T.float32]
        T.annotate_compile_flags(["--use_fast_math"])
        with T.Kernel(1):
            for i in T.serial(32):
                for j in T.serial(32):
                    B[i, j] = A[i, j] + 1.0

    assert "tilelang_compile_flags" in kernel.attrs

    compiled = tilelang.compile(kernel, out_idx=[-1])
    a = torch.randn(32, 32, device="cuda")
    b = compiled(a)
    torch.testing.assert_close(b, a + 1.0)


def test_annotations_before_tensor_type():
    """Annotations placed before tensor type annotations should work."""

    @T.prim_func
    def kernel(A, B):
        T.annotate_pass_configs({PassConfigKey.TL_ENABLE_FAST_MATH: True})
        T.annotate_compile_flags(["--use_fast_math"])
        A: T.Tensor[[32, 32], T.float32]
        B: T.Tensor[[32, 32], T.float32]
        with T.Kernel(1):
            for i in T.serial(32):
                for j in T.serial(32):
                    B[i, j] = A[i, j] + 1.0

    assert "tilelang_pass_configs" in kernel.attrs
    assert "tilelang_compile_flags" in kernel.attrs

    compiled = tilelang.compile(kernel, out_idx=[-1])
    a = torch.randn(32, 32, device="cuda")
    b = compiled(a)
    torch.testing.assert_close(b, a + 1.0)


def test_annotations_after_tensor_type():
    """Annotations placed after tensor type annotations should work."""

    @T.prim_func
    def kernel(A, B):
        A: T.Tensor[[32, 32], T.float32]
        B: T.Tensor[[32, 32], T.float32]
        T.annotate_pass_configs({PassConfigKey.TL_ENABLE_FAST_MATH: True})
        T.annotate_compile_flags(["--use_fast_math"])
        with T.Kernel(1):
            for i in T.serial(32):
                for j in T.serial(32):
                    B[i, j] = A[i, j] + 1.0

    assert "tilelang_pass_configs" in kernel.attrs
    assert "tilelang_compile_flags" in kernel.attrs

    compiled = tilelang.compile(kernel, out_idx=[-1])
    a = torch.randn(32, 32, device="cuda")
    b = compiled(a)
    torch.testing.assert_close(b, a + 1.0)


@pytest.fixture
def debug_compile_jit():
    @tilelang.jit
    def kernel():
        @T.prim_func
        def main():
            T.evaluate(0)

        return main

    return kernel


@pytest.mark.parametrize(
    "debug_key",
    [
        "tl.enable_dump_ir",
        PassConfigKey.TL_ENABLE_DUMP_IR,
        "tl.layout_visualization_enable",
        PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE,
    ],
    ids=["dump-string", "dump-enum", "layout-string", "layout-enum"],
)
@pytest.mark.parametrize(
    "attr_value,external_value,locked",
    [(True, None, True), (None, True, True), (True, False, False), (False, True, True), (None, None, False), (False, None, False)],
    ids=["attr-enables", "external-enables", "external-disables", "external-overrides", "no-debug", "attr-disabled"],
)
def test_jit_debug_lock_uses_effective_pass_configs(monkeypatch, debug_compile_jit, debug_key, attr_value, external_value, locked):
    jit_module = importlib.import_module("tilelang.jit")
    jit = debug_compile_jit
    prim_func = jit.get_tir()
    if attr_value is not None:
        prim_func = prim_func.with_attr("tilelang_pass_configs", {debug_key: attr_value})
    if external_value is not None:
        # Exercise enum/string overrides in both directions.
        external_key = debug_key.value if isinstance(debug_key, PassConfigKey) else PassConfigKey(debug_key)
        jit = replace(jit, pass_configs={external_key: external_value})
    events = []
    kernel = object()

    @contextmanager
    def recording_lock():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def cached_stub(**kwargs):
        assert events == (["enter"] if locked else [])
        effective = kwargs["pass_configs"] or {}
        assert bool(effective.get(debug_key, False)) is locked
        return kernel

    monkeypatch.setattr(jit_module, "_DEBUG_COMPILE_LOCK", recording_lock())
    monkeypatch.setattr(jit_module.JITImpl, "get_tir", lambda self: prim_func)
    # Keep the real module-level compile/attribute merge; stop before native compilation.
    monkeypatch.setattr(jit_module, "cached", cached_stub)
    assert jit.compile() is kernel
    assert events == (["enter", "exit"] if locked else [])


def test_jit_debug_root_writes_under_shared_lock_across_copies(monkeypatch, tmp_path, debug_compile_jit):
    jit_module = importlib.import_module("tilelang.jit")
    prim_func = debug_compile_jit.get_tir()
    first = replace(
        debug_compile_jit,
        debug_root_path=str(tmp_path),
        pass_configs={"tl.enable_dump_ir": False, "tl.layout_visualization_enable": False},
    )
    second = replace(first, pass_configs={**first.pass_configs, PassConfigKey.TL_ENABLE_FAST_MATH: True})
    source_file = tmp_path / "tilelang_jit_kernel_kernel.c"
    tir_file = tmp_path / "tilelang_jit_program_kernel.py"
    expected_tir = prim_func.script() + "\n"
    native_lock = jit_module._DEBUG_COMPILE_LOCK
    entered = threading.Event()
    contended = threading.Event()
    writes = []

    class ObservedDebugLock:
        owner = None
        entries = 0

        def __enter__(self):
            if not native_lock.acquire(blocking=False):
                contended.set()
                assert native_lock.acquire(timeout=10), "debug lock was not released"
            self.owner = threading.get_ident()
            self.entries += 1
            return self

        def __exit__(self, *exc):
            self.owner = None
            native_lock.release()

    lock = ObservedDebugLock()

    @contextmanager
    def checked_open(filename, mode):
        assert lock.owner == threading.get_ident()
        with open(filename, mode) as stream:
            yield stream
            assert lock.owner == threading.get_ident()
        writes.append(filename)

    def cached_stub(**kwargs):
        assert lock.owner == threading.get_ident()
        is_second = kwargs["pass_configs"].get(PassConfigKey.TL_ENABLE_FAST_MATH, False)
        if not is_second:
            entered.set()
            assert contended.wait(10), "JIT copies did not contend on the shared debug lock"
        else:
            # The first compile must finish BOTH writes before the second starts.
            assert source_file.read_text() == "// first\n"
            assert tir_file.read_text() == expected_tir
        return SimpleNamespace(get_kernel_source=lambda: "// second" if is_second else "// first")

    monkeypatch.setattr(jit_module, "_DEBUG_COMPILE_LOCK", lock)
    monkeypatch.setattr(jit_module.JITImpl, "get_tir", lambda self: prim_func)
    monkeypatch.setattr(jit_module, "cached", cached_stub)
    monkeypatch.setattr(jit_module, "open", checked_open, raising=False)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_result = pool.submit(first.compile)
        assert entered.wait(10)
        second_result = pool.submit(second.compile)
        assert first_result.result(timeout=15).get_kernel_source() == "// first"
        assert second_result.result(timeout=15).get_kernel_source() == "// second"
    assert lock.entries == 2
    assert lock.owner is None
    assert writes == [str(source_file), str(tir_file)] * 2
    assert source_file.read_text() == "// second\n"
    assert tir_file.read_text() == expected_tir


if __name__ == "__main__":
    test_out_idx_via_attr_lazy()
    test_all_attrs_together_lazy()
    test_eager_mode_attrs()
    test_out_idx_conflict_detection()
    test_no_out_idx_when_not_using_empty()
    test_pass_configs_only_lazy()
    test_compile_flags_only_lazy()
    test_annotations_before_tensor_type()
    test_annotations_after_tensor_type()
    print("All tests passed!")
