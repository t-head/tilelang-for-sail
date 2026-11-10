# PPU variant of the dynamic-shape / callee-allocated-ABI cases in
# testing/python/language/test_tilelang_language_func_attrs.py: dynamic-shape
# T.empty outputs are allocated by the TVM-FFI packed ABI on PPU (anchor and
# output prototype stay torch-visible as kDLCUDA).
import pytest
import torch
import tilelang
import tilelang.testing
from tilelang.ppu import language as T


@tilelang.testing.requires_ppu
def test_empty_dynamic_shape_is_allocated_by_tvm_ffi():
    """T.empty shape expressions should be evaluated by the packed ABI binder."""

    @tilelang.jit(target="ppu")
    def kernel(A):
        m = T.dynamic("m")
        A: T.Tensor[[m], T.float32]
        B = T.empty([m * 2 + 1], T.float32)
        with T.Kernel(1, threads=1):
            for i in T.serial(m * 2 + 1):
                B[i] = A[i % m] + 1.0
        return B

    a = torch.arange(37, dtype=torch.float32, device="cuda")
    b = kernel(a)

    assert b.shape == (75,)
    torch.testing.assert_close(b, a.repeat(3)[:75] + 1.0)


@tilelang.testing.requires_ppu
def test_empty_dynamic_shape_from_scalar_uses_ffi_allocator_anchor():
    """Scalar-only kernels should still install Torch's FFI allocator."""

    @tilelang.jit(target="ppu")
    def kernel(m: T.int32):
        B = T.empty([m + 1], T.float32)
        with T.Kernel(1, threads=1):
            for i in T.serial(m + 1):
                B[i] = T.cast(i, T.float32)
        return B

    b = kernel(23)

    assert b.shape == (24,)
    torch.testing.assert_close(b, torch.arange(24, dtype=torch.float32, device="cuda"))


@pytest.mark.parametrize(
    ("logical_dtype", "expected_torch_dtype", "expected_shape"),
    [
        (T.int4, torch.int8, (3, 8)),
        (
            T.float4_e2m1fn,
            getattr(torch, "float4_e2m1fn_x2", torch.int8),
            (3, 8),
        ),
        pytest.param(
            T.float4_e2m1fnx2,
            getattr(torch, "float4_e2m1fn_x2", torch.int8),
            (3, 16),
            marks=pytest.mark.skipif(
                not hasattr(torch, "float4_e2m1fn_x2"),
                reason="PyTorch float4_e2m1fn_x2 dtype is unavailable",
            ),
        ),
    ],
)
@tilelang.testing.requires_ppu
def test_empty_subbyte_output_uses_torch_storage_dtype(
    logical_dtype,
    expected_torch_dtype,
    expected_shape,
):
    """FFI allocation should expose packed Torch storage for sub-byte TIR."""

    @T.prim_func
    def kernel():
        output = T.empty((3, 16), logical_dtype)
        with T.Kernel(1, threads=1):
            T.evaluate(0)
        return output

    compiled = tilelang.compile(kernel, execution_backend="tvm_ffi", target="ppu")
    output = compiled()

    assert compiled.adapter._ffi_callee_allocated_output_abi
    assert "TVMFFIEnvTensorAlloc" in compiled.get_host_source()
    assert output.dtype == expected_torch_dtype
    assert output.shape == expected_shape


@tilelang.testing.requires_ppu
def test_empty_dynamic_subbyte_output_packs_symbolic_final_dimension():
    """Storage-shape packing should remain symbolic until the packed call."""

    @T.prim_func
    def kernel(n: T.int32):
        output = T.empty((3, n * 2), T.float4_e2m1fn)
        with T.Kernel(1, threads=1):
            T.evaluate(0)
        return output

    compiled = tilelang.compile(kernel, execution_backend="tvm_ffi", target="ppu")
    output = compiled(11)

    expected_dtype = getattr(torch, "float4_e2m1fn_x2", torch.int8)
    assert compiled.adapter.dynamic_symbolic_map is None
    assert output.dtype == expected_dtype
    assert output.shape == (3, 11)


@tilelang.testing.requires_ppu
def test_multiple_empty_outputs_are_returned_from_tvm_ffi():
    """Multiple FFI-allocated outputs should preserve TileLang's list API."""

    @tilelang.jit(target="ppu")
    def kernel(A):
        m = T.dynamic("m")
        A: T.Tensor[[m], T.float32]
        B = T.empty([m], T.float32)
        C = T.empty([m + 1], T.float32)
        with T.Kernel(1, threads=1):
            for i in T.serial(m):
                B[i] = A[i] + 1.0
            for i in T.serial(m + 1):
                C[i] = T.cast(i, T.float32)
        return B, C

    a = torch.arange(29, dtype=torch.float32, device="cuda")
    outputs = kernel(a)

    assert isinstance(outputs, list)
    assert [tuple(value.shape) for value in outputs] == [(29,), (30,)]
    torch.testing.assert_close(outputs[0], a + 1.0)
    torch.testing.assert_close(outputs[1], torch.arange(30, dtype=torch.float32, device="cuda"))


@tilelang.testing.requires_ppu
def test_manual_out_idx_multiple_dynamic_outputs_are_allocated_by_tvm_ffi():
    """Manual output indices should share T.empty's native allocation ABI."""

    @T.prim_func
    def kernel(A, B, C):
        m = T.dynamic("m")
        A: T.Tensor[[m], T.float32]
        B: T.Tensor[[m * 2 + 1], T.float32]
        C: T.Tensor[[m + 1], T.float32]
        with T.Kernel(1, threads=1):
            for i in T.serial(m * 2 + 1):
                B[i] = A[i % m] + 1.0
            for i in T.serial(m + 1):
                C[i] = T.cast(i, T.float32)

    assert kernel.attrs is None or "tilelang_out_idx" not in kernel.attrs

    compiled = tilelang.compile(kernel, out_idx=[1, 2], execution_backend="tvm_ffi", target="ppu")
    assert list(compiled.adapter.prim_func.attrs["tilelang_out_idx"]) == [1, 2]
    assert compiled.adapter.dynamic_symbolic_map is None

    a = torch.arange(19, dtype=torch.float32, device="cuda")
    outputs = compiled(a)

    assert isinstance(outputs, list)
    assert [tuple(value.shape) for value in outputs] == [(39,), (20,)]
    torch.testing.assert_close(outputs[0], a.repeat(3)[:39] + 1.0)
    torch.testing.assert_close(outputs[1], torch.arange(20, dtype=torch.float32, device="cuda"))


if __name__ == "__main__":
    tilelang.testing.main()
