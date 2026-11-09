import pytest
import torch

import tilelang
import tilelang.ppu.language as T
import tilelang.testing


@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "int32"])
@pytest.mark.parametrize("operation", ["sum", "max"])
@pytest.mark.parametrize("reverse", [False, True])
def test_ppu_scan_multisegment_tail(dtype, operation, reverse):
    extent = 96

    @T.prim_func
    def kernel(src: T.Tensor((extent,), dtype), dst: T.Tensor((extent,), dtype)):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((extent,), dtype)
            T.copy(src, shared)
            if operation == "sum":
                T.cumsum(shared, dim=0, reverse=reverse)
            else:
                T.cummax(shared, dim=0, reverse=reverse)
            T.copy(shared, dst)

    compiled = tilelang.compile(kernel, target="ppu", execution_backend="tvm_ffi")
    if dtype == "int32":
        src = torch.randint(-100, 100, (extent,), dtype=torch.int32, device="cuda")
    elif dtype in ("float16", "bfloat16"):
        src = torch.randint(-2, 3, (extent,), dtype=torch.int32, device="cuda").to(getattr(torch, dtype))
    else:
        src = torch.randn(extent, dtype=getattr(torch, dtype), device="cuda")
    dst = torch.empty_like(src)
    compiled(src, dst)
    torch.cuda.synchronize()
    if operation == "sum":
        expected = src.flip(0).cumsum(0).flip(0) if reverse else src.cumsum(0)
    else:
        expected = src.flip(0).cummax(0).values.flip(0) if reverse else src.cummax(0).values
    expected = expected.to(dst.dtype)
    tolerance = 1e-2 if dtype in ("float16", "bfloat16") else 1e-4
    torch.testing.assert_close(dst, expected, atol=tolerance, rtol=tolerance)


@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
@pytest.mark.parametrize("operation", ["sum", "max"])
def test_ppu_reduce_fp32x2(operation):
    rows = 128

    @T.prim_func
    def kernel(src: T.Tensor((rows, 64), "float32"), dst: T.Tensor((rows,), "float32")):
        with T.Kernel(1, threads=256):
            shared = T.alloc_shared((rows, 64), "float32")
            reduced = T.alloc_fragment((rows,), "float32")
            T.copy(src, shared, disable_tma=True)
            if operation == "sum":
                T.reduce_sum(shared, reduced, dim=1, batch=2)
            else:
                T.reduce_max(shared, reduced, dim=1, batch=2)
            T.copy(reduced, dst)

    compiled = tilelang.compile(kernel, target="ppu", execution_backend="tvm_ffi")
    assert f"{operation.capitalize()}Op_f32x2" in compiled.get_kernel_source()
    src = torch.randn(rows, 64, dtype=torch.float32, device="cuda")
    dst = torch.empty(rows, dtype=torch.float32, device="cuda")
    compiled(src, dst)
    torch.cuda.synchronize()
    expected = src.sum(dim=1) if operation == "sum" else src.max(dim=1).values
    torch.testing.assert_close(dst, expected, atol=1e-4, rtol=1e-4)


@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
@pytest.mark.parametrize("operation", ["add", "sub", "mul", "fma"])
def test_ppu_f32x2_binary_ops_runtime(operation):
    size = 64

    @T.prim_func
    def kernel(
        src: T.Tensor((size,), "float32"),
        other: T.Tensor((size,), "float32"),
        dst: T.Tensor((size,), "float32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            idx = T.Ramp(tx * 2, 1, 2)
            if operation == "add":
                dst[idx] = src[idx] + other[idx]
            elif operation == "sub":
                dst[idx] = src[idx] - other[idx]
            elif operation == "mul":
                dst[idx] = src[idx] * other[idx]
            else:
                dst[idx] = src[idx] * other[idx] + src[idx]

    compiled = tilelang.compile(kernel, target="ppu", execution_backend="tvm_ffi")
    src = torch.randn(size, dtype=torch.float32, device="cuda")
    other = torch.randn(size, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    compiled(src, other, dst)
    torch.cuda.synchronize()
    if operation == "add":
        expected = src + other
    elif operation == "sub":
        expected = src - other
    elif operation == "mul":
        expected = src * other
    else:
        expected = src * other + src
    torch.testing.assert_close(dst, expected, atol=1e-5, rtol=1e-5)
