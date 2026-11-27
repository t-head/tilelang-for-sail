"""int64 atomic add/max/min and typed atomic store on the PPU backend.

PPU counterpart of the CUDA fix #3307. int64 destinations were normalized to
`unsigned long long` in src/tl_templates/ppu/atomic.h while the object passed to
`hggc::atomic_ref` stayed a `long` lvalue, so int64 add/max/min with
return_prev failed to compile; and had the types lined up, max/min would have
ordered negative values as huge positives.
"""

import pytest
import torch

import tilelang
import tilelang.ppu.language as T
import tilelang.testing

PPU_TARGET = "ppu"


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("op", ["add", "max", "min"])
@pytest.mark.parametrize("return_prev", [False, True])
def test_atomic_int64(op, return_prev):
    atomic = getattr(T, f"atomic_{op}")

    @T.prim_func
    def main(A: T.Tensor((32,), "int64"), B: T.Tensor((32,), "int64"), Prev: T.Tensor((32,), "int64")):
        with T.Kernel(1, threads=32):
            i = T.get_thread_binding()
            if return_prev:
                Prev[i] = atomic(B[i], A[i], return_prev=True)
            else:
                atomic(B[i], A[i])

    kernel = tilelang.compile(main, target=PPU_TARGET, execution_backend="tvm_ffi")
    a = torch.tensor([1 << 40, -(1 << 40), -7, 7, -9, 9, 0, 0] * 4, dtype=torch.int64, device="cuda")
    initial = torch.tensor([-(1 << 40), 1 << 40, -3, 3, -9, 9, -1, 1] * 4, dtype=torch.int64, device="cuda")
    b, prev = initial.clone(), torch.empty_like(initial)
    kernel(a, b, prev)
    torch.cuda.synchronize()
    reference = {"add": torch.add, "max": torch.maximum, "min": torch.minimum}[op]
    torch.testing.assert_close(b, reference(initial, a), rtol=0, atol=0)
    if return_prev:
        torch.testing.assert_close(prev, initial, rtol=0, atol=0)


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float16", "bfloat16"])
@pytest.mark.parametrize("memory_order", ["relaxed", "release", "seq_cst"])
def test_atomic_store_dtype(dtype, memory_order):
    @T.prim_func
    def main(A: T.Tensor((32,), dtype), B: T.Tensor((32,), dtype)):
        with T.Kernel(1, threads=32):
            i = T.get_thread_binding()
            T.atomic_store(B[i], A[i], memory_order=memory_order)

    kernel = tilelang.compile(main, target=PPU_TARGET, execution_backend="tvm_ffi")
    values = [-7, 0, 3, 9]
    if dtype == "int64":
        values = [-(1 << 40), -7, 3, 1 << 40]
    elif dtype.startswith("float") or dtype == "bfloat16":
        # Keep the bit pattern NaN-free: torch.equal on float tensors treats
        # NaN as unequal, and the comparison below is done on raw bytes anyway.
        values += [0.5, -0.0, 1.25, -2.5]
    a = torch.tensor(values * (32 // len(values)), dtype=getattr(torch, dtype), device="cuda")
    b = torch.full_like(a, 42)
    kernel(a, b)
    torch.cuda.synchronize()
    assert torch.equal(b.view(torch.uint8), a.view(torch.uint8))


if __name__ == "__main__":
    tilelang.testing.main()
