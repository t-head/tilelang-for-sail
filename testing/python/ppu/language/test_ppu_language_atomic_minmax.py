# PPU variant of the scalar-literal atomic max/min cases in
# testing/python/language/test_tilelang_language_atomic.py: the 16-bit
# atomicCAS loop must pack the operand bits after converting to the native
# 16-bit type (previously read the low 16 bits of the raw float).
import torch
import tilelang
import tilelang.ppu.language as T
import tilelang.testing


def run_atomic_max_scalar_literal(dtype):
    @tilelang.jit(target="ppu")
    def wrapper():
        @T.prim_func
        def kernel(dst: T.Tensor((1,), dtype)):
            with T.Kernel(1, threads=1):
                T.atomic_max(dst[0], 42.0)  # python float literal is fp32

        return kernel

    kernel = wrapper()
    dst = torch.full((1,), -1e4, device="cuda", dtype=getattr(torch, dtype))
    kernel(dst)
    torch.testing.assert_close(dst, torch.full((1,), 42.0, device="cuda", dtype=getattr(torch, dtype)))


def run_atomic_min_scalar_literal(dtype):
    @tilelang.jit(target="ppu")
    def wrapper():
        @T.prim_func
        def kernel(dst: T.Tensor((1,), dtype)):
            with T.Kernel(1, threads=1):
                T.atomic_min(dst[0], 42.0)  # python float literal is fp32

        return kernel

    kernel = wrapper()
    dst = torch.full((1,), 1e4, device="cuda", dtype=getattr(torch, dtype))
    kernel(dst)
    torch.testing.assert_close(dst, torch.full((1,), 42.0, device="cuda", dtype=getattr(torch, dtype)))


@tilelang.testing.requires_ppu
def test_atomic_max_scalar_fp16():
    run_atomic_max_scalar_literal("float16")


@tilelang.testing.requires_ppu
def test_atomic_max_scalar_bf16():
    run_atomic_max_scalar_literal("bfloat16")


@tilelang.testing.requires_ppu
def test_atomic_min_scalar_fp16():
    run_atomic_min_scalar_literal("float16")


@tilelang.testing.requires_ppu
def test_atomic_min_scalar_bf16():
    run_atomic_min_scalar_literal("bfloat16")


if __name__ == "__main__":
    tilelang.testing.main()
