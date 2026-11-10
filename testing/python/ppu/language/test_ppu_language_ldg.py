# PPU variant of the ldg256 cases in
# testing/python/language/test_tilelang_language_ldg.py: 256-bit global load
# is emitted as tl::load_global_256 (two 128-bit halves on PPU).
import torch
import tilelang
import tilelang.ppu.language as T
import tilelang.testing


@tilelang.testing.requires_ppu
def test_ldg256_codegen():
    """ldg256 generates tl::load_global_256 in PPU source."""

    @tilelang.jit(target="ppu")
    def ldg256_kernel(X, Y):
        N = T.const("N")
        X: T.Tensor[[N], T.float32]
        Y: T.Tensor[[N], T.float32]

        with T.Kernel(N // 8, threads=32) as pid:
            Y[pid * 8 : pid * 8 + 8] = T.reinterpret(T.ldg256(X[pid * 8 : pid * 8 + 8]), T.float32x8)

    X = torch.randn(256, dtype=torch.float32, device="cuda")
    Y = torch.empty(256, dtype=torch.float32, device="cuda")

    ldg256_kernel(X, Y)

    src = ldg256_kernel.get_kernel_source(N=256)
    assert "load_global_256" in src, "Expected load_global_256 call in generated PPU source"
    torch.testing.assert_close(Y, X, atol=1e-5, rtol=1e-5)


@tilelang.testing.requires_ppu
def test_ldg256_stg256_fallback_on_pre_sm100():
    """Explicit 256-bit load/store on a scalar-typed pointer must route through
    the ulonglong4 overload (previously deduced as float)."""

    @tilelang.jit(target="ppu")
    def ldg256_stg256_fallback_kernel(X, Y):
        N = T.const("N")
        X: T.Tensor[[N], T.float32]
        Y: T.Tensor[[N], T.float32]

        with T.Kernel(N // 8, threads=32) as pid:
            value = T.ldg256(X[pid * 8 : pid * 8 + 8])
            T.stg256(Y[pid * 8 : pid * 8 + 8], value)

    X = torch.randn(256, dtype=torch.float32, device="cuda")
    Y = torch.empty(256, dtype=torch.float32, device="cuda")

    ldg256_stg256_fallback_kernel(X, Y)

    torch.testing.assert_close(Y, X, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    tilelang.testing.main()
