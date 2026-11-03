# PPU variant of the stg256 case in
# testing/python/language/test_tilelang_language_ldst.py: 256-bit global
# store is emitted as tl::store_global_256 (two 128-bit halves on PPU).
import torch
import tilelang
import tilelang.ppu.language as T
import tilelang.testing


@tilelang.testing.requires_ppu
def test_stg256_codegen():
    """stg256 generates tl::store_global_256 in PPU source."""

    @tilelang.jit(target="ppu")
    def stg256_kernel(X, Y):
        N = T.const("N")
        X: T.Tensor[[N], T.float32]
        Y: T.Tensor[[N], T.float32]

        with T.Kernel(N // 8, threads=32) as pid:
            val = T.ldg256(X[pid * 8 : pid * 8 + 8])
            T.stg256(Y[pid * 8 : pid * 8 + 8], val)

    X = torch.randn(256, dtype=torch.float32, device="cuda")
    Y = torch.empty(256, dtype=torch.float32, device="cuda")

    stg256_kernel(X, Y)

    src = stg256_kernel.get_kernel_source(N=256)
    assert "store_global_256" in src, "Expected store_global_256 call in generated PPU source"
    torch.testing.assert_close(Y, X, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    tilelang.testing.main()
