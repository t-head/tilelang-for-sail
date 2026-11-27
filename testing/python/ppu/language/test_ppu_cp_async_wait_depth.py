"""PPU port of the CUDA codegen fix #3547: cp_async wait depth.

`tl::cp_async_wait<N>` lowers to `ppu.cp.async.wait_group N`, whose operand
must be a compile-time immediate. A software pipeline over a dynamic trip count
emits `1 - ko` in its drain phase; the old codegen downcast that expression to
an integer and crashed (Downcast from tirx.Var to ir.IntImm failed).
"""

import tilelang
import tilelang.language as L
import tilelang.ppu.language as T
import tilelang.testing
from tvm.target import Target

PPU_TARGET = "ppu"


def _lower(kernel):
    with Target(PPU_TARGET):
        return tilelang.lower(kernel, target=PPU_TARGET, enable_device_compile=False).kernel_source


@tilelang.testing.requires_ppu
def test_cp_async_wait_constant_depth_kept():
    @T.prim_func
    def kernel(out: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=32):
            L.ptx_commit_group()
            L.ptx_wait_group(2)

    assert "tl::cp_async_wait<2>" in _lower(kernel)


@tilelang.testing.requires_ppu
def test_cp_async_wait_non_constant_depth_waits_for_all():
    # A runtime depth is only ever produced by a pipeline drain and is safe to
    # over-synchronize, so it falls back to a full wait.
    @T.prim_func
    def kernel(depth: T.int32, out: T.Tensor((4,), "int32")):
        with T.Kernel(1, threads=32):
            L.ptx_commit_group()
            L.ptx_wait_group(depth)

    source = _lower(kernel)
    assert "tl::cp_async_wait<0>" in source


if __name__ == "__main__":
    tilelang.testing.main()
