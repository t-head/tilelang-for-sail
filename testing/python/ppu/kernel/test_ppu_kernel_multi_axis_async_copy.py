"""Feature test: multi-axis 2D global->shared tiles lower to async copy in a pipeline.

Restoring the old single-active-axis rejection would leave these pipelined 2D
copies synchronous (dead prefetch); this asserts the async instruction is emitted
and the round-trip data is bitwise correct on device.
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.layout
import tilelang.testing
from tilelang import tvm


PPU_TARGET = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})


def _make_pipeline_copy_kernel(dtype, rows, inner, tiles, swizzle):
    @T.prim_func
    def main(
        A: T.Tensor((tiles, rows, inner), dtype),
        B: T.Tensor((tiles, rows, inner), dtype),
    ):
        T.func_attr({"target": PPU_TARGET})
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((rows, inner), dtype)
            if swizzle:
                T.annotate_layout({S: tilelang.layout.make_ppu_swizzled_layout(S)})
            for k in T.Pipelined(tiles, num_stages=2):
                T.copy(A[k, 0:rows, 0:inner], S[0:rows, 0:inner])
                T.copy(S[0:rows, 0:inner], B[k, 0:rows, 0:inner])

    return main


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
@pytest.mark.parametrize(
    "dtype,swizzle",
    [(T.float16, False), (T.float16, True), (T.int8, False)],
)
def test_multi_axis_async_copy_pipeline_roundtrip(dtype, swizzle):
    rows, inner, tiles = 8, 64, 4
    kernel = tilelang.compile(
        _make_pipeline_copy_kernel(dtype, rows, inner, tiles, swizzle),
        out_idx=[1],
        target=PPU_TARGET,
    )
    assert "tl::cp_async_gs<" in kernel.get_kernel_source()

    torch_dtype = dtype.as_torch()
    a = torch.randint(-4, 4, (tiles, rows, inner), device="cuda", dtype=torch.int16).to(torch_dtype)
    b = kernel(a)
    assert torch.equal(b, a)


if __name__ == "__main__":
    tilelang.testing.main()
