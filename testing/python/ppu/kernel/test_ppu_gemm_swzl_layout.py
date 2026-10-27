"""SWZL GEMM loads must match the actual shared layout (issue #43)."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import (
    make_full_bank_swizzled_layout,
    make_half_bank_swizzled_layout,
    make_linear_layout,
    make_quarter_bank_swizzled_layout,
)


@tilelang.testing.requires_ppu_compute_version(1, 5)
@pytest.mark.parametrize("dtype", ["float8_e4m3fn", "bfloat16"])
@pytest.mark.parametrize("operand", ["A", "B"])
@pytest.mark.parametrize(
    "row_bytes,layout_fn,compatible",
    [
        (128, make_full_bank_swizzled_layout, True),
        (128, make_half_bank_swizzled_layout, False),
        (128, make_quarter_bank_swizzled_layout, False),
        (128, make_linear_layout, False),
        (64, make_half_bank_swizzled_layout, True),
        (64, make_quarter_bank_swizzled_layout, False),
        (64, make_linear_layout, False),
    ],
)
def test_gemm_swzl_layout(dtype, operand, row_bytes, layout_fn, compatible):
    k = row_bytes * 8 // tilelang.tvm.DataType(dtype).bits

    @T.prim_func
    def main(
        A: T.Tensor((64, k), dtype),
        B: T.Tensor((64, k), dtype),
        C: T.Tensor((64, 64), "float32"),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, k), dtype)
            B_shared = T.alloc_shared((64, k), dtype)
            C_local = T.alloc_fragment((64, 64), "float32")
            if operand == "A":
                T.annotate_layout({A_shared: layout_fn(A_shared)})
            else:
                T.annotate_layout({B_shared: layout_fn(B_shared)})
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.clear(C_local)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C)

    torch.manual_seed(20261026)
    a = torch.randn(64, k, device="cuda").to(getattr(torch, dtype))
    b = torch.randn(64, k, device="cuda").to(getattr(torch, dtype))
    expected = a.float() @ b.float().T
    for disabled in (False, True):
        kernel = tilelang.compile(
            main,
            out_idx=[2],
            pass_configs={"tl.disable_aiu_lower": True, "tl.disable_ldmat_swzl": disabled},
        )
        source = kernel.get_kernel_source()
        loads = [line for line in source.splitlines() if "tix_ldmatrix_swzl" in line]
        selected = [line for line in loads if f"{operand}_shared" in line]
        other = "B" if operand == "A" else "A"
        # LayoutInference replaces a conflicting linear B with a new swizzled
        # buffer. That actual load is eligible even though the input annotation
        # was linear; A does not take this trans_buffer path.
        rewritten_b = operand == "B" and layout_fn is make_linear_layout
        assert bool(selected) == ((compatible or rewritten_b) and not disabled)
        if rewritten_b and not disabled:
            assert all("B_shared_" in line for line in selected)
        # A fallback for one operand must not disable a compatible other one.
        assert any(f"{other}_shared" in line for line in loads) == (not disabled)
        torch.testing.assert_close(kernel(a, b), expected, atol=0.02, rtol=0.02)


if __name__ == "__main__":
    tilelang.testing.main()
