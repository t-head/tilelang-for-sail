"""T.gemm operand dtype whitelist on the PPU backend (AllowPPUMMA in src/ppu/op/gemm.cc).

Follows the CUDA-side pattern of #3245 / #3439: an unsupported (A, B, C) dtype
configuration fails during lowering with the operand dtypes and the supported
set in the message, instead of a device-compiler static_assert from the
MmaDispatcher atom table in tl_templates/ppu/instruction/mma.h.

There are two gates, in lowering order:
- AllowPPUMMA (C++) rejects every (A, B, C) dtype configuration that has no
  native atom for the target generation, e.g. bf16 x bf16 -> fp16, or fp8 on
  ppu0010;
- the Python emitter (tilelang/ppu/intrinsics/macro/mma_macro_generator.py,
  _initialize_mma_prefix) then picks the MMA shape per generation and raises
  ValueError for dtypes it has no shape for. Only uint8 on ppu0010 reaches it:
  AllowPPUMMA admits the ppu0010 u8 atoms, but the emitter releases k32 for s8s8
  only.
"""

import pytest

import tilelang
import tilelang.ppu.language as T
import tilelang.testing
from tvm.error import InternalError
from tvm.target import Target

PPU_15 = {"kind": "ppu", "arch": "ppu_15"}
PPU_10 = {"kind": "ppu", "arch": "ppu_10"}

_REJECTED_MESSAGE = "requires a native PPU MMA atom"


def _make_gemm_kernel(dtype, accum_dtype):
    M = N = K = 64

    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), accum_dtype)):
        with T.Kernel(1, 1, threads=128):
            As = T.alloc_shared((M, K), dtype)
            Bs = T.alloc_shared((K, N), dtype)
            Cl = T.alloc_fragment((M, N), accum_dtype)
            T.clear(Cl)
            T.copy(A, As)
            T.copy(B, Bs)
            T.gemm(As, Bs, Cl)
            T.copy(Cl, C)

    return main


def _lower(kernel, arch):
    with Target(arch):
        return tilelang.lower(kernel, target=arch, enable_device_compile=False).kernel_source


# float32 x float32 -> float32 is the tf32 MMA path on the PPU backend.
_LOWERED_BY_BOTH_GENERATIONS = [
    ("float16", "float16"),
    ("float16", "float32"),
    ("bfloat16", "float32"),
    ("float32", "float32"),
    ("int8", "int32"),
]


@tilelang.testing.requires_ppu
@pytest.mark.parametrize(("dtype", "accum_dtype"), _LOWERED_BY_BOTH_GENERATIONS)
@pytest.mark.parametrize("arch", [PPU_15, PPU_10], ids=["ppu_15", "ppu_10"])
def test_gemm_supported_dtype_combo_lowers(dtype, accum_dtype, arch):
    assert "tl::mma_sync<" in _lower(_make_gemm_kernel(dtype, accum_dtype), arch)


@tilelang.testing.requires_ppu
@pytest.mark.parametrize(
    ("dtype", "accum_dtype"),
    [("float8_e4m3", "float32"), ("float8_e5m2", "float32"), ("float4_e2m1fn", "float32")],
)
def test_gemm_ppu0015_only_dtype_combo_lowers(dtype, accum_dtype):
    assert "tl::mma_sync<" in _lower(_make_gemm_kernel(dtype, accum_dtype), PPU_15)


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("arch", [PPU_15, PPU_10], ids=["ppu_15", "ppu_10"])
@pytest.mark.parametrize(
    ("dtype", "accum_dtype"),
    [
        # The emitter has an MMA shape for these three, but there is no native
        # atom; bf16 x bf16 -> fp16 is the regression upstream fixed in #3245.
        ("bfloat16", "float16"),
        ("float16", "bfloat16"),
        ("float32", "float16"),
        # No MMA shape on either generation: the whitelist rejects them first.
        ("float64", "float64"),
        ("float6_e2m3fn", "float32"),
    ],
)
def test_gemm_unsupported_dtype_combo_rejected(dtype, accum_dtype, arch):
    with pytest.raises(InternalError, match=_REJECTED_MESSAGE):
        _lower(_make_gemm_kernel(dtype, accum_dtype), arch)


@tilelang.testing.requires_ppu
@pytest.mark.parametrize(("dtype", "accum_dtype"), [("float8_e4m3", "float16"), ("uint8", "int32")])
def test_gemm_unsupported_dtype_combo_rejected_on_ppu0015(dtype, accum_dtype):
    # The s8u8 / u8s8 / u8u8 atoms of tl_templates/ppu/instruction/mma.h are
    # ppu0010-only, so ppu0015 has no native atom for these.
    with pytest.raises(InternalError, match=_REJECTED_MESSAGE):
        _lower(_make_gemm_kernel(dtype, accum_dtype), PPU_15)


@tilelang.testing.requires_ppu
def test_gemm_uint8_rejected_on_ppu0010_by_emitter():
    # The header has s8u8 / u8s8 / u8u8 atoms for ppu0010, but the emitter only
    # releases k32 for s8s8 on ppu0010, so T.gemm never reaches them.
    with pytest.raises(ValueError, match="Unsupported k_dim"):
        _lower(_make_gemm_kernel("uint8", "int32"), PPU_10)


@tilelang.testing.requires_ppu
@pytest.mark.parametrize(
    ("dtype", "accum_dtype"),
    [("float8_e4m3", "float32"), ("float8_e5m2", "float32"), ("float4_e2m1fn", "float32")],
)
def test_gemm_ppu0015_dtype_rejected_on_ppu0010(dtype, accum_dtype):
    # The fp8 / fp4 atoms exist from ppu0015 on only.
    with pytest.raises(InternalError, match=_REJECTED_MESSAGE):
        _lower(_make_gemm_kernel(dtype, accum_dtype), PPU_10)


if __name__ == "__main__":
    tilelang.testing.main()
