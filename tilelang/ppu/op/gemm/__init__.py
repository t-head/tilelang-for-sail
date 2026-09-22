"""PPU GEMM op registrations."""

from __future__ import annotations

from tilelang.tileop.gemm.registry import register_gemm_impl
from .gemm_mma import GEMM_INST_MMA_PPU, PPUGemmMMA
from tilelang.utils.target import target_is_ppu


def _match_ppu(target) -> bool:
    return target_is_ppu(target)


register_gemm_impl("ppu.mma", GEMM_INST_MMA_PPU, _match_ppu, PPUGemmMMA)
