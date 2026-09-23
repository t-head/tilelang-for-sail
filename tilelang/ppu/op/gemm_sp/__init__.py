"""PPU sparse GEMM op registrations."""

from __future__ import annotations

from tilelang.tileop.gemm_sp.registry import register_gemm_sp_impl
from tilelang.ppu.op.gemm_sp.gemm_sp_mma import GEMM_SP_INST_MMA_SP, GemmSPMMA
from tilelang.ppu.target import target_is_ppu


def _match_ppu(target) -> bool:
    return target_is_ppu(target)


register_gemm_sp_impl("ppu.mma.sp", GEMM_SP_INST_MMA_SP, _match_ppu, GemmSPMMA)
