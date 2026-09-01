from __future__ import annotations

from tvm.target import Target

from tilelang import _ffi_api
from tilelang.backend.target import TargetLike, register_target_detector, register_target_normalizer


def check_ppu_availability() -> bool:
    """
    Check if PPU is available on the system by locating the PPU SDK.
    Returns:
        bool: True if PPU is available, False otherwise.
    """
    try:
        from tilelang.contrib import hgcc

        hgcc._find_ppu_sdk()
        return True
    except Exception:
        return False


def _detect_ppu_arch() -> str | None:
    """Return the PPU architecture detected from the PPU SDK, if available."""
    from tilelang.contrib import hgcc

    compute_version = hgcc.get_target_compute_version()
    return f"{hgcc.get_target_arch(compute_version)}"


def _ppu_target_from_arch(arch: str | None) -> Target | str:
    """Build a PPU target while preserving the legacy bare string fallback."""
    if arch is None:
        return "ppu"
    return Target({"kind": "ppu", "arch": arch})


def _detect_ppu_target() -> Target | str | None:
    if not check_ppu_availability():
        return None

    arch = _detect_ppu_arch()
    return _ppu_target_from_arch(arch)


def normalize_ppu_target(target: TargetLike) -> Target | None:
    if not isinstance(target, str) or target.strip() != "ppu":
        return None
    normalized = _ppu_target_from_arch(_detect_ppu_arch())
    return normalized if isinstance(normalized, Target) else None


def target_is_ppu(target: Target) -> bool:
    return _ffi_api.TargetIsPPU(target)


def target_has_aiu_copy(target: Target) -> bool:
    return _ffi_api.TargetHasAiuCopy(target)


def target_ppu_get_warp_size(target: Target) -> int:
    return _ffi_api.TargetPPUGetWarpSize(target)


register_target_detector("ppu", _detect_ppu_target)
register_target_normalizer("ppu", normalize_ppu_target)
