from __future__ import annotations

from tvm.target import Target

from tilelang.backend.execution_backend import ExecutionBackendSpec, register_execution_backend


def _is_ppu_target(target: Target) -> bool:
    return target.kind.name == "ppu"


def _is_ppu_nvrtc_available() -> bool:
    """The PPU variant of the NVRTC backend needs the PPU SDK (hgcc + libhggc.so)."""
    try:
        from tilelang.jit.adapter.nvrtc import _is_ppu_backend_available
    except ImportError:
        return False
    return _is_ppu_backend_available()


register_execution_backend(
    "ppu",
    ExecutionBackendSpec(
        "tvm_ffi",
        supports_target=_is_ppu_target,
        enable_host_codegen=True,
        enable_device_compile=True,
    ),
)
register_execution_backend(
    "ppu",
    ExecutionBackendSpec(
        "nvrtc",
        is_available=_is_ppu_nvrtc_available,
        supports_target=_is_ppu_target,
    ),
)
register_execution_backend(
    "ppu",
    ExecutionBackendSpec("cython", supports_target=_is_ppu_target),
)
