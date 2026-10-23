from __future__ import annotations

from tilelang.backend.execution_backend import ExecutionBackendSpec


def _is_ppu_nvrtc_available() -> bool:
    """The PPU variant of the NVRTC backend needs the PPU SDK (hgcc + libhggc.so)."""
    try:
        from tilelang.jit.adapter.nvrtc import _is_ppu_backend_available
    except ImportError:
        return False
    return _is_ppu_backend_available()


PPU_EXECUTION_BACKENDS = [
    ExecutionBackendSpec(
        "tvm_ffi",
        enable_host_codegen=True,
        enable_device_compile=True,
    ),
    ExecutionBackendSpec("nvrtc", is_available=_is_ppu_nvrtc_available),
    ExecutionBackendSpec("cython"),
]
