from __future__ import annotations

from tilelang.backend.execution_backend import ExecutionBackendSpec


def _is_hgrtc_available() -> bool:
    """Hgrtc, the PPU runtime-compilation backend, needs the PPU SDK (hgcc + libhggc.so)."""
    try:
        from tilelang.jit.adapter.hgrtc import is_hgrtc_available
    except ImportError:
        return False
    return is_hgrtc_available()


PPU_EXECUTION_BACKENDS = [
    ExecutionBackendSpec(
        "tvm_ffi",
        enable_host_codegen=True,
        enable_device_compile=True,
    ),
    ExecutionBackendSpec("hgrtc", is_available=_is_hgrtc_available),
    ExecutionBackendSpec("cython"),
]
