"""Hgrtc: the PPU runtime-compilation backend for TileLang.

Hgrtc is the PPU's own runtime compilation mechanism (the analogue of
CUDA's NVRTC on the PPU toolchain): device sources are compiled to
``.hgbin`` modules by ``hgcc`` and loaded/launched at runtime through the
HGGC driver (libhggc.so). This package keeps the PPU implementation fully
independent of the CUDA ``nvrtc`` package.
"""

import logging
import os

__all__ = [
    "HGRTCKernelAdapter",
    "TLHgrtcSourceWrapper",
    "HGRTCLibraryGenerator",
    "is_hgrtc_available",
    "check_hgrtc_available",
]

logger = logging.getLogger(__name__)

HGRTC_UNAVAILABLE_MESSAGE = (
    "The PPU SDK is not available, Hgrtc backend cannot be used. "
    "Please source the PPU SDK envsetup.sh (e.g. source $PPU_SDK/envsetup.sh ppu) "
    "if you want to use the Hgrtc backend."
)


def is_hgrtc_available() -> bool:
    """Whether the PPU Hgrtc runtime-compilation backend can be used.

    Hgrtc only needs the PPU SDK (hgcc + libhggc.so), not cuda-python.
    """
    sdk = os.environ.get("PPU_SDK")
    return bool(sdk) and os.path.exists(os.path.join(sdk, "bin", "hgcc"))


def check_hgrtc_available():
    """Check if Hgrtc backend is available.

    Raises
    ------
    ImportError
        If the PPU SDK is not available
    """
    if not is_hgrtc_available():
        raise ImportError(HGRTC_UNAVAILABLE_MESSAGE)


# Conditionally import the adapter.
# Note: HGRTCKernelCache lives in tilelang.jit.adapter.hgrtc.kernel_cache and
# is imported by tilelang.cache only — importing it here would create a
# circular import (kernel_cache -> tilelang.cache -> tilelang.jit).
if is_hgrtc_available():
    from .adapter import HGRTCKernelAdapter
    from .libgen import HGRTCLibraryGenerator
    from .wrapper import TLHgrtcSourceWrapper
else:
    # Provide dummy classes that raise error on instantiation
    class HGRTCKernelAdapter:
        """Dummy HGRTCKernelAdapter that raises ImportError on instantiation."""

        def __init__(self, *args, **kwargs):
            raise ImportError(HGRTC_UNAVAILABLE_MESSAGE)

        @classmethod
        def from_database(cls, *args, **kwargs):
            raise ImportError(HGRTC_UNAVAILABLE_MESSAGE)

    class TLHgrtcSourceWrapper:
        """Dummy TLHgrtcSourceWrapper that raises ImportError on instantiation."""

        def __init__(self, *args, **kwargs):
            raise ImportError(HGRTC_UNAVAILABLE_MESSAGE)

    class HGRTCLibraryGenerator:
        """Dummy HGRTCLibraryGenerator that raises ImportError on instantiation."""

        def __init__(self, *args, **kwargs):
            raise ImportError(HGRTC_UNAVAILABLE_MESSAGE)
