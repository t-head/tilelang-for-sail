# pylint: disable=invalid-name
"""Utility to compile PPU code in-process with hgrtc (the PPU run-time compiler).

The counterpart of :mod:`tilelang.contrib.nvrtc`: device source becomes an
``.hgbin`` image in memory, so no hgcc subprocess and no temporary files are
involved. The image is loaded through the HGGC driver like any other
``.hgbin``. hgrtc keeps a disk cache keyed by source and options, so repeated
compilation of the same kernel returns almost immediately.
"""

from __future__ import annotations

import ctypes
import os
import os.path as osp
import shlex
import threading
from functools import cache

from tvm.target import Target

from tilelang.env import ACTLIZE_INCLUDE_DIR, TILELANG_TEMPLATE_PATH
from tilelang.contrib import hgcc

# hgrtc is not documented as thread-safe; serialize create/compile/destroy.
_HGRTC_LOCK = threading.Lock()

# hgrtc spells the architecture differently from hgcc's -arch=ppu_15.
_HGRTC_ARCH_BY_PPU_ARCH = {
    "ppu_10": "ppu001",
    "ppu_15": "ppu0015",
}


def _lib_candidates(ppu_sdk: str) -> list[str]:
    return [
        osp.join(ppu_sdk, "lib", "libhgrtc.so"),
        osp.join(ppu_sdk, "targets", "x86_64-linux", "lib", "libhgrtc.so"),
        "libhgrtc.so",
    ]


def _device_std_include(ppu_sdk: str) -> str:
    """Root of the SDK's device-side standard library (hggc/std)."""
    return osp.join(ppu_sdk, "targets", "x86_64-linux", "include", "hggc", "std")


def ppu_sdk() -> str:
    """PPU SDK root, raising the same message as the hgcc path when unset."""
    return hgcc._find_ppu_sdk()


def is_hgrtc_available() -> bool:
    """Whether hgrtc can be used: the PPU SDK is set up and ships libhgrtc.so
    along with the device-side standard library the templates rely on."""
    ppu_sdk = os.environ.get("PPU_SDK")
    if not ppu_sdk:
        return False
    return any(osp.isfile(candidate) for candidate in _lib_candidates(ppu_sdk)) and osp.isdir(_device_std_include(ppu_sdk))


@cache
def _get_hgrtc_lib() -> ctypes.CDLL:
    lib = None
    tried = []
    for candidate in _lib_candidates(ppu_sdk()):
        tried.append(candidate)
        try:
            lib = ctypes.CDLL(candidate)
            break
        except OSError:
            continue
    if lib is None:
        raise RuntimeError(f"Failed to load libhgrtc.so (tried: {tried})")

    c_int_p = ctypes.POINTER(ctypes.c_int)
    c_size_p = ctypes.POINTER(ctypes.c_size_t)
    c_void_pp = ctypes.POINTER(ctypes.c_void_p)
    lib.hgrtcVersion.argtypes = [c_int_p, c_int_p]
    lib.hgrtcCreateProgram.argtypes = [c_void_pp, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    lib.hgrtcDestroyProgram.argtypes = [c_void_pp]
    lib.hgrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    lib.hgrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, c_size_p]
    lib.hgrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.hgrtcGetHGBINSize.argtypes = [ctypes.c_void_p, c_size_p]
    lib.hgrtcGetHGBIN.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.hgrtcGetErrorString.argtypes = [ctypes.c_int]
    lib.hgrtcGetErrorString.restype = ctypes.c_char_p
    return lib


def _error_string(lib: ctypes.CDLL, result: int) -> str:
    text = lib.hgrtcGetErrorString(result)
    return text.decode("utf-8", errors="replace") if text else f"hgrtc error {result}"


@cache
def get_hgrtc_version() -> tuple[int, int]:
    """Return the loaded hgrtc's version, cached for this process."""
    lib = _get_hgrtc_lib()
    major, minor = ctypes.c_int(), ctypes.c_int()
    result = lib.hgrtcVersion(ctypes.byref(major), ctypes.byref(minor))
    if result != 0:
        raise RuntimeError(f"Failed to get hgrtc version: {_error_string(lib, result)}")
    return (major.value, minor.value)


def get_hgrtc_arch(target: Target | None = None) -> str:
    """Convert the PPU compilation target into the arch name hgrtc expects
    (``--ppu-arch=ppu0015`` rather than hgcc's ``-arch=ppu_15``)."""
    compute_version = hgcc.get_target_compute_version(target)
    ppu_arch = hgcc.get_target_arch(compute_version)
    hgrtc_arch = _HGRTC_ARCH_BY_PPU_ARCH.get(ppu_arch)
    if hgrtc_arch is None:
        raise ValueError(f"hgrtc has no architecture corresponding to {ppu_arch!r}")
    return hgrtc_arch


def default_compile_options(compile_flags: list[str] | str | None = None) -> list[str]:
    """The include set hgcc.compile_ppu passes to hgcc, plus the SDK's
    device-side standard library for the RTC-only templates (rtc_std.h)."""
    sdk_root = ppu_sdk()

    options = ["-std=c++20", "-lineinfo"]

    # Template path: use env var if set, otherwise fall back to repo src/
    tl_path = TILELANG_TEMPLATE_PATH
    if tl_path is None:
        tl_path = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", "src"))
    options.append("-I" + tl_path)

    options.append("-I" + osp.join(sdk_root, "targets", "x86_64-linux", "include"))
    options.append("-I" + _device_std_include(sdk_root))

    # actlize (pure PPU library, replaces cutlass)
    if ACTLIZE_INCLUDE_DIR:
        options += ["-I" + ACTLIZE_INCLUDE_DIR, "-DPPU_DEFAULT"]
    else:
        options += ["-DPPU_COMPATIBLE"]

    if compile_flags:
        if isinstance(compile_flags, str):
            options += shlex.split(compile_flags)
        else:
            for flag in compile_flags:
                options.extend(shlex.split(flag) if isinstance(flag, str) else [str(flag)])
    return options


def compile_ppu(
    code: str,
    target: Target | None = None,
    options: list[str] | None = None,
    verbose: bool = False,
) -> bytes:
    """Compile PPU code with hgrtc in-process.

    Parameters
    ----------
    code : str
        The PPU C++ source code.
    target : tvm.target.Target, optional
        The compilation target. If None, uses Target.current().
    options : list of str, optional
        Additional compiler options, appended to :func:`default_compile_options`.
        hgrtc rejects a few flags hgcc accepts (e.g. ``-O3``) with
        ``HGRTC_ERROR_INVALID_OPTION``.
    verbose : bool
        Whether to print the compile options and include the source in errors.

    Returns
    -------
    binary : bytes
        The compiled .hgbin binary data.
    """
    if options is None:
        options = []
    if isinstance(options, str):
        options = shlex.split(options)

    final_options = default_compile_options() + [f"--ppu-arch={get_hgrtc_arch(target)}"] + list(options)
    if verbose:
        print("PPU hgrtc compile options:", " ".join(final_options))

    lib = _get_hgrtc_lib()
    program = ctypes.c_void_p()
    # hgrtc may not be thread-safe; keep the whole program lifetime under the lock.
    with _HGRTC_LOCK:
        result = lib.hgrtcCreateProgram(ctypes.byref(program), code.encode("utf-8"), b"tvm_kernels", 0, None, None)
        if result != 0:
            raise RuntimeError(f"Failed to create hgrtc program: {_error_string(lib, result)}")
        try:
            opts = [opt.encode("utf-8") for opt in final_options]
            opt_array = (ctypes.c_char_p * len(opts))(*opts)
            result = lib.hgrtcCompileProgram(program, len(opts), opt_array)

            size = ctypes.c_size_t()
            lib.hgrtcGetProgramLogSize(program, ctypes.byref(size))
            log_buf = ctypes.create_string_buffer(max(size.value, 1))
            lib.hgrtcGetProgramLog(program, log_buf)
            log = log_buf.value.decode("utf-8", errors="replace")

            if result != 0:
                msg = f"hgrtc compilation failed ({result}: {_error_string(lib, result)})\n"
                if log.strip():
                    msg += f"{log}\n"
                if verbose:
                    msg += f"{code}\n"
                msg += f"Options: {' '.join(final_options)}\n"
                raise RuntimeError(msg)

            lib.hgrtcGetHGBINSize(program, ctypes.byref(size))
            hgbin_buf = ctypes.create_string_buffer(max(size.value, 1))
            lib.hgrtcGetHGBIN(program, hgbin_buf)
            binary = bytes(hgbin_buf.raw[: size.value])
        finally:
            lib.hgrtcDestroyProgram(ctypes.byref(program))

    return binary
