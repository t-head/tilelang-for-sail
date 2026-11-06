"""Hgrtc Library Generator for TileLang.

Compiles PPU kernels at runtime using hgcc and manages the resulting
.hgbin modules.

Why hgcc instead of AOT compilation:
- No offline compilation step, enables true JIT workflows
- Works with the PPU SDK toolchain only (no full TileLang native build needed)
- Allows kernel specialization based on runtime parameters

Key responsibilities:
- Compile PPU device source to .hgbin using hgcc
- Generate accompanying Python launcher code
- Load compiled .hgbin through the HGGC driver and extract kernel handles
- Manage module lifecycle (load/unload)
"""

from __future__ import annotations
import ctypes
import importlib.util
import logging
import os
import os.path as osp
import tempfile
from types import ModuleType

from tvm.target import Target

from tilelang.jit.adapter.libgen import LibraryGenerator

logger = logging.getLogger(__name__)

_hg_driver_lib = None


def _get_hg_driver():
    """Lazily load libhggc.so and perform one-time hgInit(0).

    HGGC driver must be initialized explicitly before any hgModule*/hgLaunch* call.
    """
    global _hg_driver_lib
    if _hg_driver_lib is not None:
        return _hg_driver_lib

    ppu_sdk = os.environ.get("PPU_SDK")
    if not ppu_sdk:
        raise RuntimeError(
            "PPU_SDK environment variable is not set. Please source the PPU SDK envsetup.sh (e.g. source $PPU_SDK/envsetup.sh ppu)"
        )
    candidates = [
        osp.join(ppu_sdk, "lib", "libhggc.so"),
        osp.join(ppu_sdk, "targets", "x86_64-linux", "lib", "libhggc.so"),
        "libhggc.so",
    ]
    lib = None
    for cand in candidates:
        try:
            lib = ctypes.CDLL(cand)
            break
        except OSError:
            continue
    if lib is None:
        raise RuntimeError(f"Failed to load libhggc.so (tried: {candidates})")

    lib.hgInit.argtypes = [ctypes.c_uint]
    lib.hgInit.restype = ctypes.c_int
    res = lib.hgInit(0)
    if res != 0:
        raise RuntimeError(f"hgInit(0) failed: HGresult={res}")

    # A current context is required before any hgModule*/hgLaunch* call —
    # hgModuleLoadData returns HGGC_ERROR_INVALID_CONTEXT (201) otherwise.
    # Mirrors EnsureCurrentPPUContext in src/ppu/codegen/rt_mod_ppu.cc
    # (hgInit + device context), but stays purely on driver APIs so no
    # second library (libhggcrt/libhggc_wrapper) is needed. The primary
    # context is process-wide per device, so it is shared with torch's
    # PPU runtime allocations.
    lib.hgDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    lib.hgDeviceGet.restype = ctypes.c_int
    lib.hgDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
    lib.hgDevicePrimaryCtxRetain.restype = ctypes.c_int
    lib.hgCtxSetCurrent.argtypes = [ctypes.c_void_p]
    lib.hgCtxSetCurrent.restype = ctypes.c_int
    dev = ctypes.c_int(0)
    res = lib.hgDeviceGet(ctypes.byref(dev), 0)
    if res != 0:
        raise RuntimeError(f"hgDeviceGet(0) failed: HGresult={res}")
    ctx = ctypes.c_void_p()
    res = lib.hgDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
    if res != 0:
        raise RuntimeError(f"hgDevicePrimaryCtxRetain failed: HGresult={res}")
    res = lib.hgCtxSetCurrent(ctx)
    if res != 0:
        raise RuntimeError(f"hgCtxSetCurrent failed: HGresult={res}")
    # Keep the retained primary context referenced for the process lifetime.
    lib._hg_primary_ctx = ctx

    lib.hgModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
    lib.hgModuleLoadData.restype = ctypes.c_int
    lib.hgModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    lib.hgModuleGetFunction.restype = ctypes.c_int
    lib.hgModuleUnload.argtypes = [ctypes.c_void_p]
    lib.hgModuleUnload.restype = ctypes.c_int

    _hg_driver_lib = lib
    return lib


class HGRTCLibraryGenerator(LibraryGenerator):
    """Runtime compiler and loader for hgcc-compiled PPU kernels.

    Lifecycle:
        1. compile_lib(): PPU source → .hgbin + Python launcher
        2. load_lib(): .hgbin → loaded module + kernel handles
        3. pymodule.call(): Execute kernels via Python launcher
        4. __del__: Cleanup (unload module)

    Why three files (.hg, .hgbin, .py):
        - .hg: Source for debugging, kept in temp directory
        - .hgbin: Compiled binary, loaded by the HGGC driver
        - .py: Launch code, imported as Python module

    Attributes:
        host_func: Generated Python launch code (from wrapper)
        hgmod: HGGC module handle (HGmodule)
        pymodule: Imported Python module containing call() function
    """

    host_func: str | None = None
    hgmod = None  # HGGC module handle (ctypes.c_void_p)
    pymodule: ModuleType | None = None
    pypath: str | None = None

    def __init__(self, target: Target, verbose: bool = False):
        """Initialize Hgrtc library generator.

        Args:
            target: Compilation target (must be PPU)
            verbose: Enable verbose compilation output
        """
        super().__init__(target, verbose)

    @staticmethod
    def import_from_file(module_name, file_path):
        """Dynamically import Python module from file path.

        Standard importlib pattern for loading modules outside sys.path.
        Used to import generated .py launcher code from temp directory.

        Args:
            module_name: Name to assign to imported module
            file_path: Absolute path to .py file

        Returns:
            Imported module object
        """
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to import module from file: {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def update_host_func(self, host_func: str):
        """Store generated Python launch code for later file write.

        Called by adapter after wrapper generates the launch code.
        This is the bridge between code generation and file output.

        Args:
            host_func: Python source code containing call() function
        """
        self.host_func = host_func

    def load_lib(self, lib_path: str | None = None):
        """Load a compiled .hgbin module and its Python launcher into memory.

        Why two loads:
            1. Import Python module for launch logic
            2. Load .hgbin through the HGGC driver (hgModuleLoadData)

        Args:
            lib_path: Path to .hgbin file (optional, uses self.libpath if None)

        Side effects:
            - Sets self.pymodule to imported Python module
            - Sets self.hgmod to the loaded HGGC module handle
        """
        if lib_path is None:
            lib_path = self.libpath
        else:
            self.libpath = lib_path

        # The launcher file sits next to the binary with the same base name.
        # splitext (instead of a fixed suffix replace) also covers disk
        # caches that store the binary under a preset name (e.g. autotuner
        # caches reload via kernel_lib_path): the companion is always
        # <base>.py.
        self.pypath = osp.splitext(lib_path)[0] + ".py"
        self.pymodule = self.import_from_file("kernel", self.pypath)

        # Load the hgcc-compiled .hgbin through the HGGC driver.
        self._load_module(lib_path)

    def compile_lib(self, timeout: float | None = None):
        """Compile PPU device source to .hgbin using hgcc and write output files.

        Output artifacts (all in temp directory):
            - .hg: Source code (for debugging)
            - .hgbin: Compiled binary (for execution)
            - .py: Python launcher (for calling kernels)

        Args:
            timeout: Compilation timeout in seconds (currently unsupported by hgcc)

        Side effects:
            - Writes .hg, .hgbin, .py files to temp directory
            - Sets self.srcpath, self.libpath, self.pypath
        """
        from tilelang.contrib import hgcc
        from tilelang.transform import PassConfigKey

        cfg = self.pass_configs or {}
        # Note: verbose output is controlled by self.verbose; hgcc-style
        # verbose flags can be passed through tl.device_compile_flags.
        verbose = self.verbose

        options = []
        extra_flags = cfg.get(PassConfigKey.TL_DEVICE_COMPILE_FLAGS, None)
        if extra_flags:
            import shlex

            if isinstance(extra_flags, str):
                options += shlex.split(extra_flags)
            else:
                for flag in extra_flags:
                    if isinstance(flag, str):
                        options.extend(shlex.split(flag))
                    else:
                        options.append(str(flag))

        if self.compile_flags:
            options += [item for flag in self.compile_flags for item in flag.split() if item not in options]

        # Write source code to temp file
        src = tempfile.NamedTemporaryFile(mode="w", suffix=".hg", delete=False)
        src.write(self.lib_code)
        src.close()
        self.srcpath = src.name

        binary = hgcc.compile_ppu(self.lib_code, target=self.target, options=options, verbose=verbose)

        # Write binary to .hgbin file
        self.libpath = self.srcpath.replace(".hg", ".hgbin")
        with open(self.libpath, "wb") as f:
            f.write(binary)
        # Write host function to .py file (needed by the adapter)
        self.pypath = self.srcpath.replace(".hg", ".py")
        if self.host_func is None:
            raise RuntimeError("Host function is not set, please call update_host_func() first.")
        with open(self.pypath, "w") as f:
            f.write(self.host_func)

    def _load_module(self, lib_path: str):
        """Load an hgcc-compiled .hgbin through the HGGC driver (ctypes)."""
        drv = _get_hg_driver()
        with open(lib_path, "rb") as f:
            data = f.read()
        # Keep the image buffer alive for the lifetime of the module.
        self._hgbin_buf = ctypes.create_string_buffer(data, len(data))
        module = ctypes.c_void_p()
        res = drv.hgModuleLoadData(ctypes.byref(module), ctypes.cast(self._hgbin_buf, ctypes.c_void_p))
        if res != 0 or not module:
            raise RuntimeError(f"hgModuleLoadData failed for {lib_path}: HGresult={res}")
        self.hgmod = module

    def get_kernel(self, name: str):
        """Return the HGfunction handle (ctypes.c_void_p) of a kernel in the loaded PPU module."""
        drv = _get_hg_driver()
        func = ctypes.c_void_p()
        res = drv.hgModuleGetFunction(ctypes.byref(func), self.hgmod, name.encode("utf-8"))
        if res != 0 or not func:
            raise RuntimeError(f"hgModuleGetFunction({name}) failed: HGresult={res}")
        return func

    def __del__(self):
        """Cleanup: unload the HGGC module when the object is destroyed.

        Failure to unload is logged but not raised (destructor can't fail).
        """
        hgmod = getattr(self, "hgmod", None)
        if hgmod:
            try:
                _get_hg_driver().hgModuleUnload(hgmod)
            except Exception:
                logger.warning(f"Failed to unload PPU module: {self.libpath}")
            self.hgmod = None
