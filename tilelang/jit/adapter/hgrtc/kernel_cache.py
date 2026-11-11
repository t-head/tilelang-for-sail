import os

from tilelang.cache.kernel_cache import KernelCache
from tilelang.jit import JITKernel


class HGRTCKernelCache(KernelCache):
    kernel_lib_path = "kernel.hgbin"
    kernel_py_path = "kernel.py"

    def _get_required_files(self, cache_path: str) -> list[str]:
        return super()._get_required_files(cache_path) + [os.path.join(cache_path, self.kernel_py_path)]

    def _save_so_cubin_to_disk(self, kernel: JITKernel, cache_path: str, verbose: bool = False):
        src_lib_path = kernel.adapter.libpath
        kernel_py_path = os.path.join(cache_path, self.kernel_py_path)
        # Prefer the adapter's own launcher path: the binary and its .py
        # companion always share one base name in the Hgrtc backend.
        src_py_path = getattr(getattr(kernel.adapter, "lib_generator", None), "pypath", None)
        if src_py_path is None:
            base, _ = os.path.splitext(src_lib_path)
            src_py_path = base + ".py"
        if verbose:
            self.logger.debug(f"Saving kernel hgrtc python code to file: {kernel_py_path}")
        KernelCache._safe_write_file(kernel_py_path, "wb", lambda file: file.write(KernelCache._load_binary(src_py_path)))
        super()._save_so_cubin_to_disk(kernel, cache_path, verbose)
