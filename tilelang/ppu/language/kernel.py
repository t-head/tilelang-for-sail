"""PPU dialect of ``T.Kernel`` with SIMT launch annotations."""

from __future__ import annotations

from tvm import tirx

from tilelang.language.kernel import KernelLaunchFrame, kernel_launch_factory, launch_kernel

__all__ = ["Kernel"]


@kernel_launch_factory
def Kernel(
    *blocks: int | tirx.PrimExpr,
    threads: int | list[int] | tuple[int, ...] | None = None,
    prelude: str | None = None,
    name: str | None = None,
) -> KernelLaunchFrame:
    """Construct a kernel launch frame for PPU: a grid of thread blocks.

    name : str, optional
        Device kernel name, given as an ASCII identifier. Used as-is without
        adding ``_kernel``; collisions receive a numeric suffix. When omitted,
        the name is derived from the enclosing PrimFunc's global symbol.
    """
    return launch_kernel(blocks, threads=threads, prelude=prelude, name=name)
