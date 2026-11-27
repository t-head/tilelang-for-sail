"""Address-based cache prefetch intrinsics for the PPU backend.

Ported from the CUDA front end (tilelang/cuda/language/prefetch.py). PPU has a
single hardware prefetch instruction for the global L2 form. Eviction policies
and the L1 / uniform-L1 variants have no distinct PPU encoding, so only the
policy-less L2 prefetch is exported.
"""

from typing import Literal

from tvm import tirx

from tilelang.utils.language import is_global

__all__ = ["prefetch_l2"]


def prefetch_l2(
    addr: tirx.BufferLoad,
    eviction_policy: Literal["evict_normal", "evict_last"] | None = None,
) -> tirx.Call:
    """Prefetch the global memory cache line containing ``addr`` into L2.

    Args:
        addr: A global buffer element, such as ``x[i]``. Only its address is used.
        eviction_policy: Must be None. PPU has no distinct encoding for eviction
            priorities, so a non-None value is rejected rather than silently ignored.

    Returns:
        A call to the L2 prefetch intrinsic. Prefetch is a performance hint and
        does not guarantee cache residency or synchronize memory accesses.
    """
    if eviction_policy is not None:
        raise ValueError(f"prefetch_l2 on PPU does not support eviction_policy, got {eviction_policy!r}")
    if not is_global(addr):
        raise ValueError(f"prefetch_l2 on PPU requires global memory, got scope={addr.buffer.scope()}")

    return tirx.call_intrin(
        "void",
        tirx.op.Op.get("tl.prefetch_l2"),
        tirx.address_of(addr),
        -1,
        True,
    )
