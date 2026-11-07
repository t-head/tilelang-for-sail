"""Shared helpers for read-only pre-``LowerTileOp`` analysis."""

from tvm import tirx as T


def op(node):
    """Return a TIR call's registered operation name, or an empty string."""
    return getattr(getattr(node, "op", None), "name", "")


def region(call):
    """Decode a ``tl.region`` call into buffer, origin, and extent."""
    if op(call) != "tl.region":
        raise ValueError(f"expected tl.region, got {call}")
    load = call.args[0]
    if not isinstance(load, T.BufferLoad):
        raise ValueError(f"expected BufferLoad in tl.region, got {load}")
    return load.buffer, list(load.indices), list(call.args[2:])
