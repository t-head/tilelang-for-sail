"""PPU programmatic dependent launch (PDL) language operators."""

from __future__ import annotations

from tvm import tirx

__all__ = ["pdl_sync", "pdl_trigger"]


def pdl_trigger():
    return tirx.call_intrin("void", tirx.op.Op.get("tl.pdl_trigger"))


def pdl_sync():
    return tirx.call_intrin("void", tirx.op.Op.get("tl.pdl_sync"))
