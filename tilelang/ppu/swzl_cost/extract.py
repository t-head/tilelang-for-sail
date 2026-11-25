"""Compose the read-only pre-``LowerTileOp`` SWZL analyses."""

import math

from tilelang import tvm
from tvm import tirx as T

from .address_dag import analyze
from .live_range import extract as extract_live_range


def _launch_geometry(func):
    extents = {}

    def collect(node):
        if not (isinstance(node, T.AttrStmt) and node.attr_key == "thread_extent" and node.node.thread_tag.startswith("blockIdx.")):
            return
        if not isinstance(node.value, T.IntImm):
            raise ValueError("dynamic_launch_extent")
        tag = node.node.thread_tag
        value = int(node.value)
        if tag in extents and extents[tag] != value:
            raise ValueError("inconsistent_launch_extent")
        extents[tag] = value

    T.stmt_functor.post_order_visit(func.body, collect)
    if not extents:
        raise ValueError("missing_launch_extent")
    return {
        "known": True,
        "blocks": math.prod(extents.values()),
        "dimensions": len(extents),
    }


def extract(func):
    """Extract immutable FP8 SWZL cost inputs from one PrimFunc."""
    record = extract_live_range(func)
    if not record.get("supported") or not record.get("live_range_audit", {}).get("known"):
        return record

    before = tvm.ir.structural_hash(func)
    try:
        record["address_dag_audit"] = analyze(func, record)
    except (ValueError, TypeError, KeyError, AssertionError) as exc:
        record["address_dag_audit"] = {"known": False, "reason": str(exc)}
    if before != tvm.ir.structural_hash(func):
        raise AssertionError("SWZL cost analysis mutated its input PrimFunc")

    try:
        audit = _launch_geometry(func)
    except (ValueError, TypeError, KeyError, AssertionError) as exc:
        audit = {"known": False, "reason": str(exc)}
    record["launch_audit"] = audit
    record["launch_known"] = int(audit.get("known", False))
    record["launch_blocks"] = audit.get("blocks", 0)
    if before != tvm.ir.structural_hash(func):
        raise AssertionError("SWZL launch analysis mutated its input PrimFunc")
    return record
