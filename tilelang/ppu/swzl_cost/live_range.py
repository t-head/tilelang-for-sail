"""Read-only FP8 live-range and matrix-address schedule analysis.

This runs once on the same pre-LowerTileOp TIR as the existing selector.  It
does not inspect generated CUDA/ISA or benchmark results.  In particular,
ordinary ldmatrix source addresses are modelled as sequential values: the
fixed lowering computes one address, consumes it immediately, then advances
to the next load.  The total number of addresses remains work, but is not a
simultaneous-register demand.
"""

from collections import defaultdict
import math


from tilelang import tvm
from tvm import tirx as T

from . import common as early
from .extract_copy_path import extract as previous_extract


def _integer(expr):
    if not isinstance(expr, T.IntImm):
        raise ValueError("nonconstant_geometry")
    return int(expr)


def _region(call):
    return early.region(call)


def _full_write(buf, idx, extent):
    return (
        len(idx) == len(buf.shape)
        and all(isinstance(i, T.IntImm) and int(i) == 0 for i in idx)
        and all(tvm.ir.structural_equal(s, e) for s, e in zip(buf.shape, extent))
    )


def _loads(expr, fragments):
    found = set()

    def visit(node):
        if isinstance(node, T.BufferLoad) and node.buffer.data in fragments:
            found.add(node.buffer.data)

    T.stmt_functor.post_order_visit(expr, visit)
    return found


def _contains_gemm(stmt):
    found = [False]

    def visit(node):
        if isinstance(node, T.Call) and early.op(node) == "tl.tileop.gemm":
            found[0] = True

    T.stmt_functor.post_order_visit(stmt, visit)
    return found[0]


def _phase_key(call):
    regions = [_region(call.args[i]) for i in (0, 1, 2)]
    offsets = []
    for _, idx, extent in regions:
        if len(idx) > 2 and any(_integer(e) != 1 for e in extent[:-2]):
            raise ValueError("non_page_leading_region")
        offsets.append(tuple(_integer(i) for i in idx[-2:]))
    return (
        regions[0][0].data,
        regions[1][0].data,
        regions[2][0].data,
        *(_integer(x) for x in call.args[5:8]),
        bool(_integer(call.args[3])),
        bool(_integer(call.args[4])),
        tuple(offsets),
    )


def _phase_signature(phase):
    return (
        phase["rows"],
        phase["columns"],
        phase["inner"],
        phase["transpose_a"],
        phase["transpose_b"],
    )


def _match_phase_keys(phases, ordered_keys):
    """Match semantic phases to physical GEMM identities conservatively.

    The semantic extractor may preserve multiple loop regions for one GEMM
    identity when a pipelined steady state and its bounded tails cannot be
    proven equivalent.  Live-range facts are buffer-level facts, so those
    regional phases may share the same physical key.  Distinct physical keys
    with the same public geometry remain ambiguous and are rejected.
    """
    if len(ordered_keys) == len(phases) and all(_phase_signature(phase) == key[3:8] for phase, key in zip(phases, ordered_keys)):
        return ordered_keys

    by_signature = defaultdict(list)
    for key in ordered_keys:
        by_signature[key[3:8]].append(key)
    matched = []
    for phase in phases:
        candidates = by_signature[_phase_signature(phase)]
        if len(candidates) != 1:
            raise ValueError(f"ambiguous_phase_identity:{len(candidates)}")
        matched.append(candidates[0])
    if set(matched) != set(ordered_keys):
        raise ValueError(f"phase_identity_mismatch:{len(ordered_keys)}!={len(phases)}")
    return matched


def annotate_address_schedule(phase):
    """Add exact fixed-lowering schedule facts to one semantic phase."""
    k_slices = max(1, phase["inner"] / 32)
    eligible_operands = int(bool(phase["a_swizzle"])) + int(bool(phase["b_swizzle"]))
    # A and B loads are emitted sequentially inside each ki.  Even when both
    # operands qualify, their source-address values do not overlap.
    phase["ordinary_address_peak"] = int(phase["eligible_packets"] > 0)
    phase["ordinary_address_work"] = phase["eligible_packets"]
    phase["ordinary_address_operands"] = eligible_operands
    phase["ordinary_address_work_per_slice"] = phase["eligible_packets"] / k_slices
    # Operand fragments exist on both paths.  Only the path-specific rebound
    # estimate is exposed here; it lives until the immediately following MMA.
    phase["swzl_binding_peak_words"] = phase["reorder_words"] / k_slices
    phase["matrix_operand_words"] = 4 * phase["matrix_packets"] / k_slices
    return phase


def analyze(func, record):
    fragments = {}
    layouts = {}
    gemms = []
    threads = []

    def collect(node):
        if hasattr(node, "annotations"):
            layouts.update(node.annotations.get("layout_map", {}))
        for buf in getattr(node, "alloc_buffers", []):
            if buf.scope() == "local.fragment":
                fragments[buf.data] = buf
        if isinstance(node, T.AttrStmt) and node.attr_key == "thread_extent" and node.node.thread_tag == "threadIdx.x":
            threads.append(_integer(node.value))
        if isinstance(node, T.Call) and early.op(node) == "tl.tileop.gemm":
            gemms.append(node)

    T.stmt_functor.post_order_visit(func.body, collect)
    if len(set(threads)) != 1:
        raise ValueError("unsupported_threads")
    nt = threads[0]

    def words(data):
        buf = fragments[data]
        layout = layouts.get(buf)
        if layout is None:
            raise ValueError("missing_fragment_layout")
        return math.prod(_integer(s) for s in layout.get_output_shape()) * tvm.DataType(buf.dtype).bits / 32

    ordered_keys = []
    calls_by_key = defaultdict(list)
    for call in gemms:
        key = _phase_key(call)
        if key not in calls_by_key:
            ordered_keys.append(key)
        calls_by_key[key].append(call)

    # Recompute the same conservative may-liveness, retaining buffer identity
    # so live-through access intensity can be measured rather than guessed.
    phase_live = defaultdict(set)

    def transfer(stmt, out):
        if isinstance(stmt, T.SeqStmt):
            live = set(out)
            for child in reversed(stmt.seq):
                live = transfer(child, live)
            return live
        if isinstance(stmt, T.For):
            live = set(out)
            for _ in range(len(fragments) + 2):
                nxt = set(out) | transfer(stmt.body, live)
                if nxt == live:
                    return live | _loads(stmt.min, fragments) | _loads(stmt.extent, fragments)
                live = nxt
            raise ValueError("liveness_did_not_converge")
        if isinstance(stmt, T.IfThenElse):
            other = transfer(stmt.else_case, out) if stmt.else_case is not None else set(out)
            return transfer(stmt.then_case, out) | other | _loads(stmt.condition, fragments)
        if isinstance(stmt, T.BufferStore):
            return set(out) | _loads(stmt.value, fragments) | set().union(*(_loads(i, fragments) for i in stmt.indices))
        if isinstance(stmt, T.Evaluate):
            call = stmt.value
            reads, writes = set(), set()
            if isinstance(call, T.Call) and early.op(call).startswith("tl.tileop."):
                for arg in call.args:
                    if isinstance(arg, T.Call) and early.op(arg) == "tl.region":
                        buf, idx, extent = _region(arg)
                        if buf.data not in fragments:
                            continue
                        rw = _integer(arg.args[1])
                        if rw & 1:
                            reads.add(buf.data)
                        if rw & 2 and _full_write(buf, idx, extent):
                            writes.add(buf.data)
                live = (set(out) - writes) | reads
                if early.op(call) == "tl.tileop.gemm":
                    phase_live[_phase_key(call)].update(set(out) | reads)
                return live
            return set(out) | _loads(call, fragments)
        if hasattr(stmt, "block"):
            return transfer(stmt.block, out)
        if hasattr(stmt, "body"):
            return transfer(stmt.body, out)
        return set(out)

    transfer(func.body, set())

    # Count non-GEMM fragment traffic in 32-bit words per thread.  Static
    # element loops are weighted; a loop that contains GEMM represents the
    # dynamic execution envelope and is deliberately counted once.
    reads = defaultdict(float)
    writes = defaultdict(float)
    known = [True]

    def count_expr(expr, weight):
        def visit(node):
            if isinstance(node, T.BufferLoad) and node.buffer.data in fragments:
                reads[node.buffer.data] += weight * tvm.DataType(node.buffer.dtype).bits / 32

        T.stmt_functor.post_order_visit(expr, visit)

    def count_stmt(stmt, weight=1.0):
        if isinstance(stmt, T.SeqStmt):
            for child in stmt.seq:
                count_stmt(child, weight)
            return
        if isinstance(stmt, T.For):
            child_weight = weight
            if not _contains_gemm(stmt.body):
                if isinstance(stmt.extent, T.IntImm):
                    child_weight *= int(stmt.extent)
                else:
                    known[0] = False
            count_stmt(stmt.body, child_weight)
            return
        if isinstance(stmt, T.IfThenElse):
            count_expr(stmt.condition, weight)
            count_stmt(stmt.then_case, weight)
            if stmt.else_case is not None:
                count_stmt(stmt.else_case, weight)
            return
        if isinstance(stmt, T.BufferStore):
            if stmt.buffer.data in fragments:
                writes[stmt.buffer.data] += weight * tvm.DataType(stmt.buffer.dtype).bits / 32
            count_expr(stmt.value, weight)
            for index in stmt.indices:
                count_expr(index, weight)
            return
        if isinstance(stmt, T.Evaluate):
            call = stmt.value
            if isinstance(call, T.Call) and early.op(call).startswith("tl.tileop."):
                if early.op(call) == "tl.tileop.gemm":
                    return
                for arg in call.args:
                    if not (isinstance(arg, T.Call) and early.op(arg) == "tl.region"):
                        continue
                    buf, _, extent = _region(arg)
                    if buf.data not in fragments:
                        continue
                    rw = _integer(arg.args[1])
                    volume = math.prod(_integer(e) for e in extent) * tvm.DataType(buf.dtype).bits / (32 * nt)
                    if rw & 1:
                        reads[buf.data] += weight * volume
                    if rw & 2:
                        writes[buf.data] += weight * volume
                return
            count_expr(call, weight)
            return
        if hasattr(stmt, "block"):
            count_stmt(stmt.block, weight)
        elif hasattr(stmt, "body"):
            count_stmt(stmt.body, weight)

    count_stmt(func.body)

    phases = record["phases"]
    matched_keys = _match_phase_keys(phases, ordered_keys)
    for phase, key in zip(phases, matched_keys):
        annotate_address_schedule(phase)
        active = key[2]
        live = set(phase_live[key])
        other = live - {active}
        phase["fragment_access_known"] = int(known[0])
        phase["active_non_gemm_read_words"] = reads[active]
        phase["active_non_gemm_write_words"] = writes[active]
        phase["live_through_read_words"] = sum(reads[data] for data in other)
        phase["live_through_write_words"] = sum(writes[data] for data in other)
        phase["live_through_touch_words"] = phase["live_through_read_words"] + phase["live_through_write_words"]
        phase["live_through_touch_density"] = phase["live_through_touch_words"] / max(1, phase["other_live_words"])
        phase["live_through_buffers"] = len(other)

    return {
        "known": bool(known[0]),
        "semantic_phases": len(phases),
        "fragment_buffers": len(fragments),
        "non_gemm_read_words": sum(reads.values()),
        "non_gemm_write_words": sum(writes.values()),
    }


def extract(func):
    record = previous_extract(func)
    if not record.get("supported"):
        return record
    before = tvm.ir.structural_hash(func)
    try:
        record["live_range_audit"] = analyze(func, record)
    except (ValueError, TypeError, KeyError, AssertionError) as exc:
        record["live_range_audit"] = {"known": False, "reason": str(exc)}
        for phase in record["phases"]:
            annotate_address_schedule(phase)
            phase["fragment_access_known"] = 0
    assert before == tvm.ir.structural_hash(func)
    return record
