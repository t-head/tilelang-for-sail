"""Read-only copy address arithmetic before LowerTileOp.

Count simplified expression-DAG operations for the planned per-thread copy
iterations. Loop-invariant work is a hoisting opportunity, not proof the SDK
will hoist it or a prediction of cycles/VGPRs. Count non-power-of-two division
separately instead of assigning a guessed instruction latency.
"""

import math
from tilelang import tvm
from tvm import tirx as T
from .copy_plan import footprint as base_footprint, integer

FIELDS = ("ops", "invariant_ops", "unique_invariant_ops", "dynamic_ops", "nonpow_divmod")


def expression_nodes(expr, loop_vars):
    nodes = {}
    dynamic = set()
    nonpow = set()
    arithmetic = (T.Add, T.Sub, T.Mul, T.FloorDiv, T.FloorMod, T.Min, T.Max, T.Cast)

    def visit(node):
        accepted = isinstance(node, arithmetic)
        if isinstance(node, T.Call):
            accepted = node.op.name in ("tirx.bitwise_xor", "tirx.bitwise_and", "tirx.bitwise_or", "tirx.shift_left", "tirx.shift_right")
        if not accepted:
            return
        key = (tvm.ir.structural_hash(node), str(node))
        nodes[key] = node
        vs = []
        T.stmt_functor.post_order_visit(node, lambda n: vs.append(n) if isinstance(n, T.Var) else None)
        if any(v in loop_vars for v in vs):
            dynamic.add(key)
        if isinstance(node, (T.FloorDiv, T.FloorMod)):
            divisor = int(node.b) if isinstance(node.b, T.IntImm) else 0
            if divisor <= 0 or divisor & (divisor - 1):
                nonpow.add(key)

    T.stmt_functor.post_order_visit(expr, visit)
    return set(nodes), dynamic, nonpow


def analyze(src, dst, layouts, threads, analyzer, plan, loop_vars):
    width = plan["vector_bytes"]
    iterations = plan["iterations"]
    if iterations > 2048:
        raise ValueError("copy_arith_iteration_limit")
    tid = T.Var("copy_arith_lane", "int32")
    analyzer.bind(tid, tvm.ir.Range.from_min_extent(0, threads))
    total = {k: 0 for k in FIELDS}
    unique_invariant = set()
    for iteration in range(iterations):
        linear = tid * width + iteration * threads * width
        nodes = set()
        dynamic = set()
        nonpow = set()
        for buf, origin, extent in (src, dst):
            shape = [integer(e) for e in extent]
            stride = math.prod(shape)
            indices = []
            first = True
            for base, e in zip(origin, shape):
                stride //= e
                if e == 1:
                    indices.append(base)
                else:
                    # The copy guard excludes padded lanes. The leading
                    # coordinate is division only, never an extra wrap.
                    coord = T.floordiv(linear, stride)
                    indices.append(base + (coord if first else T.floormod(coord, e)))
                    first = False
            layout = layouts.get(buf)
            if layout is not None:
                mapping = dict(zip(layout.get_forward_vars(), indices))
                indices = [T.stmt_functor.substitute(v, mapping) for v in layout.get_forward_index()]
                physical = layout.get_output_shape()
                strides = []
                stride = 1
                for dim in reversed(physical):
                    strides.insert(0, stride)
                    stride *= dim
            else:
                strides = list(buf.strides)
                if not strides:
                    strides = []
                    stride = 1
                    for dim in reversed(buf.shape):
                        strides.insert(0, stride)
                        stride *= dim
            expr = analyzer.simplify(sum(i * s for i, s in zip(indices, strides)))
            n, d, np = expression_nodes(expr, loop_vars)
            nodes |= n
            dynamic |= d
            nonpow |= np
        invariant = nodes - dynamic
        unique_invariant |= invariant
        total["ops"] += len(nodes)
        total["dynamic_ops"] += len(dynamic)
        total["invariant_ops"] += len(invariant)
        total["nonpow_divmod"] += len(nonpow)
    total["unique_invariant_ops"] = len(unique_invariant)
    return total


def footprint(src, dst, layouts, threads, analyzer, loop_vars=()):
    result = base_footprint(src, dst, layouts, threads, analyzer)
    if result["known"]:
        try:
            data = analyze(src, dst, layouts, threads, analyzer, result, set(loop_vars))
            result.update({"arith_" + k: v for k, v in data.items()})
            result["arith_known"] = True
        except (ValueError, TypeError, KeyError) as exc:
            result["arith_known"] = False
            result["arith_reason"] = str(exc)
    return result
