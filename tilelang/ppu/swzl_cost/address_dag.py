"""Static ordinary-ldmatrix address expression DAG analysis.

Expressions are reconstructed from the actual inferred shared layout and the
fixed TensorCoreIntrinEmitter lane maps.  Counts describe pre-lowering
expression structure only; generated CUDA/ISA and measurements are absent.
"""

from collections import Counter, defaultdict


from tilelang import tvm
from tvm import tirx as T
from tilelang.ppu.intrinsics.layout.mma_layout import (
    ldmatrix_32x16_to_shared_16x32_layout_a as lane_a,
    ldmatrix_32x16_to_shared_16x32_layout_b as lane_b,
)

from . import common as early
from .live_range import _match_phase_keys, _phase_key


BINARY = (T.Add, T.Sub, T.Mul, T.FloorDiv, T.FloorMod, T.Min, T.Max)


def flatten_symbolic(layout, indices, analyzer):
    mapping = dict(zip(layout.get_forward_vars(), indices))
    value = T.IntImm("int32", 0)
    for expression, size in zip(layout.get_forward_index(), layout.get_output_shape()):
        mapped = T.stmt_functor.substitute(expression, mapping)
        value = value * int(size) + mapped
    return analyzer.simplify(value)


def _node_key(node):
    return str(node)


def operation_nodes(expression):
    result = {}

    def visit(node):
        if isinstance(node, (T.IntImm, T.Var, T.BufferLoad)):
            return
        if isinstance(node, T.Cast):
            return
        if isinstance(node, (*BINARY, T.Call)):
            result[_node_key(node)] = node

    T.stmt_functor.post_order_visit(expression, visit)
    return result


def variables(expression):
    result = set()

    def visit(node):
        if isinstance(node, T.Var):
            result.add(str(node.name))

    T.stmt_functor.post_order_visit(expression, visit)
    return result


def sethi_ullman(expression):
    if isinstance(expression, (T.IntImm, T.Var, T.BufferLoad)):
        return 1
    if isinstance(expression, T.Cast):
        return sethi_ullman(expression.value)
    if isinstance(expression, BINARY):
        left, right = sethi_ullman(expression.a), sethi_ullman(expression.b)
        return left + 1 if left == right else max(left, right)
    if isinstance(expression, T.Call):
        values = sorted((sethi_ullman(arg) for arg in expression.args), reverse=True)
        return max((value + index for index, value in enumerate(values)), default=1)
    return 1


def dag_stats(expressions):
    if not expressions:
        return dict(
            known=True, expressions=0, total_ops=0, unique_ops=0, shared_ops=0, lane_ki_shared_ops=0, peak_temporaries=0, working_set=0
        )
    nodes = [operation_nodes(expression) for expression in expressions]
    frequency = Counter(key for item in nodes for key in item)
    representatives = {}
    for item in nodes:
        representatives.update(item)
    shared = {key for key, count in frequency.items() if count > 1}
    lane_ki = {key for key in shared if variables(representatives[key]) <= {"swzl_tx", "swzl_ki"}}
    peak = max(sethi_ullman(expression) for expression in expressions)
    # Shared values can be retained while the current expression needs its
    # own evaluation stack.  This is a bounded candidate working set, not a
    # claim that the backend retains every value.
    return dict(
        known=True,
        expressions=len(expressions),
        total_ops=sum(len(item) for item in nodes),
        unique_ops=len(frequency),
        shared_ops=len(shared),
        lane_ki_shared_ops=len(lane_ki),
        peak_temporaries=peak,
        working_set=peak + len(lane_ki),
    )


def operand_expressions(call, layouts, threads, operand, analyzer):
    is_a = operand == "a"
    arg = 0 if is_a else 1
    buffer, origin, extent = early.region(call.args[arg])
    layout = layouts.get(buffer)
    if layout is None:
        raise ValueError("missing_shared_layout")
    m, n, k = map(int, call.args[5:8])
    warps = threads // 32
    wm = min(warps, m // 16)
    if not wm or warps % wm:
        raise ValueError("unsupported_warp_partition")
    wn = warps // wm
    size = m if is_a else n
    perwarp = size // (wm if is_a else wn)
    if perwarp % 16 or k % 32:
        raise ValueError("non_integral_matrix_tiles")
    tx = T.Var("swzl_tx", "int32")
    ki = T.Var("swzl_ki", "int32")
    lane = T.floormod(tx, 32)
    warp = T.floormod(T.floordiv(tx, 32), wm) if is_a else T.floormod(T.floordiv(tx, 32 * wm), wn)
    row, column = (lane_a if is_a else lane_b)(lane, 0)
    result = []
    for tile in range(perwarp // 16):
        indices = list(origin[:-2]) + [
            origin[-2] + warp * perwarp + tile * 16 + row,
            origin[-1] + ki * 32 + column,
        ]
        result.append(flatten_symbolic(layout, indices, analyzer))
    return result


def analyze(func, record):
    layouts, calls, threads = {}, [], []

    def visit(node):
        if hasattr(node, "annotations"):
            layouts.update(node.annotations.get("layout_map", {}))
        if isinstance(node, T.AttrStmt) and node.attr_key == "thread_extent" and node.node.thread_tag == "threadIdx.x":
            threads.append(int(node.value))
        if isinstance(node, T.Call) and early.op(node) == "tl.tileop.gemm":
            calls.append(node)

    T.stmt_functor.post_order_visit(func.body, visit)
    if len(set(threads)) != 1:
        raise ValueError("unsupported_threads")
    grouped = defaultdict(list)
    order = []
    for call in calls:
        key = _phase_key(call)
        if key not in grouped:
            order.append(key)
        grouped[key].append(call)
    matched_keys = _match_phase_keys(record["phases"], order)
    analyzer = tvm.arith.Analyzer()
    for phase, key in zip(record["phases"], matched_keys):
        call = grouped[key][0]
        expressions = []
        operand_stats = {}
        for operand in ("a", "b"):
            if not phase[operand + "_swizzle"]:
                operand_stats[operand] = dag_stats([])
                continue
            current = operand_expressions(call, layouts, threads[0], operand, analyzer)
            expressions.extend(current)
            operand_stats[operand] = dag_stats(current)
        stats = dag_stats(expressions)
        for name, value in stats.items():
            phase["address_dag_" + name] = value
        for operand, values in operand_stats.items():
            for name, value in values.items():
                phase[f"address_dag_{operand}_{name}"] = value
    return dict(known=True, phases=len(record["phases"]))
