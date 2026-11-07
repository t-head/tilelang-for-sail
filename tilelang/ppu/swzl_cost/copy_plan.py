"""Read-only transaction footprint, not a new copy lowering.

For equal-shaped non-fragment FP8 regions, prove contiguous/aligned chunks in
both physical layouts, then mirror SelectMinPaddingVectorSize. Unknown IR is
reported explicitly, never inferred from a benchmark name or measured code.
The returned counts are instruction-demand proxies, not peak physical VGPRs.
"""

import math
from tilelang import tvm
from tvm import tirx as T


def integer(x):
    if not isinstance(x, T.IntImm):
        raise ValueError("symbolic_copy_geometry")
    return int(x)


def min_padding_width(total, threads, maximum):
    widths = []
    width = maximum
    while width:
        widths.append(width)
        width //= 2
    return min(widths, key=lambda w: (((total + threads * w - 1) // (threads * w)) * threads * w - total, -w))


def footprint(src, dst, layouts, threads, analyzer):
    try:
        return _footprint(src, dst, layouts, threads, analyzer)
    except (ValueError, TypeError, KeyError) as exc:
        return dict(known=False, reason=str(exc))


def _footprint(src, dst, layouts, threads, analyzer):
    if str(src[0].dtype) != "float8_e4m3fn" or str(dst[0].dtype) != "float8_e4m3fn":
        raise ValueError("copy_dtype_not_e4m3fn")
    shapes = [[integer(e) for e in item[2]] for item in (src, dst)]
    logical = [[e for e in shape if e != 1] for shape in shapes]
    if logical[0] != logical[1] or not logical[0]:
        raise ValueError("copy_region_mapping_not_proven")
    total = math.prod(logical[0])
    linear = T.Var("copy_cost_linear", "int32")
    analyzer.bind(linear, tvm.ir.Range.from_min_extent(0, total))
    offsets = []
    predicates = []
    for (buffer, origin, extent), shape in zip((src, dst), shapes):
        indices = []
        stride = math.prod(shape)
        for base, e in zip(origin, shape):
            stride //= e
            indices.append(base + T.floormod(T.floordiv(linear, stride), e) if e != 1 else base)
        predicates.append([i >= 0 for i in indices] + [i < s for i, s in zip(indices, buffer.shape)])
        layout = layouts.get(buffer)
        if layout is not None:
            variables = list(layout.get_forward_vars())
            indices = [T.stmt_functor.substitute(v, dict(zip(variables, indices))) for v in layout.get_forward_index()]
            physical = list(layout.get_output_shape())
            strides = []
            stride = 1
            for extent in reversed(physical):
                strides.insert(0, stride)
                stride *= extent
        else:
            strides = list(buffer.strides)
            if not strides:
                strides = []
                stride = 1
                for extent in reversed(buffer.shape):
                    strides.insert(0, stride)
                    stride *= extent
        offsets.append(analyzer.simplify(sum(i * s for i, s in zip(indices, strides))))
    maximum = 1
    for width in (16, 8, 4, 2):
        if total % width:
            continue
        group = T.Var("copy_cost_group", "int32")
        analyzer.bind(group, tvm.ir.Range.from_min_extent(0, total // width))
        ok = True
        for offset in offsets:
            base = analyzer.simplify(T.stmt_functor.substitute(offset, {linear: group * width}))
            if not analyzer.can_prove(T.floormod(base, width) == 0):
                ok = False
                break
            for lane in range(1, width):
                value = analyzer.simplify(T.stmt_functor.substitute(offset, {linear: group * width + lane}))
                if not analyzer.can_prove(value - base == lane):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        for terms in predicates:
            for term in terms:
                start = analyzer.simplify(T.stmt_functor.substitute(term, {linear: group * width}))
                end = analyzer.simplify(T.stmt_functor.substitute(term, {linear: group * width + width - 1}))
                if not (tvm.ir.structural_equal(start, end) or analyzer.can_prove(start == end)):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            maximum = width
            break
    width = min_padding_width(total, threads, maximum)
    iterations = math.ceil(total / (threads * width))
    padded = iterations * threads * width - total
    return dict(
        known=True,
        vector_bytes=width,
        max_vector_bytes=maximum,
        iterations=iterations,
        padding_elements=padded,
        lane_guard=int(padded > 0),
        instruction_demand=total / threads / width,
        value_words=total / threads / width * max(1, width / 4),
        byte_demand=total / threads,
    )
