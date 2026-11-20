"""Classify the fixed PPU15 FP8 copy path without speculative lowering.

Mirrors SelectCopyInstForLowering and the guards in Copy::LowerAiu at6451a379.
The admitted domain is default/pipeline-marked two-active-axis copies; custom
copy hints and unsupported syntax are unknown, never guessed. No kernel names,
measurements or generated CUDA enter this analysis.
"""

import math
from tilelang import tvm, _ffi_api
from tvm import tirx as T
from .copy_arith import FIELDS, footprint as simt_footprint


def plan(src, dst, layouts, threads, annotations, aiu_enabled):
    def simt(reason):
        return dict(known=True, path="simt", reason=reason, aiu_packets=0, aiu_bytes=0, aiu_rows=0, aiu_width=0)

    keys = set(map(str, annotations))
    if keys - {"no_implicit_async_commit_wait"}:
        return dict(known=False, reason="custom_copy_annotation")
    srcbuf, srcidx, srcext = src
    dstbuf, dstidx, dstext = dst
    if str(srcbuf.dtype) != "float8_e4m3fn" or str(dstbuf.dtype) != "float8_e4m3fn":
        return dict(known=False, reason="not_exact_e4m3fn")
    if not all(isinstance(e, T.IntImm) for e in srcext):
        return dict(known=False, reason="dynamic_copy_extent")
    active = [int(e) for e in srcext if int(e) != 1]
    if len(active) != 2:
        return dict(known=False, reason="non_2d_copy_domain")
    if not aiu_enabled:
        # A two-active-axis FP8 auto-CPAsync candidate is rejected by
        # HasAlignedPPUAsyncCopyTransactions/GetSingleActiveAxis at this
        # fixed baseline. Default non-pipelined copies are normal too.
        return simt("aiu_disabled_2d_simt")
    if srcbuf in layouts:
        return simt("nonlinear_global_layout")
    rank = len(srcbuf.shape)
    if rank < 2 or rank > 5:
        return simt("global_rank_outside_2_5")
    strides = list(srcbuf.strides)
    if not strides:
        strides = []
        stride = T.IntImm("int64", 1)
        for dim in reversed(srcbuf.shape):
            strides.insert(0, stride)
            stride *= dim
    analyzer = tvm.arith.Analyzer()
    strides = [analyzer.simplify(s) for s in strides]
    if not all(isinstance(s, T.IntImm) for s in strides):
        return dict(known=False, reason="symbolic_global_stride")
    if int(strides[-1]) != 1:
        return dict(known=False, reason="nonunit_inner_stride")
    if any(int(s) % 16 or int(s) >= (1 << 63) for s in strides[:-1]):
        return simt("unsupported_global_stride")
    if [int(e) for e in dstext if int(e) != 1] != active:
        return dict(known=False, reason="region_mapping")
    if dstbuf not in layouts:
        return simt("no_shared_layout")
    mode = _ffi_api.detect_swizzle_mode(layouts[dstbuf], dstbuf)
    width = 64 if mode.is_swizzle_64b() else 128 if mode.is_swizzle_128b() else 0
    if not width:
        return simt("not_64_128_swizzled")
    inner = int(srcext[-1])
    outer = active[0]
    if inner % width:
        return dict(known=False, reason="aiu_box_not_divisible")
    if outer < 8:
        return simt("too_few_rows")
    inner_splits = inner // width
    warps = max(1, threads // 32)
    target = max(1, warps // inner_splits)
    outer_splits = max(f for f in range(1, min(outer, target) + 1) if outer % f == 0)
    rows = outer // outer_splits
    if rows < 8:
        outer_splits = max(f for f in range(1, outer // 8 + 1) if outer % f == 0)
        rows = outer // outer_splits
    participating = inner_splits * outer_splits
    if participating > warps:
        return dict(known=False, reason="aiu_insufficient_warps")
    return dict(
        known=True,
        path="aiu",
        reason="proven_aiu",
        aiu_packets=participating / warps,
        aiu_bytes=inner * outer / warps,
        aiu_rows=rows,
        aiu_width=width,
    )


def footprint(src, dst, layouts, threads, analyzer, loop_vars=(), annotations=None):
    cfg = tvm.transform.PassContext.current().config
    path = plan(src, dst, layouts, threads, annotations or {}, not cfg.get("tl.disable_aiu_lower", True))
    if path["known"] and path["path"] == "aiu":
        result = dict(
            known=True,
            arith_known=True,
            vector_bytes=16,
            max_vector_bytes=16,
            iterations=0,
            instruction_demand=0,
            value_words=0,
            lane_guard=0,
            padding_elements=0,
            byte_demand=math.prod(int(e) for e in dst[2]) / threads,
        )
        for key in FIELDS:
            result["arith_" + key] = 0
        # No SIMT subword expansion; the separate path_aiu_width gives the
        # actual AIU box width (64/128B), not a vector-register transaction.
    else:
        result = simt_footprint(src, dst, layouts, threads, analyzer, loop_vars)
    result.update({"path_" + k: v for k, v in path.items()})
    return result
