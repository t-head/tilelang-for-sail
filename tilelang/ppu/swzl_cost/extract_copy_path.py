"""FP8 shared-to-fragment cost features, extracted once before LowerTileOp.

No operator name, benchmark result, runtime trip value or generated code enters
this module. Pipelined clones are counted per loop region, not as extra GEMMs.
Liveness is a conservative buffer-level backward analysis, not physical VGPRs.
"""

from collections import Counter, defaultdict
import math
from . import common as early
from .copy_path import FIELDS, footprint
from tilelang import tvm, _ffi_api
from tvm import tirx as T


def _integer(expr):
    if not isinstance(expr, T.IntImm):
        raise ValueError("nonconstant_geometry")
    return int(expr)


def _region(call):
    b, idx, extent = early.region(call)
    return b, idx, extent


def _full_write(buf, idx, extent):
    return (
        len(idx) == len(buf.shape)
        and all(isinstance(i, T.IntImm) and int(i) == 0 for i in idx)
        and all(tvm.ir.structural_equal(s, e) for s, e in zip(buf.shape, extent))
    )


def extract(func):
    try:
        return _extract(func)
    except (ValueError, TypeError, KeyError) as exc:
        return dict(supported=False, reason="unsupported_ir", detail=str(exc))


def _extract(func):
    target = func.attrs.get("target") if func.attrs else None
    if target is None or target.kind.name != "ppu" or str(target.attrs.get("arch", "")) != "ppu_15":
        return dict(supported=False, reason="target_keep_existing")
    loops = []
    predicates = []
    gemms = []
    copies = []
    fragments = {}
    layouts = {}
    threads = []
    analyzer = tvm.arith.Analyzer()

    def before(x):
        if isinstance(x, T.AttrStmt) and x.attr_key == "thread_extent":
            analyzer.bind(x.node.var, tvm.ir.Range.from_min_extent(0, x.value))
            if x.node.thread_tag == "threadIdx.x":
                threads.append(_integer(x.value))
        if hasattr(x, "annotations"):
            layouts.update(x.annotations.get("layout_map", {}))
        for b in getattr(x, "alloc_buffers", []):
            if b.scope() == "local.fragment":
                fragments[b.data] = b
        if isinstance(x, T.For):
            loops.append(x)
            analyzer.bind(x.loop_var, tvm.ir.Range.from_min_extent(x.min, x.extent))
        if isinstance(x, T.SBlockRealize):
            predicates.append(x.predicate)
        if not isinstance(x, T.Call):
            return
        name = early.op(x)
        if name == "tl.tileop.gemm":
            gemms.append((x, tuple(loops), any(not isinstance(p, T.IntImm) or not int(p) for p in predicates)))
        if name == "tl.tileop.copy":
            a, b = [_region(x.args[i]) for i in (0, 1)]
            guard = not all(analyzer.can_prove(i >= 0) and analyzer.can_prove(i + e <= s) for i, e, s in zip(a[1], a[2], a[0].shape))
            copies.append((a, b, tuple(loops), guard, dict(getattr(x, "annotations", {}))))

    def after(x):
        if isinstance(x, T.For):
            loops.pop()
        if isinstance(x, T.SBlockRealize):
            predicates.pop()

    T.stmt_functor.ir_transform(func.body, before, after)
    if not gemms:
        return dict(supported=False, reason="no_gemm")
    # The selected policy is function-wide.  Keep the existing policy when a
    # non-FP8 shared-to-fragment copy is present, otherwise an FP8 GEMM could
    # accidentally change the lowering of an unrelated BF16/INT8 fragment.
    if any(
        a[0].scope().startswith("shared")
        and b[0].scope() == "local.fragment"
        and (str(a[0].dtype) != "float8_e4m3fn" or str(b[0].dtype) != "float8_e4m3fn")
        for a, b, _, _, _ in copies
    ):
        return dict(supported=False, reason="mixed_fragment_dtype_keep_existing")
    # A mixed function must be untouched too: the decision is function-wide.
    if any(str(_region(g.args[i])[0].dtype) != "float8_e4m3fn" for g, _, _ in gemms for i in (0, 1)):
        return dict(supported=False, reason="dtype_keep_existing")
    if len(set(threads)) != 1 or threads[0] % 32:
        return dict(supported=False, reason="unsupported_threads")
    nt = threads[0]
    warps = nt // 32

    def words(b):
        layout = layouts.get(b)
        if layout is None:
            raise ValueError("missing_fragment_layout")
        return math.prod(_integer(s) for s in layout.get_output_shape()) * tvm.DataType(b.dtype).bits / 32

    def stage(ls):
        return max((int(l.annotations.get("tl_pipelined_num_stages", 0)) for l in ls), default=0)

    def context(ls):
        pipeline = [l for l in ls if l.annotations.get("tl_pipelined_num_stages", 0)]
        return pipeline[-1] if pipeline else (ls[-1] if ls else None)

    keys = {}
    nodes = {}
    counts = defaultdict(Counter)
    guarded = defaultdict(list)
    snapshots = {}
    call_contexts = {}

    def kind(b, ext, trans, is_a):
        if trans if is_a else not trans:
            return 0
        if b not in layouts or len(b.shape) < 2 or _integer(b.shape[-2]) % 16 or _integer(ext[-1]) * 8 < 256:
            return 0
        s = _ffi_api.detect_swizzle_mode(layouts[b], b)
        return 64 if s.is_swizzle_64b() else 128 if s.is_swizzle_128b() else 0

    for call, ls, predicate in gemms:
        regs = [_region(call.args[i]) for i in (0, 1, 2)]
        a, b, c = [r[0] for r in regs]
        m, n, k = [_integer(x) for x in call.args[5:8]]
        ta, tb = [bool(_integer(x)) for x in call.args[3:5]]
        if _integer(call.args[8]) != 1 or not all(x.scope().startswith("shared") for x in (a, b)):
            return dict(supported=False, reason="operand_path_keep_existing")
        # Ignore ONLY explicit leading page axes. Different tile offsets or
        # operands remain different phases; repeated calls in one loop count.
        offsets = []
        for _buf, idx, ext in regs:
            if len(idx) > 2 and any(_integer(e) != 1 for e in ext[:-2]):
                return dict(supported=False, reason="non_page_leading_region")
            offsets.append(tuple(_integer(i) for i in idx[-2:]))
        key = (a.data, b.data, c.data, m, n, k, ta, tb, tuple(offsets))
        keys[call] = key
        counts[key][context(ls)] += 1
        guarded[(key, context(ls))].append(predicate)
        wm = min(warps, m // 16)
        if not wm or warps % wm:
            return dict(supported=False, reason="unsupported_warp_partition")
        wn = warps // wm
        am = m / wm
        bn = n / wn
        al, bl = am * k / 512, bn * k / 512
        la, lb = kind(a, regs[0][2], ta, True), kind(b, regs[1][2], tb, False)
        phase = dict(
            rows=m,
            columns=n,
            inner=k,
            warps=warps,
            mma_atoms=am * bn / 256 * k / 32,
            acc_words=m * n / nt,
            matrix_packets=al + bl,
            a_packets=al,
            b_packets=bl,
            a_swizzle=la,
            b_swizzle=lb,
            eligible_packets=al * bool(la) + bl * bool(lb),
            reorder_words=4 * (al * bool(la) * ta + bl * bool(lb) * tb),
            stages=stage(ls),
            pages=max(math.prod(_integer(s) for s in z[0].shape) / math.prod(_integer(e) for e in z[2]) for z in regs[:2]),
            static_loop=int(bool(ls) and all(isinstance(l.extent, T.IntImm) for l in ls)),
            static_trip_count=math.prod(int(l.extent) for l in ls) if all(isinstance(l.extent, T.IntImm) for l in ls) else 0,
            transpose_a=ta,
            transpose_b=tb,
        )
        snapshots[(key, context(ls))] = dict(phase)
        call_contexts[call] = context(ls)
        if key in nodes:
            phase["stages"] = max(phase["stages"], nodes[key]["stages"])
            phase["pages"] = max(phase["pages"], nodes[key]["pages"])
            phase["static_loop"] = min(phase["static_loop"], nodes[key]["static_loop"])
            phase["static_trip_count"] = min(phase["static_trip_count"], nodes[key]["static_trip_count"])
        nodes[key] = phase

    def multiplicity(key):
        regions = counts[key]
        if len(regions) == 1:
            return next(iter(regions.values()))
        # Only normalize the bounded tail of one annotated pipeline. Never
        # merge two unrelated steady loops just because their GEMMs look alike.
        steady = [
            l
            for l in regions
            if l is not None
            and l.annotations.get("tl_pipelined_num_stages", 0)
            and (not isinstance(l.extent, T.IntImm) or int(l.extent) > int(l.annotations["tl_pipelined_num_stages"]))
        ]
        if len(steady) != 1:
            raise ValueError("ambiguous_repeated_gemm_regions")
        main = steady[0]
        limit = int(main.annotations["tl_pipelined_num_stages"])
        for loop, count in regions.items():
            if loop == main:
                continue
            if count > regions[main]:
                raise ValueError("unmatched_pipeline_tail")
            if loop is None:
                if not all(guarded[(key, loop)]):
                    raise ValueError("unguarded_repeated_gemm")
            elif not (
                isinstance(loop.extent, T.IntImm)
                and int(loop.extent) <= limit
                and int(loop.annotations.get("tl_pipelined_num_stages", 0)) == limit
            ):
                raise ValueError("unmatched_pipeline_tail")
        return regions[main]

    weights = {}
    for key in list(nodes):
        try:
            weights[key] = multiplicity(key)
        except ValueError:
            # An unproven clone is still real work. Preserve each loop region
            # separately instead of conflating it with the steady-state phase
            # or rejecting the entire otherwise legal FP8 function.
            for ctx, count in counts[key].items():
                regional = key + (ctx,)
                nodes[regional] = snapshots[(key, ctx)]
                weights[regional] = count
            del nodes[key]
            for call, base in list(keys.items()):
                if base == key:
                    keys[call] = key + (call_contexts[call],)
    # Backward may-liveness on actual statements, with full-buffer kills and
    # loop fixed points. Ignore SBlock aggregate metadata (not real accesses).
    phase_live = defaultdict(set)

    def loads(expr):
        found = set()

        def visit(x):
            if isinstance(x, T.BufferLoad) and x.buffer.data in fragments:
                found.add(x.buffer.data)

        T.stmt_functor.post_order_visit(expr, visit)
        return found

    def transfer(s, out):
        if isinstance(s, T.SeqStmt):
            live = set(out)
            for child in reversed(s.seq):
                live = transfer(child, live)
            return live
        if isinstance(s, T.For):
            live = set(out)
            for _ in range(len(fragments) + 2):
                nxt = set(out) | transfer(s.body, live)
                if nxt == live:
                    return live | loads(s.min) | loads(s.extent)
                live = nxt
            raise ValueError("liveness_did_not_converge")
        if isinstance(s, T.IfThenElse):
            return transfer(s.then_case, out) | (transfer(s.else_case, out) if s.else_case is not None else set(out)) | loads(s.condition)
        if isinstance(s, T.BufferStore):
            return set(out) | loads(s.value) | set().union(*(loads(i) for i in s.indices))
        if isinstance(s, T.Evaluate):
            call = s.value
            reads = set()
            writes = set()
            if isinstance(call, T.Call) and early.op(call).startswith("tl.tileop."):
                for arg in call.args:
                    if isinstance(arg, T.Call) and early.op(arg) == "tl.region":
                        b, idx, ext = _region(arg)
                        if b.data not in fragments:
                            continue
                        rw = _integer(arg.args[1])
                        if rw & 1:
                            reads.add(b.data)
                        if rw & 2 and _full_write(b, idx, ext):
                            writes.add(b.data)
                live = (set(out) - writes) | reads
                if call in keys:
                    phase_live[keys[call]].update(set(out) | reads)
                return live
            return set(out) | loads(call)
        if hasattr(s, "block"):
            return transfer(s.block, out)
        if hasattr(s, "body"):
            return transfer(s.body, out)
        return set(out)

    transfer(func.body, set())
    streams = {}
    copy_edges = []
    plans = {}
    for a, b, ls, guard, annotations in copies:
        copy_edges.append((a[0].data, b[0].data))
        if a[0].scope() == "global" and b[0].scope().startswith("shared") and ls:
            size = math.prod(_integer(e) for e in b[2]) / nt / 16
            plan = footprint(a, b, layouts, nt, analyzer, loop_vars=[l.loop_var for l in ls], annotations=annotations)
            plan["audit_destination"] = str(b[0].name)
            # Explicit custom planning annotations must not be guessed.
            previous = plans.get(b[0].data)
            if previous is None or plan.get("instruction_demand", 0) > previous.get("instruction_demand", 0):
                plans[b[0].data] = plan
            streams[b[0].data] = (max(size, streams.get(b[0].data, (0, False))[0]), guard or streams.get(b[0].data, (0, False))[1])
    cfg = tvm.transform.PassContext.current().config

    # Reachability through copies, never an Attention-specific predicate.
    def downstream(source):
        seen = {source}
        changed = True
        while changed:
            changed = False
            for a, b in copy_edges:
                if a in seen and b not in seen:
                    seen.add(b)
                    changed = True
        return seen

    phases = []
    for key, p in nodes.items():
        cdata = key[2]
        reachable = downstream(cdata)
        p["feeds_gemm"] = int(any(other != key and any(x in reachable for x in other[:2]) for other in nodes))
        p["multiplicity"] = weights[key]
        p["live_words"] = sum(words(fragments[b]) for b in phase_live[key])
        p["other_live_words"] = max(0, p["live_words"] - p["acc_words"])
        p["copy_packets"] = sum(v[0] for b, v in streams.items() if b in key[:2])
        p["copy_guard"] = int(
            any(v[1] for b, v in streams.items() if b in key[:2]) and not cfg.get("tl.disable_safe_memory_legalize", False)
        )
        selected = [v for b, v in plans.items() if b in key[:2]]
        p["copy_plan_known"] = int(all(v["known"] for v in selected))
        p["copy_instructions"] = sum(v.get("instruction_demand", 0) for v in selected)
        p["copy_value_words"] = sum(v.get("value_words", 0) for v in selected)
        p["copy_iterations"] = sum(v.get("iterations", 0) for v in selected)
        p["copy_lane_guard"] = int(any(v.get("lane_guard", 0) for v in selected))
        p["copy_min_width"] = min((v.get("vector_bytes", 1) for v in selected), default=16)
        p["copy_arith_known"] = int(all(v.get("arith_known", False) for v in selected))
        for field in FIELDS:
            p["copy_arith_" + field] = sum(v.get("arith_" + field, 0) for v in selected)
        p["copy_path_known"] = int(all(v.get("path_known", False) for v in selected))
        for field in ("aiu_packets", "aiu_bytes", "aiu_rows"):
            p["copy_" + field] = sum(v.get("path_" + field, 0) for v in selected)
        p["copy_aiu_streams"] = sum(v.get("path_path") == "aiu" for v in selected)
        phases.append(p)
    return dict(
        supported=any(p["eligible_packets"] for p in phases) and all(p["copy_path_known"] and p["copy_arith_known"] for p in phases),
        reason="fp8_semantic_features",
        dtype="float8_e4m3fn",
        phases=phases,
        aiu=int(not cfg.get("tl.disable_aiu_lower", True)),
        audit=dict(
            static_calls=len(gemms),
            semantic_phases=len(phases),
            fragment_buffers=len(fragments),
            allocated_fragment_words=sum(words(b) for b in fragments.values()),
            steady_copy_packets=sum(v[0] for v in streams.values()),
            copy_plans=list(plans.values()),
        ),
    )
