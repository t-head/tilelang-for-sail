"""PPU tests for `LowerTileOp` copy annotations affecting cp.async sync (path 2)."""

import pytest

import tilelang.language as T
import tilelang.layout
import tilelang.ppu.transform
import tilelang.testing
from tilelang import tvm
from tilelang.layout import make_full_bank_swizzled_layout, make_half_bank_swizzled_layout
from tvm.tirx.stmt_functor import post_order_visit


def _count_calls(func: tvm.tirx.PrimFunc):
    counts = {}

    def _visit(node):
        if isinstance(node, tvm.tirx.Call) and isinstance(node.op, tvm.ir.Op):
            name = str(node.op.name)
            counts[name] = counts.get(name, 0) + 1

    post_order_visit(func.body, _visit)
    return counts


@pytest.mark.parametrize("row_bytes", [64, 128])
@pytest.mark.parametrize("existing_alignment", [None, 16, 64, 256])
def test_ppu_lower_tile_op_preserves_smem_alignment(row_bytes, existing_alignment):
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})
    k = row_bytes // 2
    layout = make_full_bank_swizzled_layout if row_bytes == 128 else make_half_bank_swizzled_layout

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            a_pad = T.alloc_shared((16,), "uint8")
            b_A = T.alloc_shared((64, k), "bfloat16")
            c_B = T.alloc_shared((64, k), "bfloat16")
            acc = T.alloc_fragment((64, 64), "float32")
            T.annotate_layout({b_A: layout(b_A), c_B: layout(c_B)})
            T.clear(a_pad)
            T.clear(acc)
            T.gemm(b_A, c_B, acc, transpose_B=True)

    if existing_alignment is not None:
        main = main.with_attr("tl.smem_alignment_map", {"a_pad": 256, "b_A": existing_alignment})
    with target:
        mod = tvm.tirx.transform.BindTarget(target)(tvm.IRModule.from_expr(main))
        mod = tilelang.transform.MaterializeKernelLaunch()(mod)
        mod = tilelang.ppu.transform.LayoutInference()(mod)
        lowered = tilelang.ppu.transform.LowerTileOp()(mod)["main"]

    assert _count_calls(lowered).get("tl.tix_ldmatrix_swzl", 0) > 0
    expected = {"b_A": max(existing_alignment or 0, row_bytes), "c_B": row_bytes}
    if existing_alignment is not None:
        expected["a_pad"] = 256
    assert {str(key): int(value) for key, value in lowered.attrs["tl.smem_alignment_map"].items()} == expected


def _vectorized_cp_async_element_counts(func: tvm.tirx.PrimFunc):
    mod = tilelang.transform.VectorizeLoop()(tvm.IRModule.from_expr(func))
    element_counts = []

    def _visit(node):
        if isinstance(node, tvm.tirx.Call) and isinstance(node.op, tvm.ir.Op) and str(node.op.name) == "tl.ptx_cp_async":
            assert isinstance(node.args[2], tvm.tirx.IntImm)
            element_counts.append(int(node.args[2].value))

    post_order_visit(mod["main"].body, _visit)
    return element_counts


def _shared_sync_stores(func: tvm.tirx.PrimFunc):
    stores = []

    def _visit(node):
        if isinstance(node, tvm.tirx.BufferStore) and node.buffer.scope().startswith("shared"):
            stores.append(node)

    post_order_visit(func.body, _visit)
    return stores


def _lower_pipeline_managed_subword_copy(
    extent: int,
    *,
    dtype="uint16",
    src_offset: int = 0,
    dst_offset: int = 0,
    src_elem_offset: int = 0,
    dst_elem_offset: int = 0,
    coalesced_width=None,
):
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})

    @T.prim_func
    def before(
        A_handle: T.handle,
        B_handle: T.handle,
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        A = T.match_buffer(
            A_handle,
            (extent + src_offset,),
            dtype=dtype,
            elem_offset=src_elem_offset,
            align=128,
            offset_factor=1,
        )
        B = T.match_buffer(B_handle, (extent,), dtype=dtype, align=128)
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        S = T.sblock_alloc_buffer(
            (extent + dst_offset,),
            dtype=dtype,
            scope="shared",
            elem_offset=dst_elem_offset,
            align=128,
            offset_factor=1,
        )
        if coalesced_width is None:
            T.copy(
                A[src_offset : src_offset + extent],
                S[dst_offset : dst_offset + extent],
                annotations={"no_implicit_async_commit_wait": T.int32(1)},
            )
        else:
            T.copy(
                A[src_offset : src_offset + extent],
                S[dst_offset : dst_offset + extent],
                coalesced_width=coalesced_width,
                annotations={"no_implicit_async_commit_wait": T.int32(1)},
            )
        if tx < extent:
            B[tx] = S[dst_offset + tx]

    mod = tvm.IRModule.from_expr(before)
    with target:
        mod = tilelang.ppu.transform.LayoutInference()(mod)
        return tilelang.ppu.transform.LowerTileOp()(mod)["main"]


def _lower_pipeline_managed_padded_2d_copy():
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})

    @T.prim_func
    def before(
        A: T.Tensor((2, 5), "uint8"),
        B: T.Tensor((2, 4), "uint8"),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        S = T.sblock_alloc_buffer((2, 5), dtype="uint8", scope="shared", align=128)
        T.copy(
            A[0:2, 0:4],
            S[0:2, 0:4],
            annotations={"no_implicit_async_commit_wait": T.int32(1)},
        )
        if tx < 8:
            B[tx // 4, tx % 4] = S[tx // 4, tx % 4]

    mod = tvm.IRModule.from_expr(before)
    with target:
        mod = tilelang.ppu.transform.LayoutInference()(mod)
        return tilelang.ppu.transform.LowerTileOp()(mod)["main"]


def _lower_pipeline_managed_padded_1d_layout_copy(extent=64):
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})

    @T.prim_func
    def before(
        A: T.Tensor((extent,), "uint8"),
        B: T.Tensor((extent,), "uint8"),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        S = T.sblock_alloc_buffer((extent,), dtype="uint8", scope="shared", align=128)
        with T.sblock("root"):
            T.annotate_layout({S: T.Layout((extent,), lambda i: (i // 4) * 5 + i % 4)})
            T.copy(
                A[0:extent],
                S[0:extent],
                annotations={"no_implicit_async_commit_wait": T.int32(1)},
            )
            if tx < extent:
                B[tx] = S[tx]

    mod = tvm.IRModule.from_expr(before)
    with target:
        mod = tilelang.ppu.transform.LayoutInference()(mod)
        return tilelang.ppu.transform.LowerTileOp()(mod)["main"]


def _lower_pipeline_managed_chunk_permuted_1d_layout_copy():
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})

    @T.prim_func
    def before(
        A: T.Tensor((64,), "uint8"),
        B: T.Tensor((64,), "uint8"),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        S = T.sblock_alloc_buffer((64,), dtype="uint8", scope="shared", align=128)
        with T.sblock("root"):
            T.annotate_layout(
                {
                    S: T.Layout(
                        (64,),
                        lambda i: (i // 8) * 8 + (1 - (i // 4) % 2) * 4 + i % 4,
                    )
                }
            )
            T.copy(
                A[0:64],
                S[0:64],
                annotations={"no_implicit_async_commit_wait": T.int32(1)},
            )
            if tx < 64:
                B[tx] = S[tx]

    mod = tvm.IRModule.from_expr(before)
    with target:
        mod = tilelang.ppu.transform.LayoutInference()(mod)
        return tilelang.ppu.transform.LowerTileOp()(mod)["main"]


def test_ppu_lower_tile_op_respects_copy_annotation_for_pipeline_managed_cp_async():
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})

    @T.prim_func
    def before(
        A: T.Tensor((16,), T.float32),
        B: T.Tensor((16,), T.float32),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 16)
        S = T.alloc_buffer((16,), dtype=T.float32, scope="shared")
        T.copy(
            A[0:16],
            S,
            annotations={"no_implicit_async_commit_wait": T.int32(1)},
        )
        B[tx] = S[tx]

    mod = tvm.IRModule.from_expr(before)
    with target:
        mod = tilelang.ppu.transform.LowerTileOp()(mod)
    calls = _count_calls(mod["main"])

    assert calls.get("tl.ptx_cp_async", 0) > 0
    assert calls.get("tirx.ptx_commit_group", 0) == 0
    assert calls.get("tirx.ptx_wait_group", 0) == 0


def test_ppu_lower_tile_op_respects_copy_annotation_for_explicit_async_copy():
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})

    @T.prim_func
    def before(
        A: T.Tensor((16,), T.float32),
        B: T.Tensor((16,), T.float32),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 16)
        S = T.alloc_buffer((16,), dtype=T.float32, scope="shared")
        T.async_copy(
            A[0:16],
            S,
            annotations={"no_implicit_async_commit_wait": T.int32(1)},
        )
        B[tx] = S[tx]

    mod = tvm.IRModule.from_expr(before)
    with target:
        mod = tilelang.ppu.transform.LowerTileOp()(mod)
    calls = _count_calls(mod["main"])

    assert calls.get("tl.ptx_cp_async", 0) > 0
    assert calls.get("tirx.ptx_commit_group", 0) == 0
    assert calls.get("tirx.ptx_wait_group", 0) == 0


def test_ppu_lower_tile_op_respects_parallel_loop_async_annotation_without_pipeline_context():
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})

    @T.prim_func
    def before(
        A: T.Tensor((16,), T.float32),
        B: T.Tensor((16,), T.float32),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 16)
        S = T.alloc_buffer((16,), dtype=T.float32, scope="shared")
        for i in T.parallel(
            16,
            annotations={"parallel_async_without_async_commit_wait": T.bool(True)},
        ):
            S[i] = A[i]
        B[tx] = S[tx]

    mod = tvm.IRModule.from_expr(before)
    with target:
        mod = tilelang.ppu.transform.LayoutInference()(mod)
        mod = tilelang.ppu.transform.LowerTileOp()(mod)
    calls = _count_calls(mod["main"])

    assert calls.get("tl.ptx_cp_async", 0) > 0
    assert calls.get("tirx.ptx_commit_group", 0) == 0
    assert calls.get("tirx.ptx_wait_group", 0) == 0


def test_ppu_pipeline_managed_subword_copy_uses_minimum_legal_async_width():
    dtype_widths = {
        T.float4_e2m1fn: 8,
        "uint8": 4,
        "int8": 4,
        "float8_e4m3": 4,
        "float8_e4m3fn": 4,
        "float8_e5m2": 4,
        "float8_e8m0fnu": 4,
        "uint16": 2,
        "int16": 2,
        "float16": 2,
        "bfloat16": 2,
    }
    for dtype, expected_elements in dtype_widths.items():
        for extent in (64, 128):
            lowered = _lower_pipeline_managed_subword_copy(extent, dtype=dtype)
            calls = _count_calls(lowered)
            assert calls.get("tl.ptx_cp_async", 0) > 0, (dtype, extent)
            assert _vectorized_cp_async_element_counts(lowered) == [expected_elements], (dtype, extent)
            assert not _shared_sync_stores(lowered), (dtype, extent)
            assert calls.get("tirx.ptx_commit_group", 0) == 0
            assert calls.get("tirx.ptx_wait_group", 0) == 0

    # A naturally wider copy remains wider; the new policy is a floor and must
    # not cap an existing legal width.
    naturally_wider = _lower_pipeline_managed_subword_copy(512)
    calls = _count_calls(naturally_wider)
    assert calls.get("tl.ptx_cp_async", 0) > 0
    assert _vectorized_cp_async_element_counts(naturally_wider) == [4]

    # Copies outside the one-wave promotion limit still pass through the
    # alignment gate before preserving the common vectorizer's natural width.
    for side in ("src_offset", "dst_offset"):
        kwargs = {side: 2}
        naturally_wider = _lower_pipeline_managed_subword_copy(512, **kwargs)
        assert _count_calls(naturally_wider).get("tl.ptx_cp_async", 0) > 0, kwargs

    # Aligned region minima remain eligible.  Buffer-level elem_offset is
    # intentionally handled more conservatively below because LowerAccessPtr
    # does not currently fold it into the final pointer.
    aligned = (
        {"dtype": "uint8", "src_offset": 4},
        {"dtype": "uint8", "dst_offset": 4},
        {"dtype": "uint16", "src_offset": 2},
        {"dtype": "uint16", "dst_offset": 2},
        {"dtype": T.float4_e2m1fn, "src_offset": 8},
        {"dtype": T.float4_e2m1fn, "dst_offset": 8},
    )
    for kwargs in aligned:
        lowered = _lower_pipeline_managed_subword_copy(64, **kwargs)
        assert _count_calls(lowered).get("tl.ptx_cp_async", 0) > 0, kwargs

    # A layout may permute complete four-byte chunks without making the copy
    # unsafe.  Admit it as long as every final transaction start is aligned;
    # the common vectorizer separately proves unit stride inside each chunk.
    lowered = _lower_pipeline_managed_chunk_permuted_1d_layout_copy()
    assert _count_calls(lowered).get("tl.ptx_cp_async", 0) > 0
    assert _vectorized_cp_async_element_counts(lowered) == [4]
    assert not _shared_sync_stores(lowered)


def test_ppu_pipeline_managed_subword_copy_rejects_ragged_length():
    cases = tuple(
        {"extent": 64 + remainder, "dtype": dtype}
        for dtype, width in (
            (T.float4_e2m1fn, 8),
            ("uint8", 4),
            ("uint16", 2),
        )
        for remainder in range(1, width)
    )
    for kwargs in cases:
        lowered = _lower_pipeline_managed_subword_copy(**kwargs)
        calls = _count_calls(lowered)
        assert calls.get("tl.ptx_cp_async", 0) == 0, kwargs
        assert _shared_sync_stores(lowered), kwargs

    # A copy larger than one wave remains outside width promotion, but a known
    # incomplete four-byte transaction is still a safety failure rather than
    # an instruction to leave natural vectorization unchanged.
    lowered = _lower_pipeline_managed_subword_copy(129, dtype="uint16")
    assert _count_calls(lowered).get("tl.ptx_cp_async", 0) == 0
    assert _shared_sync_stores(lowered)


def test_ppu_pipeline_managed_subword_copy_rejects_misaligned_start():
    cases = tuple(
        {"extent": 64, "dtype": dtype, side: offset}
        for dtype, width in (
            (T.float4_e2m1fn, 8),
            ("uint8", 4),
            ("uint16", 2),
        )
        for side in ("src_offset", "dst_offset")
        for offset in range(1, width)
    )
    for kwargs in cases:
        lowered = _lower_pipeline_managed_subword_copy(**kwargs)
        calls = _count_calls(lowered)
        assert calls.get("tl.ptx_cp_async", 0) == 0, kwargs
        assert _shared_sync_stores(lowered), kwargs


def test_ppu_pipeline_managed_subword_copy_rejects_unqualified_cases():
    cases = (
        {"extent": 64, "dtype": "uint8", "src_elem_offset": 4},
        {"extent": 64, "dtype": "uint8", "dst_elem_offset": 4},
        {"extent": 64, "dtype": "uint16", "src_elem_offset": 2},
        {"extent": 64, "dtype": "uint16", "dst_elem_offset": 2},
        {"extent": 64, "dtype": T.float4_e2m1fn, "src_elem_offset": 8},
        {"extent": 64, "dtype": T.float4_e2m1fn, "dst_elem_offset": 8},
        {"extent": 64, "coalesced_width": 1},
        {"extent": 384},
        {"extent": 64, "dtype": T.float4_e2m1_unpacked},
    )
    for kwargs in cases:
        lowered = _lower_pipeline_managed_subword_copy(**kwargs)
        calls = _count_calls(lowered)
        assert calls.get("tl.ptx_cp_async", 0) == 0, kwargs
        assert _shared_sync_stores(lowered), kwargs

    # The first address and total byte count are both 4-byte aligned, but the
    # padded row stride makes the second transaction start at byte offset 5.
    # V1 therefore admits only a single active region axis.
    lowered = _lower_pipeline_managed_padded_2d_copy()
    assert _count_calls(lowered).get("tl.ptx_cp_async", 0) == 0
    assert _shared_sync_stores(lowered)

    # A single logical axis is not sufficient: this explicit layout is
    # contiguous only inside each group of four, with physical starts at
    # 0, 5, 10, ... .  Every transaction start must remain four-byte aligned.
    lowered = _lower_pipeline_managed_padded_1d_layout_copy()
    assert _count_calls(lowered).get("tl.ptx_cp_async", 0) == 0
    assert _shared_sync_stores(lowered)

    # The same final-layout check also guards copies outside the one-wave
    # auto-promotion limit, where natural vectorization remains possible.
    lowered = _lower_pipeline_managed_padded_1d_layout_copy(512)
    assert _count_calls(lowered).get("tl.ptx_cp_async", 0) == 0
    assert _shared_sync_stores(lowered)


def test_ppu_pipeline_managed_subword_copy_rejects_unsafe_offset_when_naturally_wide():
    # Without an explicit safety-failure result from the width planner, the
    # common vectorizer can still select a naturally wide cp.async even though
    # LowerAccessPtr does not fold Buffer::elem_offset into the final pointer.
    for side in ("src_elem_offset", "dst_elem_offset"):
        lowered = _lower_pipeline_managed_subword_copy(512, dtype="uint16", **{side: 2})
        assert _count_calls(lowered).get("tl.ptx_cp_async", 0) == 0, side
        assert _shared_sync_stores(lowered), side

    # A larger copy is not auto-promoted, but it can still become a naturally
    # wide cp.async.  Reject a two-byte region start before that vectorization.
    for side in ("src_offset", "dst_offset"):
        lowered = _lower_pipeline_managed_subword_copy(512, dtype="uint16", **{side: 1})
        assert _count_calls(lowered).get("tl.ptx_cp_async", 0) == 0, side
        assert _shared_sync_stores(lowered), side


def test_ppu_pipeline_subword_copy_is_annotated_and_lowered_end_to_end():
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})

    def make_pipeline(dtype, extent, alignment_elements):
        storage_extent = ((extent + alignment_elements - 1) // alignment_elements) * alignment_elements

        @T.prim_func
        def before(
            A: T.Tensor((4, storage_extent), dtype),
            B: T.Tensor((4, extent), dtype),
        ):
            T.func_attr({"global_symbol": "main", "target": target})
            T.launch_thread("blockIdx.x", 1)
            tx = T.launch_thread("threadIdx.x", 128)
            S = T.alloc_buffer((storage_extent,), dtype=dtype, scope="shared")
            for ko in T.Pipelined(4, num_stages=2):
                T.copy(A[ko, 0:extent], S[0:extent])
                if tx < extent:
                    B[ko, tx] = S[tx]

        return before

    for dtype, extent, expected_elements, expect_async in (
        (T.float4_e2m1fn, 64, 8, True),
        ("uint8", 64, 4, True),
        ("uint16", 64, 2, True),
        (T.float4_e2m1fn, 67, 8, False),
        ("uint8", 67, 4, False),
        ("uint16", 65, 2, False),
    ):
        before = make_pipeline(dtype, extent, expected_elements)
        mod = tvm.IRModule.from_expr(before)
        with target:
            mod = tilelang.transform.IfStmtBinding()(mod)
            mod = tilelang.transform.PipelinePlanning()(mod)
            mod = tilelang.transform.InjectSoftwarePipeline()(mod)
            mod = tilelang.transform.Simplify()(mod)
            mod = tilelang.ppu.transform.LayoutInference()(mod)
            mod = tilelang.ppu.transform.LowerTileOp()(mod)

        calls = _count_calls(mod["main"])
        if expect_async:
            assert calls.get("tl.ptx_cp_async", 0) > 0, dtype
            element_counts = _vectorized_cp_async_element_counts(mod["main"])
            assert element_counts, dtype
            assert all(count == expected_elements for count in element_counts), (
                dtype,
                element_counts,
            )
            assert not _shared_sync_stores(mod["main"]), dtype
        else:
            assert calls.get("tl.ptx_cp_async", 0) == 0, dtype
            assert _shared_sync_stores(mod["main"]), dtype


def _lower_multi_axis_2d_copy(
    rows: int,
    inner: int,
    *,
    src_stride: int = None,
    dst_stride: int = None,
    dtype: str = "uint8",
    layout=None,
    threads: int = 128,
):
    src_stride = inner if src_stride is None else src_stride
    dst_stride = inner if dst_stride is None else dst_stride
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})

    @T.prim_func
    def before(
        A: T.Tensor((rows, src_stride), dtype),
        B: T.Tensor((rows, inner), dtype),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", threads)
        S = T.sblock_alloc_buffer((rows, dst_stride), dtype=dtype, scope="shared", align=128)
        with T.sblock("root"):
            if layout is not None:
                T.annotate_layout({S: layout(S)})
            T.copy(
                A[0:rows, 0:inner],
                S[0:rows, 0:inner],
                annotations={"no_implicit_async_commit_wait": T.int32(1)},
            )
            if tx < rows * inner:
                B[tx // inner, tx % inner] = S[tx // inner, tx % inner]

    mod = tvm.IRModule.from_expr(before)
    with target:
        mod = tilelang.ppu.transform.LayoutInference()(mod)
        return tilelang.ppu.transform.LowerTileOp()(mod)["main"]


def _row_xor_layout(buffer):
    # XOR the inner index with a row-dependent multiple of four.  Additive row
    # terms (XOR-on-row times cols, floormod, select, ...) stay provable because
    # modular analysis tracks their coefficient; XOR on the inner index has an
    # unknown modular set, so the whole-tile symbolic proof fails and validation
    # must enumerate concrete rows.  The mask's low two bits are zero, so every
    # transaction start stays 4-aligned and each enumerated row passes.
    rows, cols = buffer.shape
    return T.Layout((rows, cols), lambda i, j: i * cols + (j ^ (i * 4)))


def test_ppu_multi_axis_aligned_2d_copy_emits_cp_async():
    # One-wave scale: the width planner promotes to whole four-byte transactions.
    lowered = _lower_multi_axis_2d_copy(2, 4, src_stride=8, dst_stride=8)
    calls = _count_calls(lowered)
    assert calls.get("tl.ptx_cp_async", 0) > 0
    assert not _shared_sync_stores(lowered)
    assert calls.get("tirx.ptx_commit_group", 0) == 0
    assert calls.get("tirx.ptx_wait_group", 0) == 0

    # Beyond one wave: natural vectorization still emits async copies once every
    # row start is proven aligned (contiguous rows here).
    for dtype in ("uint8", "float16"):
        lowered = _lower_multi_axis_2d_copy(8, 64, dtype=dtype, threads=64)
        calls = _count_calls(lowered)
        assert calls.get("tl.ptx_cp_async", 0) > 0, dtype
        assert not _shared_sync_stores(lowered), dtype


def test_ppu_multi_axis_swizzled_2d_copy_emits_cp_async():
    lowered = _lower_multi_axis_2d_copy(
        8,
        64,
        dtype="float16",
        threads=64,
        layout=lambda S: tilelang.layout.make_ppu_swizzled_layout(S),
    )
    calls = _count_calls(lowered)
    assert calls.get("tl.ptx_cp_async", 0) > 0
    assert not _shared_sync_stores(lowered)


def test_ppu_multi_axis_row_enumerated_copy_stays_synchronous():
    # The row-XOR layout defeats the whole-tile symbolic proof, so validation
    # falls back to enumerating concrete rows; every row is 4-aligned and the
    # enumeration accepts.  The common vectorizer still cannot prove unit stride
    # through the XOR, so the established synchronous lowering remains in effect
    # (no async instruction, real shared stores).  The assertions pin that
    # conservative outcome: any lowering change that starts emitting async on
    # these layouts without a sound stride proof will fail here.
    for rows, inner, threads in ((2, 16, 128), (8, 64, 128)):
        lowered = _lower_multi_axis_2d_copy(rows, inner, threads=threads, layout=_row_xor_layout)
        calls = _count_calls(lowered)
        assert calls.get("tl.ptx_cp_async", 0) == 0, (rows, inner)
        assert _shared_sync_stores(lowered), (rows, inner)


def test_ppu_multi_axis_copy_rejects_misaligned_row_stride():
    # Inner extent tiles into whole transactions, but the padded row stride puts
    # some row starts off a four-byte boundary; both width regimes must reject.
    # Pad source and destination independently so each side's row validation is
    # exercised on its own (padding both lets the source check short-circuit).
    for rows, inner, stride, threads in ((2, 4, 6, 128), (4, 64, 66, 64)):
        for src_stride, dst_stride in ((stride, inner), (inner, stride)):
            lowered = _lower_multi_axis_2d_copy(rows, inner, src_stride=src_stride, dst_stride=dst_stride, threads=threads)
            calls = _count_calls(lowered)
            assert calls.get("tl.ptx_cp_async", 0) == 0, (rows, inner, stride, src_stride, dst_stride)
            assert _shared_sync_stores(lowered), (rows, inner, stride, src_stride, dst_stride)


def test_ppu_multi_axis_copy_rejects_mismatched_iteration_domain():
    # The SIMT loop iterates the shared destination's extents, so a (2, 4) ->
    # (2, 8) copy visits source transaction starts beyond the validated source
    # range.  With this source layout the source-range check alone passes
    # (offsets 0 and 12 are aligned) and the contiguous destination is provably
    # aligned, while the loop also visits misaligned source starts 5 and 17;
    # the domain-mismatch guard must keep the copy synchronous.
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})

    @T.prim_func
    def before(
        A: T.Tensor((2, 4), "uint8"),
        B: T.Tensor((2, 8), "uint8"),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        S = T.sblock_alloc_buffer((2, 8), dtype="uint8", scope="shared", align=128)
        with T.sblock("root"):
            T.annotate_layout({A: T.Layout((2, 4), lambda i, j: 12 * i + j + j // 4)})
            T.copy(
                A[0:2, 0:4],
                S[0:2, 0:8],
                annotations={"no_implicit_async_commit_wait": T.int32(1)},
            )
            if tx < 16:
                B[tx // 8, tx % 8] = S[tx // 8, tx % 8]

    mod = tvm.IRModule.from_expr(before)
    with target:
        mod = tilelang.ppu.transform.LayoutInference()(mod)
        lowered = tilelang.ppu.transform.LowerTileOp()(mod)["main"]
    calls = _count_calls(lowered)
    assert calls.get("tl.ptx_cp_async", 0) == 0
    assert _shared_sync_stores(lowered)


def test_ppu_multi_axis_copy_with_interleaved_unit_axes_emits_cp_async():
    # Unit axes sit at different positions on the two sides (src axis 0, dst
    # axis 2); MakeIndices turns them into constant indices and pairs the
    # remaining non-unit extents (8, 64), which match.  The domain guard must
    # skip the unit axes and accept, so async copy is still emitted.  Pins the
    # skip branches against an over-conservative per-dimension comparison.
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})

    @T.prim_func
    def before(
        A: T.Tensor((1, 8, 64), "uint8"),
        B: T.Tensor((1, 8, 64), "uint8"),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        S = T.sblock_alloc_buffer((8, 64, 1), dtype="uint8", scope="shared", align=128)
        with T.sblock("root"):
            T.copy(
                A[0:1, 0:8, 0:64],
                S[0:8, 0:64, 0:1],
                annotations={"no_implicit_async_commit_wait": T.int32(1)},
            )
            for r in T.serial(4):
                idx = tx * 4 + r
                B[0, idx // 64, idx % 64] = S[idx // 64, idx % 64, 0]

    mod = tvm.IRModule.from_expr(before)
    with target:
        mod = tilelang.ppu.transform.LayoutInference()(mod)
        lowered = tilelang.ppu.transform.LowerTileOp()(mod)["main"]
    calls = _count_calls(lowered)
    assert calls.get("tl.ptx_cp_async", 0) > 0
    assert not _shared_sync_stores(lowered)


def test_ppu_multi_axis_copy_enumeration_budget_boundary():
    # uint8 inner extent 64 -> 16 transactions per row.  The row-XOR layout
    # defeats the symbolic proof, so both cases reach the enumeration budget:
    # 256 rows hit the 4096-transaction budget exactly; 272 rows exceed it and
    # must be rejected against the remaining budget before any multiplication.
    # 272 (not 258/264) keeps a per-thread element count divisible by four, so
    # the width planner still offers a legal four-element transaction width.
    #
    # Observability limit: both sides of the boundary lower to the same
    # synchronous IR (verified structurally identical apart from extents).
    # The vectorizer cannot prove unit stride through the inner-index XOR, so
    # even an accepted tile falls back to synchronous stores, and no layout
    # that defeats the symbolic proof keeps a provable stride (additive
    # 4-aligned row terms stay symbolically provable; swizzled layouts are
    # provable too and never reach enumeration).  These assertions therefore
    # pin the conservative fallback and bound the enumeration work; catching
    # an off-by-one in the budget comparison itself would need a C++-level
    # validator test (no such test infrastructure in-tree yet).
    at_budget = _lower_multi_axis_2d_copy(256, 64, layout=_row_xor_layout)
    calls = _count_calls(at_budget)
    assert calls.get("tl.ptx_cp_async", 0) == 0
    assert _shared_sync_stores(at_budget)

    over_budget = _lower_multi_axis_2d_copy(272, 64, layout=_row_xor_layout)
    calls = _count_calls(over_budget)
    assert calls.get("tl.ptx_cp_async", 0) == 0
    assert _shared_sync_stores(over_budget)


def test_ppu_multi_axis_copy_rejects_huge_extents_before_multiplying():
    # Enumeration budget on huge outer extents: 4096 x 4096 outer rows with one
    # transaction per row must be rejected without attempting 2^24 row
    # enumerations (removing the budget guard would hang the lowering below).
    # Pre- and post-multiplication rejection are observationally equivalent at
    # this size; a true int64-wrap case cannot be expressed through the
    # frontend (T.Layout iteration vars are int32, and larger affine layout
    # offsets overflow the simplifier's int32 arithmetic first), so the
    # overflow-safety of the factor-by-factor guard is verified by construction
    # rather than by this test.
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})

    @T.prim_func
    def before(
        A: T.Tensor((4096, 4096, 4), "uint8"),
        B: T.Tensor((4096, 4096, 4), "uint8"),
    ):
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        S = T.sblock_alloc_buffer((4096, 4096, 4), dtype="uint8", scope="shared", align=128)
        with T.sblock("root"):
            T.annotate_layout(
                {
                    S: T.Layout(
                        (4096, 4096, 4),
                        lambda i, j, k: (i * 4096 + j) * 4 + (k ^ (i * 4)),
                    )
                }
            )
            T.copy(
                A[0:4096, 0:4096, 0:4],
                S[0:4096, 0:4096, 0:4],
                annotations={"no_implicit_async_commit_wait": T.int32(1)},
            )
            if tx < 4:
                B[0, 0, tx] = S[0, 0, tx]

    mod = tvm.IRModule.from_expr(before)
    with target:
        mod = tilelang.ppu.transform.LayoutInference()(mod)
        lowered = tilelang.ppu.transform.LowerTileOp()(mod)["main"]
    calls = _count_calls(lowered)
    assert calls.get("tl.ptx_cp_async", 0) == 0
    assert _shared_sync_stores(lowered)


if __name__ == "__main__":
    tilelang.testing.main()
