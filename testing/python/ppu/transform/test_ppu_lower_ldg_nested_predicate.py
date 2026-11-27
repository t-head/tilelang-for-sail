"""PPU LowerLDGSTG: a nested predicated load keeps the enclosing store guard.

PPU counterpart of the CUDA fix #3390.
"""

import pytest

import tilelang as tl
import tilelang.language as T
import tilelang.ppu.transform as P
import tilelang.testing
from tilelang import tvm as tvm
from tilelang.transform import PassConfigKey
from tvm import tirx
from tvm.target import Target


def _apply_passes(mod, enable_non_predicated=False, enable_predicated=False):
    mod = tvm.tirx.transform.BindTarget(Target({"kind": "ppu", "arch": "ppu_15"}))(mod)
    mod = tl.transform.FlattenBuffer()(mod)
    mod = tl.transform.VectorizeLoop()(mod)
    with tvm.transform.PassContext(
        config={
            PassConfigKey.TL_ENABLE_LOWER_LDGSTG: enable_non_predicated,
            PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: enable_predicated,
        }
    ):
        mod = P.LowerLDGSTG()(mod)
    return mod


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("lanes", [1, 2, 4, 8])
@pytest.mark.parametrize("enable_non_predicated", [False, True])
@pytest.mark.parametrize("enclosing_store", [False, True])
def test_nested_predicated_load_preserves_store_guard(lanes, enable_non_predicated, enclosing_store):
    """A nested load must retain both predicates, for scalar and vector widths."""
    iterator = T.serial if lanes == 1 else T.vectorized

    @T.prim_func
    def func(
        A: T.Buffer((128,), "float32"),
        B: T.Buffer((128,), "float32"),
        outer: T.int32,
        inner: T.int32,
    ):
        for i in T.thread_binding(128 // lanes, "threadIdx.x"):
            for j in iterator(lanes):
                if enclosing_store:
                    with T.If(outer > 0), T.Then():
                        B[i * lanes + j] = T.if_then_else(inner > 0, A[i * lanes + j], T.float32(0))
                else:
                    B[i * lanes + j] = T.if_then_else(inner > 0, A[i * lanes + j], T.float32(0))

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    lowered = _apply_passes(mod, enable_non_predicated=enable_non_predicated, enable_predicated=True)
    loads, stores = [], []

    def visitor(node):
        if isinstance(node, tirx.Call) and hasattr(node.op, "name"):
            if node.op.name == f"tl.ldg{32 * lanes}":
                loads.append(node)
            elif node.op.name == f"tl.stg{32 * lanes}":
                stores.append(node)

    tirx.stmt_functor.post_order_visit(lowered["main"].body, visitor)
    assert len(loads) == 1
    outer, inner = func.params[2:]
    expected = tirx.And(outer > 0, inner > 0) if enclosing_store else inner > 0
    tvm.ir.assert_structural_equal(loads[0].args[-1], expected)
    if enclosing_store:
        assert len(stores) == 1
        tvm.ir.assert_structural_equal(stores[0].args[-1], outer > 0)


@tilelang.testing.requires_ppu
def test_nested_predicated_load_does_not_leak_store_guard():
    """A subsequent load uses its own condition, not a preceding store guard."""

    @T.prim_func
    def func(
        A: T.Buffer((128,), "float32"),
        B: T.Buffer((128,), "float32"),
        outer: T.int32,
        inner: T.int32,
    ):
        for i in T.thread_binding(128, "threadIdx.x"):
            with T.If(outer > 0), T.Then():
                B[i] = T.if_then_else(inner > 0, A[i], T.float32(0))
            B[i] = T.if_then_else(inner > 0, A[i], T.float32(0))

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    lowered = _apply_passes(mod, enable_predicated=True)
    predicates = []

    def visitor(node):
        if isinstance(node, tirx.Call) and hasattr(node.op, "name") and node.op.name == "tl.ldg32":
            predicates.append(node.args[-1])

    tirx.stmt_functor.post_order_visit(lowered["main"].body, visitor)
    assert len(predicates) == 2
    outer, inner = func.params[2:]
    tvm.ir.assert_structural_equal(predicates[0], tirx.And(outer > 0, inner > 0))
    tvm.ir.assert_structural_equal(predicates[1], inner > 0)


if __name__ == "__main__":
    tilelang.testing.main()
