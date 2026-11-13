"""Focused tests for the pre-LowerTileOp FP8 SWZL cost decision."""

import pytest

import tilelang
import tilelang.ppu.language as T
import tilelang.ppu.pipeline as ppu_pipeline
import tilelang.ppu.swzl_cost as swzl_cost
import tilelang.ppu.swzl_cost.model as swzl_model
from tilelang import tvm
from tilelang.layout import make_full_bank_swizzled_layout
from tilelang.ppu.swzl_cost.live_range import _match_phase_keys
from tilelang.ppu.swzl_cost.model import calibrated_decide, dynamic_four_warp_packet_reuse_choice


def _module(functions=1):
    result = {}
    for index in range(functions):
        name = "main" if index == 0 else f"other_{index}"
        result[name] = tvm.tirx.PrimFunc([], tvm.tirx.Evaluate(0)).with_attr("global_symbol", name)
    return tvm.IRModule(result)


def _install_fake_lower(monkeypatch, observed_disable):
    def factory():
        def run(mod):
            config = tvm.transform.PassContext.current().config
            observed_disable.append(bool(config.get(tilelang.PassConfigKey.TL_DISABLE_LDMAT_SWZL, False)))
            return mod

        return run

    monkeypatch.setattr(tilelang.ppu.transform, "LowerTileOp", factory)


def _real_fp8_module(with_bf16_fragment_copy=False):
    @T.prim_func
    def main(
        a: T.Tensor((128, 128), T.float8_e4m3fn),
        b: T.Tensor((128, 128), T.float8_e4m3fn),
        c: T.Tensor((128, 128), T.float32),
    ):
        with T.Kernel(1, threads=128):
            a_shared = T.alloc_shared((128, 128), T.float8_e4m3fn)
            b_shared = T.alloc_shared((128, 128), T.float8_e4m3fn)
            c_local = T.alloc_fragment((128, 128), T.float32)
            T.copy(a, a_shared)
            T.copy(b, b_shared)
            T.clear(c_local)
            T.gemm(
                a_shared,
                b_shared,
                c_local,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            T.copy(c_local, c)
            if with_bf16_fragment_copy:
                extra_shared = T.alloc_shared((32, 64), T.bfloat16)
                extra_fragment = T.alloc_fragment((16, 16), T.bfloat16)
                base = T.Fragment(
                    (16, 16),
                    forward_fn=lambda i, j: (
                        4 * (i % 8) + j % 8 // 2,
                        4 * (i // 8) + 2 * (j // 8) + j % 2,
                    ),
                )
                T.annotate_layout(
                    {
                        extra_shared: make_full_bank_swizzled_layout(extra_shared),
                        extra_fragment: base.replicate(4),
                    }
                )
                T.copy(extra_shared[:16, :16], extra_fragment)

    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})
    mod = tvm.IRModule.from_expr(main)
    with target, tvm.transform.PassContext(opt_level=3):
        mod = tvm.tirx.transform.BindTarget(target)(mod)
        mod = tilelang.transform.MaterializeKernelLaunch()(mod)
        mod = tilelang.ppu.transform.LayoutInference()(mod)
    return target, mod


def _lowered_ops(mod, target):
    with target, tvm.transform.PassContext(opt_level=3):
        mod = ppu_pipeline._lower_tile_op_with_swzl_cost_model(mod)
    ops = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and isinstance(node.op, tvm.ir.Op):
            ops.append(node.op.name)

    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    return ops


def test_swzl_cost_model_runs_by_default(monkeypatch):
    observed_disable = []
    _install_fake_lower(monkeypatch, observed_disable)
    monkeypatch.setattr(swzl_cost, "extract", lambda _func: {"supported": True})
    monkeypatch.setattr(swzl_cost, "decide", lambda _record: {"choice": "ordinary"})

    with tvm.transform.PassContext():
        result = ppu_pipeline._lower_tile_op_with_swzl_cost_model(_module())

    assert len(result.functions) == 1
    assert observed_disable == [True]


def test_swzl_cost_model_can_be_disabled(monkeypatch):
    observed_disable = []
    _install_fake_lower(monkeypatch, observed_disable)
    monkeypatch.setattr(
        swzl_cost,
        "extract",
        lambda _func: pytest.fail("disabled cost model must not inspect the function"),
    )
    config = {tilelang.PassConfigKey.TL_DISABLE_LDMAT_SWZL_COST_MODEL: True}

    with tvm.transform.PassContext(config=config):
        ppu_pipeline._lower_tile_op_with_swzl_cost_model(_module())

    assert observed_disable == [False]


@pytest.mark.parametrize(
    "choice,expected_disable",
    [("keep_existing", False), ("swzl", False), ("ordinary", True)],
)
def test_swzl_cost_model_makes_one_function_wide_decision(monkeypatch, choice, expected_disable):
    observed_disable = []
    calls = {"extract": 0, "decide": 0}
    _install_fake_lower(monkeypatch, observed_disable)

    def extract(_func):
        calls["extract"] += 1
        return {"supported": True, "phases": []}

    def decide(_record):
        calls["decide"] += 1
        return {"choice": choice, "reason": "test"}

    monkeypatch.setattr(swzl_cost, "extract", extract)
    monkeypatch.setattr(swzl_cost, "decide", decide)
    with tvm.transform.PassContext():
        result = ppu_pipeline._lower_tile_op_with_swzl_cost_model(_module())
        assert not tvm.transform.PassContext.current().config.get(tilelang.PassConfigKey.TL_DISABLE_LDMAT_SWZL, False)

    assert len(result.functions) == 1
    assert calls == {"extract": 1, "decide": 1}
    assert observed_disable == [expected_disable]


def test_explicit_swzl_disable_precedes_cost_model(monkeypatch):
    observed_disable = []
    _install_fake_lower(monkeypatch, observed_disable)
    monkeypatch.setattr(
        swzl_cost,
        "extract",
        lambda _func: pytest.fail("explicit disable must bypass the cost model"),
    )
    config = {
        tilelang.PassConfigKey.TL_DISABLE_LDMAT_SWZL: True,
    }

    with tvm.transform.PassContext(config=config):
        ppu_pipeline._lower_tile_op_with_swzl_cost_model(_module())

    assert observed_disable == [True]


def test_explicit_false_allows_nested_ordinary_without_mutating_outer_context(monkeypatch):
    observed_disable = []
    _install_fake_lower(monkeypatch, observed_disable)
    monkeypatch.setattr(swzl_cost, "extract", lambda _func: {"supported": True})
    monkeypatch.setattr(swzl_cost, "decide", lambda _record: {"choice": "ordinary"})
    config = {
        tilelang.PassConfigKey.TL_DISABLE_LDMAT_SWZL: False,
    }

    with tvm.transform.PassContext(config=config):
        ppu_pipeline._lower_tile_op_with_swzl_cost_model(_module())
        outer_config = tvm.transform.PassContext.current().config
        assert tilelang.PassConfigKey.TL_DISABLE_LDMAT_SWZL in outer_config
        assert not bool(outer_config[tilelang.PassConfigKey.TL_DISABLE_LDMAT_SWZL])

    assert observed_disable == [True]


def test_multiple_functions_keep_existing_swzl_policy(monkeypatch):
    observed_disable = []
    _install_fake_lower(monkeypatch, observed_disable)
    monkeypatch.setattr(
        swzl_cost,
        "extract",
        lambda _func: pytest.fail("module-wide policy requires exactly one PrimFunc"),
    )
    with tvm.transform.PassContext():
        ppu_pipeline._lower_tile_op_with_swzl_cost_model(_module(functions=2))

    assert observed_disable == [False]


def test_analysis_failure_keeps_existing_swzl_policy(monkeypatch):
    observed_disable = []
    _install_fake_lower(monkeypatch, observed_disable)
    monkeypatch.setattr(swzl_cost, "extract", lambda _func: (_ for _ in ()).throw(ValueError("new IR")))

    with tvm.transform.PassContext():
        ppu_pipeline._lower_tile_op_with_swzl_cost_model(_module())

    assert observed_disable == [False]


def test_unsupported_record_keeps_existing_swzl_policy():
    assert swzl_cost.decide({"supported": False, "reason": "dtype_keep_existing"}) == {
        "choice": "keep_existing",
        "reason": "dtype_keep_existing",
    }


def test_missing_model_keeps_existing_swzl_policy():
    assert calibrated_decide(None, {"supported": True}) == {
        "choice": "keep_existing",
        "reason": "missing_model_keep_existing",
    }


@pytest.mark.parametrize("contents", [None, "{", "{}", '{"bias": 0, "rate": 1, "trees": [{"value": "bad"}]}'])
def test_model_load_failure_returns_none(tmp_path, contents):
    path = tmp_path / "model.json"
    if contents is not None:
        path.write_text(contents)

    assert swzl_model._load_model(path) is None


def test_valid_model_loads(tmp_path):
    path = tmp_path / "model.json"
    path.write_text('{"bias": 0, "rate": 1, "trees": [{"value": -1}]}')

    assert swzl_model._load_model(path) == {
        "bias": 0,
        "rate": 1,
        "trees": [{"value": -1}],
    }


def test_real_fp8_tir_selects_and_lowers_ordinary_path(monkeypatch):
    target, mod = _real_fp8_module()
    record = swzl_cost.extract(mod["main"])
    monkeypatch.setattr(
        swzl_model,
        "_MODEL",
        {"bias": 1.0, "rate": 0.0, "trees": [{"value": 0.0}]},
    )

    assert record["supported"]
    assert any(phase["reorder_words"] for phase in record["phases"])
    assert swzl_cost.decide(record)["choice"] == "ordinary"
    ops = _lowered_ops(mod, target)
    matrix_loads = [op for op in ops if op in ("tl.ptx_ldmatrix", "tl.tix_ldmatrix_swzl")]
    assert matrix_loads
    assert set(matrix_loads) == {"tl.ptx_ldmatrix"}


def test_real_mixed_dtype_tir_keeps_existing_swzl_path(monkeypatch):
    target, mod = _real_fp8_module(with_bf16_fragment_copy=True)
    record = swzl_cost.extract(mod["main"])
    monkeypatch.setattr(
        swzl_model,
        "_MODEL",
        {"bias": 1.0, "rate": 0.0, "trees": [{"value": 0.0}]},
    )

    assert record == {
        "supported": False,
        "reason": "mixed_fragment_dtype_keep_existing",
    }
    assert swzl_cost.decide(record) == {
        "choice": "keep_existing",
        "reason": "mixed_fragment_dtype_keep_existing",
    }
    ops = _lowered_ops(mod, target)
    matrix_loads = [op for op in ops if op in ("tl.ptx_ldmatrix", "tl.tix_ldmatrix_swzl")]
    assert matrix_loads
    assert set(matrix_loads) == {"tl.tix_ldmatrix_swzl"}


@pytest.mark.parametrize("tree_value,choice", [(1.0, "ordinary"), (0.0, "swzl"), (-1.0, "swzl")])
def test_calibrated_score_sign_has_one_clear_path_mapping(tree_value, choice):
    model = {"bias": 0.0, "rate": 1.0, "trees": [{"value": tree_value}]}
    record = {"supported": True, "phases": [{"reorder_words": 1}]}

    result = calibrated_decide(model, record, feature_fn=lambda _record: {})

    assert result["choice"] == choice
    assert result["net_log_risk"] == tree_value


def test_no_rebinding_cost_keeps_existing_swzl_policy():
    record = {
        "supported": True,
        "phases": [{"reorder_words": 0}],
    }

    assert swzl_cost.decide(record) == {
        "choice": "keep_existing",
        "reason": "no_rebinding_cost_keep_existing",
    }


@pytest.mark.parametrize("audit_name", ["live_range_audit", "address_dag_audit", "launch_audit"])
def test_unknown_analysis_keeps_existing_swzl_policy(audit_name):
    record = {
        "supported": True,
        "phases": [{"reorder_words": 1}],
        audit_name: {"known": False, "reason": "unsupported_test_shape"},
    }

    assert swzl_cost.decide(record) == {
        "choice": "keep_existing",
        "reason": f"{audit_name}_unknown_keep_existing",
    }


def test_repeated_semantic_phases_share_unique_physical_gemm_identity():
    first = ("a0", "b0", "c0", 256, 32, 256, False, True, ())
    second = ("a1", "b1", "c1", 256, 256, 32, False, True, ())
    phases = [
        {"rows": key[3], "columns": key[4], "inner": key[5], "transpose_a": key[6], "transpose_b": key[7]}
        for key in (first, first, second, second)
    ]

    assert _match_phase_keys(phases, [first, second]) == [first, first, second, second]


def test_repeated_semantic_phase_rejects_ambiguous_physical_identity():
    first = ("a0", "b0", "c0", 256, 32, 256, False, True, ())
    duplicate_geometry = ("a1", "b1", "c1", 256, 32, 256, False, True, ())
    phase = {"rows": 256, "columns": 32, "inner": 256, "transpose_a": False, "transpose_b": True}

    with pytest.raises(ValueError, match="ambiguous_phase_identity:2"):
        _match_phase_keys([phase, phase, phase], [first, duplicate_geometry])


def _packet_reuse_record(packets_per_warp, reorder_operands, reuse):
    binding = 4 * packets_per_warp
    return {
        "aiu": 0,
        "launch_known": 1,
        "launch_blocks": 32,
        "phases": [
            {
                "reorder_words": binding,
                "static_trip_count": 0,
                "feeds_gemm": 0,
                "warps": 4,
                "stages": 0,
                "copy_guard": 1,
                "copy_min_width": 16,
                "copy_lane_guard": 0,
                "other_live_words": 0,
                "address_dag_working_set": 30 if reorder_operands == 1 else 50,
                "address_dag_expressions": packets_per_warp + (reuse if reorder_operands == 2 else 0),
                "swzl_binding_peak_words": binding,
                "copy_value_words": binding,
                "copy_iterations": packets_per_warp,
                "mma_atoms": binding * reuse,
                "acc_words": 2 * binding * reuse,
                "live_words": 2 * binding * reuse,
                "matrix_operand_words": binding + 4 * reuse,
                "ordinary_address_operands": reorder_operands,
            }
        ],
    }


def _hard_rule_phase(**updates):
    phase = {
        "reorder_words": 32,
        "static_trip_count": 0,
        "feeds_gemm": 0,
        "warps": 2,
        "stages": 0,
        "copy_guard": 0,
        "copy_min_width": 16,
        "copy_lane_guard": 0,
        "other_live_words": 0,
        "live_through_buffers": 0,
        "address_dag_working_set": 32,
        "address_dag_expressions": 8,
        "swzl_binding_peak_words": 32,
        "copy_value_words": 32,
        "copy_iterations": 8,
        "mma_atoms": 32,
        "live_words": 32,
        "matrix_operand_words": 32,
        "ordinary_address_operands": 1,
    }
    phase.update(updates)
    return phase


def _hard_rule_cases():
    scalar = {
        "aiu": 0,
        "phases": [
            _hard_rule_phase(
                reorder_words=8,
                warps=2,
                stages=1,
                copy_guard=1,
                copy_min_width=8,
                copy_lane_guard=1,
                live_through_buffers=3,
                other_live_words=8,
                address_dag_working_set=8,
                swzl_binding_peak_words=8,
                copy_value_words=16,
                copy_iterations=8,
                mma_atoms=4,
            )
        ],
    }
    mixed = {
        "aiu": 0,
        "phases": [
            _hard_rule_phase(feeds_gemm=0, stages=1, copy_guard=1, swzl_binding_peak_words=16),
            _hard_rule_phase(feeds_gemm=1, stages=1, copy_guard=1, swzl_binding_peak_words=16),
        ],
    }
    allocator = {
        "aiu": 0,
        "launch_known": 1,
        "launch_blocks": 52,
        "phases": [
            _hard_rule_phase(
                reorder_words=60,
                warps=4,
                copy_guard=1,
                swzl_binding_peak_words=60,
                copy_value_words=1,
                copy_iterations=1,
                mma_atoms=60,
                live_words=128,
                matrix_operand_words=60,
                address_dag_working_set=20,
            )
        ],
    }
    repeated = {
        "aiu": 0,
        "phases": [
            _hard_rule_phase(
                reorder_words=20,
                warps=1,
                swzl_binding_peak_words=20,
                address_dag_working_set=24,
            ),
            _hard_rule_phase(
                reorder_words=20,
                warps=1,
                swzl_binding_peak_words=20,
                address_dag_working_set=24,
            ),
        ],
    }
    compact = {
        "aiu": 0,
        "phases": [
            _hard_rule_phase(
                static_trip_count=1,
                warps=8,
                copy_min_width=8,
                swzl_binding_peak_words=32,
                mma_atoms=32,
            )
        ],
    }
    return [
        (scalar, "swzl", "fp8_scalar_nonfeed_reuse_frontier"),
        (mixed, "swzl", "fp8_mixed_feed_binding_balance_frontier"),
        (_packet_reuse_record(20, 1, 1), "ordinary", "fp8_dynamic_four_warp_packet_reuse_cost"),
        (allocator, "swzl", "fp8_dynamic_four_warp_allocator_cliff_frontier"),
        (repeated, "swzl", "fp8_repeated_phase_full_fragment_reuse_frontier"),
        (compact, "swzl", "fp8_static_one_step_compact_fragment_frontier"),
    ]


@pytest.mark.parametrize("record,choice,reason", _hard_rule_cases())
def test_each_hard_rule_has_an_explicit_choice_and_reason(record, choice, reason):
    assert swzl_model._hard_rule_decide(record) == {"choice": choice, "reason": reason}


@pytest.mark.parametrize(
    "packets_per_warp,reorder_operands,reuse,choice",
    [
        # Representative measured scheduler buckets.
        pytest.param(20, 1, 1, "ordinary", id="p20-single-r1"),
        pytest.param(20, 2, 2, "swzl", id="p20-dual-r2"),
        pytest.param(29, 1, 4, "swzl", id="p29-single-r4"),
        pytest.param(29, 2, 1, "ordinary", id="p29-dual-r1"),
        pytest.param(30, 2, 8, "swzl", id="p30-dual-r8"),
        pytest.param(31, 1, 4, "swzl", id="p31-single-r4"),
        # Bank-placement transition before the high-work frontier.
        pytest.param(48, 1, 4, "swzl", id="p48-single-r4"),
        pytest.param(48, 2, 4, "ordinary", id="p48-dual-r4"),
        pytest.param(49, 2, 2, "swzl", id="p49-dual-r2"),
        # High-work frontier and independent holdout.
        pytest.param(50, 1, 1, "swzl", id="p50-single-r1"),
        pytest.param(50, 1, 8, "ordinary", id="p50-single-r8"),
        pytest.param(50, 2, 8, "swzl", id="p50-dual-r8"),
        pytest.param(52, 1, 4, "ordinary", id="p52-single-r4"),
        pytest.param(52, 2, 4, "swzl", id="p52-dual-r4"),
        # The p54/p55 blind counterexamples that define the final pressure rule.
        pytest.param(54, 1, 3, "swzl", id="p54-single-r3"),
        pytest.param(54, 1, 5, "ordinary", id="p54-single-r5"),
        pytest.param(54, 1, 6, "ordinary", id="p54-single-r6"),
        pytest.param(54, 1, 7, "ordinary", id="p54-single-r7"),
        pytest.param(54, 2, 3, "swzl", id="p54-dual-r3"),
        pytest.param(54, 2, 7, "swzl", id="p54-dual-r7"),
        pytest.param(55, 1, 3, "swzl", id="p55-single-r3"),
        pytest.param(55, 1, 5, "ordinary", id="p55-single-r5"),
        pytest.param(55, 1, 6, "ordinary", id="p55-single-r6"),
        pytest.param(55, 1, 7, "ordinary", id="p55-single-r7"),
        pytest.param(55, 2, 3, "swzl", id="p55-dual-r3"),
        pytest.param(55, 2, 7, "swzl", id="p55-dual-r7"),
    ],
)
def test_packet_reuse_cost_boundaries(packets_per_warp, reorder_operands, reuse, choice):
    record = _packet_reuse_record(packets_per_warp, reorder_operands, reuse)
    assert dynamic_four_warp_packet_reuse_choice(record) == choice


def test_unmeasured_scheduler_bucket_does_not_extrapolate():
    record = _packet_reuse_record(20, 1, 9)
    assert dynamic_four_warp_packet_reuse_choice(record) is None
