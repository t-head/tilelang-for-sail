"""Focused tests for the pre-LowerTileOp FP8 SWZL cost decision."""

import pytest

import tilelang
import tilelang.ppu.pipeline as ppu_pipeline
import tilelang.ppu.swzl_cost as swzl_cost
from tilelang import tvm
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
