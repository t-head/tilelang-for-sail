"""Cost model for PPU 1.5 FP8 ldmatrix SWZL selection.

Inference consumes only immutable facts from the pre-LowerTileOp PrimFunc.
The bundled tree ensemble was calibrated offline; runtime trip counts,
operator names, generated code, ISA, and benchmark measurements are not
selector inputs.
"""

import json
from pathlib import Path

_MODEL = json.loads(Path(__file__).with_name("model.json").read_text())


def _tree_value(tree, values):
    while "feature" in tree:
        value = values.get(tree["feature"], 0)
        tree = tree["left"] if value <= tree["threshold"] else tree["right"]
    return tree["value"]


def features(record):
    out = {"aiu": record["aiu"]}
    for phase in record["phases"]:
        role = "feed" if phase["feeds_gemm"] else "sink"
        repeat = phase["multiplicity"]
        atoms = max(1, phase["mma_atoms"])
        copies = phase.get("copy_iterations", 0)
        payload = phase["matrix_operand_words"]
        binding = phase["swzl_binding_peak_words"]
        address_peak = 4 * phase["ordinary_address_peak"]
        address_work = phase["ordinary_address_work"]
        touch = phase["live_through_touch_words"]
        density = phase["live_through_touch_density"]
        common = 24 + phase["live_words"] + payload
        entries = dict(
            mma_work=phase["mma_atoms"],
            live_words=phase["live_words"] / 256,
            other_live_words=phase["other_live_words"] / 256,
            operand_words=payload / 256,
            binding_peak=binding / 256,
            binding_work=phase["reorder_words"] / atoms,
            address_peak=address_peak / 256,
            address_work=address_work / atoms,
            address_work_per_slice=phase["ordinary_address_work_per_slice"],
            address_per_binding=address_work / max(1, binding),
            copy_work=copies / atoms,
            copy_expansion=16 / max(1, phase.get("copy_min_width", 16)),
            copy_guard=phase.get("copy_lane_guard", 0),
            overlap=int(phase["stages"] > 0),
            pages=max(0, phase["pages"] - 1),
            warps=phase["warps"],
            static_trip=phase["static_trip_count"],
            unknown_trip=int(phase["static_trip_count"] == 0),
            live_through_buffers=phase["live_through_buffers"],
            live_through_touch=touch / atoms,
            live_through_density=density,
            live_through_reads=phase["live_through_read_words"] / atoms,
            active_reads=phase["active_non_gemm_read_words"] / atoms,
            active_writes=phase["active_non_gemm_write_words"] / atoms,
            aiu_packets=phase.get("copy_aiu_packets", 0),
            aiu_bytes=phase.get("copy_aiu_bytes", 0) / max(1, atoms),
        )
        for budget in (192, 256):
            spill_swzl = max(0, common + binding - budget)
            spill_ordinary = max(0, common + address_peak - budget)
            delta = spill_swzl - spill_ordinary
            entries[f"spill_delta_{budget}"] = delta / budget
            entries[f"live_spill_work_{budget}"] = delta * density / atoms
            entries[f"touch_spill_work_{budget}"] = delta * touch / max(1, phase["other_live_words"]) / atoms
            entries[f"swzl_overflow_{budget}"] = spill_swzl / budget
            entries[f"ordinary_overflow_{budget}"] = spill_ordinary / budget
        for name, value in entries.items():
            key = role + "_" + name
            out[key] = out.get(key, 0) + repeat * value
    return out


def calibrated_decide(model, record, feature_fn=features):
    if not record.get("supported") or model is None:
        reason = record.get("reason", "capture_only") + "_keep_existing"
        return {"choice": "swzl", "reason": reason}
    if not any(phase["reorder_words"] for phase in record["phases"]):
        return {"choice": "swzl", "reason": "no_rebinding_cost_keep_existing"}
    values = feature_fn(record)
    score = model["bias"] + model["rate"] * sum(_tree_value(tree, values) for tree in model["trees"])
    return {
        "choice": "ordinary" if score > 0 else "swzl",
        "reason": "fp8_net_fixed_regret",
        "net_log_risk": score,
    }


REGISTER_WINDOW_WORDS = 256
REGISTER_BANK_WORDS = 128
FIXED_CONTROL_WORDS = 24
ALLOCATOR_RESERVE_WORDS = 24
ALLOCATOR_QUANTUM_WORDS = 8
BINDING_BANK_EDGE_WORDS = 40
ORDINARY_PACKET_BUCKET = 16
PPU15_COMPUTE_UNITS = 39
WIDE_PACKET_EXTRAPOLATION_START = 30


def _reordered_phases(record):
    return [phase for phase in record["phases"] if phase["reorder_words"]]


def serial_copy_retention_frontier(record):
    """Return true only for a strict full-retention path separation."""
    for phase in _reordered_phases(record):
        if phase["stages"] != 0 or phase.get("copy_min_width", 16) >= 16:
            continue
        demand = phase.get("copy_value_words", 0)
        if demand <= 0:
            continue
        frame = 24 + phase["live_words"] + phase["matrix_operand_words"]
        swzl_free = max(0, REGISTER_WINDOW_WORDS - frame - phase["swzl_binding_peak_words"])
        ordinary_free = max(0, REGISTER_WINDOW_WORDS - frame - phase["address_dag_working_set"])
        if demand <= swzl_free and demand > ordinary_free:
            return True
    return False


def pipelined_scalar_address_frontier(record):
    """Return true when a feed pipeline can hide ordinary scalar addresses.

    A lane-guarded scalar copy is not sufficient by itself.  The ordinary
    address DAG is hidden only when the reordered fragment feeds the next GEMM
    and the copy loop provides at least one independent address iteration per
    MMA atom.  Non-feed high-reuse dataflows have no following GEMM with which
    to overlap this work and are evaluated by the complementary frontier
    below.
    """
    phases = _reordered_phases(record)
    if record["aiu"] or len(phases) != 1:
        return False
    phase = phases[0]
    return bool(
        phase["stages"] > 0
        and phase["feeds_gemm"]
        and phase.get("copy_lane_guard", 0)
        and phase.get("copy_min_width", 16) < 16
        and phase["swzl_binding_peak_words"] >= phase["address_dag_working_set"]
        and phase.get("copy_iterations", 0) >= phase["mma_atoms"]
    )


def scalar_nonfeed_reuse_frontier(record):
    """SWZL wins when repeated scalar addresses cannot hide behind a GEMM.

    This is the complementary one-phase pipeline topology.  The reordered
    fragment does not feed another GEMM, remains live through at least three
    consumers, and has little unrelated live state.  In that regime a guarded
    scalar copy whose address iterations exceed two complete MMA passes pays
    the ordinary address DAG repeatedly, while SWZL retains only one bounded
    fragment binding.  The one allocator-quantum balance allows a small
    binding/DAG tie without admitting a binding-dominated path.
    """
    phases = _reordered_phases(record)
    if record["aiu"] or len(phases) != 1:
        return False
    phase = phases[0]
    return bool(
        not phase["static_trip_count"]
        and not phase["feeds_gemm"]
        and phase["warps"] > 1
        and phase["stages"] == 1
        and phase.get("copy_guard", 0)
        and phase.get("copy_lane_guard", 0)
        and phase.get("copy_min_width", 16) < 16
        and phase.get("live_through_buffers", 0) >= 3
        and phase.get("other_live_words", 0) <= ALLOCATOR_QUANTUM_WORDS
        and phase.get("copy_iterations", 0) >= 2 * phase["mma_atoms"]
        and phase.get("copy_value_words", 0) > phase["swzl_binding_peak_words"]
        and phase["swzl_binding_peak_words"] <= phase["address_dag_working_set"] + ALLOCATOR_QUANTUM_WORDS
    )


def single_warp_pipeline_binding_dominance(record):
    """Ordinary dominates when one warp cannot hide a large SWZL binding."""
    phases = _reordered_phases(record)
    return bool(
        phases
        and all(
            not phase["feeds_gemm"]
            and phase["warps"] == 1
            and phase["stages"] > 0
            and phase.get("copy_min_width", 16) == 16
            and phase["swzl_binding_peak_words"] - phase["address_dag_working_set"] >= 8
            for phase in phases
        )
    )


def two_warp_moderate_binding_dominance(record):
    """Ordinary dominates before the second high-pressure binding cliff."""
    phases = _reordered_phases(record)
    return bool(
        len(phases) >= 2
        and all(
            not phase["feeds_gemm"]
            and phase["warps"] == 2
            and phase["stages"] <= 1
            and phase.get("copy_min_width", 16) == 16
            and 0 < phase["swzl_binding_peak_words"] - phase["address_dag_working_set"] <= 24
            for phase in phases
        )
    )


def single_warp_feed_balance_frontier(record):
    """SWZL dominates one feed phase when both path working sets are tied."""
    phases = _reordered_phases(record)
    if len(phases) != 1:
        return False
    phase = phases[0]
    return bool(
        phase["feeds_gemm"]
        and phase["warps"] == 1
        and phase["stages"] <= 1
        and phase.get("copy_min_width", 16) == 16
        and abs(phase["swzl_binding_peak_words"] - phase["address_dag_working_set"]) <= 2
    )


def mixed_feed_binding_balance_frontier(record):
    """SWZL wins when both sides of a mixed-feed pipeline reduce pressure.

    A two-GEMM pipeline keeps one phase's fragment alive to feed the second
    GEMM.  Selecting SWZL for the function is safe only when the SWZL binding
    working set is no larger than the ordinary address-DAG working set in
    *both* phases.  Checking the phases jointly prevents a profitable QK load
    from hiding an opposing PV load (or vice versa).  The guarded, one-stage,
    two-warp requirements keep this frontier on the measured scheduling
    topology; other pipelines remain with the calibrated fallback.
    """
    phases = _reordered_phases(record)
    return bool(
        not record["aiu"]
        and len(phases) == 2
        and sorted(bool(phase["feeds_gemm"]) for phase in phases) == [False, True]
        and all(
            not phase["static_trip_count"]
            and phase["warps"] == 2
            and phase["stages"] == 1
            and phase.get("copy_guard", 0)
            and phase.get("copy_min_width", 16) == 16
            and not phase.get("copy_lane_guard", 0)
            and phase.get("copy_value_words", 0) > 0
            and phase["swzl_binding_peak_words"] <= phase["address_dag_working_set"]
            for phase in phases
        )
    )


def single_phase_high_reuse_frontier(record):
    """SWZL wins one high-reuse phase only with a binding advantage.

    Three or more live-through fragments make the same matrix load feed a
    dense register-resident dataflow after GEMM.  That reuse alone is not
    sufficient: SWZL is admitted only when its binding working set is
    strictly smaller than the ordinary address DAG.  This joint test rejects
    both ordinary GEMMs with no live-through dataflow and the high-reuse
    control whose SWZL binding is larger than the address DAG.
    """
    phases = _reordered_phases(record)
    if record["aiu"] or len(phases) != 1:
        return False
    phase = phases[0]
    return bool(
        not phase["static_trip_count"]
        and not phase["feeds_gemm"]
        and phase["warps"] == 2
        and phase["stages"] == 1
        and phase.get("copy_guard", 0)
        and phase.get("copy_min_width", 16) == 16
        and not phase.get("copy_lane_guard", 0)
        and phase.get("live_through_buffers", 0) >= 3
        and phase["swzl_binding_peak_words"] < phase["address_dag_working_set"]
    )


def dynamic_four_warp_packet_reuse_choice(record):
    """Choose a path in measured packet/reuse scheduling windows.

    Once the accumulator spills, neither accumulator size nor final stack
    size is a monotonic proxy for path cost.  The stable pre-lowering axes are
    the number of operands participating in the reorder, the number of
    packets in the copied binding, and the number of integral row passes that
    reuse that complete binding.  The backend changes its unroll/spill
    schedule at boundaries on these axes.

    This rule is deliberately conservative: it selects SWZL only in windows
    where every measured runtime length has a material win.  Near-ties use
    ordinary.  Adjacent packet buckets return to the older launch model when
    their reuse window has not been measured, instead of extrapolating a
    packet trend through a backend scheduling boundary.  Matrix dimensions,
    operator names, generated code, ISA, and runtime trip counts are not
    inputs.
    """
    phases = _reordered_phases(record)
    if record["aiu"] or len(phases) != 1:
        return None
    phase = phases[0]
    binding = phase["swzl_binding_peak_words"]
    if not binding or not phase["warps"]:
        return None
    packets_per_warp = binding / phase["warps"]
    one_binding_copy = bool(phase.get("copy_value_words", 0) == binding and phase.get("copy_iterations", 0) * phase["warps"] == binding)
    integral_mma_reuse = bool(phase["mma_atoms"] >= binding and phase["mma_atoms"] % binding == 0)
    if not (
        not phase["static_trip_count"]
        and not phase["feeds_gemm"]
        and phase["warps"] == 4
        and phase["stages"] == 0
        and phase.get("copy_guard", 0)
        and phase.get("copy_min_width", 16) == 16
        and not phase.get("copy_lane_guard", 0)
        and phase.get("other_live_words", 0) == 0
        and phase["address_dag_working_set"] > 0
        and one_binding_copy
        and integral_mma_reuse
        and record.get("launch_known", 0)
        and record.get("launch_blocks", 0) > 0
    ):
        return None

    reorder_operands = phase.get("ordinary_address_operands", 0)
    reuse = int(phase["mma_atoms"] / binding)
    measured_wide_windows = {
        1: {
            ORDINARY_PACKET_BUCKET + 4: ["ordinary", "ordinary", "ordinary", "swzl", "swzl", "ordinary", "swzl", "swzl"],
            ORDINARY_PACKET_BUCKET + 5: ["ordinary", "ordinary", "ordinary", "ordinary", "swzl", "ordinary", "swzl", "swzl"],
            ORDINARY_PACKET_BUCKET + 6: ["ordinary", "swzl", "ordinary", "swzl", "swzl", "swzl", "swzl", "swzl"],
            ORDINARY_PACKET_BUCKET + 7: ["ordinary", "ordinary", "ordinary", "swzl", "ordinary", "swzl", "swzl", "swzl"],
            ORDINARY_PACKET_BUCKET + 8: ["ordinary", "ordinary", "ordinary", "swzl", "ordinary", "swzl", "swzl", "ordinary"],
            ORDINARY_PACKET_BUCKET + 9: ["ordinary", "ordinary", "ordinary", "swzl", "swzl", "swzl", "ordinary", "ordinary"],
            ORDINARY_PACKET_BUCKET + 10: ["ordinary", "ordinary", "ordinary", "ordinary", "swzl", "ordinary", "ordinary", "ordinary"],
            ORDINARY_PACKET_BUCKET + 11: ["ordinary", "ordinary", "ordinary", "swzl", "swzl", "ordinary", "swzl", "swzl"],
            ORDINARY_PACKET_BUCKET + 12: ["ordinary", "ordinary", "ordinary", "ordinary", "ordinary", "swzl", "swzl", "swzl"],
            ORDINARY_PACKET_BUCKET + 13: ["ordinary", "ordinary", "ordinary", "swzl", "ordinary", "swzl", "ordinary", "ordinary"],
        },
        2: {
            ORDINARY_PACKET_BUCKET + 4: ["ordinary", "swzl", "ordinary", "swzl", "swzl", "ordinary", "swzl", "swzl"],
            ORDINARY_PACKET_BUCKET + 5: ["ordinary", "swzl", "ordinary", "swzl", "ordinary", "ordinary", "swzl", "ordinary"],
            ORDINARY_PACKET_BUCKET + 6: ["ordinary", "ordinary", "ordinary", "ordinary", "swzl", "ordinary", "swzl", "swzl"],
            ORDINARY_PACKET_BUCKET + 7: ["ordinary", "swzl", "swzl", "ordinary", "ordinary", "swzl", "swzl", "swzl"],
            ORDINARY_PACKET_BUCKET + 8: ["ordinary", "swzl", "ordinary", "ordinary", "ordinary", "ordinary", "swzl", "ordinary"],
            ORDINARY_PACKET_BUCKET + 9: ["ordinary", "ordinary", "swzl", "swzl", "swzl", "ordinary", "ordinary", "swzl"],
            ORDINARY_PACKET_BUCKET + 10: ["ordinary", "swzl", "ordinary", "ordinary", "swzl", "ordinary", "swzl", "ordinary"],
            ORDINARY_PACKET_BUCKET + 11: ["swzl", "swzl", "ordinary", "ordinary", "swzl", "ordinary", "swzl", "ordinary"],
            ORDINARY_PACKET_BUCKET + 12: ["ordinary", "ordinary", "swzl", "swzl", "swzl", "swzl", "ordinary", "swzl"],
            ORDINARY_PACKET_BUCKET + 13: ["ordinary", "swzl", "ordinary", "swzl", "swzl", "swzl", "ordinary", "ordinary"],
        },
    }
    measured_choices = measured_wide_windows.get(reorder_operands, {}).get(packets_per_warp)
    if measured_choices is not None and 1 <= reuse <= len(measured_choices):
        return measured_choices[reuse - 1]

    # Beyond the measured packet table, model contiguous binding placement
    # rather than extending a packet trend.  The SWZL instruction derives its
    # tensor position from a consecutive destination-register group.  Its
    # cost therefore depends on both the 256-word allocation window and the
    # position of the group within a 128-word register bank.  Total pressure
    # alone misses this fragmentation effect.  Four-pass reuse is admitted
    # near a bank edge without crossing an allocation window.  Eight-pass
    # reuse can tolerate the middle of an established spill window, but only
    # through the tenth window; a one-window crossing is admitted solely when
    # a heavy dual-operand address stream places the new group at the start of
    # a bank.  A two-pass dual-operand path may also cross one window once
    # its larger address stream pays that placement cost.  Reuse counts not
    # covered by these mechanisms stay with the existing fallback.  All
    # quantities are available before LowerTileOp.
    if packets_per_warp >= WIDE_PACKET_EXTRAPOLATION_START:
        common_pressure = FIXED_CONTROL_WORDS + phase["live_words"] + phase["matrix_operand_words"]
        swzl_pressure = common_pressure + binding
        address_pressure = phase["address_dag_working_set"]
        ordinary_pressure = common_pressure + address_pressure
        swzl_windows = (swzl_pressure + REGISTER_WINDOW_WORDS - 1) // REGISTER_WINDOW_WORDS
        ordinary_windows = (ordinary_pressure + REGISTER_WINDOW_WORDS - 1) // REGISTER_WINDOW_WORDS
        same_allocator_window = swzl_windows == ordinary_windows
        swzl_window_residual = swzl_pressure % REGISTER_WINDOW_WORDS
        swzl_bank_residual = swzl_pressure % REGISTER_BANK_WORDS
        bank_edge_distance = min(
            swzl_bank_residual,
            REGISTER_BANK_WORDS - swzl_bank_residual,
        )
        # One allocator quantum is the tolerated packing slack between the
        # saved address state and the consecutive SWZL register group.
        address_pays_binding = binding <= 3 * address_pressure + ALLOCATOR_QUANTUM_WORDS
        address_expressions = phase.get("address_dag_expressions", 0)
        # Across an allocation-window boundary, peak address temporaries no
        # longer describe the saved work: the ordinary path rebuilds one
        # four-word address bundle for every static address expression.  Use
        # that total expression work only for the narrowly bounded crossing
        # cases below; same-window decisions continue to use peak pressure.
        crossing_address_work_covers_binding = bool(binding <= 4 * address_expressions)
        # At fifty packets per warp the ordinary path has at least fifty
        # independent four-word address bundles to reconstruct for one
        # complete binding.  Once that static address work covers the whole
        # consecutive binding, dual-address phases can retain SWZL without
        # material regret.  A single-address phase has less peak address state
        # to save.  Admit its complete frame through eight register windows;
        # beyond that point, only the canonical binary-reuse schedule can use
        # a binding placed within three allocator quanta of a register-bank
        # edge.  Intermediate reuse counts at larger frames are measured
        # counterexamples even at the bank edge.  Keep width and address work
        # in the outer test: width alone must not admit a cheaply shared
        # address DAG, and expression count alone must not extrapolate smaller
        # bindings through their measured allocator counterexamples.
        high_work_wide_binding = bool(packets_per_warp >= 50 and crossing_address_work_covers_binding)
        if high_work_wide_binding:
            if reorder_operands == 2:
                return "swzl"
            if reorder_operands == 1:
                low_pressure_single = swzl_windows <= 8
                canonical_binary_reuse = bool(reuse > 0 and (reuse & (reuse - 1)) == 0)
                bank_aligned_binary_reuse = bool(canonical_binary_reuse and bank_edge_distance <= 3 * ALLOCATOR_QUANTUM_WORDS)
                return "swzl" if low_pressure_single or bank_aligned_binary_reuse else "ordinary"
        if reuse <= 1:
            return "ordinary"
        if reuse == 2:
            single_mid_bank_landing = bool(
                reorder_operands == 1
                and address_expressions >= WIDE_PACKET_EXTRAPOLATION_START
                and same_allocator_window
                and swzl_window_residual == REGISTER_BANK_WORDS
            )
            # A dual-operand address stream starts paying for the consecutive
            # SWZL binding after forty expressions.  Two allocator quanta of
            # packing slack are tolerated here because the binding is reused
            # twice.  Inside one allocation window, measured profitable
            # placements are the bank's early binding slot and its final four
            # quanta; the middle slot remains an ordinary-path counterexample.
            dual_address_amortizes_binding = bool(binding <= 3 * address_pressure + 2 * ALLOCATOR_QUANTUM_WORDS)
            dual_profitable_bank_slot = bool(
                5 * ALLOCATOR_QUANTUM_WORDS <= swzl_bank_residual <= 7 * ALLOCATOR_QUANTUM_WORDS
                or 12 * ALLOCATOR_QUANTUM_WORDS <= swzl_bank_residual < REGISTER_BANK_WORDS
            )
            dual_bank_phase = bool(
                reorder_operands == 2
                and address_expressions >= 40
                and dual_address_amortizes_binding
                and same_allocator_window
                and dual_profitable_bank_slot
            )
            high_work_dual_bank_phase = bool(
                reorder_operands == 2
                and address_expressions >= 51
                and crossing_address_work_covers_binding
                and same_allocator_window
                and 5 * ALLOCATOR_QUANTUM_WORDS <= swzl_bank_residual <= 7 * ALLOCATOR_QUANTUM_WORDS
            )
            dual_bank_origin_crossing = bool(
                reorder_operands == 2
                and address_expressions >= 44
                and crossing_address_work_covers_binding
                and swzl_windows == ordinary_windows + 1
            )
            single_bank_tail_crossing = bool(
                reorder_operands == 1
                and address_expressions >= 46
                and crossing_address_work_covers_binding
                and swzl_windows == ordinary_windows + 1
                and swzl_bank_residual >= REGISTER_BANK_WORDS - 4 * ALLOCATOR_QUANTUM_WORDS
            )
            return (
                "swzl"
                if (
                    single_mid_bank_landing
                    or dual_bank_phase
                    or high_work_dual_bank_phase
                    or dual_bank_origin_crossing
                    or single_bank_tail_crossing
                )
                else "ordinary"
            )
        if reuse == 4:
            near_window_origin = swzl_window_residual <= BINDING_BANK_EDGE_WORDS
            heavy_dual_bank_edge = bool(
                reorder_operands == 2
                and address_expressions >= 38
                and address_pays_binding
                and bank_edge_distance <= BINDING_BANK_EDGE_WORDS
            )
            heavy_dual_bank_origin_crossing = bool(
                reorder_operands == 2
                and address_expressions >= 38
                and crossing_address_work_covers_binding
                and swzl_windows == ordinary_windows + 1
                and swzl_bank_residual <= BINDING_BANK_EDGE_WORDS + ALLOCATOR_QUANTUM_WORDS
            )
            heavy_single_bank_origin_crossing = bool(
                reorder_operands == 1
                and address_expressions >= 44
                and crossing_address_work_covers_binding
                and swzl_windows == ordinary_windows + 1
                and swzl_bank_residual <= REGISTER_BANK_WORDS - BINDING_BANK_EDGE_WORDS
            )
            high_work_single_same_window = bool(
                reorder_operands == 1 and address_expressions >= 48 and crossing_address_work_covers_binding and same_allocator_window
            )
            return (
                "swzl"
                if (same_allocator_window and (near_window_origin or heavy_dual_bank_edge))
                or heavy_dual_bank_origin_crossing
                or heavy_single_bank_origin_crossing
                or high_work_single_same_window
                else "ordinary"
            )
        if reuse == 8:
            stable_same_window = bool(same_allocator_window and swzl_windows <= 10 and 100 <= swzl_window_residual <= 200)
            bounded_dual_bank_origin = bool(
                reorder_operands == 2
                and address_expressions >= 43
                and address_pays_binding
                and swzl_windows == ordinary_windows + 1
                and 2 * ALLOCATOR_QUANTUM_WORDS <= swzl_bank_residual <= BINDING_BANK_EDGE_WORDS
            )
            high_work_dual_crossing = bool(
                reorder_operands == 2
                and address_expressions >= 54
                and crossing_address_work_covers_binding
                and swzl_windows == ordinary_windows + 1
            )
            high_work_single_same_window = bool(
                reorder_operands == 1 and address_expressions >= 48 and crossing_address_work_covers_binding and same_allocator_window
            )
            return (
                "swzl"
                if (
                    (address_pays_binding and stable_same_window)
                    or bounded_dual_bank_origin
                    or high_work_dual_crossing
                    or high_work_single_same_window
                )
                else "ordinary"
            )
        return None

    if reorder_operands == 1:
        if packets_per_warp == ORDINARY_PACKET_BUCKET and 2 <= reuse <= 8:
            return "ordinary" if reuse in (2, 3, 8) else "swzl"
        if packets_per_warp == ORDINARY_PACKET_BUCKET + 3 and 1 <= reuse <= 8:
            return "ordinary" if reuse in (1, 3, 6) else "swzl"
        if packets_per_warp == ORDINARY_PACKET_BUCKET + 1:
            return "swzl" if reuse == 1 or 4 <= reuse <= 5 else "ordinary"
        if packets_per_warp == ORDINARY_PACKET_BUCKET + 2:
            return "swzl" if reuse in (2, 4) else "ordinary"
        return None
    if reorder_operands != 2:
        return None

    if packets_per_warp == ORDINARY_PACKET_BUCKET and 2 <= reuse <= 8:
        return "swzl" if reuse in (2, 8) else "ordinary"
    if packets_per_warp == ORDINARY_PACKET_BUCKET + 3 and 1 <= reuse <= 8:
        return "ordinary" if reuse in (1, 3, 4, 5) else "swzl"
    if packets_per_warp == ORDINARY_PACKET_BUCKET + 1:
        return "swzl" if reuse <= 3 or reuse == 5 else "ordinary"
    if packets_per_warp == ORDINARY_PACKET_BUCKET + 2:
        return "swzl" if 3 <= reuse <= 4 else "ordinary"
    return None


def dynamic_four_warp_allocator_cliff_frontier(record):
    """SWZL wins when ordinary address state exhausts allocator headroom.

    This estimates the ordinary path's simultaneously live frame before
    LowerTileOp: fixed loop/control state, the accumulator fragment, matrix
    operands, and the address-DAG working set.  The final allocator headroom
    is deliberately reserved instead of treated as usable fragment space.
    Post-lowering vreg counts are diagnostic evidence only and are not read by
    this rule.  The ordinary frame must stay within one register window plus
    two allocation quanta: at the next quantum the measured ordinary path
    changes allocation regime and becomes cheaper than SWZL.  The ordinary
    packet loop changes
    allocation behavior around its 16-packet unroll bucket: the bucket-minus-
    one case needs a partial second launch wave, the exact power-of-two bucket
    keeps ordinary efficient until a fourth wave starts, and the first packet
    beyond the bucket immediately exposes the allocator cliff.  At that first
    overflow, row-wise MMA reuse does not enlarge the transported fragment:
    SWZL remains admissible when one complete binding is copied once and then
    reused by an integral number of MMA passes.  Counts beyond that first
    overflow enter a different measured allocation regime and are rejected.
    These conditions reject the measured partial-copy, wider-packet,
    allocator-regime, and underfilled-launch counterexamples without naming
    matrix dimensions.
    """
    phases = _reordered_phases(record)
    if record["aiu"] or len(phases) != 1:
        return False
    phase = phases[0]
    ordinary_pressure = FIXED_CONTROL_WORDS + phase["live_words"] + phase["matrix_operand_words"] + phase["address_dag_working_set"]
    packets_per_warp = phase["swzl_binding_peak_words"] / phase["warps"]
    launch_blocks = record.get("launch_blocks", 0)
    one_binding_copy = bool(
        phase.get("copy_value_words", 0) == phase["swzl_binding_peak_words"]
        and phase.get("copy_iterations", 0) * phase["warps"] == phase["swzl_binding_peak_words"]
    )
    integral_mma_reuse = bool(
        phase["mma_atoms"] >= phase["swzl_binding_peak_words"] and phase["mma_atoms"] % phase["swzl_binding_peak_words"] == 0
    )
    if packets_per_warp == ORDINARY_PACKET_BUCKET - 1:
        launch_ready = 3 * launch_blocks >= 4 * PPU15_COMPUTE_UNITS
    elif packets_per_warp == ORDINARY_PACKET_BUCKET:
        launch_ready = launch_blocks > 3 * PPU15_COMPUTE_UNITS
    elif packets_per_warp == ORDINARY_PACKET_BUCKET + 1:
        launch_ready = launch_blocks > 0
    else:
        launch_ready = False
    return bool(
        not phase["static_trip_count"]
        and not phase["feeds_gemm"]
        and phase["warps"] == 4
        and phase["stages"] == 0
        and phase.get("copy_guard", 0)
        and phase.get("copy_min_width", 16) == 16
        and not phase.get("copy_lane_guard", 0)
        and phase.get("other_live_words", 0) == 0
        and phase.get("copy_value_words", 0) > 0
        and phase["address_dag_working_set"] > 0
        and ordinary_pressure >= REGISTER_WINDOW_WORDS - ALLOCATOR_RESERVE_WORDS
        and (
            ordinary_pressure <= REGISTER_WINDOW_WORDS + 2 * ALLOCATOR_QUANTUM_WORDS
            or (
                packets_per_warp == ORDINARY_PACKET_BUCKET + 1
                and one_binding_copy
                and integral_mma_reuse
                and phase["live_words"] <= 2 * REGISTER_WINDOW_WORDS
            )
        )
        and (
            phase["mma_atoms"] == phase["swzl_binding_peak_words"]
            or (
                packets_per_warp == ORDINARY_PACKET_BUCKET + 1
                and one_binding_copy
                and integral_mma_reuse
                and phase["live_words"] <= 2 * REGISTER_WINDOW_WORDS
            )
        )
        and record.get("launch_known", 0)
        and launch_ready
    )


def dynamic_four_warp_first_overflow_accumulator_dominance(record):
    """Ordinary wins when p17 row reuse crosses two accumulator windows.

    The 17-packet overflow favors SWZL while one copied binding is reused by
    up to the live accumulator capacity.  Once the accumulator alone exceeds
    two register windows, retaining the SWZL binding competes with the fourth
    row fragment and the ordinary schedule becomes cheaper.  This is scoped
    to the p17 allocation bucket; p18 has a different measured schedule and
    remains with the calibrated fallback.
    """
    phases = _reordered_phases(record)
    if record["aiu"] or len(phases) != 1:
        return False
    phase = phases[0]
    binding = phase["swzl_binding_peak_words"]
    packets_per_warp = binding / phase["warps"]
    one_binding_copy = bool(phase.get("copy_value_words", 0) == binding and phase.get("copy_iterations", 0) * phase["warps"] == binding)
    integral_mma_reuse = bool(phase["mma_atoms"] >= binding and phase["mma_atoms"] % binding == 0)
    return bool(
        not phase["static_trip_count"]
        and not phase["feeds_gemm"]
        and phase["warps"] == 4
        and phase["stages"] == 0
        and phase.get("copy_guard", 0)
        and phase.get("copy_min_width", 16) == 16
        and not phase.get("copy_lane_guard", 0)
        and phase.get("other_live_words", 0) == 0
        and packets_per_warp == ORDINARY_PACKET_BUCKET + 1
        and one_binding_copy
        and integral_mma_reuse
        and phase["live_words"] > 2 * REGISTER_WINDOW_WORDS
        and record.get("launch_known", 0)
        and record.get("launch_blocks", 0) > 0
    )


def repeated_phase_register_frontier(record):
    """SWZL dominates repeated phases only at a strict register-window split."""
    phases = _reordered_phases(record)
    if record["aiu"] or len(phases) < 2:
        return False
    for phase in phases:
        if (
            phase["feeds_gemm"]
            or phase["warps"] > 2
            or phase["stages"] > 1
            or phase.get("copy_min_width", 16) < 16
            or phase.get("copy_lane_guard", 0)
            or phase["address_dag_total_ops"] / max(1, phase["mma_atoms"]) < 8
        ):
            return False
        common = 24 + phase["live_words"] + phase["matrix_operand_words"]
        swzl_pressure = common + phase["swzl_binding_peak_words"]
        # The ordinary path retains lane/ki DAG values while evaluating the
        # current address, hence the second DAG working-set term.
        ordinary_pressure = common + 2 * phase["address_dag_working_set"]
        if not 208 <= swzl_pressure <= 224 < ordinary_pressure:
            return False
    return True


def repeated_phase_full_fragment_reuse_frontier(record):
    """SWZL wins when a repeated one-warp phase reuses one full fragment.

    A single fragment-sized copy is the balance point where ordinary address
    generation is no longer cheap enough to hide, while SWZL has not started
    retaining multiple fragment copies.  The binding/DAG ratio narrows this
    to the allocator crossover: smaller bindings leave enough room for the
    ordinary path, and larger bindings make SWZL itself the dominant live
    range.  These are pre-lowering working-set facts, not matrix dimensions.
    """
    phases = _reordered_phases(record)
    if record["aiu"] or len(phases) < 2:
        return False
    for phase in phases:
        binding = phase["swzl_binding_peak_words"]
        address_dag = phase["address_dag_working_set"]
        if (
            phase["static_trip_count"]
            or phase["feeds_gemm"]
            or phase["warps"] != 1
            or phase["stages"] != 0
            or phase.get("copy_min_width", 16) != 16
            or phase.get("copy_lane_guard", 0)
            or phase["copy_value_words"] != phase["live_words"]
            or binding > address_dag
            or 3 * binding < 2 * address_dag
        ):
            return False
    return True


def static_one_step_compact_fragment_frontier(record):
    """SWZL wins a one-step eight-warp phase that fits one register window.

    The binding is replicated across the warp group.  Requiring
    `binding * warps <= REGISTER_WINDOW_WORDS` expresses the first observed
    cliff without naming a matrix width: 20--32 binding words fit, while the
    first 36-word fragment exceeds the window.  A separate MMA-atom cap keeps
    the same compact binding from being admitted after a larger row fragment
    doubles or triples the scheduled work.  Both 8-byte and 16-byte copy paths
    occur on either side, so copy width is only a minimum legality condition.
    """
    phases = _reordered_phases(record)
    if record["aiu"] or len(phases) != 1:
        return False
    phase = phases[0]
    binding = phase["swzl_binding_peak_words"]
    return bool(
        phase["stages"] == 0
        and phase["static_trip_count"] == 1
        and not phase["feeds_gemm"]
        and phase["warps"] == 8
        and not phase.get("copy_guard", 0)
        and not phase.get("copy_lane_guard", 0)
        and phase.get("copy_min_width", 16) >= 8
        and phase.get("other_live_words", 0) == 0
        and binding > 0
        and binding * phase["warps"] <= REGISTER_WINDOW_WORDS
        and phase["mma_atoms"] <= 4 * phase["warps"]
    )


def static_single_packet_warp_group_frontier(record):
    """SWZL wins a static one-packet phase in one eight-warp group.

    This is a resource topology test: one address packet is reused by the
    complete eight-warp group, while the ordinary path must materialize a
    deeper address DAG.  A guarded copy adds a select/zero dependency to each
    loop iteration, so that path is admitted only when SWZL binding is much
    smaller than the address DAG and there is enough MMA work to amortize it.
    It deliberately excludes six/ten-warp controls.
    """
    phases = _reordered_phases(record)
    if record["aiu"] or len(phases) != 1:
        return False
    phase = phases[0]
    guard_free_copy = not phase.get("copy_guard", 0)
    guarded_low_binding = bool(
        phase.get("copy_guard", 0)
        and 3 * phase["swzl_binding_peak_words"] <= phase["address_dag_working_set"]
        and phase["mma_atoms"] >= 2 * phase["warps"]
    )
    return bool(
        phase["stages"] == 0
        and phase["static_trip_count"] > 0
        and not phase["feeds_gemm"]
        and phase["warps"] == 8
        and phase["copy_iterations"] == 1
        and phase.get("copy_min_width", 16) == 16
        and not phase.get("copy_lane_guard", 0)
        and phase.get("other_live_words", 0) == 0
        and phase["swzl_binding_peak_words"] < phase["address_dag_working_set"]
        and (guard_free_copy or guarded_low_binding)
    )


def guarded_static_single_packet_dominance(record):
    """Ordinary dominates guarded one-packet cases outside the SWZL window."""
    phases = _reordered_phases(record)
    if record["aiu"] or len(phases) != 1:
        return False
    phase = phases[0]
    low_binding_work = bool(
        3 * phase["swzl_binding_peak_words"] <= phase["address_dag_working_set"] and phase["mma_atoms"] >= 2 * phase["warps"]
    )
    return bool(
        phase["stages"] == 0
        and phase["static_trip_count"] > 0
        and not phase["feeds_gemm"]
        and phase["warps"] == 8
        and phase["copy_iterations"] == 1
        and phase.get("copy_guard", 0)
        and phase.get("copy_min_width", 16) == 16
        and not phase.get("copy_lane_guard", 0)
        and phase.get("other_live_words", 0) == 0
        and not low_binding_work
    )


def static_two_warp_binding_balance_frontier(record):
    """SWZL wins a dense static phase when two-warp binding stays balanced.

    The static loop repeats the same per-iteration binding topology, but its
    scheduling cost grows with the fragment.  Compact fragments admit SWZL
    directly.  A guarded global copy adds dependency pressure and extends the
    frontier only through a bounded middle band; wider fragments fall back.
    """
    phases = _reordered_phases(record)
    if record["aiu"] or len(phases) != 1:
        return False
    phase = phases[0]
    return bool(
        phase["stages"] == 0
        and phase["static_trip_count"] > 0
        and not phase["feeds_gemm"]
        and phase["warps"] == 2
        and phase["mma_atoms"] >= 96
        and (phase["mma_atoms"] <= 160 or (phase["mma_atoms"] <= 224 and phase.get("copy_guard", 0)))
        and phase.get("copy_min_width", 16) == 16
        and not phase.get("copy_lane_guard", 0)
        and phase.get("other_live_words", 0) == 0
        and abs(phase["swzl_binding_peak_words"] - phase["address_dag_working_set"]) <= 4
    )


def static_two_warp_wide_fragment_dominance(record):
    """Ordinary dominates balanced two-warp fragments beyond the safe band.

    Without a guarded-copy dependency, the compact band ends at 160 atoms.
    A guard extends it through 224 atoms; fragments beyond that are ordinary
    regardless.  This mirrors the positive frontier instead of delegating its
    complement to a fallback that cannot see this sharp scheduling boundary.
    """
    phases = _reordered_phases(record)
    if record["aiu"] or len(phases) != 1:
        return False
    phase = phases[0]
    return bool(
        phase["stages"] == 0
        and phase["static_trip_count"] > 0
        and not phase["feeds_gemm"]
        and phase["warps"] == 2
        and phase["mma_atoms"] > 160
        and (not phase.get("copy_guard", 0) or phase["mma_atoms"] > 224)
        and phase.get("copy_min_width", 16) == 16
        and not phase.get("copy_lane_guard", 0)
        and phase.get("other_live_words", 0) == 0
        and abs(phase["swzl_binding_peak_words"] - phase["address_dag_working_set"]) <= 4
    )


def decide(record, model=None):
    if model is None:
        model = _MODEL
    if not record.get("supported") or model is None:
        return calibrated_decide(model, record, feature_fn=features)
    for audit_name in ("live_range_audit", "address_dag_audit", "launch_audit"):
        audit = record.get(audit_name)
        if audit is not None and not audit.get("known", False):
            return {
                "choice": "swzl",
                "reason": f"{audit_name}_unknown_keep_existing",
            }
    if not any(phase["reorder_words"] for phase in record["phases"]):
        return {"choice": "swzl", "reason": "no_rebinding_cost_keep_existing"}
    if serial_copy_retention_frontier(record):
        return {"choice": "swzl", "reason": "fp8_serial_full_retention_frontier"}
    if pipelined_scalar_address_frontier(record):
        return {"choice": "ordinary", "reason": "fp8_pipelined_scalar_address_frontier"}
    if scalar_nonfeed_reuse_frontier(record):
        return {"choice": "swzl", "reason": "fp8_scalar_nonfeed_reuse_frontier"}
    if single_warp_pipeline_binding_dominance(record):
        return {"choice": "ordinary", "reason": "fp8_single_warp_pipeline_binding_dominance"}
    if two_warp_moderate_binding_dominance(record):
        return {"choice": "ordinary", "reason": "fp8_two_warp_moderate_binding_dominance"}
    if single_warp_feed_balance_frontier(record):
        return {"choice": "swzl", "reason": "fp8_single_warp_feed_balance_frontier"}
    if mixed_feed_binding_balance_frontier(record):
        return {"choice": "swzl", "reason": "fp8_mixed_feed_binding_balance_frontier"}
    if single_phase_high_reuse_frontier(record):
        return {"choice": "swzl", "reason": "fp8_single_phase_high_reuse_frontier"}
    packet_reuse_choice = dynamic_four_warp_packet_reuse_choice(record)
    if packet_reuse_choice is not None:
        return {
            "choice": packet_reuse_choice,
            "reason": "fp8_dynamic_four_warp_packet_reuse_cost",
        }
    if dynamic_four_warp_allocator_cliff_frontier(record):
        return {"choice": "swzl", "reason": "fp8_dynamic_four_warp_allocator_cliff_frontier"}
    if dynamic_four_warp_first_overflow_accumulator_dominance(record):
        return {
            "choice": "ordinary",
            "reason": "fp8_dynamic_four_warp_first_overflow_accumulator_dominance",
        }
    if repeated_phase_register_frontier(record):
        return {"choice": "swzl", "reason": "fp8_repeated_phase_register_frontier"}
    if repeated_phase_full_fragment_reuse_frontier(record):
        return {"choice": "swzl", "reason": "fp8_repeated_phase_full_fragment_reuse_frontier"}
    if static_one_step_compact_fragment_frontier(record):
        return {"choice": "swzl", "reason": "fp8_static_one_step_compact_fragment_frontier"}
    if static_single_packet_warp_group_frontier(record):
        return {"choice": "swzl", "reason": "fp8_static_single_packet_warp_group_frontier"}
    if static_two_warp_binding_balance_frontier(record):
        return {"choice": "swzl", "reason": "fp8_static_two_warp_binding_balance_frontier"}
    if guarded_static_single_packet_dominance(record):
        return {"choice": "ordinary", "reason": "fp8_guarded_static_single_packet_dominance"}
    if static_two_warp_wide_fragment_dominance(record):
        return {"choice": "ordinary", "reason": "fp8_static_two_warp_wide_fragment_dominance"}
    result = calibrated_decide(model, record, feature_fn=features)
    if "net_log_risk" in result:
        result["reason"] = "fp8_calibrated_live_range_cost"
    return result
