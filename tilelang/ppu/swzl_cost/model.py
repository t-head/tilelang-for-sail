"""Cost model for PPU 1.5 FP8 ldmatrix SWZL selection.

Inference consumes only immutable facts from the pre-LowerTileOp PrimFunc.
The bundled tree ensemble was calibrated offline; runtime trip counts,
operator names, generated code, ISA, and benchmark measurements are not
selector inputs.
"""

import json
import math
from pathlib import Path


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _valid_tree(tree):
    if not isinstance(tree, dict):
        return False
    if "feature" not in tree:
        return _finite_number(tree.get("value"))
    return bool(
        isinstance(tree.get("feature"), str)
        and _finite_number(tree.get("threshold"))
        and _valid_tree(tree.get("left"))
        and _valid_tree(tree.get("right"))
    )


def _load_model(path=None):
    """Load the optional calibrated model without making compilation fail."""
    path = Path(path) if path is not None else Path(__file__).with_name("model.json")
    try:
        model = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not (
        isinstance(model, dict)
        and _finite_number(model.get("bias"))
        and _finite_number(model.get("rate"))
        and isinstance(model.get("trees"), list)
        and all(_valid_tree(tree) for tree in model["trees"])
    ):
        return None
    return model


_MODEL = _load_model()


def _tree_value(tree, values):
    while "feature" in tree:
        value = values.get(tree["feature"], 0)
        tree = tree["left"] if value <= tree["threshold"] else tree["right"]
    return tree["value"]


def calibrated_score(model, values):
    """Return the calibrated ordinary-vs-SWZL risk margin.

    Positive scores select the ordinary ldmatrix path.  Zero and negative
    scores select SWZL.  The magnitude is a calibrated log-risk margin, not a
    latency estimate.
    """
    return model["bias"] + model["rate"] * sum(_tree_value(tree, values) for tree in model["trees"])


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
    """Choose from the numerical model after legality and hard rules."""
    if not record.get("supported"):
        return {
            "choice": "keep_existing",
            "reason": record.get("reason", "unsupported_keep_existing"),
        }
    if model is None:
        return {"choice": "keep_existing", "reason": "missing_model_keep_existing"}
    if not any(phase["reorder_words"] for phase in record["phases"]):
        return {"choice": "keep_existing", "reason": "no_rebinding_cost_keep_existing"}
    values = feature_fn(record)
    score = calibrated_score(model, values)
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


def _scalar_nonfeed_reuse_choice(record):
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
        return None
    phase = phases[0]
    eligible = bool(
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
    return "swzl" if eligible else None


def _mixed_feed_binding_balance_choice(record):
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
    eligible = bool(
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
    return "swzl" if eligible else None


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


def _dynamic_four_warp_allocator_cliff_choice(record):
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
        return None
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
    eligible = bool(
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
    return "swzl" if eligible else None


def _repeated_phase_full_fragment_reuse_choice(record):
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
        return None
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
            return None
    return "swzl"


def _static_one_step_compact_fragment_choice(record):
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
        return None
    phase = phases[0]
    binding = phase["swzl_binding_peak_words"]
    eligible = bool(
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
    return "swzl" if eligible else None


# Ordered hard overrides for measured backend scheduling boundaries.  These
# rules return a path or None; they intentionally do not manufacture a numeric
# latency score.  Only the calibrated fallback below emits net_log_risk.
_HARD_RULES = (
    ("fp8_scalar_nonfeed_reuse_frontier", _scalar_nonfeed_reuse_choice),
    ("fp8_mixed_feed_binding_balance_frontier", _mixed_feed_binding_balance_choice),
    ("fp8_dynamic_four_warp_packet_reuse_cost", dynamic_four_warp_packet_reuse_choice),
    ("fp8_dynamic_four_warp_allocator_cliff_frontier", _dynamic_four_warp_allocator_cliff_choice),
    ("fp8_repeated_phase_full_fragment_reuse_frontier", _repeated_phase_full_fragment_reuse_choice),
    ("fp8_static_one_step_compact_fragment_frontier", _static_one_step_compact_fragment_choice),
)


def _hard_rule_decide(record):
    for reason, rule in _HARD_RULES:
        choice = rule(record)
        if choice is not None:
            return {"choice": choice, "reason": reason}
    return None


def decide(record, model=None):
    """Select the lowering path through eligibility, hard rules, then score."""
    if model is None:
        model = _MODEL
    if not record.get("supported") or model is None:
        return calibrated_decide(model, record, feature_fn=features)
    for audit_name in ("live_range_audit", "address_dag_audit", "launch_audit"):
        audit = record.get(audit_name)
        if audit is not None and not audit.get("known", False):
            return {
                "choice": "keep_existing",
                "reason": f"{audit_name}_unknown_keep_existing",
            }
    if not any(phase["reorder_words"] for phase in record["phases"]):
        return {"choice": "keep_existing", "reason": "no_rebinding_cost_keep_existing"}
    hard_decision = _hard_rule_decide(record)
    if hard_decision is not None:
        return hard_decision
    result = calibrated_decide(model, record, feature_fn=features)
    if "net_log_risk" in result:
        result["reason"] = "fp8_calibrated_live_range_cost"
    return result
