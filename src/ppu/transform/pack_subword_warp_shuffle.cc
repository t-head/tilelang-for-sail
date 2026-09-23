/*!
 * \file pack_subword_warp_shuffle.cc
 * \brief Fuse proven scalar subword relayouts into packed 32-bit shuffles.
 *
 * Match by the source-lane mapping of four consecutive destination bytes,
 * not by an assumed architecture-specific MMA layout.  An unconditional
 * mapping packs four source values into one uint32, shuffles once per distinct
 * source lane, then gathers the desired bytes.  PPU0015 E4M3FN admits up to
 * four source lanes; PPU0010/0015 INT8 admits at most two per candidate.
 * Other mappings retain scalar shuffles.  Conditional scalar shuffles are
 * evaluated before the lane-varying select so the full-warp mask stays valid.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <array>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "op/builtin.h"
#include "ppu/target_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

bool IsPackedShuffleFP8(DataType dtype) {
  return dtype.is_scalar() &&
         (dtype.is_float8_e4m3() || dtype.is_float8_e4m3fn() ||
          dtype.is_float8_e5m2());
}

struct ScalarShuffle {
  Call call;
  PrimExpr value;
  PrimExpr lane;
};

struct ConditionalSourceLaneStore {
  BufferStore store;
  PrimExpr condition;
  ScalarShuffle true_shuffle;
  ScalarShuffle false_shuffle;
  ffi::Map<ffi::String, ffi::Any> cast_annotations;
};

struct BoundConditionalSourceLaneStore {
  ConditionalSourceLaneStore relayout;
  size_t next_index;
};

struct UnconditionalSourceLaneStore {
  BufferStore store;
  ScalarShuffle shuffle;
  ffi::Map<ffi::String, ffi::Any> cast_annotations;
};

struct BoundUnconditionalSourceLaneStore {
  UnconditionalSourceLaneStore relayout;
  size_t next_index;
};

std::optional<ScalarShuffle> MatchShuffle(const PrimExpr &expr) {
  const auto *call = expr.as<CallNode>();
  if (call == nullptr || !call->op.same_as(tl::shfl_sync()) ||
      call->args.size() != 4 || call->dtype.bits() != 32 ||
      !call->dtype.is_scalar()) {
    return std::nullopt;
  }
  return ScalarShuffle{GetRef<Call>(call), call->args[1], call->args[2]};
}

bool MatchConditional(const PrimExpr &expr, PrimExpr *condition,
                      PrimExpr *true_value, PrimExpr *false_value) {
  if (const auto *select = expr.as<SelectNode>()) {
    *condition = select->condition;
    *true_value = select->true_value;
    *false_value = select->false_value;
    return true;
  }
  if (const auto *call = expr.as<CallNode>();
      call != nullptr && call->op.same_as(builtin::if_then_else()) &&
      call->args.size() == 3) {
    *condition = call->args[0];
    *true_value = call->args[1];
    *false_value = call->args[2];
    return true;
  }
  return false;
}

std::optional<ConditionalSourceLaneStore>
MatchConditionalSourceLaneStore(const Stmt &stmt) {
  const auto *store_node = stmt.as<BufferStoreNode>();
  if (store_node == nullptr || store_node->indices.size() != 1 ||
      store_node->predicate.defined() ||
      store_node->buffer.scope() != "local") {
    return std::nullopt;
  }

  const auto *cast = store_node->value.as<CastNode>();
  if (cast == nullptr || cast->dtype != store_node->buffer->dtype) {
    return std::nullopt;
  }

  PrimExpr condition;
  PrimExpr true_value;
  PrimExpr false_value;
  if (!MatchConditional(cast->value, &condition, &true_value, &false_value)) {
    return std::nullopt;
  }

  auto true_shuffle = MatchShuffle(true_value);
  auto false_shuffle = MatchShuffle(false_value);
  if (!true_shuffle.has_value() || !false_shuffle.has_value()) {
    return std::nullopt;
  }

  return ConditionalSourceLaneStore{GetRef<BufferStore>(store_node),
                                    condition, std::move(true_shuffle.value()),
                                    std::move(false_shuffle.value()),
                                    cast->annotations};
}

std::optional<UnconditionalSourceLaneStore>
MatchUnconditionalSourceLaneStore(const Stmt &stmt) {
  const auto *store_node = stmt.as<BufferStoreNode>();
  if (store_node == nullptr || store_node->indices.size() != 1 ||
      store_node->predicate.defined() ||
      store_node->buffer.scope() != "local") {
    return std::nullopt;
  }

  const auto *cast = store_node->value.as<CastNode>();
  if (cast == nullptr || cast->dtype != store_node->buffer->dtype) {
    return std::nullopt;
  }
  auto shuffle = MatchShuffle(cast->value);
  if (!shuffle.has_value()) {
    return std::nullopt;
  }
  return UnconditionalSourceLaneStore{GetRef<BufferStore>(store_node),
                                      std::move(shuffle.value()),
                                      cast->annotations};
}

class PackSubwordWarpShuffleRewriter : public StmtExprMutator {
 public:
  explicit PackSubwordWarpShuffleRewriter(int ppu_arch)
      : ppu_arch_(ppu_arch) {}

  Stmt VisitStmt_(const ForNode *op) final {
    Stmt visited = StmtExprMutator::VisitStmt_(op);
    const auto *loop = visited.as<ForNode>();
    if (loop == nullptr || loop->thread_binding.defined() ||
        loop->step.defined()) {
      return visited;
    }

    const auto *extent = loop->extent.as<IntImmNode>();
    if (extent == nullptr || extent->value < 4 || extent->value % 4 != 0) {
      return visited;
    }

    auto normalized_store = InlineFlatBinds(loop->body);
    if (!normalized_store.has_value()) {
      return visited;
    }

    Var pack_var(loop->loop_var->name_hint + "_pack", loop->loop_var.dtype());
    PrimExpr pack_base =
        loop->min + pack_var * make_const(pack_var.dtype(), 4);
    std::array<ConditionalSourceLaneStore, 4> group;
    bool conditional_group_matched = true;
    for (size_t j = 0; j < 4; ++j) {
      Map<Var, PrimExpr> substitution{
          {loop->loop_var,
           pack_base + make_const(pack_var.dtype(), static_cast<int64_t>(j))}};
      Stmt scalar_store = Substitute(normalized_store.value(), substitution);
      auto matched = MatchConditionalSourceLaneStore(scalar_store);
      if (!matched.has_value()) {
        conditional_group_matched = false;
        break;
      }
      group[j] = std::move(matched.value());
    }

    if (conditional_group_matched) {
      PackDecision decision = ShouldPack(group);
      if (decision != PackDecision::kKeepScalar) {
        return For(pack_var, make_zero(pack_var.dtype()),
                   make_const(loop->extent.dtype(), extent->value / 4),
                   loop->kind, PackConditionalLaneGroup(group, decision),
                   loop->thread_binding, loop->annotations, loop->step,
                   loop->span);
      }
      auto scalar = MatchConditionalSourceLaneStore(normalized_store.value());
      if (scalar.has_value() &&
          SubwordPackFactor(scalar->store->buffer->dtype) == 4) {
        return For(loop->loop_var, loop->min, loop->extent, loop->kind,
                   SafeScalarConditionalStore(scalar.value()),
                   loop->thread_binding, loop->annotations, loop->step,
                   loop->span);
      }
      return visited;
    }

    // One shuffle per byte can use one to four distinct source lanes.  Pack
    // the source bytes first, then gather each result byte from its lane.
    Map<Var, PrimExpr> first_substitution{{loop->loop_var, loop->min}};
    auto first_unconditional = MatchUnconditionalSourceLaneStore(
        Substitute(normalized_store.value(), first_substitution));
    if (!first_unconditional.has_value()) {
      return visited;
    }
    int factor = SubwordPackFactor(first_unconditional->store->buffer->dtype);
    if (factor == 0 || extent->value % factor != 0) {
      return visited;
    }
    std::vector<UnconditionalSourceLaneStore> unconditional_group;
    unconditional_group.reserve(factor);
    Var unconditional_pack_var(loop->loop_var->name_hint + "_subword_pack",
                               loop->loop_var.dtype());
    PrimExpr unconditional_base =
        loop->min + unconditional_pack_var *
                        make_const(unconditional_pack_var.dtype(), factor);
    for (int j = 0; j < factor; ++j) {
      Map<Var, PrimExpr> substitution{
          {loop->loop_var,
           unconditional_base + make_const(unconditional_pack_var.dtype(), j)}};
      auto item = MatchUnconditionalSourceLaneStore(
          Substitute(normalized_store.value(), substitution));
      if (!item.has_value()) {
        return visited;
      }
      unconditional_group.push_back(std::move(item.value()));
    }
    if (!ShouldPack(unconditional_group)) {
      return visited;
    }
    return For(unconditional_pack_var, make_zero(unconditional_pack_var.dtype()),
               make_const(loop->extent.dtype(), extent->value / factor),
               loop->kind, PackUnconditionalLaneGroup(unconditional_group),
               loop->thread_binding, loop->annotations, loop->step,
               loop->span);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> seq;
    seq.reserve(op->seq.size());
    for (const Stmt &stmt : op->seq) {
      seq.push_back(VisitStmt(stmt));
    }

    Array<Stmt> rewritten;
    rewritten.reserve(seq.size());
    size_t i = 0;
    while (i < seq.size()) {
      std::array<ConditionalSourceLaneStore, 4> group;
      size_t cursor = i;
      bool matched = true;
      for (size_t j = 0; j < 4; ++j) {
        auto item = MatchBoundConditionalSourceLaneStore(seq, cursor);
        if (!item.has_value()) {
          matched = false;
          break;
        }
        group[j] = std::move(item->relayout);
        cursor = item->next_index;
      }
      if (matched) {
        PackDecision decision = ShouldPack(group);
        if (decision != PackDecision::kKeepScalar) {
          rewritten.push_back(PackConditionalLaneGroup(group, decision));
          i = cursor;
          continue;
        }
      }

      if (auto first = MatchBoundUnconditionalSourceLaneStore(seq, i)) {
        int factor = SubwordPackFactor(first->relayout.store->buffer->dtype);
        if (factor != 0) {
          std::vector<UnconditionalSourceLaneStore> unconditional_group;
          unconditional_group.reserve(factor);
          size_t unconditional_cursor = i;
          bool unconditional_matched = true;
          for (int j = 0; j < factor; ++j) {
            auto item = MatchBoundUnconditionalSourceLaneStore(
                seq, unconditional_cursor);
            if (!item.has_value()) {
              unconditional_matched = false;
              break;
            }
            unconditional_group.push_back(std::move(item->relayout));
            unconditional_cursor = item->next_index;
          }
          if (unconditional_matched) {
            if (ShouldPack(unconditional_group)) {
              rewritten.push_back(
                  PackUnconditionalLaneGroup(unconditional_group));
              i = unconditional_cursor;
              continue;
            }
          }
        }
      }
      if (auto scalar = MatchBoundConditionalSourceLaneStore(seq, i)) {
        if (SubwordPackFactor(scalar->relayout.store->buffer->dtype) == 4) {
          rewritten.push_back(SafeScalarConditionalStore(scalar->relayout));
          i = scalar->next_index;
          continue;
        }
      }
      rewritten.push_back(seq[i]);
      ++i;
    }
    return SeqStmt::Flatten(rewritten);
  }

 private:
  enum class PackDecision { kKeepScalar, kGeneric, kFP8Fast };

  Stmt SafeScalarConditionalStore(const ConditionalSourceLaneStore &item) {
    Var true_value("scalar_true", item.true_shuffle.call->dtype);
    Var false_value("scalar_false", item.false_shuffle.call->dtype);
    PrimExpr result = Cast(item.store->buffer->dtype,
                           Select(item.condition, true_value, false_value),
                           item.cast_annotations);
    Stmt store = BufferStore(item.store->buffer, result, item.store->indices,
                             std::nullopt, item.store->span);
    return SeqStmt({Bind(true_value, item.true_shuffle.call),
                    Bind(false_value, item.false_shuffle.call), store});
  }

  template <typename BoundStore, typename Matcher>
  std::optional<BoundStore> MatchBoundStoreImpl(const Array<Stmt> &seq,
                                                size_t start,
                                                Matcher matcher) {
    Map<Var, PrimExpr> bindings;
    std::vector<Var> bound_vars;
    size_t cursor = start;
    while (cursor < seq.size()) {
      const auto *bind = seq[cursor].as<BindNode>();
      if (bind == nullptr) {
        break;
      }
      bound_vars.push_back(bind->var);
      bindings.Set(bind->var, Substitute(bind->value, bindings));
      ++cursor;
    }
    if (cursor == seq.size()) {
      return std::nullopt;
    }

    auto matched = matcher(Substitute(seq[cursor], bindings));
    if (!matched.has_value()) {
      return std::nullopt;
    }

    // A Bind statement scopes its value over the remaining sequence.  It is
    // safe to erase only when none of its variables escape this scalar store.
    for (const Var &var : bound_vars) {
      for (size_t j = cursor + 1; j < seq.size(); ++j) {
        if (UsesVar(seq[j], [node = var.get()](const VarNode *candidate) {
              return candidate == node;
            })) {
          return std::nullopt;
        }
      }
    }
    return BoundStore{std::move(matched.value()), cursor + 1};
  }

  std::optional<BoundUnconditionalSourceLaneStore>
  MatchBoundUnconditionalSourceLaneStore(const Array<Stmt> &seq, size_t start) {
    return MatchBoundStoreImpl<BoundUnconditionalSourceLaneStore>(
        seq, start, [](const Stmt &stmt) {
          return MatchUnconditionalSourceLaneStore(stmt);
        });
  }

  std::optional<BoundConditionalSourceLaneStore>
  MatchBoundConditionalSourceLaneStore(const Array<Stmt> &seq, size_t start) {
    return MatchBoundStoreImpl<BoundConditionalSourceLaneStore>(
        seq, start, [](const Stmt &stmt) {
          return MatchConditionalSourceLaneStore(stmt);
        });
  }

  std::optional<Stmt> InlineFlatBinds(const Stmt &body) {
    if (body.as<BufferStoreNode>() != nullptr) {
      return body;
    }
    const auto *seq = body.as<SeqStmtNode>();
    if (seq == nullptr || seq->seq.empty()) {
      return std::nullopt;
    }

    Map<Var, PrimExpr> bindings;
    for (size_t i = 0; i < seq->seq.size(); ++i) {
      const Stmt &stmt = seq->seq[i];
      if (const auto *bind = stmt.as<BindNode>()) {
        if (i + 1 == seq->seq.size()) {
          return std::nullopt;
        }
        bindings.Set(bind->var, Substitute(bind->value, bindings));
        continue;
      }
      if (i + 1 != seq->seq.size()) {
        return std::nullopt;
      }
      return Substitute(stmt, bindings);
    }
    return std::nullopt;
  }

  bool Equal(const PrimExpr &lhs, const PrimExpr &rhs) {
    return ffi::StructuralEqual()(analyzer_.Simplify(lhs),
                                  analyzer_.Simplify(rhs));
  }

  bool SameCallControl(const Call &lhs, const Call &rhs) {
    return Equal(lhs->args[0], rhs->args[0]) &&
           Equal(lhs->args[3], rhs->args[3]);
  }

  bool SameAnnotations(const ffi::Map<ffi::String, ffi::Any> &lhs,
                       const ffi::Map<ffi::String, ffi::Any> &rhs) const {
    return ffi::StructuralEqual()(lhs, rhs);
  }

  // FP8 on PPU0015 and signed INT8 on PPU0010/0015 have a proven four-byte
  // carrier.  This is a dtype/target gate, not a source-lane restriction.
  int SubwordPackFactor(DataType dtype) const {
    if (ppu_arch_ == 15 && IsPackedShuffleFP8(dtype)) {
      return 4;
    }
    if ((ppu_arch_ == 10 || ppu_arch_ == 15) && dtype.is_int() &&
        dtype.bits() == 8 && dtype.is_scalar()) {
      return 4;
    }
    return 0;
  }

  size_t CountDistinctSourceLanes(
      const std::array<PrimExpr, 4> &source_lanes) {
    std::vector<PrimExpr> distinct;
    for (const PrimExpr &lane : source_lanes) {
      bool seen = false;
      for (const PrimExpr &existing : distinct) {
        if (analyzer_.CanProveEqual(lane, existing)) {
          seen = true;
          break;
        }
      }
      if (!seen) {
        distinct.push_back(lane);
      }
    }
    return distinct.size();
  }

  bool ShouldPack(const std::vector<UnconditionalSourceLaneStore> &group) {
    if (group.empty()) {
      return false;
    }
    const Buffer &buffer = group[0].store->buffer;
    DataType dtype = buffer->dtype;
    int factor = SubwordPackFactor(dtype);
    if (factor != 4 || static_cast<int>(group.size()) != factor) {
      return false;
    }
    PrimExpr base = group[0].store->indices[0];
    if (!analyzer_.CanProveEqual(
            FloorMod(base, make_const(base.dtype(), factor)),
            make_zero(base.dtype()))) {
      return false;
    }
    const Call &control = group[0].shuffle.call;
    std::array<PrimExpr, 4> lanes;
    for (int j = 0; j < factor; ++j) {
      if (!group[j].store->buffer.same_as(buffer) ||
          !SameAnnotations(group[j].cast_annotations,
                           group[0].cast_annotations) ||
          !SameCallControl(group[j].shuffle.call, control) ||
          !analyzer_.CanProveEqual(
              group[j].store->indices[0],
              base + make_const(base.dtype(), static_cast<int64_t>(j)))) {
        return false;
      }
      lanes[j] = group[j].shuffle.lane;
    }
    if (ppu_arch_ == 15 && dtype.is_float8_e4m3fn()) {
      return true;
    }
    return dtype.is_int() && dtype.bits() == 8 &&
           CountDistinctSourceLanes(lanes) <= 2;
  }

  PackDecision ShouldPack(
      const std::array<ConditionalSourceLaneStore, 4> &group) {
    const Buffer &buffer = group[0].store->buffer;
    DataType dtype = buffer->dtype;
    if (SubwordPackFactor(dtype) != 4) {
      return PackDecision::kKeepScalar;
    }
    PrimExpr base = group[0].store->indices[0];
    if (!analyzer_.CanProveEqual(FloorMod(base, make_const(base.dtype(), 4)),
                                 make_zero(base.dtype()))) {
      return PackDecision::kKeepScalar;
    }
    const Call &control = group[0].true_shuffle.call;
    std::array<PrimExpr, 4> true_lanes;
    std::array<PrimExpr, 4> false_lanes;
    for (size_t j = 0; j < 4; ++j) {
      if (!group[j].store->buffer.same_as(buffer) ||
          !Equal(group[j].condition, group[0].condition) ||
          !SameAnnotations(group[j].cast_annotations,
                           group[0].cast_annotations) ||
          !SameCallControl(group[j].true_shuffle.call, control) ||
          !SameCallControl(group[j].false_shuffle.call, control) ||
          !analyzer_.CanProveEqual(
              group[j].store->indices[0],
              base + make_const(base.dtype(), static_cast<int64_t>(j)))) {
        return PackDecision::kKeepScalar;
      }
      true_lanes[j] = group[j].true_shuffle.lane;
      false_lanes[j] = group[j].false_shuffle.lane;
    }
    if (dtype.is_int() && dtype.bits() == 8) {
      return CountDistinctSourceLanes(true_lanes) <= 2 &&
                     CountDistinctSourceLanes(false_lanes) <= 2
                 ? PackDecision::kGeneric
                 : PackDecision::kKeepScalar;
    }
    if (ppu_arch_ != 15 || !dtype.is_float8_e4m3fn()) {
      return PackDecision::kKeepScalar;
    }

    // The observed C-to-A layout permits a cheaper two-shuffle gather.
    PrimExpr lane0 = true_lanes[0];
    PrimExpr lane1 = true_lanes[2];
    bool same_two_lanes =
        Equal(true_lanes[0], true_lanes[1]) &&
        Equal(true_lanes[2], true_lanes[3]) &&
        Equal(false_lanes[0], true_lanes[0]) &&
        Equal(false_lanes[1], true_lanes[1]) &&
        Equal(false_lanes[2], true_lanes[2]) &&
        Equal(false_lanes[3], true_lanes[3]) &&
        analyzer_.CanProveEqual(lane1,
                                lane0 + make_const(lane0.dtype(), 1));
    bool same_two_lane_values =
        Equal(group[0].true_shuffle.value, group[2].true_shuffle.value) &&
        Equal(group[1].true_shuffle.value, group[3].true_shuffle.value) &&
        Equal(group[0].false_shuffle.value, group[2].false_shuffle.value) &&
        Equal(group[1].false_shuffle.value, group[3].false_shuffle.value);
    return same_two_lanes && same_two_lane_values ? PackDecision::kFP8Fast
                                                   : PackDecision::kGeneric;
  }

  // Source byte j stays in byte position j before the shuffle.  For each
  // distinct source lane, fetch the packed word and merge only the bytes
  // owned by that lane.  Four target bytes admit one to four source lanes.
  PrimExpr GatherPackedLanes(const PrimExpr &packed_source,
                             const std::array<PrimExpr, 4> &source_lanes,
                             const Call &control_ref) {
    std::vector<PrimExpr> distinct_lanes;
    std::array<size_t, 4> lane_index;
    for (size_t j = 0; j < 4; ++j) {
      size_t index = 0;
      for (; index < distinct_lanes.size(); ++index) {
        if (analyzer_.CanProveEqual(source_lanes[j], distinct_lanes[index])) {
          break;
        }
      }
      if (index == distinct_lanes.size()) {
        distinct_lanes.push_back(source_lanes[j]);
      }
      lane_index[j] = index;
    }

    auto shuffle_word = [&](const PrimExpr &lane) {
      return Call(DataType::UInt(32), tl::shfl_sync(),
                  {control_ref->args[0], packed_source, lane,
                   control_ref->args[3]});
    };
    PrimExpr result = shuffle_word(distinct_lanes[0]);
    for (size_t index = 1; index < distinct_lanes.size(); ++index) {
      uint32_t selector = 0;
      for (size_t j = 0; j < 4; ++j) {
        uint32_t source_byte = lane_index[j] == index ? 4 + j : j;
        selector |= source_byte << (4 * j);
      }
      result = Call(DataType::UInt(32), builtin::call_pure_extern(),
                    {StringImm("__byte_perm"), result,
                     shuffle_word(distinct_lanes[index]),
                     make_const(DataType::UInt(32), selector)});
    }
    return result;
  }

  Stmt PackUnconditionalLaneGroup(
      const std::vector<UnconditionalSourceLaneStore> &group) {
    const Buffer &buffer = group[0].store->buffer;
    DataType subword_dtype = buffer->dtype;
    int factor = SubwordPackFactor(subword_dtype);
    PrimExpr base = group[0].store->indices[0];
    const Call &scalar_call = group[0].shuffle.call;
    Array<PrimExpr> source_values;
    source_values.reserve(factor);
    std::array<PrimExpr, 4> source_lanes;
    for (int j = 0; j < factor; ++j) {
      source_values.push_back(group[j].shuffle.value);
      source_lanes[j] = group[j].shuffle.lane;
    }

    DataType packed_dtype = subword_dtype.with_lanes(factor);
    PrimExpr source_vector = Shuffle::Concat(source_values);
    PrimExpr cast_vector =
        Cast(packed_dtype, source_vector, group[0].cast_annotations);
    PrimExpr source_u32_value =
        Call(DataType::UInt(32), builtin::reinterpret(), {cast_vector});
    Var source_u32("packed_subword", DataType::UInt(32));
    PrimExpr packed_shuffle =
        GatherPackedLanes(source_u32, source_lanes, scalar_call);
    PrimExpr result_vector =
        Call(packed_dtype, builtin::reinterpret(), {packed_shuffle});
    PrimExpr ramp = Ramp(base, make_const(base.dtype(), 1), factor);
    Stmt store = BufferStore(buffer, result_vector, {ramp}, std::nullopt,
                             group[0].store->span);
    return SeqStmt({Bind(source_u32, source_u32_value), store});
  }

  // General conditional case: gather each branch independently, then select
  // at the destination lane.  The specialized FP8 C-to-A case below uses a
  // cheaper shared source word when its stronger mapping proof holds.
  Stmt PackConditionalLaneGroupGeneric(
      const std::array<ConditionalSourceLaneStore, 4> &group) {
    DataType vector_dtype = group[0].store->buffer->dtype.with_lanes(4);
    Array<PrimExpr> true_values;
    Array<PrimExpr> false_values;
    std::array<PrimExpr, 4> true_lanes;
    std::array<PrimExpr, 4> false_lanes;
    for (size_t j = 0; j < 4; ++j) {
      true_values.push_back(group[j].true_shuffle.value);
      false_values.push_back(group[j].false_shuffle.value);
      true_lanes[j] = group[j].true_shuffle.lane;
      false_lanes[j] = group[j].false_shuffle.lane;
    }
    auto make_word = [&](const Array<PrimExpr> &values) {
      PrimExpr vector = Cast(vector_dtype, Shuffle::Concat(values),
                             group[0].cast_annotations);
      return Call(DataType::UInt(32), builtin::reinterpret(), {vector});
    };
    Var true_word("packed_true", DataType::UInt(32));
    Var false_word("packed_false", DataType::UInt(32));
    const Call &control_ref = group[0].true_shuffle.call;
    PrimExpr true_result =
        GatherPackedLanes(true_word, true_lanes, control_ref);
    PrimExpr false_result =
        GatherPackedLanes(false_word, false_lanes, control_ref);
    // Evaluate both shuffles while the full warp is active.  Emitting them
    // inside a C++ conditional expression would make one branch's lanes
    // inactive despite the full-warp shuffle mask.
    Var true_gather("gathered_true", DataType::UInt(32));
    Var false_gather("gathered_false", DataType::UInt(32));
    PrimExpr selected = Select(group[0].condition, true_gather, false_gather);
    PrimExpr result = Call(vector_dtype, builtin::reinterpret(), {selected});
    PrimExpr base = group[0].store->indices[0];
    PrimExpr ramp = Ramp(base, make_const(base.dtype(), 1), 4);
    Stmt store = BufferStore(group[0].store->buffer, result, {ramp},
                             std::nullopt, group[0].store->span);
    return SeqStmt({Bind(true_word, make_word(true_values)),
                    Bind(false_word, make_word(false_values)),
                    Bind(true_gather, true_result),
                    Bind(false_gather, false_result), store});
  }

  Stmt PackConditionalLaneGroup(
      const std::array<ConditionalSourceLaneStore, 4> &group,
      PackDecision decision) {
    const Buffer &buffer = group[0].store->buffer;
    DataType subword_dtype = buffer->dtype;
    PrimExpr base = group[0].store->indices[0];
    if (decision == PackDecision::kGeneric) {
      return PackConditionalLaneGroupGeneric(group);
    }
    PrimExpr lane0 = group[0].true_shuffle.lane;
    PrimExpr lane1 = group[2].true_shuffle.lane;
    const Call &control_ref = group[0].true_shuffle.call;

    Array<PrimExpr> source_values{
        group[0].true_shuffle.value, group[1].true_shuffle.value,
        group[0].false_shuffle.value, group[1].false_shuffle.value};
    PrimExpr source_f32x4 = Shuffle::Concat(source_values);
    DataType fp8x4_dtype = subword_dtype.with_lanes(4);
    PrimExpr source_fp8x4 = Cast(fp8x4_dtype, source_f32x4,
                                group[0].cast_annotations);
    PrimExpr source_u32_value =
        Call(DataType::UInt(32), builtin::reinterpret(), {source_fp8x4});
    Var source_u32("packed_fp8", DataType::UInt(32));

    const Call &scalar_call = control_ref;
    auto packed_shuffle = [&](const PrimExpr &lane) {
      return Call(DataType::UInt(32), tl::shfl_sync(),
                  {scalar_call->args[0], source_u32, lane,
                   scalar_call->args[3]});
    };
    PrimExpr shuffled0 = packed_shuffle(lane0);
    PrimExpr shuffled1 = packed_shuffle(lane1);

    PrimExpr selector =
        Select(group[0].condition,
               make_const(DataType::UInt(32), 0x5410),
               make_const(DataType::UInt(32), 0x7632));
    PrimExpr packed_result = Call(
        DataType::UInt(32), builtin::call_pure_extern(),
        {StringImm("__byte_perm"), shuffled0, shuffled1, selector});
    PrimExpr result_fp8x4 =
        Call(fp8x4_dtype, builtin::reinterpret(), {packed_result});

    PrimExpr ramp = Ramp(base, make_const(base.dtype(), 1), 4);
    Stmt store = BufferStore(buffer, result_fp8x4, {ramp}, std::nullopt,
                             group[0].store->span);
    return SeqStmt({Bind(source_u32, source_u32_value), store});
  }

  arith::Analyzer analyzer_;
  int ppu_arch_;
};

}  // namespace

using namespace tirx::transform;

tvm::transform::Pass PackSubwordWarpShuffle() {
  auto pass_func = [=](PrimFunc f, const IRModule &m,
                       const PassContext &ctx) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.has_value() || !TargetIsPPU(target.value())) {
      return f;
    }
    int ppu_arch = GetPPUArchInt(target.value());
    if (ppu_arch != 10 && ppu_arch != 15) {
      return f;
    }
    auto *n = f.CopyOnWrite();
    n->body = PackSubwordWarpShuffleRewriter(ppu_arch)(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.PackSubwordWarpShuffle", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.ppu.transform.PackSubwordWarpShuffle",
                        PackSubwordWarpShuffle);
}

}  // namespace tl
}  // namespace tvm
