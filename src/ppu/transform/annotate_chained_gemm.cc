/*!
 * \file annotate_chained_gemm.cc
 * \brief Add "a_from_gemm_c" annotation to gemm Calls whose A comes from a
 * prior GEMM's C (via copies or element-wise BufferStore). Runs after
 * software pipelining and before layout inference.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_set>

#include "../../op/copy.h"
#include "../../op/gemm.h"
#include "../../op/operator.h"
#include "../../op/utils.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

constexpr const char *kAFromGemmC = "a_from_gemm_c";

// Traces GEMM C -> A chains in program order. The chained set grows
// monotonically across SBlocks; matching uses buffer data var identity.
class ChainedGemmAnnotator : public StmtExprMutator {
public:
  Stmt VisitStmt_(const BufferStoreNode *op) final {
    Stmt result = StmtExprMutator::VisitStmt_(op);
    const auto *store = result.as<BufferStoreNode>();
    if (!store)
      return result;
    if (!IsFragmentBuffer(store->buffer) && !IsLocalBuffer(store->buffer)) {
      return result;
    }
    bool rhs_has_chain = false;
    PostOrderVisit(store->value, [&](const ObjectRef &node) {
      if (rhs_has_chain)
        return;
      if (const auto *load = node.as<BufferLoadNode>()) {
        if (load->buffer.defined() &&
            chained_data_vars_.count(load->buffer->data.get()) != 0) {
          rhs_has_chain = true;
        }
      }
    });
    if (rhs_has_chain) {
      chained_data_vars_.insert(store->buffer->data.get());
    }
    return result;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = Downcast<Call>(StmtExprMutator::VisitExpr_(op));
    if (!call->op.as<Op>().has_value()) {
      return call;
    }
    TileOperator tile_op = ParseOperator(call);
    if (!tile_op.defined()) {
      return call;
    }

    if (const auto *gemm = tile_op.as<GemmNode>()) {
      return HandleGemm(call, gemm);
    }
    if (const auto *copy = tile_op.as<CopyNode>()) {
      HandleCopy(copy);
    }
    return call;
  }

private:
  bool IsChained(const Buffer &buffer) const {
    return buffer.defined() &&
           chained_data_vars_.count(buffer->data.get()) != 0;
  }

  void MarkChained(const Buffer &buffer) {
    if (buffer.defined()) {
      chained_data_vars_.insert(buffer->data.get());
    }
  }

  PrimExpr HandleGemm(const Call &call, const GemmNode *gemm) {
    PrimExpr result = call;
    if (IsChained(gemm->a_) && !call->annotations.count(kAFromGemmC)) {
      Map<String, ObjectRef> annotations = call->annotations;
      annotations.Set(kAFromGemmC, IntImm(DataType::Int(32), 1));
      result = Call(call->dtype, call->op, call->args, annotations, call->span);
    }
    MarkChained(gemm->c_);
    return result;
  }

  void HandleCopy(const CopyNode *copy) {
    if (IsChained(copy->src) &&
        (IsFragmentBuffer(copy->dst) || IsLocalBuffer(copy->dst))) {
      MarkChained(copy->dst);
    }
  }

  std::unordered_set<const VarNode *> chained_data_vars_;
};

} // namespace

using namespace tirx::transform;

tvm::transform::Pass AnnotateChainedGemm() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    auto *n = f.CopyOnWrite();
    n->body = ChainedGemmAnnotator()(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AnnotateChainedGemm", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.ppu.transform.AnnotateChainedGemm",
                        AnnotateChainedGemm);
}

} // namespace tl
} // namespace tvm
