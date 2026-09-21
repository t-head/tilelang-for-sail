/*!
 * \brief Wrap threadIdx variables in aiu_load and tix_ldmatrix_swzl address
 * arguments with ppu_to_uniform_b32.
 * \file inject_ppu_uniform.cc
 */
#include <tvm/tirx/expr.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <unordered_set>
#include <utility>

#include "op/builtin.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

class UniformWrapMutator : public StmtExprMutator {
public:
  explicit UniformWrapMutator(
      const std::unordered_set<const VarNode *> &thread_vars)
      : thread_vars_(thread_vars) {}

  PrimExpr Rewrite(const PrimExpr &expr) { return VisitExpr(expr); }

  PrimExpr VisitExpr_(const VarNode *op) final {
    if (thread_vars_.count(op)) {
      Var var = GetRef<Var>(op);
      return Call(DataType::Int(32), tl::ppu_to_uniform_b32(),
                  {Cast(DataType::Int(32), var)});
    }
    return StmtExprMutator::VisitExpr_(op);
  }

private:
  const std::unordered_set<const VarNode *> &thread_vars_;
};

class PpuUniformInjector : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f) {
    PrimFuncNode *fptr = f.CopyOnWrite();
    PpuUniformInjector injector;
    fptr->body = injector.VisitStmt(f->body);
    return f;
  }

private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (std::string(iv->thread_tag).rfind("threadIdx", 0) == 0) {
        thread_vars_.insert(iv->var.get());
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tl::ppu_aiu_load()) ||
        op->op.same_as(tl::tix_ldmatrix_swzl())) {
      // Wrap threadIdx in all arguments with ppu_to_uniform_b32
      Array<PrimExpr> new_args;
      new_args.reserve(op->args.size());
      UniformWrapMutator wrapper(thread_vars_);
      for (const PrimExpr &arg : op->args) {
        new_args.push_back(wrapper.Rewrite(arg));
      }
      Call call = GetRef<Call>(op);
      auto *n = call.CopyOnWrite();
      n->args = new_args;
      return call;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  std::unordered_set<const VarNode *> thread_vars_;
};

using namespace tirx::transform;

static tvm::transform::Pass InjectPpuUniform() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return PpuUniformInjector::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ppu.InjectPpuUniform", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.ppu.transform.InjectPpuUniform",
                        InjectPpuUniform);
}

} // namespace tl
} // namespace tvm
