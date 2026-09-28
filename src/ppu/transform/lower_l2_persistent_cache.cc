/*!
 * \file tl/transform/lower_l2_persistent_cache.cc
 * \brief L2 persistent cache annotation materialization.
 *
 * This pass reads the ``l2_persistent_map`` PrimFunc attribute and emits
 * prologue/epilogue calls to
 * __tvm_ppu_stream_set/reset_access_policy_window.
 */

#include "support/check.h"
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>

#include "ppu/runtime/runtime.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

class LowerL2PersistentCache : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc &f) {
    PrimFuncNode *fptr = f.CopyOnWrite();
    LowerL2PersistentCache substituter;
    fptr->body = substituter.VisitStmt(f->body);

    // Collect prologue/epilogue statements for host-side setup/teardown
    Array<Stmt> prologue_stmts;
    Array<Stmt> epilogue_stmts;

    // Additionally, if L2 persistent cache annotations were lowered earlier,
    // materialize TVM FFI calls to set the stream access policy window.
    if (f->attrs.defined() && f->attrs->dict.count("l2_persistent_map")) {
      auto l2_map =
          f->GetAttr<Map<String, Array<PrimExpr>>>("l2_persistent_map");
      if (l2_map.defined()) {
        // Build a lookup from buffer name to Buffer object
        std::unordered_map<std::string, Buffer> name2buf;
        for (const auto &kv : f->buffer_map) {
          name2buf.emplace(kv.second->name, kv.second);
        }
        for (const auto &kv : l2_map.value()) {
          const std::string buf_name = kv.first;
          const Array<PrimExpr> &args = kv.second;
          if (name2buf.count(buf_name) == 0) {
            continue;
          }
          const Buffer &buf = name2buf.at(buf_name);
          // Build base pointer expression.
          PrimExpr base_ptr = buf->data;
          if (buf->elem_offset.defined() && !is_zero(buf->elem_offset)) {
            PrimExpr byte_offset =
                buf->elem_offset *
                IntImm(buf->elem_offset.dtype(), buf->dtype.bytes());
            base_ptr =
                Call(DataType::Handle(), builtin::handle_add_byte_offset(),
                     {base_ptr, byte_offset});
          }
          // Args packed: func_name, base_ptr, num_bytes, hit_ratio
          Array<PrimExpr> packed_args;
          packed_args.push_back(
              StringImm(tvm_ppu_stream_set_access_policy_window));
          packed_args.push_back(base_ptr);
          // size_in_bytes (args[1]) then hit_ratio (args[0])
          ICHECK_GE(args.size(), 2);
          packed_args.push_back(args[1]);
          packed_args.push_back(args[0]);
          prologue_stmts.push_back(Evaluate(Call(
              DataType::Int(32), builtin::tvm_call_packed(), packed_args)));
        }
        // Add a single epilogue call to reset the access policy window and
        // restore L2 limit
        Array<PrimExpr> reset_args;
        reset_args.push_back(
            StringImm(tvm_ppu_stream_reset_access_policy_window));
        epilogue_stmts.push_back(Evaluate(
            Call(DataType::Int(32), builtin::tvm_call_packed(), reset_args)));
      }
    }

    // Stitch prologue statements before the original body
    if (!prologue_stmts.empty()) {
      // Chain the Let/Evaluate statements sequentially
      Stmt seq = prologue_stmts.size() == 1 ? prologue_stmts[0]
                                            : SeqStmt(prologue_stmts);
      fptr->body = SeqStmt({seq, fptr->body});
    }
    if (!epilogue_stmts.empty()) {
      Stmt seq_end = epilogue_stmts.size() == 1 ? epilogue_stmts[0]
                                                : SeqStmt(epilogue_stmts);
      fptr->body = SeqStmt({fptr->body, seq_end});
    }
    return f;
  }

private:
  LowerL2PersistentCache() = default;
};

using namespace tirx::transform;

static tvm::transform::Pass LowerL2PersistentCache() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    return LowerL2PersistentCache::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerL2PersistentCache", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.ppu.transform.LowerL2PersistentCache",
                        LowerL2PersistentCache);
}

} // namespace tl
} // namespace tvm
