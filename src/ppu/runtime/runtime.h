/*!
 * \file tl/ppu/runtime.h
 * \brief PPU runtime function names.
 *
 */

#ifndef TVM_TL_BACKEND_PPU_RUNTIME_RUNTIME_H_
#define TVM_TL_BACKEND_PPU_RUNTIME_RUNTIME_H_

namespace tvm {
namespace tl {

constexpr const char *tvm_ppu_stream_set_access_policy_window =
    "__tvm_ppu_stream_set_access_policy_window";
constexpr const char *tvm_ppu_stream_reset_access_policy_window =
    "__tvm_ppu_stream_reset_access_policy_window";

} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_PPU_RUNTIME_RUNTIME_H_
