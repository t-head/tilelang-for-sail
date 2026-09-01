/*!
 * \file tl/ppu/target_utils.h
 * \brief PPU target attribute helpers.
 */

#ifndef TVM_TL_PPU_TARGET_UTILS_H_
#define TVM_TL_PPU_TARGET_UTILS_H_

#include <tvm/runtime/data_type.h>
#include <tvm/target/target.h>

namespace tvm {
namespace tl {

bool TargetIsPPU(Target target);

int TargetPPUGetWarpSize(Target target);
bool TargetHasAiuCopy(Target target);
bool TargetPPUHasAsyncCopy(Target target);
bool TargetPPUHasLdmatrix(Target target);
bool TargetPPUHasStmatrix(Target target);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_PPU_TARGET_UTILS_H_
