/*!
 * \file tl/ppu/op/transpose.cc
 * \brief PPU implementation registration for tl.transpose lowering.
 */

#include "backend/common/op/transpose.h"

#include "backend/common/target_utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchPpuTransposeTarget(Target target) { return TargetIsPPU(target); }

bool RegisterPpuTranspose() {
  RegisterTransposeImpl(TransposeImpl{
      "ppu.Transpose",
      MatchPpuTransposeTarget,
      backend::Transpose::Lower,
  });
  return true;
}

const bool ppu_transpose_registered = RegisterPpuTranspose();

} // namespace

} // namespace tl
} // namespace tvm
