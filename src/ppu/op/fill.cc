/*!
 * \file tl/ppu/op/fill.cc
 * \brief PPU implementation registration for tl.fill lowering.
 */

#include "backend/common/op/fill.h"

#include "backend/common/target_utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchPpuFillTarget(Target target) { return TargetIsPPU(target); }

bool RegisterPpuFill() {
  RegisterFillImpl(FillImpl{
      "ppu.Fill",
      MatchPpuFillTarget,
      backend::Fill::Lower,
  });
  return true;
}

const bool ppu_fill_registered = RegisterPpuFill();

} // namespace

} // namespace tl
} // namespace tvm
