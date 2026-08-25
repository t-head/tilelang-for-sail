/*!
 * \file tl/ppu/op/scan.cc
 * \brief PPU implementation registration for tl scan lowering.
 */

#include "backend/common/op/scan.h"

#include "backend/common/target_utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchPpuScanTarget(Target target) { return TargetIsPPU(target); }

bool RegisterPpuScan() {
  RegisterCumSumImpl(CumSumImpl{
      "ppu.CumSum",
      MatchPpuScanTarget,
      backend::scan::LowerCumSum,
  });
  RegisterCumMaxImpl(CumMaxImpl{
      "ppu.CumMax",
      MatchPpuScanTarget,
      backend::scan::LowerCumMax,
  });
  return true;
}

const bool ppu_scan_registered = RegisterPpuScan();

} // namespace

} // namespace tl
} // namespace tvm
