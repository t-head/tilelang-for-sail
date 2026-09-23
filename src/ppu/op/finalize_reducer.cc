/*!
 * \file tl/ppu/op/finalize_reducer.cc
 * \brief PPU implementation registration for tl.finalize_reducer AllReduce lowering.
 */

#include "backend/common/op/finalize_reducer.h"

#include "ppu/target_utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tirx;

namespace ppu {

struct FinalizeReducer : backend::FinalizeReducerLowerer<FinalizeReducer> {
  static int WarpSize(Target target) { return TargetPPUGetWarpSize(target); }

  static std::string MakeBatchAllReduce(std::string reducer,
                                        int reducing_threads, int scale,
                                        PrimExpr thread_offset,
                                        PrimExpr all_threads, int batch,
                                        int workspace_stride, Target target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset;
    ss << ", tl::SyncThreadsBarrier";
    ss << ", " << batch << ", " << workspace_stride << ">::run_batch";
    return ss.str();
  }

  static std::string MakeScalarAllReduce(std::string reducer,
                                         int reducing_threads, int scale,
                                         PrimExpr thread_offset,
                                         PrimExpr all_threads, Target target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset;
    ss << ">::run";
    return ss.str();
  }
};

} // namespace ppu

namespace {

bool MatchPpuFinalizeReducerTarget(Target target) { return TargetIsPPU(target); }

bool RegisterPpuFinalizeReducer() {
  RegisterFinalizeReducerImpl(FinalizeReducerImpl{
      "ppu.FinalizeReducer",
      MatchPpuFinalizeReducerTarget,
      ppu::FinalizeReducer::Lower,
  });
  return true;
}

const bool ppu_finalize_reducer_registered = RegisterPpuFinalizeReducer();

} // namespace

} // namespace tl
} // namespace tvm
