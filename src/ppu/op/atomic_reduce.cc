/*!
 * \file tl/ppu/op/atomic_reduce.cc
 * \brief PPU implementation registration for tl.atomicmax/tl.atomicmin lowering.
 */

#include "backend/common/op/atomic_reduce.h"

#include "backend/common/target_utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchPpuAtomicReduceTarget(Target target) { return TargetIsPPU(target); }

bool RegisterPpuAtomicReduce() {
  RegisterAtomicReduceImpl(AtomicReduceImpl{
      "ppu.AtomicReduce",
      MatchPpuAtomicReduceTarget,
      backend::AtomicReduce::InferLayout,
      backend::AtomicReduce::Lower,
  });
  return true;
}

const bool ppu_atomic_reduce_registered = RegisterPpuAtomicReduce();

} // namespace

} // namespace tl
} // namespace tvm
