/*!
 * \file tl/ppu/target_utils.cc
 * \brief PPU target attribute helpers.
 */

#include "ppu/target_utils.h"

#include <tvm/ffi/reflection/registry.h>

#include <string>

#include "dlpack/dlpack.h"
#include "support/check.h"

namespace tvm {
namespace tl {
namespace {

int GetPPUArchInt(Target target) {
  if (!TargetIsPPU(target))
    return 0;
  auto s = target->GetAttr<ffi::String>("arch");
  ICHECK(s.has_value());
  const std::string arch_str = s.value();
  size_t pos = arch_str.rfind('_');
  ICHECK(pos != std::string::npos && pos + 1 < arch_str.size())
      << "arch string must contain '_' followed by a version number";
  const std::string suffix = arch_str.substr(pos + 1);
  ICHECK(!suffix.empty() &&
         suffix.find_first_not_of("0123456789") == std::string::npos)
      << "arch version suffix must be a number";
  return std::stoi(suffix);
}

} // namespace

bool TargetIsPPU(Target target) {
  return target->GetTargetDeviceType() == kDLPPU;
}

bool TargetHasAiuCopy(Target target) {
  if (!TargetIsPPU(target))
    return false;
  int arch = GetPPUArchInt(target);
  return arch >= 15;
}

bool TargetPPUHasAsyncCopy(Target target) {
  if (!TargetIsPPU(target))
    return false;
  return true;
}

int TargetPPUGetWarpSize(Target target) {
  (void)target;
  return 32;
}

bool TargetPPUHasLdmatrix(Target target) {
  if (!TargetIsPPU(target))
    return false;
  return true;
}

bool TargetPPUHasStmatrix(Target target) {
  return false;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.TargetIsPPU",
           [](Target target) { return TargetIsPPU(target); })
      .def("tl.TargetHasAiuCopy",
           [](Target target) { return TargetHasAiuCopy(target); })
      .def("tl.TargetPPUGetWarpSize",
           [](Target target) { return TargetPPUGetWarpSize(target); })
      .def("tl.TargetPPUHasAsyncCopy",
           [](Target target) { return TargetPPUHasAsyncCopy(target); })
      .def("tl.TargetPPUHasLdmatrix",
           [](Target target) { return TargetPPUHasLdmatrix(target); })
      .def("tl.TargetPPUHasStmatrix",
           [](Target target) { return TargetPPUHasStmatrix(target); });
}

} // namespace tl
} // namespace tvm
