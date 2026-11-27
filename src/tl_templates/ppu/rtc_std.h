/*
 * Standard library shim for hgrtc, the PPU run-time compiler.
 *
 * hgrtc has no host libstdc++ to fall back on, and the SDK's device-side
 * standard library (hggc/std) keeps its names in hggc::std once any of its
 * wrapper headers has been included. The PPU templates spell those names as
 * std::, so pull in the subset they use. This header is only reachable in
 * run-time-compiled translation units and needs the SDK include root on the
 * search path, which tilelang/contrib/hgrtc.py always provides.
 */

#pragma once

#if defined(__HGGCCC_RTC__)

#include <hggc/std/cstdint>
#include <hggc/std/limits>
#include <hggc/std/type_traits>
#include <hggc/std/utility>

// Adding names to namespace std is reserved for the implementation. It is
// confined to run-time-compiled units, which carry no host standard library
// that these names could clash with.
namespace std {

using hggc::std::conditional_t;
using hggc::std::extent_v;
using hggc::std::index_sequence;
using hggc::std::integral_constant;
using hggc::std::is_arithmetic_v;
using hggc::std::is_integral_v;
using hggc::std::is_pointer_v;
using hggc::std::is_same_v;
using hggc::std::is_void_v;
using hggc::std::make_index_sequence;
using hggc::std::numeric_limits;
using hggc::std::remove_extent_t;

} // namespace std

// The SDK's cooperative_groups / hggc_pipeline headers cast pointers to
// uintptr_t in the global namespace and rely on the host <cstdint> for it.
using hggc::std::uintptr_t;

#endif // __HGGCCC_RTC__
