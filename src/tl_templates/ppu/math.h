#pragma once

#include "common.h"

#include <cutlass/fast_math.h>

#define hexp cutlass::fast_exp
#define hlog cutlass::fast_log
#define hsqrt cutlass::fast_sqrt
#define hsin cutlass::fast_sin
#define hcos cutlass::fast_cos

namespace cutlass {
TL_DEVICE
bfloat16_t fast_exp(bfloat16_t x) { return bfloat16_t(fast_exp(float(x))); }

TL_DEVICE
half_t fast_log(half_t x) { return half_t(fast_log(float(x))); }

TL_DEVICE
bfloat16_t fast_log(bfloat16_t x) { return bfloat16_t(fast_log(float(x))); }

TL_DEVICE
half_t fast_sqrt(half_t x) { return half_t(fast_sqrt(float(x))); }

TL_DEVICE
bfloat16_t fast_sqrt(bfloat16_t x) { return bfloat16_t(fast_sqrt(float(x))); }

TL_DEVICE
half_t fast_sin(half_t x) { return half_t(fast_sin(float(x))); }

TL_DEVICE
bfloat16_t fast_sin(bfloat16_t x) { return bfloat16_t(fast_sin(float(x))); }

TL_DEVICE
half_t fast_cos(half_t x) { return half_t(fast_cos(float(x))); }

TL_DEVICE
bfloat16_t fast_cos(bfloat16_t x) { return bfloat16_t(fast_cos(float(x))); }

TL_DEVICE
bfloat16_t fast_tanh(bfloat16_t x) { return bfloat16_t(fast_tanh(float(x))); }
} // namespace cutlass

namespace tl {
TL_DEVICE float fast_rcp(float x) {
  float ret;
  asm volatile("ppu.rcp.approx.ftz.f32 %0, %1;" : "=f"(ret) : "f"(x));
  return ret;
}

// Pass operands by value: checking NaN must not re-evaluate the caller's
// expression (which can, for example, be a returning atomic operation).
// NaN detection uses the self-inequality idiom, which needs no isnan
// intrinsic on the device.
TL_DEVICE float clamp(float x, float lo, float hi) {
  if (x != x)
    return x;
  if (lo != lo)
    return lo;
  if (hi != hi)
    return hi;
  return fminf(fmaxf(x, lo), hi);
}

TL_DEVICE double clamp(double x, double lo, double hi) {
  if (x != x)
    return x;
  if (lo != lo)
    return lo;
  if (hi != hi)
    return hi;
  return fmin(fmax(x, lo), hi);
}

// Narrow formats, including FP8/FP16/BF16, are exactly representable in
// float.
template <typename T> TL_DEVICE T clamp(T x, T lo, T hi) {
  return T(clamp(float(x), float(lo), float(hi)));
}

TL_DEVICE __half2 clamp2(__half2 x, __half2 lo, __half2 hi) {
  return min2_nan(max2_nan(x, lo), hi);
}

TL_DEVICE __ppu_bfloat162 clamp2(__ppu_bfloat162 x, __ppu_bfloat162 lo,
                                 __ppu_bfloat162 hi) {
  return min2_nan(max2_nan(x, lo), hi);
}
} // namespace tl
