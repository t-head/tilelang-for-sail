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
// intrinsic on the device. NaN in either operand propagates to the result;
// numeric lanes clamp to [-limit, limit], preserving the sign of zero. The
// caller supplies a positive limit, matching the CUDA clamp_symmetric.
TL_DEVICE float clamp_symmetric(float x, float limit) {
  if (x != x)
    return x;
  if (limit != limit)
    return limit;
  return fminf(limit, fmaxf(-limit, x));
}

// Narrow formats, including FP8/FP16/BF16, are exactly representable in
// float.
template <typename T> TL_DEVICE T clamp_symmetric(T x, T limit) {
  return T(clamp_symmetric(float(x), float(limit)));
}

// Flip the per-lane sign bits of a packed x2 operand.
TL_DEVICE __half2 neg2(__half2 a) {
  uint1 bits = to_uint1(a);
  bits.x ^= 0x80008000u;
  return from_uint1<__half2>(bits);
}

TL_DEVICE __ppu_bfloat162 neg2(__ppu_bfloat162 a) {
  uint1 bits = to_uint1(a);
  bits.x ^= 0x80008000u;
  return from_uint1<__ppu_bfloat162>(bits);
}

TL_DEVICE __half2 clamp_symmetric2(__half2 x, __half2 limit) {
  return min2_nan(max2_nan(neg2(limit), x), limit);
}

TL_DEVICE __ppu_bfloat162 clamp_symmetric2(__ppu_bfloat162 x,
                                           __ppu_bfloat162 limit) {
  return min2_nan(max2_nan(neg2(limit), x), limit);
}
} // namespace tl
