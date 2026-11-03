#pragma once

#include "common.h"
#include <cute/numeric/numeric_types.hpp>
#include <hggc_fp8.h>

using fp8_e4_t = tl::float_e4m3_t;
using fp8_e5_t = tl::float_e5m2_t;

using fp8_e8_t = __hg_fp8_e8m0;
#define TL_HAS_FP8_E8M0 1

#define TL_FP8_VECTOR_COPY(Type, Storage)                                      \
  TL_DEVICE Type() = default;                                                  \
  TL_DEVICE Type(const Type &other) { *this = other; }                         \
  TL_DEVICE Type &operator=(const Type &other) {                               \
    static_assert(sizeof(Type) == sizeof(Storage), "FP8 copy size mismatch");  \
    static_assert(alignof(Type) >= alignof(Storage),                           \
                  "FP8 copy alignment mismatch");                              \
    *reinterpret_cast<Storage *>(this) =                                       \
        *reinterpret_cast<const Storage *>(&other);                            \
    return *this;                                                              \
  }

struct __HGGC_ALIGN__(2) fp8_e4_2_t {
  fp8_e4_t x;
  fp8_e4_t y;

  TL_FP8_VECTOR_COPY(fp8_e4_2_t, uint16_t)
};

struct __HGGC_ALIGN__(4) fp8_e4_4_t {
  fp8_e4_t x;
  fp8_e4_t y;
  fp8_e4_t z;
  fp8_e4_t w;

  TL_FP8_VECTOR_COPY(fp8_e4_4_t, uint32_t)
};

struct __HGGC_ALIGN__(8) fp8_e4_8_t {
  fp8_e4_4_t x;
  fp8_e4_4_t y;

  TL_FP8_VECTOR_COPY(fp8_e4_8_t, uint2)
};

struct __HGGC_ALIGN__(16) fp8_e4_16_t {
  fp8_e4_8_t x;
  fp8_e4_8_t y;

  TL_FP8_VECTOR_COPY(fp8_e4_16_t, uint4)
};

struct __HGGC_ALIGN__(32) fp8_e4_32_t {
  fp8_e4_16_t x;
  fp8_e4_16_t y;

  TL_FP8_VECTOR_COPY(fp8_e4_32_t, ulonglong4)
};

struct __HGGC_ALIGN__(2) fp8_e5_2_t {
  fp8_e5_t x;
  fp8_e5_t y;

  TL_FP8_VECTOR_COPY(fp8_e5_2_t, uint16_t)
};

struct __HGGC_ALIGN__(4) fp8_e5_4_t {
  fp8_e5_t x;
  fp8_e5_t y;
  fp8_e5_t z;
  fp8_e5_t w;

  TL_FP8_VECTOR_COPY(fp8_e5_4_t, uint32_t)
};

struct __HGGC_ALIGN__(8) fp8_e5_8_t {
  fp8_e5_4_t x;
  fp8_e5_4_t y;

  TL_FP8_VECTOR_COPY(fp8_e5_8_t, uint2)
};

struct __HGGC_ALIGN__(16) fp8_e5_16_t {
  fp8_e5_8_t x;
  fp8_e5_8_t y;

  TL_FP8_VECTOR_COPY(fp8_e5_16_t, uint4)
};

struct __HGGC_ALIGN__(32) fp8_e5_32_t {
  fp8_e5_16_t x;
  fp8_e5_16_t y;

  TL_FP8_VECTOR_COPY(fp8_e5_32_t, ulonglong4)
};

struct __HGGC_ALIGN__(2) fp8_e8_2_t {
  fp8_e8_t x;
  fp8_e8_t y;

  TL_FP8_VECTOR_COPY(fp8_e8_2_t, uint16_t)
};

struct __HGGC_ALIGN__(4) fp8_e8_4_t {
  fp8_e8_t x;
  fp8_e8_t y;
  fp8_e8_t z;
  fp8_e8_t w;

  TL_FP8_VECTOR_COPY(fp8_e8_4_t, uint32_t)
};

struct __HGGC_ALIGN__(8) fp8_e8_8_t {
  fp8_e8_4_t x;
  fp8_e8_4_t y;

  TL_FP8_VECTOR_COPY(fp8_e8_8_t, uint2)
};

struct __HGGC_ALIGN__(16) fp8_e8_16_t {
  fp8_e8_8_t x;
  fp8_e8_8_t y;

  TL_FP8_VECTOR_COPY(fp8_e8_16_t, uint4)
};

struct __HGGC_ALIGN__(32) fp8_e8_32_t {
  fp8_e8_16_t x;
  fp8_e8_16_t y;

  TL_FP8_VECTOR_COPY(fp8_e8_32_t, ulonglong4)
};

#undef TL_FP8_VECTOR_COPY

// e4m3x2 -> float2
TL_DEVICE float2
__tl_cvt_fp8x2_to_float2(const __hg_fp8x2_storage_t x,
                         const __hg_fp8_interpretation_t fp8_interpretation) {
  half2 tmp = __hg_cvt_fp8x2_to_halfraw2(x, fp8_interpretation);
  float2 result;
  result.x = (float)tmp.x;
  result.y = (float)tmp.y;
  return result;
}

// e4m3x2/e5m2x2 -> half2
TL_DEVICE half2
__tl_cvt_fp8x2_to_half2(const __hg_fp8x2_storage_t x,
                        const __hg_fp8_interpretation_t fp8_interpretation) {
  return __hg_cvt_fp8x2_to_halfraw2(x, fp8_interpretation);
}

// half2 -> e4m3x2/e5m2x2
// No direct HG intrinsic; half -> float is exact, so compose through float2
// to keep a single rounding step at the fp8 conversion.
TL_DEVICE __hg_fp8x2_storage_t __tl_cvt_half2_to_fp8x2(
    const half2 x, const __hg_fp8_interpretation_t fp8_interpretation) {
  return __hg_cvt_float2_to_fp8x2(__half22float2(x), __HG_SATFINITE,
                                  fp8_interpretation);
}

// bfloat162 -> e4m3x2/e5m2x2
// bfloat16 -> float is exact, so compose through float2 for a single
// rounding step.
TL_DEVICE __hg_fp8x2_storage_t __tl_cvt_bfloat162_to_fp8x2(
    const __ppu_bfloat162 x,
    const __hg_fp8_interpretation_t fp8_interpretation) {
  return __hg_cvt_float2_to_fp8x2(__bfloat1622float2(x), __HG_SATFINITE,
                                  fp8_interpretation);
}

// e4m3x2/e5m2x2 -> bfloat162
// fp8 values are exactly representable in float, so composing through float2
// is exact.
TL_DEVICE __ppu_bfloat162 __tl_cvt_fp8x2_to_bfloat162(
    const __hg_fp8x2_storage_t x,
    const __hg_fp8_interpretation_t fp8_interpretation) {
  return __float22bfloat162_rn(__tl_cvt_fp8x2_to_float2(x, fp8_interpretation));
}

// ============================================================================
// FP8 <-> Half/BFloat16 Scalar Conversions
// ============================================================================

// fp8 (e4m3/e5m2) -> half. No scalar HG cvt exists; zero-extend into the
// paired hardware cvt and take the low lane.
TL_DEVICE half
__tl_cvt_fp8_to_half(const __hg_fp8_storage_t x,
                     const __hg_fp8_interpretation_t fp8_interpretation) {
  __hg_fp8x2_storage_t x2 = static_cast<__hg_fp8x2_storage_t>(x);
  return __hg_cvt_fp8x2_to_halfraw2(x2, fp8_interpretation).x;
}

// half -> fp8. half -> float is exact, so the only rounding happens at the
// fp8 hardware cvt (satfinite).
TL_DEVICE __hg_fp8_storage_t __tl_cvt_half_to_fp8(
    const half x, const __hg_fp8_interpretation_t fp8_interpretation) {
  __hg_fp8x2_storage_t r = __hg_cvt_float2_to_fp8x2(
      make_float2(__half2float(x), 0.0f), __HG_SATFINITE, fp8_interpretation);
  return static_cast<__hg_fp8_storage_t>(r & 0xFF);
}

// fp8 -> bfloat16. fp8 -> half -> float is exact, and float -> bf16 of an
// exact fp8 value is exact (bf16's 7 mantissa bits cover fp8's 3).
TL_DEVICE __ppu_bfloat16
__tl_cvt_fp8_to_bfloat16(const __hg_fp8_storage_t x,
                         const __hg_fp8_interpretation_t fp8_interpretation) {
  return __float2bfloat16(
      __half2float(__tl_cvt_fp8_to_half(x, fp8_interpretation)));
}

// bfloat16 -> fp8. bf16 -> f32 is an exact 16-bit shift; the only rounding
// happens at the fp8 hardware cvt (satfinite).
TL_DEVICE __hg_fp8_storage_t
__tl_cvt_bfloat16_to_fp8(const __ppu_bfloat16 x,
                         const __hg_fp8_interpretation_t fp8_interpretation) {
  __ppu_bfloat16_raw raw = *reinterpret_cast<const __ppu_bfloat16_raw *>(&x);
  const unsigned int u = ((unsigned int)raw.x) << 16U;
  const float f = *reinterpret_cast<const float *>(&u);
  __hg_fp8x2_storage_t r = __hg_cvt_float2_to_fp8x2(
      make_float2(f, 0.0f), __HG_SATFINITE, fp8_interpretation);
  return static_cast<__hg_fp8_storage_t>(r & 0xFF);
}

// ============================================================================
// FP8 E8M0 Related Conversions
// ============================================================================
#if TL_HAS_FP8_E8M0

// fp8_e8m0 -> bfloat16
TL_DEVICE __ppu_bfloat16
__tl_cvt_e8m0_to_bfloat16(const __hg_fp8_storage_t src) {
  __ppu_bfloat16_raw raw = __hg_cvt_e8m0_to_bf16raw(src);
  return *reinterpret_cast<const __ppu_bfloat16 *>(&raw);
}

// fp8_e8m0x2 -> bfloat16x2
TL_DEVICE __ppu_bfloat162
__tl_cvt_e8m0x2_to_bfloat162(const __hg_fp8x2_storage_t src) {
  __ppu_bfloat162_raw raw = __hg_cvt_e8m0x2_to_bf162raw(src);
  return *reinterpret_cast<const __ppu_bfloat162 *>(&raw);
}

// bfloat16 -> fp8_e8m0
TL_DEVICE
__hg_fp8_storage_t __tl_cvt_bfloat16_to_e8m0(const __ppu_bfloat16 src) {
  __ppu_bfloat16_raw raw = *reinterpret_cast<const __ppu_bfloat16_raw *>(&src);
  return __hg_cvt_bfloat16raw_to_e8m0(raw, __HG_SATFINITE, hggcRoundPosInf);
}

// bfloat162 -> fp8_e8m0x2
TL_DEVICE __hg_fp8x2_storage_t
__tl_cvt_bfloat162_to_e8m0x2(const __ppu_bfloat162 src) {
  __ppu_bfloat162_raw raw =
      *reinterpret_cast<const __ppu_bfloat162_raw *>(&src);
  return __hg_cvt_bfloat162raw_to_e8m0x2(raw, __HG_SATFINITE, hggcRoundPosInf);
}

// float -> fp8_e8m0
TL_DEVICE __hg_fp8_storage_t __tl_cvt_float_to_e8m0(const float src) {
  return __hg_cvt_float_to_e8m0(src, __HG_SATFINITE, hggcRoundPosInf);
}

// float2 -> fp8_e8m0x2
TL_DEVICE __hg_fp8x2_storage_t __tl_cvt_float2_to_e8m0x2(const float2 src) {
  return __hg_cvt_float2_to_e8m0x2(src, __HG_SATFINITE, hggcRoundPosInf);
}

// double -> fp8_e8m0
TL_DEVICE __hg_fp8_storage_t __tl_cvt_double_to_e8m0(const double src) {
  return __hg_cvt_double_to_e8m0(src, __HG_SATFINITE, hggcRoundPosInf);
}

// double2 -> fp8_e8m0x2
TL_DEVICE __hg_fp8x2_storage_t __tl_cvt_double2_to_e8m0x2(const double2 src) {
  return __hg_cvt_double2_to_e8m0x2(src, __HG_SATFINITE, hggcRoundPosInf);
}

#endif
