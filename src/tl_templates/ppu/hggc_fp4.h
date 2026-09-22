#pragma once

#include "common.h"

// Fp4 Support
#include <hggc_fp16.h>
#include <hggc_bf16.h>
#include <hggc_fp8.h> // brings in enum hggcRoundMode (hgrt/hggc_device_types.h)

// ============================================================================
// FP4 core types (host + device visible)
// ============================================================================

typedef unsigned char __hg_fp4_storage_t;
typedef unsigned char __hg_fp4x2_storage_t;
typedef unsigned short __hg_fp4x4_storage_t;

typedef enum __hg_fp4_interpretation_t {
  __HG_E2M1 = 0, /**< Stands for fp4 numbers of e2m1 kind. */
} __hg_fp4_interpretation_t;

// ============================================================================
// FP4 conversions (device only)
// ============================================================================

#if defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100)

// double -> fp4_e2m1, round-to-nearest-even (or towards-zero), satfinite
TL_DEVICE __hg_fp4_storage_t
__hg_cvt_double_to_fp4(const double x,
                       const __hg_fp4_interpretation_t fp4_interpretation,
                       const enum hggcRoundMode rounding) {
  (void)fp4_interpretation; // only __HG_E2M1 is defined
  unsigned char res;
  unsigned long long int xbits;
  (void)memcpy(&xbits, &x, sizeof(x));

  // fp4_interpretation == __HG_E2M1
  const unsigned char FP4_MAXNORM = 0x7U;
  const unsigned char FP4_MANTISSA_MASK = 0x1U;
  const unsigned short int FP4_EXP_BIAS = 1U;
  const unsigned long long int FP4_SIGNIFICAND_BITS = 2ULL;
  const unsigned long long int FP4_MINDENORM_O2 =
      0x3FD0000000000000ULL; // mindenorm/2 = 2^-2
  const unsigned long long int FP4_OVERFLOW_THRESHOLD =
      0x4018000000000000ULL; // maxnorm = 6.0
  const unsigned long long int FP4_MINNORM =
      0x3FF0000000000000ULL; // minnorm = 2^0
  const unsigned long long int DP_INF_BITS = 0x7FF0000000000000ULL;

  // 1/2 LSB of the target format, positioned in double precision mantissa
  // helpful in midpoints detection during round-to-nearest-even step
  const unsigned long long int FP4_DP_HALF_ULP =
      1ULL << (53ULL - FP4_SIGNIFICAND_BITS - 1ULL);
  // prepare sign bit in target format
  unsigned char sign = (unsigned char)((xbits >> 63ULL) << 3U);
  // prepare exponent field in target format
  const unsigned char exp =
      (unsigned char)((((unsigned short int)(xbits >> 52ULL)) & 0x7FFU) -
                      1023U + FP4_EXP_BIAS);
  // round mantissa to target format width, rounding towards zero
  const unsigned char mantissa =
      (unsigned char)(xbits >> (53ULL - FP4_SIGNIFICAND_BITS)) &
      FP4_MANTISSA_MASK;
  const unsigned long long int absx = xbits & 0x7FFFFFFFFFFFFFFFULL;

  if (absx <= FP4_MINDENORM_O2) {
    // zero or underflow
    res = 0U;
  } else if (absx > FP4_OVERFLOW_THRESHOLD) {
    // overflow or NaN
    if (absx > DP_INF_BITS) {
      // NaN converts to positive FP4_MAXNORM
      sign = 0U;
    }
    res = FP4_MAXNORM;
  } else if (absx >= FP4_MINNORM) {
    res = (unsigned char)((exp << (FP4_SIGNIFICAND_BITS - 1U)) | mantissa);
    // rounded-off bits
    const unsigned long long int round =
        xbits & ((FP4_DP_HALF_ULP << 1ULL) - 1ULL);
    if (rounding == hggcRoundNearest) {
      // round-to-nearest-even adjustment
      if ((round > FP4_DP_HALF_ULP) ||
          ((round == FP4_DP_HALF_ULP) && (mantissa & 1U))) {
        res = (unsigned char)(res + 1U);
      }
    }
  } else { // Denormal range
    const unsigned char shift = (unsigned char)(1U - exp);
    // add implicit leading bit, then additional round-off due to
    // denormalization
    res = (unsigned char)((mantissa | (1U << (FP4_SIGNIFICAND_BITS - 1U))) >>
                          shift);
    if (rounding == hggcRoundNearest) {
      // rounded-off bits, including implicit leading bit
      const unsigned long long int round =
          (xbits | (1ULL << (53ULL - 1ULL))) &
          ((FP4_DP_HALF_ULP << (shift + 1ULL)) - 1ULL);
      // round-to-nearest-even adjustment
      if ((round > (FP4_DP_HALF_ULP << shift)) ||
          ((round == (FP4_DP_HALF_ULP << shift)) && (res & 1U))) {
        res = (unsigned char)(res + 1U);
      }
    }
  }

  res |= sign;

  return (__hg_fp4_storage_t)res;
}

// double2 -> fp4_e2m1x2
TL_DEVICE __hg_fp4x2_storage_t
__hg_cvt_double2_to_fp4x2(const double2 x,
                          const __hg_fp4_interpretation_t fp4_interpretation,
                          const enum hggcRoundMode rounding) {
  __hg_fp4x2_storage_t storage = (__hg_fp4x2_storage_t)__hg_cvt_double_to_fp4(
      x.y, fp4_interpretation, rounding);
  storage = (__hg_fp4x2_storage_t)(storage << 4U);
  storage = (__hg_fp4x2_storage_t)(storage |
                                   __hg_cvt_double_to_fp4(x.x,
                                                          fp4_interpretation,
                                                          rounding));
  return storage;
}

// float -> fp4_e2m1 (via double, exact widening)
TL_DEVICE __hg_fp4_storage_t
__hg_cvt_float_to_fp4(const float x,
                      const __hg_fp4_interpretation_t fp4_interpretation,
                      const enum hggcRoundMode rounding) {
  return __hg_cvt_double_to_fp4((double)x, fp4_interpretation, rounding);
}

// float2 -> fp4_e2m1x2
TL_DEVICE __hg_fp4x2_storage_t
__hg_cvt_float2_to_fp4x2(const float2 x,
                         const __hg_fp4_interpretation_t fp4_interpretation,
                         const enum hggcRoundMode rounding) {
  __hg_fp4x2_storage_t storage = (__hg_fp4x2_storage_t)__hg_cvt_float_to_fp4(
      x.y, fp4_interpretation, rounding);
  storage = (__hg_fp4x2_storage_t)(storage << 4U);
  storage = (__hg_fp4x2_storage_t)(storage |
                                   __hg_cvt_float_to_fp4(x.x,
                                                         fp4_interpretation,
                                                         rounding));
  return storage;
}

// half -> fp4_e2m1 (via float)
TL_DEVICE __hg_fp4_storage_t
__hg_cvt_halfraw_to_fp4(const __half_raw x,
                        const __hg_fp4_interpretation_t fp4_interpretation,
                        const enum hggcRoundMode rounding) {
  const float fx = __half2float(*reinterpret_cast<const __half *>(&x));
  return __hg_cvt_float_to_fp4(fx, fp4_interpretation, rounding);
}

// half2 -> fp4_e2m1x2
TL_DEVICE __hg_fp4x2_storage_t
__hg_cvt_halfraw2_to_fp4x2(const __half2_raw x,
                           const __hg_fp4_interpretation_t fp4_interpretation,
                           const enum hggcRoundMode rounding) {
  __half_raw raw;
  raw.x = x.x;
  const __hg_fp4_storage_t lo =
      __hg_cvt_halfraw_to_fp4(raw, fp4_interpretation, rounding);
  raw.x = x.y;
  const __hg_fp4_storage_t hi =
      __hg_cvt_halfraw_to_fp4(raw, fp4_interpretation, rounding);
  return (__hg_fp4x2_storage_t)((hi << 4U) | lo);
}

// bfloat16 -> fp4_e2m1 (via float; bf16 -> f32 is an exact 16-bit shift)
TL_DEVICE __hg_fp4_storage_t
__hg_cvt_bfloat16raw_to_fp4(const __ppu_bfloat16_raw x,
                            const __hg_fp4_interpretation_t fp4_interpretation,
                            const enum hggcRoundMode rounding) {
  const unsigned int u = ((unsigned int)x.x) << 16U;
  const float fx = *reinterpret_cast<const float *>(&u);
  return __hg_cvt_float_to_fp4(fx, fp4_interpretation, rounding);
}

// bfloat162 -> fp4_e2m1x2
TL_DEVICE __hg_fp4x2_storage_t __hg_cvt_bfloat16raw2_to_fp4x2(
    const __ppu_bfloat162_raw x,
    const __hg_fp4_interpretation_t fp4_interpretation,
    const enum hggcRoundMode rounding) {
  __ppu_bfloat16_raw raw;
  raw.x = x.y;
  __hg_fp4x2_storage_t storage =
      (__hg_fp4x2_storage_t)__hg_cvt_bfloat16raw_to_fp4(raw,
                                                        fp4_interpretation,
                                                        rounding);
  storage = (__hg_fp4x2_storage_t)(storage << 4U);
  raw.x = x.x;
  storage = (__hg_fp4x2_storage_t)(storage |
                                   __hg_cvt_bfloat16raw_to_fp4(raw,
                                                               fp4_interpretation,
                                                               rounding));
  return storage;
}

// fp4_e2m1 -> half. Exact: e2m1 has only 8 magnitudes, all representable in
// f16. Normal e2m1 exponent rebases by +14 (half bias 15 vs e2m1 bias 1);
// the only subnormal pattern is 0.5.
TL_DEVICE __half_raw
__hg_cvt_fp4_to_halfraw(const __hg_fp4_storage_t x,
                        const __hg_fp4_interpretation_t fp4_interpretation) {
  (void)fp4_interpretation; // only __HG_E2M1 is defined
  __half_raw res;
  const unsigned int sign = ((unsigned int)x & 0x8U) << 12U;
  const unsigned int exp = ((unsigned int)x >> 1U) & 0x3U;
  const unsigned int man = (unsigned int)x & 0x1U;
  unsigned int bits;
  if (exp == 0U) {
    bits = man ? 0x3800U : 0x0000U; // 0.5 or 0.0
  } else {
    bits = ((exp + 14U) << 10U) | (man << 9U);
  }
  res.x = (unsigned short)(bits | sign);
  return res;
}

// fp4_e2m1x2 -> half2
TL_DEVICE __half2_raw
__hg_cvt_fp4x2_to_halfraw2(const __hg_fp4x2_storage_t x,
                           const __hg_fp4_interpretation_t fp4_interpretation) {
  __half2_raw res;
  res.x =
      __hg_cvt_fp4_to_halfraw((__hg_fp4_storage_t)x, fp4_interpretation).x;
  res.y = __hg_cvt_fp4_to_halfraw((__hg_fp4_storage_t)(x >> 4U),
                                  fp4_interpretation)
              .x;
  return res;
}

#endif // defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100)

// ============================================================================
// C++ fp4 structs (host + device visible)
// ============================================================================

struct __HGGC_ALIGN__(1) __hg_fp4_e2m1 {
  __hg_fp4_storage_t __x;

  __hg_fp4_e2m1() = default;

#if defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100)
  // Constructor from float, satfinite + round-to-nearest-even
  TL_DEVICE explicit __hg_fp4_e2m1(const float f) {
    __x = __hg_cvt_float_to_fp4(f, __HG_E2M1, hggcRoundNearest);
  }

  // Constructor from double, satfinite + round-to-nearest-even
  TL_DEVICE explicit __hg_fp4_e2m1(const double f) {
    __x = __hg_cvt_double_to_fp4(f, __HG_E2M1, hggcRoundNearest);
  }

  // Conversion to float (via half)
  TL_DEVICE explicit operator float() const {
    const __half_raw raw = __hg_cvt_fp4_to_halfraw(__x, __HG_E2M1);
    return __half2float(*reinterpret_cast<const __half *>(&raw));
  }
#endif // __HGGC_ARCH__
};

struct __HGGC_ALIGN__(1) __hg_fp4x2_e2m1 {
  __hg_fp4x2_storage_t __x;

  __hg_fp4x2_e2m1() = default;
};

// ============================================================================
// tilelang FP4 wrappers (host + device visible)
// ============================================================================

// Wrapper for __hg_fp4_e2m1 with implicit conversions
struct fp4_e2_t {
  __hg_fp4_storage_t __x;

  TL_DEVICE fp4_e2_t() = default;

  // Constructor from __hg_fp4_e2m1
  TL_DEVICE fp4_e2_t(__hg_fp4_e2m1 x) : __x(x.__x) {}

  // Constructor from storage type
  TL_DEVICE fp4_e2_t(__hg_fp4_storage_t x) : __x(x) {}

#if defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100)
  // Constructor from float
  TL_DEVICE explicit fp4_e2_t(float x) {
    __hg_fp4_e2m1 tmp(x);
    __x = tmp.__x;
  }

  // Conversion to __hg_fp4_e2m1
  TL_DEVICE operator __hg_fp4_e2m1() const {
    __hg_fp4_e2m1 tmp;
    tmp.__x = __x;
    return tmp;
  }

  // Conversion to float
  TL_DEVICE operator float() const {
    __hg_fp4_e2m1 tmp;
    tmp.__x = __x;
    return float(tmp);
  }

  // Implicit conversion to half_t (cutlass::half_t)
  TL_DEVICE operator half_t() const { return half_t(float(*this)); }

  // Implicit conversion to __half
  TL_DEVICE operator __half() const { return __half(float(*this)); }
#endif // __HGGC_ARCH__
};

// Tag for tcgen05 unpacked FP4 shared-memory layout. The hardware atom carries
// 16 4-bit payload values in the low 64 bits of a 128-bit aligned region.
struct float4_e2m1_unpacked_t {
  uint8_t __x;
};

class fp4_e2_2_t {
public:
  __hg_fp4x2_storage_t __x;

  TL_DEVICE fp4_e2_2_t() = default;
  TL_DEVICE fp4_e2_2_t(__hg_fp4x2_storage_t data) : __x(data) {}
  TL_DEVICE fp4_e2_2_t(__hg_fp4x2_e2m1 data) : __x(data.__x) {}

  // Get low 4 bits (first fp4)
  TL_DEVICE fp4_e2_t x() const {
    return fp4_e2_t(__hg_fp4_storage_t(__x & 0x0F));
  }

  // Get high 4 bits (second fp4)
  TL_DEVICE fp4_e2_t y() const {
    return fp4_e2_t(__hg_fp4_storage_t((__x >> 4) & 0x0F));
  }

  // Set low 4 bits (first fp4)
  TL_DEVICE void set_x(fp4_e2_t val) { __x = (__x & 0xF0) | (val.__x & 0x0F); }

  // Set high 4 bits (second fp4)
  TL_DEVICE void set_y(fp4_e2_t val) {
    __x = (__x & 0x0F) | ((val.__x & 0x0F) << 4);
  }
};

struct __HGGC_ALIGN__(2) fp4_e2_4_t {
  fp4_e2_2_t x;
  fp4_e2_2_t y;
};

struct __HGGC_ALIGN__(4) fp4_e2_8_t {
  fp4_e2_4_t x;
  fp4_e2_4_t y;
};

struct __HGGC_ALIGN__(8) fp4_e2_16_t {
  fp4_e2_8_t x;
  fp4_e2_8_t y;
};

struct __HGGC_ALIGN__(16) fp4_e2_32_t {
  fp4_e2_16_t x;
  fp4_e2_16_t y;

  TL_DEVICE fp4_e2_32_t &operator=(const ulonglong4 &rhs) {
    x.x = *(fp4_e2_8_t *)&rhs.x;
    x.y = *(fp4_e2_8_t *)&rhs.y;
    y.x = *(fp4_e2_8_t *)&rhs.z;
    y.y = *(fp4_e2_8_t *)&rhs.w;
    return *this;
  }
};

struct __HGGC_ALIGN__(32) fp4_e2_64_t {
  fp4_e2_32_t x;
  fp4_e2_32_t y;
};

// Pack two fp4_e2_t values.
TL_DEVICE fp4_e2_2_t make_fp4_e2_2_t(fp4_e2_t x, fp4_e2_t y) {
  __hg_fp4x2_storage_t packed = (x.__x & 0x0F) | ((y.__x & 0x0F) << 4);
  fp4_e2_2_t result;
  result.__x = packed;
  return result;
}

// Pack four fp4_e2_t values.
TL_DEVICE fp4_e2_4_t make_fp4_e2_4_t(fp4_e2_t x0, fp4_e2_t x1, fp4_e2_t x2,
                                     fp4_e2_t x3) {
  fp4_e2_4_t result;
  result.x = make_fp4_e2_2_t(x0, x1);
  result.y = make_fp4_e2_2_t(x2, x3);
  return result;
}

// Pack eight fp4_e2_t values.
TL_DEVICE fp4_e2_8_t make_fp4_e2_8_t(fp4_e2_t x0, fp4_e2_t x1, fp4_e2_t x2,
                                     fp4_e2_t x3, fp4_e2_t x4, fp4_e2_t x5,
                                     fp4_e2_t x6, fp4_e2_t x7) {
  fp4_e2_8_t result;
  result.x = make_fp4_e2_4_t(x0, x1, x2, x3);
  result.y = make_fp4_e2_4_t(x4, x5, x6, x7);
  return result;
}

// Pack sixteen fp4_e2_t values.
TL_DEVICE fp4_e2_16_t make_fp4_e2_16_t(fp4_e2_t x0, fp4_e2_t x1, fp4_e2_t x2,
                                       fp4_e2_t x3, fp4_e2_t x4, fp4_e2_t x5,
                                       fp4_e2_t x6, fp4_e2_t x7, fp4_e2_t y0,
                                       fp4_e2_t y1, fp4_e2_t y2, fp4_e2_t y3,
                                       fp4_e2_t y4, fp4_e2_t y5, fp4_e2_t y6,
                                       fp4_e2_t y7) {
  fp4_e2_16_t result;
  result.x = make_fp4_e2_8_t(x0, x1, x2, x3, x4, x5, x6, x7);
  result.y = make_fp4_e2_8_t(y0, y1, y2, y3, y4, y5, y6, y7);
  return result;
}

// Pack thirty-two fp4_e2_t values.
TL_DEVICE fp4_e2_32_t make_fp4_e2_32_t(
    fp4_e2_t x0, fp4_e2_t x1, fp4_e2_t x2, fp4_e2_t x3, fp4_e2_t x4,
    fp4_e2_t x5, fp4_e2_t x6, fp4_e2_t x7, fp4_e2_t x8, fp4_e2_t x9,
    fp4_e2_t x10, fp4_e2_t x11, fp4_e2_t x12, fp4_e2_t x13, fp4_e2_t x14,
    fp4_e2_t x15, fp4_e2_t y0, fp4_e2_t y1, fp4_e2_t y2, fp4_e2_t y3,
    fp4_e2_t y4, fp4_e2_t y5, fp4_e2_t y6, fp4_e2_t y7, fp4_e2_t y8,
    fp4_e2_t y9, fp4_e2_t y10, fp4_e2_t y11, fp4_e2_t y12, fp4_e2_t y13,
    fp4_e2_t y14, fp4_e2_t y15) {
  fp4_e2_32_t result;
  result.x = make_fp4_e2_16_t(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11,
                              x12, x13, x14, x15);
  result.y = make_fp4_e2_16_t(y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11,
                              y12, y13, y14, y15);
  return result;
}

// ============================================================================
// FP4 <-> Half/Float/Double/BFloat16 Conversions (device only)
// ============================================================================

#if defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100)

// fp4_e2m1 -> half
TL_DEVICE __half __tl_cvt_fp4_to_half(const __hg_fp4_storage_t src) {
  __half_raw raw = __hg_cvt_fp4_to_halfraw(src, __HG_E2M1);
  __half result;
  result = *reinterpret_cast<__half *>(&raw);
  return result;
}

// fp4_e2m1x2 (1 byte) -> half2
TL_DEVICE half2 __tl_cvt_fp4x2_to_half2(const __hg_fp4x2_storage_t src) {
  __half2_raw raw = __hg_cvt_fp4x2_to_halfraw2(src, __HG_E2M1);
  half2 result;
  result = *reinterpret_cast<half2 *>(&raw);
  return result;
}

// half -> fp4_e2m1
TL_DEVICE __hg_fp4_storage_t __tl_cvt_half_to_fp4(const __half src) {
  __half_raw raw = *reinterpret_cast<const __half_raw *>(&src);
  return __hg_cvt_halfraw_to_fp4(raw, __HG_E2M1, hggcRoundNearest);
}

// half2 -> fp4_e2m1x2 (1 byte)
TL_DEVICE __hg_fp4x2_storage_t __tl_cvt_half2_to_fp4x2(const half2 src) {
  __half2_raw raw = *reinterpret_cast<const __half2_raw *>(&src);
  return __hg_cvt_halfraw2_to_fp4x2(raw, __HG_E2M1, hggcRoundNearest);
}

// ============================================================================
// FP4 <-> Float Conversions
// ============================================================================

// fp4_e2m1 -> float
TL_DEVICE float __tl_cvt_fp4_to_float(const __hg_fp4_storage_t src) {
  return __half2float(__tl_cvt_fp4_to_half(src));
}

// fp4_e2m1x2 (1 byte) -> float2
TL_DEVICE float2 __tl_cvt_fp4x2_to_float2(const __hg_fp4x2_storage_t src) {
  half2 tmp = __tl_cvt_fp4x2_to_half2(src);
  float2 result;
  result.x = __half2float(tmp.x);
  result.y = __half2float(tmp.y);
  return result;
}

// float -> fp4_e2m1
TL_DEVICE __hg_fp4_storage_t __tl_cvt_float_to_fp4(const float src) {
  return __hg_cvt_float_to_fp4(src, __HG_E2M1, hggcRoundNearest);
}

// float2 -> fp4_e2m1x2 (1 byte)
TL_DEVICE __hg_fp4x2_storage_t __tl_cvt_float2_to_fp4x2(const float2 src) {
  return __hg_cvt_float2_to_fp4x2(src, __HG_E2M1, hggcRoundNearest);
}

// ============================================================================
// FP4 <-> Double Conversions
// ============================================================================

// fp4_e2m1 -> double
TL_DEVICE double __tl_cvt_fp4_to_double(const __hg_fp4_storage_t src) {
  return static_cast<double>(__tl_cvt_fp4_to_float(src));
}

// fp4_e2m1x2 -> double2
TL_DEVICE double2 __tl_cvt_fp4x2_to_double2(const __hg_fp4x2_storage_t src) {
  float2 tmp = __tl_cvt_fp4x2_to_float2(src);
  double2 result;
  result.x = static_cast<double>(tmp.x);
  result.y = static_cast<double>(tmp.y);
  return result;
}

// double -> fp4_e2m1
TL_DEVICE __hg_fp4_storage_t __tl_cvt_double_to_fp4(const double src) {
  return __hg_cvt_double_to_fp4(src, __HG_E2M1, hggcRoundNearest);
}

// double2 -> fp4_e2m1x2
TL_DEVICE __hg_fp4x2_storage_t __tl_cvt_double2_to_fp4x2(const double2 src) {
  return __hg_cvt_double2_to_fp4x2(src, __HG_E2M1, hggcRoundNearest);
}

// ============================================================================
// FP4 <-> BFloat16 Conversions
// ============================================================================

// fp4_e2m1 -> bfloat16
TL_DEVICE __ppu_bfloat16 __tl_cvt_fp4_to_bfloat16(const __hg_fp4_storage_t src) {
  return __float2bfloat16(__tl_cvt_fp4_to_float(src));
}

// fp4_e2m1x2 -> bfloat162
TL_DEVICE __ppu_bfloat162
__tl_cvt_fp4x2_to_bfloat162(const __hg_fp4x2_storage_t src) {
  float2 tmp = __tl_cvt_fp4x2_to_float2(src);
  return __floats2bfloat162_rn(tmp.x, tmp.y);
}

// bfloat16 -> fp4_e2m1
TL_DEVICE __hg_fp4_storage_t __tl_cvt_bfloat16_to_fp4(const __ppu_bfloat16 src) {
  __ppu_bfloat16_raw raw = *reinterpret_cast<const __ppu_bfloat16_raw *>(&src);
  return __hg_cvt_bfloat16raw_to_fp4(raw, __HG_E2M1, hggcRoundNearest);
}

// bfloat162 -> fp4_e2m1x2
TL_DEVICE __hg_fp4x2_storage_t
__tl_cvt_bfloat162_to_fp4x2(const __ppu_bfloat162 src) {
  __ppu_bfloat162_raw raw = *reinterpret_cast<const __ppu_bfloat162_raw *>(&src);
  return __hg_cvt_bfloat16raw2_to_fp4x2(raw, __HG_E2M1, hggcRoundNearest);
}

#endif // defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100)

// ============================================================================
// FP4 Packed Buffer Access Helpers (host + device visible)
// ============================================================================
// These helpers are used for accessing individual fp4 elements from packed
// fp4_e2_2_t storage, where each byte stores 2 fp4 values.

// Load a single fp4 element from packed storage
// packed: pointer to fp4_e2_2_t array
// idx: logical index of the fp4 element
TL_DEVICE fp4_e2_t tl_fp4_packed_load(fp4_e2_2_t *packed, int idx) {
  return (idx & 1) ? packed[idx >> 1].y() : packed[idx >> 1].x();
}

// Store a single fp4 element to packed storage
// packed: pointer to fp4_e2_2_t array
// idx: logical index of the fp4 element
// val: value to store
TL_DEVICE void tl_fp4_packed_store(fp4_e2_2_t *packed, int idx, fp4_e2_t val) {
  if (idx & 1) {
    packed[idx >> 1].set_y(val);
  } else {
    packed[idx >> 1].set_x(val);
  }
}
