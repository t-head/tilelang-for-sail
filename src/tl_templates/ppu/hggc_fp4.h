#pragma once
// PPU FP4 E2M1 type support for TileLang
// Storage types, low-level conversions, wrapper structs, and pack utilities.

#include "common.h"

#include <hggc_fp16.h>
#include <hggc_bf16.h>
#include <hggc_fp8.h>  // hggcRoundMode via hgrt/hggc_device_types.h

// ---------------------------------------------------------------------------
// PPU FP4 storage typedefs
// ---------------------------------------------------------------------------

typedef unsigned char  __hg_fp4_storage_t;
typedef unsigned char  __hg_fp4x2_storage_t;
typedef unsigned short __hg_fp4x4_storage_t;

typedef enum __hg_fp4_interpretation_t {
  __HG_E2M1 = 0,  // e2m1 encoding
} __hg_fp4_interpretation_t;

// ---------------------------------------------------------------------------
// PPU low-level __hg_cvt_* primitives (device only)
// Scalar converters listed first, then their x2 vector counterparts.
// ---------------------------------------------------------------------------

#if defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100)

/// double -> fp4 e2m1 with satfinite and configurable rounding
TL_DEVICE __hg_fp4_storage_t
__hg_cvt_double_to_fp4(const double x,
                       const __hg_fp4_interpretation_t fp4_interpretation,
                       const enum hggcRoundMode rounding) {
  (void)fp4_interpretation;
  unsigned char out;
  unsigned long long int raw_bits;
  (void)memcpy(&raw_bits, &x, sizeof(x));

  const unsigned char FP4_MAXNORM = 0x7U;
  const unsigned char FP4_MANTISSA_MASK = 0x1U;
  const unsigned short int FP4_EXP_BIAS = 1U;
  const unsigned long long int FP4_SIGNIFICAND_BITS = 2ULL;
  const unsigned long long int FP4_MINDENORM_O2 =
      0x3FD0000000000000ULL;
  const unsigned long long int FP4_OVERFLOW_THRESHOLD =
      0x4018000000000000ULL;
  const unsigned long long int FP4_MINNORM =
      0x3FF0000000000000ULL;
  const unsigned long long int DP_INF_BITS = 0x7FF0000000000000ULL;

  const unsigned long long int FP4_DP_HALF_ULP =
      1ULL << (53ULL - FP4_SIGNIFICAND_BITS - 1ULL);

  // extract sign, exponent, and mantissa for target format
  unsigned char sign_bits = (unsigned char)((raw_bits >> 63ULL) << 3U);
  const unsigned char exp_field =
      (unsigned char)((((unsigned short int)(raw_bits >> 52ULL)) & 0x7FFU) -
                      1023U + FP4_EXP_BIAS);
  const unsigned char mant =
      (unsigned char)(raw_bits >> (53ULL - FP4_SIGNIFICAND_BITS)) &
      FP4_MANTISSA_MASK;
  const unsigned long long int abs_val = raw_bits & 0x7FFFFFFFFFFFFFFFULL;

  if (abs_val <= FP4_MINDENORM_O2) {
    // underflow to zero
    out = 0U;
  } else if (abs_val > FP4_OVERFLOW_THRESHOLD) {
    // overflow or NaN -> clamp to maxnorm
    if (abs_val > DP_INF_BITS) {
      sign_bits = 0U;  // NaN -> positive maxnorm
    }
    out = FP4_MAXNORM;
  } else if (abs_val >= FP4_MINNORM) {
    // normal range
    out = (unsigned char)((exp_field << (FP4_SIGNIFICAND_BITS - 1U)) | mant);
    const unsigned long long int round_off =
        raw_bits & ((FP4_DP_HALF_ULP << 1ULL) - 1ULL);
    if (rounding == hggcRoundNearest) {
      if ((round_off > FP4_DP_HALF_ULP) ||
          ((round_off == FP4_DP_HALF_ULP) && (mant & 1U))) {
        out = (unsigned char)(out + 1U);
      }
    }
  } else {
    // denormal range
    const unsigned char denorm_shift = (unsigned char)(1U - exp_field);
    out = (unsigned char)((mant | (1U << (FP4_SIGNIFICAND_BITS - 1U))) >>
                          denorm_shift);
    if (rounding == hggcRoundNearest) {
      const unsigned long long int round_off =
          (raw_bits | (1ULL << (53ULL - 1ULL))) &
          ((FP4_DP_HALF_ULP << (denorm_shift + 1ULL)) - 1ULL);
      if ((round_off > (FP4_DP_HALF_ULP << denorm_shift)) ||
          ((round_off == (FP4_DP_HALF_ULP << denorm_shift)) && (out & 1U))) {
        out = (unsigned char)(out + 1U);
      }
    }
  }

  out |= sign_bits;
  return (__hg_fp4_storage_t)out;
}

/// float -> fp4 e2m1 (widened to double internally)
TL_DEVICE __hg_fp4_storage_t
__hg_cvt_float_to_fp4(const float x,
                      const __hg_fp4_interpretation_t fp4_interpretation,
                      const enum hggcRoundMode rounding) {
  return __hg_cvt_double_to_fp4((double)x, fp4_interpretation, rounding);
}

/// half_raw -> fp4 e2m1 (via float widening)
TL_DEVICE __hg_fp4_storage_t
__hg_cvt_halfraw_to_fp4(const __half_raw x,
                        const __hg_fp4_interpretation_t fp4_interpretation,
                        const enum hggcRoundMode rounding) {
  const float fval = __half2float(*reinterpret_cast<const __half *>(&x));
  return __hg_cvt_float_to_fp4(fval, fp4_interpretation, rounding);
}

/// bfloat16_raw -> fp4 e2m1 (bf16 to float is a 16-bit left shift)
TL_DEVICE __hg_fp4_storage_t
__hg_cvt_bfloat16raw_to_fp4(const __ppu_bfloat16_raw x,
                            const __hg_fp4_interpretation_t fp4_interpretation,
                            const enum hggcRoundMode rounding) {
  const unsigned int bits32 = ((unsigned int)x.x) << 16U;
  const float fval = *reinterpret_cast<const float *>(&bits32);
  return __hg_cvt_float_to_fp4(fval, fp4_interpretation, rounding);
}

/// double2 -> fp4x2 (pack two converted nibbles into one byte)
TL_DEVICE __hg_fp4x2_storage_t
__hg_cvt_double2_to_fp4x2(const double2 x,
                          const __hg_fp4_interpretation_t fp4_interpretation,
                          const enum hggcRoundMode rounding) {
  __hg_fp4x2_storage_t packed = (__hg_fp4x2_storage_t)__hg_cvt_double_to_fp4(
      x.y, fp4_interpretation, rounding);
  packed = (__hg_fp4x2_storage_t)(packed << 4U);
  packed = (__hg_fp4x2_storage_t)(packed |
                                  __hg_cvt_double_to_fp4(x.x,
                                                         fp4_interpretation,
                                                         rounding));
  return packed;
}

/// float2 -> fp4x2
TL_DEVICE __hg_fp4x2_storage_t
__hg_cvt_float2_to_fp4x2(const float2 x,
                         const __hg_fp4_interpretation_t fp4_interpretation,
                         const enum hggcRoundMode rounding) {
  __hg_fp4x2_storage_t packed = (__hg_fp4x2_storage_t)__hg_cvt_float_to_fp4(
      x.y, fp4_interpretation, rounding);
  packed = (__hg_fp4x2_storage_t)(packed << 4U);
  packed = (__hg_fp4x2_storage_t)(packed |
                                  __hg_cvt_float_to_fp4(x.x,
                                                        fp4_interpretation,
                                                        rounding));
  return packed;
}

/// half2_raw -> fp4x2
TL_DEVICE __hg_fp4x2_storage_t
__hg_cvt_halfraw2_to_fp4x2(const __half2_raw x,
                           const __hg_fp4_interpretation_t fp4_interpretation,
                           const enum hggcRoundMode rounding) {
  __half_raw hr;
  hr.x = x.x;
  const __hg_fp4_storage_t lo_nibble =
      __hg_cvt_halfraw_to_fp4(hr, fp4_interpretation, rounding);
  hr.x = x.y;
  const __hg_fp4_storage_t hi_nibble =
      __hg_cvt_halfraw_to_fp4(hr, fp4_interpretation, rounding);
  return (__hg_fp4x2_storage_t)((hi_nibble << 4U) | lo_nibble);
}

/// bfloat162_raw -> fp4x2
TL_DEVICE __hg_fp4x2_storage_t __hg_cvt_bfloat16raw2_to_fp4x2(
    const __ppu_bfloat162_raw x,
    const __hg_fp4_interpretation_t fp4_interpretation,
    const enum hggcRoundMode rounding) {
  __ppu_bfloat16_raw br;
  br.x = x.y;
  __hg_fp4x2_storage_t packed =
      (__hg_fp4x2_storage_t)__hg_cvt_bfloat16raw_to_fp4(br,
                                                         fp4_interpretation,
                                                         rounding);
  packed = (__hg_fp4x2_storage_t)(packed << 4U);
  br.x = x.x;
  packed = (__hg_fp4x2_storage_t)(packed |
                                  __hg_cvt_bfloat16raw_to_fp4(br,
                                                              fp4_interpretation,
                                                              rounding));
  return packed;
}

/// fp4 e2m1 -> half_raw. Exact conversion: e2m1 has only 8 magnitudes.
/// Normal exponent rebias: +14 (f16 bias 15 vs e2m1 bias 1).
TL_DEVICE __half_raw
__hg_cvt_fp4_to_halfraw(const __hg_fp4_storage_t x,
                        const __hg_fp4_interpretation_t fp4_interpretation) {
  (void)fp4_interpretation;
  __half_raw out;
  const unsigned int sign_bits = ((unsigned int)x & 0x8U) << 12U;
  const unsigned int exp_field = ((unsigned int)x >> 1U) & 0x3U;
  const unsigned int mant_bit = (unsigned int)x & 0x1U;
  unsigned int half_bits;
  if (exp_field == 0U) {
    half_bits = mant_bit ? 0x3800U : 0x0000U;
  } else {
    half_bits = ((exp_field + 14U) << 10U) | (mant_bit << 9U);
  }
  out.x = (unsigned short)(half_bits | sign_bits);
  return out;
}

/// fp4x2 -> half2_raw (unpack each nibble and convert)
TL_DEVICE __half2_raw
__hg_cvt_fp4x2_to_halfraw2(const __hg_fp4x2_storage_t x,
                           const __hg_fp4_interpretation_t fp4_interpretation) {
  __half2_raw out;
  out.x =
      __hg_cvt_fp4_to_halfraw((__hg_fp4_storage_t)x, fp4_interpretation).x;
  out.y = __hg_cvt_fp4_to_halfraw((__hg_fp4_storage_t)(x >> 4U),
                                  fp4_interpretation)
              .x;
  return out;
}

#endif  // __HGGC_ARCH__ >= 100

// ---------------------------------------------------------------------------
// PPU native FP4 C++ structs
// ---------------------------------------------------------------------------

struct __HGGC_ALIGN__(1) __hg_fp4_e2m1 {
  __hg_fp4_storage_t __x;

  __hg_fp4_e2m1() = default;

#if defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100)
  /// Construct from float (satfinite, round-to-nearest)
  TL_DEVICE explicit __hg_fp4_e2m1(const float input_val) {
    __x = __hg_cvt_float_to_fp4(input_val, __HG_E2M1, hggcRoundNearest);
  }

  /// Construct from double (satfinite, round-to-nearest)
  TL_DEVICE explicit __hg_fp4_e2m1(const double input_val) {
    __x = __hg_cvt_double_to_fp4(input_val, __HG_E2M1, hggcRoundNearest);
  }

  /// Convert to float (fp4 -> half -> float)
  TL_DEVICE explicit operator float() const {
    const __half_raw hr = __hg_cvt_fp4_to_halfraw(__x, __HG_E2M1);
    return __half2float(*reinterpret_cast<const __half *>(&hr));
  }
#endif
};

struct __HGGC_ALIGN__(1) __hg_fp4x2_e2m1 {
  __hg_fp4x2_storage_t __x;
  __hg_fp4x2_e2m1() = default;
};

// ---------------------------------------------------------------------------
// TileLang FP4 <-> scalar type conversions (device only)
// Layout: all from-fp4 conversions first, then all to-fp4 conversions.
// ---------------------------------------------------------------------------

#if defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100)

// -- From FP4 ---------------------------------------------------------------

/// @brief fp4_e2m1 -> half
TL_DEVICE __half __tl_cvt_fp4_to_half(const __hg_fp4_storage_t input) {
  __half_raw bits = __hg_cvt_fp4_to_halfraw(input, __HG_E2M1);
  return *reinterpret_cast<__half *>(&bits);
}

/// @brief fp4_e2m1x2 -> half2
TL_DEVICE half2 __tl_cvt_fp4x2_to_half2(const __hg_fp4x2_storage_t input) {
  __half2_raw bits = __hg_cvt_fp4x2_to_halfraw2(input, __HG_E2M1);
  return *reinterpret_cast<half2 *>(&bits);
}

/// @brief fp4_e2m1 -> float (via half intermediate)
TL_DEVICE float __tl_cvt_fp4_to_float(const __hg_fp4_storage_t input) {
  return __half2float(__tl_cvt_fp4_to_half(input));
}

/// @brief fp4_e2m1x2 -> float2 (via half2 intermediate)
TL_DEVICE float2 __tl_cvt_fp4x2_to_float2(const __hg_fp4x2_storage_t input) {
  half2 h2 = __tl_cvt_fp4x2_to_half2(input);
  float2 out;
  out.x = __half2float(h2.x);
  out.y = __half2float(h2.y);
  return out;
}

/// @brief fp4_e2m1 -> bfloat16 (via float intermediate)
TL_DEVICE __ppu_bfloat16
__tl_cvt_fp4_to_bfloat16(const __hg_fp4_storage_t input) {
  return __float2bfloat16(__tl_cvt_fp4_to_float(input));
}

/// @brief fp4_e2m1x2 -> bfloat162 (via float2 intermediate)
TL_DEVICE __ppu_bfloat162
__tl_cvt_fp4x2_to_bfloat162(const __hg_fp4x2_storage_t input) {
  float2 f2 = __tl_cvt_fp4x2_to_float2(input);
  return __floats2bfloat162_rn(f2.x, f2.y);
}

/// @brief fp4_e2m1 -> double (via float intermediate)
TL_DEVICE double __tl_cvt_fp4_to_double(const __hg_fp4_storage_t input) {
  return (double)__tl_cvt_fp4_to_float(input);
}

/// @brief fp4_e2m1x2 -> double2 (via float2 intermediate)
TL_DEVICE double2
__tl_cvt_fp4x2_to_double2(const __hg_fp4x2_storage_t input) {
  float2 f2 = __tl_cvt_fp4x2_to_float2(input);
  double2 out;
  out.x = (double)(f2.x);
  out.y = (double)(f2.y);
  return out;
}

// -- To FP4 -----------------------------------------------------------------

/// @brief half -> fp4_e2m1
TL_DEVICE __hg_fp4_storage_t __tl_cvt_half_to_fp4(const __half input) {
  return __hg_cvt_halfraw_to_fp4(
      *reinterpret_cast<const __half_raw *>(&input), __HG_E2M1,
      hggcRoundNearest);
}

/// @brief half2 -> fp4_e2m1x2
TL_DEVICE __hg_fp4x2_storage_t __tl_cvt_half2_to_fp4x2(const half2 input) {
  return __hg_cvt_halfraw2_to_fp4x2(
      *reinterpret_cast<const __half2_raw *>(&input), __HG_E2M1,
      hggcRoundNearest);
}

/// @brief float -> fp4_e2m1
TL_DEVICE __hg_fp4_storage_t __tl_cvt_float_to_fp4(const float input) {
  return __hg_cvt_float_to_fp4(input, __HG_E2M1, hggcRoundNearest);
}

/// @brief float2 -> fp4_e2m1x2
TL_DEVICE __hg_fp4x2_storage_t
__tl_cvt_float2_to_fp4x2(const float2 input) {
  return __hg_cvt_float2_to_fp4x2(input, __HG_E2M1, hggcRoundNearest);
}

/// @brief bfloat16 -> fp4_e2m1
TL_DEVICE __hg_fp4_storage_t
__tl_cvt_bfloat16_to_fp4(const __ppu_bfloat16 input) {
  return __hg_cvt_bfloat16raw_to_fp4(
      *reinterpret_cast<const __ppu_bfloat16_raw *>(&input), __HG_E2M1,
      hggcRoundNearest);
}

/// @brief bfloat162 -> fp4_e2m1x2
TL_DEVICE __hg_fp4x2_storage_t
__tl_cvt_bfloat162_to_fp4x2(const __ppu_bfloat162 input) {
  return __hg_cvt_bfloat16raw2_to_fp4x2(
      *reinterpret_cast<const __ppu_bfloat162_raw *>(&input), __HG_E2M1,
      hggcRoundNearest);
}

/// @brief double -> fp4_e2m1
TL_DEVICE __hg_fp4_storage_t __tl_cvt_double_to_fp4(const double input) {
  return __hg_cvt_double_to_fp4(input, __HG_E2M1, hggcRoundNearest);
}

/// @brief double2 -> fp4_e2m1x2
TL_DEVICE __hg_fp4x2_storage_t
__tl_cvt_double2_to_fp4x2(const double2 input) {
  return __hg_cvt_double2_to_fp4x2(input, __HG_E2M1, hggcRoundNearest);
}

#endif  // __HGGC_ARCH__ >= 100

// ---------------------------------------------------------------------------
// TileLang FP4 wrapper structs
// ---------------------------------------------------------------------------

/// Wrapper for __hg_fp4_e2m1 providing implicit type conversions
struct fp4_e2_t {
  __hg_fp4_storage_t __x;

  TL_DEVICE fp4_e2_t() = default;

  /// Construct from raw storage byte
  TL_DEVICE fp4_e2_t(__hg_fp4_storage_t raw) : __x(raw) {}

  /// Construct from native PPU fp4 type
  TL_DEVICE fp4_e2_t(__hg_fp4_e2m1 input) : __x(input.__x) {}

#if defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100)
  /// Construct from float (satfinite, round-to-nearest)
  TL_DEVICE explicit fp4_e2_t(float fval) {
    __hg_fp4_e2m1 native(fval);
    __x = native.__x;
  }

  /// Convert to float
  TL_DEVICE operator float() const {
    __hg_fp4_e2m1 native;
    native.__x = __x;
    return float(native);
  }

  /// Convert to native PPU fp4 type
  TL_DEVICE operator __hg_fp4_e2m1() const {
    __hg_fp4_e2m1 native;
    native.__x = __x;
    return native;
  }

  /// Convert to __half
  TL_DEVICE operator __half() const { return __half(float(*this)); }

  /// Convert to half_t
  TL_DEVICE operator half_t() const { return half_t(float(*this)); }
#endif
};

/// Packed pair of fp4 values stored in a single byte
class fp4_e2_2_t {
public:
  __hg_fp4x2_storage_t __x;

  TL_DEVICE fp4_e2_2_t() = default;
  TL_DEVICE fp4_e2_2_t(__hg_fp4x2_e2m1 input) : __x(input.__x) {}
  TL_DEVICE fp4_e2_2_t(__hg_fp4x2_storage_t raw_bits) : __x(raw_bits) {}

  /// Write low nibble (element 0)
  TL_DEVICE void set_x(fp4_e2_t elem) {
    __x = (__x & 0xF0) | (elem.__x & 0x0F);
  }

  /// Write high nibble (element 1)
  TL_DEVICE void set_y(fp4_e2_t elem) {
    __x = (__x & 0x0F) | ((elem.__x & 0x0F) << 4);
  }

  /// Read low nibble (element 0)
  TL_DEVICE fp4_e2_t x() const {
    return fp4_e2_t((__hg_fp4_storage_t)(__x & 0x0F));
  }

  /// Read high nibble (element 1)
  TL_DEVICE fp4_e2_t y() const {
    return fp4_e2_t((__hg_fp4_storage_t)((__x >> 4) & 0x0F));
  }
};

// ---------------------------------------------------------------------------
// Aligned FP4 vector types
// ---------------------------------------------------------------------------

struct __HGGC_ALIGN__(2) fp4_e2_4_t {
  fp4_e2_2_t x, y;
};

struct __HGGC_ALIGN__(4) fp4_e2_8_t {
  fp4_e2_4_t x, y;
};

struct __HGGC_ALIGN__(8) fp4_e2_16_t {
  fp4_e2_8_t x, y;
};

struct __HGGC_ALIGN__(16) fp4_e2_32_t {
  fp4_e2_16_t x, y;

  TL_DEVICE fp4_e2_32_t &operator=(const ulonglong4 &src) {
    x.x = *(fp4_e2_8_t *)&src.x;
    x.y = *(fp4_e2_8_t *)&src.y;
    y.x = *(fp4_e2_8_t *)&src.z;
    y.y = *(fp4_e2_8_t *)&src.w;
    return *this;
  }
};

struct __HGGC_ALIGN__(32) fp4_e2_64_t {
  fp4_e2_32_t x, y;
};

/// Tag for unpacked FP4 shared-memory layout
/// (16 nibbles in low 64 bits of a 128-bit aligned region)
struct float4_e2m1_unpacked_t {
  uint8_t __x;
};

// ---------------------------------------------------------------------------
// Packed element load/store helpers
// ---------------------------------------------------------------------------

/// Load one fp4 element from packed fp4_e2_2_t array by logical index
TL_DEVICE fp4_e2_t tl_fp4_packed_load(fp4_e2_2_t *buf, int pos) {
  return (pos & 1) ? buf[pos >> 1].y() : buf[pos >> 1].x();
}

/// Store one fp4 element into packed fp4_e2_2_t array by logical index
TL_DEVICE void tl_fp4_packed_store(fp4_e2_2_t *buf, int pos, fp4_e2_t elem) {
  (pos & 1) ? buf[pos >> 1].set_y(elem) : buf[pos >> 1].set_x(elem);
}

// ---------------------------------------------------------------------------
// Pack multiple fp4_e2_t values into wider containers
// ---------------------------------------------------------------------------

/// Pack two fp4 values into one byte
TL_DEVICE fp4_e2_2_t make_fp4_e2_2_t(fp4_e2_t lo, fp4_e2_t hi) {
  fp4_e2_2_t out;
  out.__x = (__hg_fp4x2_storage_t)((lo.__x & 0x0F) | ((hi.__x & 0x0F) << 4));
  return out;
}

/// Pack four fp4 values
TL_DEVICE fp4_e2_4_t make_fp4_e2_4_t(fp4_e2_t a0, fp4_e2_t a1,
                                     fp4_e2_t a2, fp4_e2_t a3) {
  fp4_e2_4_t out;
  out.x = make_fp4_e2_2_t(a0, a1);
  out.y = make_fp4_e2_2_t(a2, a3);
  return out;
}

/// Pack eight fp4 values
TL_DEVICE fp4_e2_8_t make_fp4_e2_8_t(fp4_e2_t a0, fp4_e2_t a1,
                                     fp4_e2_t a2, fp4_e2_t a3,
                                     fp4_e2_t a4, fp4_e2_t a5,
                                     fp4_e2_t a6, fp4_e2_t a7) {
  fp4_e2_8_t out;
  out.x = make_fp4_e2_4_t(a0, a1, a2, a3);
  out.y = make_fp4_e2_4_t(a4, a5, a6, a7);
  return out;
}

/// Pack sixteen fp4 values
TL_DEVICE fp4_e2_16_t make_fp4_e2_16_t(fp4_e2_t a0, fp4_e2_t a1,
                                       fp4_e2_t a2, fp4_e2_t a3,
                                       fp4_e2_t a4, fp4_e2_t a5,
                                       fp4_e2_t a6, fp4_e2_t a7,
                                       fp4_e2_t b0, fp4_e2_t b1,
                                       fp4_e2_t b2, fp4_e2_t b3,
                                       fp4_e2_t b4, fp4_e2_t b5,
                                       fp4_e2_t b6, fp4_e2_t b7) {
  fp4_e2_16_t out;
  out.x = make_fp4_e2_8_t(a0, a1, a2, a3, a4, a5, a6, a7);
  out.y = make_fp4_e2_8_t(b0, b1, b2, b3, b4, b5, b6, b7);
  return out;
}

/// Pack thirty-two fp4 values
TL_DEVICE fp4_e2_32_t make_fp4_e2_32_t(
    fp4_e2_t a0,  fp4_e2_t a1,  fp4_e2_t a2,  fp4_e2_t a3,
    fp4_e2_t a4,  fp4_e2_t a5,  fp4_e2_t a6,  fp4_e2_t a7,
    fp4_e2_t a8,  fp4_e2_t a9,  fp4_e2_t a10, fp4_e2_t a11,
    fp4_e2_t a12, fp4_e2_t a13, fp4_e2_t a14, fp4_e2_t a15,
    fp4_e2_t b0,  fp4_e2_t b1,  fp4_e2_t b2,  fp4_e2_t b3,
    fp4_e2_t b4,  fp4_e2_t b5,  fp4_e2_t b6,  fp4_e2_t b7,
    fp4_e2_t b8,  fp4_e2_t b9,  fp4_e2_t b10, fp4_e2_t b11,
    fp4_e2_t b12, fp4_e2_t b13, fp4_e2_t b14, fp4_e2_t b15) {
  fp4_e2_32_t out;
  out.x = make_fp4_e2_16_t(a0, a1, a2, a3, a4, a5, a6, a7,
                           a8, a9, a10, a11, a12, a13, a14, a15);
  out.y = make_fp4_e2_16_t(b0, b1, b2, b3, b4, b5, b6, b7,
                           b8, b9, b10, b11, b12, b13, b14, b15);
  return out;
}
