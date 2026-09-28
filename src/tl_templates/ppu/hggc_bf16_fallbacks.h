#pragma once

#include "hggc_bf16_wrapper.h"
#include <hggc_fp16.h>

namespace ppu_bf16 {

#ifdef ENABLE_BF16
inline __device__ float2 bf1622float2(const __ppu_bfloat162 val) {
  return __bfloat1622float2(val);
}

inline __device__ int16_t bf1622int16(__ppu_bfloat162 val) {
  val = __hmin2(val, make_bfloat162(127., 127.));
  val = __hmax2(val, make_bfloat162(-128., -128.));
  union {
    int8_t int8[2];
    int16_t int16;
  };
  int8[0] = static_cast<int8_t>(static_cast<short>(val.x));
  int8[1] = static_cast<int8_t>(static_cast<short>(val.y));
  return int16;
}

inline __device__ __ppu_bfloat162 float22bf162(const float2 val) {
  return __float22bfloat162_rn(val);
}

inline __device__ __ppu_bfloat162 bf162bf162(const __ppu_bfloat16 val) {
  return __bfloat162bfloat162(val);
}

inline __device__ __ppu_bfloat162 bf16hadd2(const __ppu_bfloat162 x,
                                            const __ppu_bfloat162 y) {
  return __hadd2(x, y);
}

inline __device__ __ppu_bfloat16 bf16hadd(const __ppu_bfloat16 x,
                                          const __ppu_bfloat16 y) {
  return __hadd(x, y);
}

inline __device__ __ppu_bfloat162 bf16hsub2(const __ppu_bfloat162 x,
                                            const __ppu_bfloat162 y) {
  return __hsub2(x, y);
}

inline __device__ __ppu_bfloat16 bf16hsub(const __ppu_bfloat16 x,
                                          const __ppu_bfloat16 y) {
  return __hsub(x, y);
}

inline __device__ __ppu_bfloat162 bf16hmul2(const __ppu_bfloat162 x,
                                            const __ppu_bfloat162 y) {
  return __hmul2(x, y);
}

inline __device__ __ppu_bfloat16 bf16hmul(const __ppu_bfloat16 x,
                                          const __ppu_bfloat16 y) {
  return __hmul(x, y);
}

inline __device__ __ppu_bfloat162 bf16hfma2(const __ppu_bfloat162 x,
                                            const __ppu_bfloat162 y,
                                            const __ppu_bfloat162 z) {
  return __hfma2(x, y, z);
}

inline __device__ __ppu_bfloat16 bf16hfma(const __ppu_bfloat16 x,
                                          const __ppu_bfloat16 y,
                                          const __ppu_bfloat16 z) {
  return __hfma(x, y, z);
}

inline __device__ __ppu_bfloat162 bf16exp2(const __ppu_bfloat162 x) {
  return h2exp(x);
}

inline __device__ __ppu_bfloat16 bf16hadd(__ppu_bfloat16 a, __ppu_bfloat16 b,
                                          __ppu_bfloat16 c) {
  return a + b + c;
}

inline __device__ __ppu_bfloat16 bf16hadd(__ppu_bfloat16 a, __ppu_bfloat16 b,
                                          __ppu_bfloat16 c, __ppu_bfloat16 d) {
  return (__ppu_bfloat16)((float)a + (float)b + (float)c + (float)d);
}

inline __device__ __ppu_bfloat162 bf16hadd2(__ppu_bfloat162 a,
                                            __ppu_bfloat162 b,
                                            __ppu_bfloat162 c) {
  return a + b + c;
}

inline __device__ __ppu_bfloat16 bf16hmul(__ppu_bfloat16 a, __ppu_bfloat16 b,
                                          __ppu_bfloat16 c) {
  return a * b * c;
}

inline __device__ __ppu_bfloat162 bf16hmul2(__ppu_bfloat162 a,
                                            __ppu_bfloat162 b,
                                            __ppu_bfloat162 c) {
  return a * b * c;
}

inline __device__ __ppu_bfloat162 bf16hfma2(__ppu_bfloat162 a,
                                            __ppu_bfloat162 b,
                                            __ppu_bfloat162 c,
                                            __ppu_bfloat162 d) {
  return a * b * c + d;
}

#endif // ENABLE_BF16

} // namespace ppu_bf16
