#pragma once

#include "mma_base.h"

#if __HGGC_ARCH__ == 100
#include <cute/arch/mma_ppu0010.hpp>
#elif __HGGC_ARCH__ == 150
#include <cute/arch/mma_ppu0015.hpp>
#endif

namespace tl {
namespace detail {

#define TL_PPU_DEFINE_MMA_DISPATCHER(ATypeEnum, BTypeEnum, CTypeEnum, MValue,  \
                                     NValue, KValue, TransAValue, TransBValue, \
                                     SaturateValue, ImplType)                  \
  template <>                                                                  \
  struct MmaDispatcher<DataType::ATypeEnum, DataType::BTypeEnum,               \
                       DataType::CTypeEnum, MValue, NValue, KValue,            \
                       TransAValue, TransBValue, SaturateValue> {              \
    using Impl = ImplType;                                                     \
    using Traits = MmaImplTraits<Impl>;                                        \
    using CRegType = typename Traits::DReg;                                    \
    using ARegType = typename Traits::AReg;                                    \
    using BRegType = typename Traits::BReg;                                    \
    static_assert(                                                             \
        std::is_same_v<typename Traits::DReg, typename Traits::CReg>,          \
        "tl::mma_sync requires matching accumulator/output regs");             \
    static TL_DEVICE void exec(CRegType *d, const ARegType *a,                 \
                               const BRegType *b, const CRegType *c) {         \
      call_fma<Impl>(d, a, b, c);                                              \
    }                                                                          \
  };

#if __HGGC_ARCH__ == 100
TL_PPU_DEFINE_MMA_DISPATCHER(kFloat16, kFloat16, kFloat16, 16, 16, 16, false,
                             true, false,
                             cute::PPU0010_16x16x16_F16F16F16F16_TN)
TL_PPU_DEFINE_MMA_DISPATCHER(kFloat16, kFloat16, kFloat32, 16, 16, 16, false,
                             true, false,
                             cute::PPU0010_16x16x16_F32F16F16F32_TN)
TL_PPU_DEFINE_MMA_DISPATCHER(kBFloat16, kBFloat16, kFloat32, 16, 16, 16, false,
                             true, false,
                             cute::PPU0010_16x16x16_F32BF16BF16F32_TN)
TL_PPU_DEFINE_MMA_DISPATCHER(kTensorFloat32, kTensorFloat32, kFloat32, 16, 16,
                             8, false, true, false,
                             cute::PPU0010_16x16x8_F32TF32TF32F32_TN)
TL_PPU_DEFINE_MMA_DISPATCHER(kInt8, kInt8, kInt32, 16, 16, 32, false, true,
                             false, cute::PPU0010_16x16x32_S32S8S8S32_TN)
TL_PPU_DEFINE_MMA_DISPATCHER(kInt8, kUInt8, kInt32, 16, 16, 32, false, true,
                             false, cute::PPU0010_16x16x32_S32S8U8S32_TN)
TL_PPU_DEFINE_MMA_DISPATCHER(kUInt8, kInt8, kInt32, 16, 16, 32, false, true,
                             false, cute::PPU0010_16x16x32_S32U8S8S32_TN)
TL_PPU_DEFINE_MMA_DISPATCHER(kUInt8, kUInt8, kInt32, 16, 16, 32, false, true,
                             false, cute::PPU0010_16x16x32_S32U8U8S32_TN)
#elif __HGGC_ARCH__ == 150
TL_PPU_DEFINE_MMA_DISPATCHER(kFloat16, kFloat16, kFloat16, 16, 16, 16, false,
                             true, false,
                             cute::PPU0015_16x16x16_F16F16F16F16_TN)
TL_PPU_DEFINE_MMA_DISPATCHER(kFloat16, kFloat16, kFloat32, 16, 16, 16, false,
                             true, false,
                             cute::PPU0015_16x16x16_F32F16F16F32_TN)
TL_PPU_DEFINE_MMA_DISPATCHER(kBFloat16, kBFloat16, kFloat32, 16, 16, 16, false,
                             true, false,
                             cute::PPU0015_16x16x16_F32BF16BF16F32_TN)
TL_PPU_DEFINE_MMA_DISPATCHER(kTensorFloat32, kTensorFloat32, kFloat32, 16, 16,
                             8, false, true, false,
                             cute::PPU0015_16x16x8_F32TF32TF32F32_TN)
TL_PPU_DEFINE_MMA_DISPATCHER(kFloat32, kFloat32, kFloat32, 16, 16, 8, false,
                             true, false, cute::PPU0015_16x16x8_F32F32F32F32_TN)
TL_PPU_DEFINE_MMA_DISPATCHER(kFloat8_e4m3, kFloat8_e4m3, kFloat32, 16, 16, 32,
                             false, true, false,
                             cute::PPU0015_16x16x32_F32E4M3E4M3F32_TN)
TL_PPU_DEFINE_MMA_DISPATCHER(kFloat8_e4m3, kFloat8_e5m2, kFloat32, 16, 16, 32,
                             false, true, false,
                             cute::PPU0015_16x16x32_F32E4M3E5M2F32_TN)
TL_PPU_DEFINE_MMA_DISPATCHER(kFloat8_e5m2, kFloat8_e4m3, kFloat32, 16, 16, 32,
                             false, true, false,
                             cute::PPU0015_16x16x32_F32E5M2E4M3F32_TN)
TL_PPU_DEFINE_MMA_DISPATCHER(kFloat8_e5m2, kFloat8_e5m2, kFloat32, 16, 16, 32,
                             false, true, false,
                             cute::PPU0015_16x16x32_F32E5M2E5M2F32_TN)
TL_PPU_DEFINE_MMA_DISPATCHER(kInt8, kInt8, kInt32, 16, 16, 32, false, true,
                             false, cute::PPU0015_16x16x32_S32S8S8S32_TN)

// 16x8 MMA fallback: actlize provides the native PPU0015 m16n8 MMA atoms
// (cute/arch/mma_ppu0015.hpp) that lower to the m16n8k* tensor-core
// instruction family. TransA/TransB follow the existing F32F16F16F32 TN
// entry above (TransA=false, TransB=true). fp8 is covered at k32 with F32
// accumulation only: fp8 k16 (unreachable: k_dim is always 32 for fp8) and
// fp16 accumulation stay unwired.
TL_PPU_DEFINE_MMA_DISPATCHER(kFloat16, kFloat16, kFloat32, 16, 8, 16, false,
                             true, false, cute::PPU0015_16x8x16_F32F16F16F32_TN)
// F16 accumulate: the atom packs the four f16 outputs into 2 b32 registers
// (D/C both uint32_t[2]), so the DReg == CReg assertion still holds.
TL_PPU_DEFINE_MMA_DISPATCHER(kFloat16, kFloat16, kFloat16, 16, 8, 16, false,
                             true, false, cute::PPU0015_16x8x16_F16F16F16F16_TN)
TL_PPU_DEFINE_MMA_DISPATCHER(kBFloat16, kBFloat16, kFloat32, 16, 8, 16, false,
                             true, false,
                             cute::PPU0015_16x8x16_F32BF16BF16F32_TN)
TL_PPU_DEFINE_MMA_DISPATCHER(kTensorFloat32, kTensorFloat32, kFloat32, 16, 8, 8,
                             false, true, false,
                             cute::PPU0015_16x8x8_F32TF32TF32F32_TN)
// FP8 x FP8 -> F32 m16n8k32 over the E4M3/E5M2 operand mixes
// (cute/arch/mma_ppu0015.hpp): the four atoms share the S8 k32 atom register
// geometry (D/C float[4], A uint32_t[4], B uint32_t[2]) and the same
// ALayout/BLayout/CLayout, so the plain dispatcher macro applies and the
// b_bits == 8 shared-load layouts are reused as-is.
TL_PPU_DEFINE_MMA_DISPATCHER(kFloat8_e4m3, kFloat8_e4m3, kFloat32, 16, 8, 32,
                             false, true, false,
                             cute::PPU0015_16x8x32_F32E4M3E4M3F32_TN)
TL_PPU_DEFINE_MMA_DISPATCHER(kFloat8_e4m3, kFloat8_e5m2, kFloat32, 16, 8, 32,
                             false, true, false,
                             cute::PPU0015_16x8x32_F32E4M3E5M2F32_TN)
TL_PPU_DEFINE_MMA_DISPATCHER(kFloat8_e5m2, kFloat8_e4m3, kFloat32, 16, 8, 32,
                             false, true, false,
                             cute::PPU0015_16x8x32_F32E5M2E4M3F32_TN)
TL_PPU_DEFINE_MMA_DISPATCHER(kFloat8_e5m2, kFloat8_e5m2, kFloat32, 16, 8, 32,
                             false, true, false,
                             cute::PPU0015_16x8x32_F32E5M2E5M2F32_TN)

TL_PPU_DEFINE_MMA_DISPATCHER(kInt8, kInt8, kInt32, 16, 8, 32, false, true,
                             false, cute::PPU0015_16x8x32_S32S8S8S32_TN)

// FP4 e2m1 MMA: the atom carries 4 extra E8M0 scale registers (SRegisters),
// so it cannot use TL_PPU_DEFINE_MMA_DISPATCHER (call_fma only unpacks
// D/A/B/C register groups). The scale-less entry point retains the historical
// scale=1 defaults, while exec_scaled forwards runtime S0/S1 and selectors.
template <>
struct MmaDispatcher<DataType::kFloat4_e2m1fn, DataType::kFloat4_e2m1fn,
                     DataType::kFloat32, 16, 16, 64, false, true, false> {
  using Impl = cute::PPU0015_16x16x64_F32F4F4F32_TN;
  using Traits = MmaImplTraits<Impl>;
  using CRegType = typename Traits::DReg;
  using ARegType = typename Traits::AReg;
  using BRegType = typename Traits::BReg;
  static_assert(std::is_same_v<typename Traits::DReg, typename Traits::CReg>,
                "tl::mma_sync requires matching accumulator/output regs");
  static TL_DEVICE void exec(CRegType *d, const ARegType *a, const BRegType *b,
                             const CRegType *c) {
    exec_scaled(d, a, b, c, 0x7f7f7f7fu, 0x7f7f7f7fu, 0u, 0u);
  }
  static TL_DEVICE void exec_scaled(CRegType *d, const ARegType *a,
                                    const BRegType *b, const CRegType *c,
                                    uint32_t scale_a, uint32_t scale_b,
                                    uint32_t scale_a_selector,
                                    uint32_t scale_b_selector) {
    Impl::fma(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], a[0], a[1], a[2],
              a[3], b[0], b[1], b[2], b[3], c[0], c[1], c[2], c[3], c[4], c[5],
              c[6], c[7], scale_a, scale_b, scale_a_selector, scale_b_selector);
  }
};
#endif

#undef TL_PPU_DEFINE_MMA_DISPATCHER

} // namespace detail
} // namespace tl
