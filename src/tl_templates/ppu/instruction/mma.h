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
                                     NValue, KValue, TransAValue,              \
                                     TransBValue, SaturateValue, ImplType)      \
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
                             false,
                             cute::PPU0010_16x16x32_S32S8S8S32_TN)
TL_PPU_DEFINE_MMA_DISPATCHER(kInt8, kUInt8, kInt32, 16, 16, 32, false, true,
                             false,
                             cute::PPU0010_16x16x32_S32S8U8S32_TN)
TL_PPU_DEFINE_MMA_DISPATCHER(kUInt8, kInt8, kInt32, 16, 16, 32, false, true,
                             false,
                             cute::PPU0010_16x16x32_S32U8S8S32_TN)
TL_PPU_DEFINE_MMA_DISPATCHER(kUInt8, kUInt8, kInt32, 16, 16, 32, false, true,
                             false,
                             cute::PPU0010_16x16x32_S32U8U8S32_TN)
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
                             true, false,
                             cute::PPU0015_16x16x8_F32F32F32F32_TN)
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
                             false,
                             cute::PPU0015_16x16x32_S32S8S8S32_TN)
#endif

#undef TL_PPU_DEFINE_MMA_DISPATCHER

} // namespace detail
} // namespace tl
