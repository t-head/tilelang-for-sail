#pragma once

#include "../common.h"
#include <cute/arch/mma_ppu0010.hpp>
#include <cute/arch/mma_ppu0015.hpp>
#ifndef __HGGCCC_RTC__
#include <type_traits>
#include <utility>
#endif

namespace tl {

#ifndef TL_ALWAYS_FALSE_V_DEFINED
#define TL_ALWAYS_FALSE_V_DEFINED
template <class> inline constexpr bool always_false_v = false;
#endif

namespace detail {

template <class Impl> struct MmaImplTraits {
  using DReg = std::remove_extent_t<typename Impl::DRegisters>;
  using AReg = std::remove_extent_t<typename Impl::ARegisters>;
  using BReg = std::remove_extent_t<typename Impl::BRegisters>;
  using CReg = std::remove_extent_t<typename Impl::CRegisters>;

  static constexpr int kDRegs = std::extent_v<typename Impl::DRegisters>;
  static constexpr int kARegs = std::extent_v<typename Impl::ARegisters>;
  static constexpr int kBRegs = std::extent_v<typename Impl::BRegisters>;
  static constexpr int kCRegs = std::extent_v<typename Impl::CRegisters>;
};

template <class Impl, size_t... DIdx, size_t... AIdx, size_t... BIdx,
          size_t... CIdx>
TL_DEVICE void
call_fma_impl(typename MmaImplTraits<Impl>::DReg *d,
              const typename MmaImplTraits<Impl>::AReg *a,
              const typename MmaImplTraits<Impl>::BReg *b,
              const typename MmaImplTraits<Impl>::CReg *c,
              std::index_sequence<DIdx...>, std::index_sequence<AIdx...>,
              std::index_sequence<BIdx...>, std::index_sequence<CIdx...>) {
  Impl::fma(d[DIdx]..., a[AIdx]..., b[BIdx]..., c[CIdx]...);
}

template <class Impl>
TL_DEVICE void call_fma(typename MmaImplTraits<Impl>::DReg *d,
                        const typename MmaImplTraits<Impl>::AReg *a,
                        const typename MmaImplTraits<Impl>::BReg *b,
                        const typename MmaImplTraits<Impl>::CReg *c) {
  call_fma_impl<Impl>(d, a, b, c,
                      std::make_index_sequence<MmaImplTraits<Impl>::kDRegs>{},
                      std::make_index_sequence<MmaImplTraits<Impl>::kARegs>{},
                      std::make_index_sequence<MmaImplTraits<Impl>::kBRegs>{},
                      std::make_index_sequence<MmaImplTraits<Impl>::kCRegs>{});
}

template <DataType AType, DataType BType, DataType CType, int M, int N, int K,
          bool TransA, bool TransB, bool Saturate>
struct MmaDispatcher {
  using CRegType = void;
  using ARegType = void;
  using BRegType = void;

  static TL_DEVICE void exec(CRegType *, const ARegType *, const BRegType *,
                             const CRegType *) {
    static_assert(always_false_v<std::integral_constant<int, M>>,
                  "tl::mma_sync: unsupported configuration");
  }

  static TL_DEVICE void exec_scaled(CRegType *, const ARegType *,
                                    const BRegType *, const CRegType *,
                                    uint32_t, uint32_t, uint32_t, uint32_t) {
    static_assert(always_false_v<std::integral_constant<int, M>>,
                  "tl::mma_sync_scaled: unsupported configuration");
  }
};

} // namespace detail

template <DataType AType, DataType BType, DataType CType, int M, int N, int K,
          bool TransA, bool TransB, bool Saturate = false>
TL_DEVICE void mma_sync(
    typename detail::MmaDispatcher<AType, BType, CType, M, N, K, TransA, TransB,
                                   Saturate>::CRegType *c,
    const typename detail::MmaDispatcher<AType, BType, CType, M, N, K, TransA,
                                         TransB, Saturate>::ARegType *a,
    const typename detail::MmaDispatcher<AType, BType, CType, M, N, K, TransA,
                                         TransB, Saturate>::BRegType *b) {
  using Dispatcher = detail::MmaDispatcher<AType, BType, CType, M, N, K, TransA,
                                           TransB, Saturate>;
  static_assert(!std::is_void_v<typename Dispatcher::CRegType>,
                "tl::mma_sync: unsupported configuration");
  Dispatcher::exec(c, a, b, c);
}

template <DataType AType, DataType BType, DataType CType, int M, int N, int K,
          bool TransA, bool TransB, bool Saturate = false>
TL_DEVICE void mma_sync_scaled(
    typename detail::MmaDispatcher<AType, BType, CType, M, N, K, TransA, TransB,
                                   Saturate>::CRegType *c,
    const typename detail::MmaDispatcher<AType, BType, CType, M, N, K, TransA,
                                         TransB, Saturate>::ARegType *a,
    const typename detail::MmaDispatcher<AType, BType, CType, M, N, K, TransA,
                                         TransB, Saturate>::BRegType *b,
    uint32_t scale_a, uint32_t scale_b, uint32_t scale_a_selector,
    uint32_t scale_b_selector) {
  using Dispatcher = detail::MmaDispatcher<AType, BType, CType, M, N, K, TransA,
                                           TransB, Saturate>;
  static_assert(!std::is_void_v<typename Dispatcher::CRegType>,
                "tl::mma_sync_scaled: unsupported configuration");
  Dispatcher::exec_scaled(c, a, b, c, scale_a, scale_b, scale_a_selector,
                          scale_b_selector);
}

} // namespace tl
