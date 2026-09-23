#pragma once

#include "common.h"
#include <cutlass/arch/barrier.h>

// Reuse cutlass advanced barrier abstraction
using Barrier = cutlass::arch::ClusterTransactionBarrier;

namespace tl {

TL_DEVICE void mbarrier_init(uint64_t &smem_barrier, uint32_t arrive_count) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  asm volatile("ppu.awbar.init.shared.b64 [%1], %0;"
               :
               : "r"(arrive_count), "r"(smem_int_ptr));
}

TL_DEVICE uint32_t mbarrier_test_wait(uint64_t &smem_barrier, int phase_bit) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  uint32_t waitComplete;
  asm volatile("{\n"
               ".reg .pred P1; \n\t"
               "ppu.awbar.test_wait.parity.shared::blk.b64 P1, [%1], %2;\n\t"
               "ppu.selp.b32 %0, 1, 0, P1; \n\t"
               "}\n"
               : "=r"(waitComplete)
               : "r"(smem_int_ptr), "r"(phase_bit));
  return waitComplete;
}

TL_DEVICE void mbarrier_wait(uint64_t &smem_barrier, int phase_bit) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  // Arbitrarily large timer value after which try-wait expires and re-tries.
  uint32_t ticks = 0x989680;
  asm volatile(
      "{\n"
      ".reg .pred                P1;\n"
      "LAB_WAIT:\n"
      "ppu.awbar.test_wait.parity.shared::blk.b64 P1, [%0], %1;\n"
      "@P1                       ppu.bra DONE;\n"
      "ppu.nanosleep.u32 %2;\n" // wait a few nanoseconds on current PPU
                                // architectures to save instruction issue slots
      "ppu.bra                   LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(smem_int_ptr),
      "r"(phase_bit), "r"(ticks));
}

TL_DEVICE void mbarrier_arrive(uint64_t &smem_barrier) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  asm volatile("ppu.awbar.arrive.shared.b64 _, [%0];" : : "r"(smem_int_ptr));
}

template <typename BarrierType = uint64_t>
TL_DEVICE void mbarrier_cp_async_arrive(BarrierType &smem_mbar) {
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  asm volatile("ppu.cp.async.awbar.arrive.shared.b64 [%0];"
               :
               : "r"(smem_int_mbar));
}

template <typename BarrierType = uint64_t>
TL_DEVICE void mbarrier_cp_async_arrive_noinc(BarrierType &smem_mbar) {
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  asm volatile("{\n\t"
               "ppu.cp.async.awbar.arrive.noinc.shared.b64 [%0];\n\t"
               "}"
               :
               : "r"(smem_int_mbar));
  cutlass::arch::synclog_emit_cpasync_barrier_arrive(__LINE__, smem_int_mbar);
}

TL_DEVICE void syncthreads_partial(uint64_t &smem_barrier) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  uint64_t state = 0;
  asm volatile("{\n"
               ".reg .pred                P1;\n"
               "ppu.awbar.arrive.shared.b64 %0, [%1];\n"
               "LAB_WAIT:\n"
               "ppu.awbar.test_wait.shared.b64 P1, [%1], %0;\n"
               "@!P1                      ppu.bra LAB_WAIT;\n"
               "}\n"
               : "+l"(state)
               : "r"(smem_int_ptr));
}
} // namespace tl
