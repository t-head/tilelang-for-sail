/*!
 * \file tl/ppu/op/gemm.cc
 * \brief PPU implementation registration for tl.gemm instruction selection.
 */

#include "op/gemm.h"
#include "support/check.h"
#include <tvm/runtime/logging.h>

#include "op/builtin.h"
#include "op/utils.h"
#include "ppu/target_utils.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace ppu {

namespace {

constexpr const char *kPpuMMA = "ppu.mma";
// Own key so ReuseExistingSharedLayout returns false: the pinned B layout of
// the PPU0010 chained RS transB lowering must reach LayoutInference, which
// converts a conflicting retained layout instead of silently keeping it.
constexpr const char *kPpuMMAChainedRS = "ppu.mma.rs_chained";
constexpr const char *kPpuMMABlockScaled = "ppu.mma.blockscaled";

// Annotation set by AnnotateChainedGemm on gemms whose A comes from a prior
// GEMM's C fragment.
constexpr const char *kAFromGemmC = "a_from_gemm_c";

void FatalUnsupportedPpuArchGemm(const GemmNode &op, Target target,
                                 const char *inst_name) {
  LOG(FATAL) << inst_name << " requires an architecture newer than PPU's "
             << "supported ppu0010/ppu0015 targets. Got target=" << target
             << ", A(scope=" << op.a_.scope() << ", dtype=" << op.a_->dtype
             << "), B(scope=" << op.b_.scope() << ", dtype=" << op.b_->dtype
             << "), C(scope=" << op.c_.scope() << ", dtype=" << op.c_->dtype
             << "), M=" << op.m_ << ", N=" << op.n_ << ", K=" << op.k_ << ".";
}

// Single whitelist of the T.gemm operand dtype configurations. It mirrors the
// native MMA atom table (MmaDispatcher specializations) in
// tl_templates/ppu/instruction/mma.h, so an unsupported combination fails here
// with the operand dtypes in the message instead of a device-compiler
// static_assert deep inside the template.
bool AllowPPUMMA(const GemmNode &op, int arch) {
  DataType a = op.a_->dtype, b = op.b_->dtype, c = op.c_->dtype;
  if (a.is_float8() && b.is_float8()) {
    // Only the e4m3/e5m2 atoms exist, and only from ppu0015 on.
    auto is_e4m3_or_e5m2 = [](DataType dtype) {
      return dtype.is_float8_e4m3() || dtype.is_float8_e4m3fn() ||
             dtype.is_float8_e5m2();
    };
    return arch >= 15 && c == DataType::Float(32) && is_e4m3_or_e5m2(a) &&
           is_e4m3_or_e5m2(b);
  }
  if (a.is_float4() && b.is_float4()) {
    // The scale-less FP4 atom; runtime scales go through T.gemm_blockscaled.
    return arch >= 15 && c == DataType::Float(32);
  }
  if (a != b) {
    return false;
  }
  if (a == DataType::Float(16)) {
    return c == DataType::Float(16) || c == DataType::Float(32);
  }
  // float32 operands are emitted as the tf32 atom (see the PPU MMA emitter).
  if (a == DataType::Float(32) || a.is_tfloat32()) {
    return c == DataType::Float(32);
  }
  if (a == DataType::BFloat(16)) {
    return c == DataType::Float(32);
  }
  if (a == DataType::Int(8)) {
    return c == DataType::Int(32);
  }
  // The u8 atoms (s8u8/u8s8/u8u8) only exist on ppu0010. T.gemm does not reach
  // them today: the Python emitter releases k32 for s8s8 only.
  if (a == DataType::UInt(8)) {
    return arch == 10 && c == DataType::Int(32);
  }
  return false;
}

void FatalMmaSyncUnavailable(const GemmNode &op, Target target) {
  LOG(FATAL) << "T.gemm requires a native PPU MMA atom, but the operand dtype "
                "configuration is not supported. Got target="
             << target << ", A(scope=" << op.a_.scope()
             << ", dtype=" << op.a_->dtype << "), B(scope=" << op.b_.scope()
             << ", dtype=" << op.b_->dtype << "), C(scope=" << op.c_.scope()
             << ", dtype=" << op.c_->dtype << "), M=" << op.m_
             << ", N=" << op.n_ << ", K=" << op.k_
             << ". Supported: "
                "float16->float16/float32, bfloat16->float32, "
                "float32/tfloat32->float32, int8->int32; ppu0015 "
                "additionally float8_e4m3/e5m2 and float4_e2m1fn -> float32.";
}

std::pair<int, int>
ComputeDefaultWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                            int num_warps, int k_n_per_warp) {
  int m_warp = 1, n_warp = 1;
  constexpr int kMPerWarp = 16;

  ICHECK(M % kMPerWarp == 0)
      << "M must be divisible by " << kMPerWarp << ", but got " << M;
  ICHECK(N % k_n_per_warp == 0)
      << "N must be divisible by " << k_n_per_warp << ", but got " << N;

  if (policy.IsFullRow()) {
    int best_m = 1;
    int best_n = 1;
    int max_m_warp = M / kMPerWarp;
    int max_n_warp = N / k_n_per_warp;
    if (max_m_warp > num_warps) {
      best_m = num_warps;
    } else if (max_m_warp * max_n_warp > num_warps) {
      best_m = max_m_warp;
      int n = num_warps / best_m;
      if (n != 0) {
        best_n = n;
      }
    } else {
      best_m = max_m_warp;
      best_n = max_n_warp;
    }
    m_warp = best_m;
    n_warp = best_n;
  } else if (policy.IsFullCol()) {
    int best_m = 1;
    int best_n = 1;
    int max_m_warp = M / kMPerWarp;
    int max_n_warp = N / k_n_per_warp;
    if (max_n_warp > num_warps) {
      best_n = num_warps;
    } else if (max_m_warp * max_n_warp > num_warps) {
      best_n = max_n_warp;
      int m = num_warps / best_n;
      if (m != 0) {
        best_m = m;
      }
    } else {
      best_m = max_m_warp;
      best_n = max_n_warp;
    }
    m_warp = best_m;
    n_warp = best_n;
  } else if (policy.IsSquare()) {
    int max_m_warps = M / kMPerWarp;
    float ideal_ratio = N > 0 ? static_cast<float>(M) / N : 1.0f;

    int best_m = 1;
    int best_n = 1;
    float best_balance = std::numeric_limits<float>::max();
    for (int m = 1; m <= max_m_warps && m <= num_warps; m++) {
      int n = num_warps / m;

      float m_per_warp = static_cast<float>(M) / (m * kMPerWarp);
      float n_per_warp = static_cast<float>(N) / (n * k_n_per_warp);
      if (m_per_warp < 1 || n_per_warp < 1)
        continue;
      if (m * n != num_warps)
        continue;

      float balance = std::abs(m_per_warp / n_per_warp - ideal_ratio);
      if (balance < best_balance) {
        best_balance = balance;
        best_m = m;
        best_n = n;
      }
    }

    m_warp = best_m;
    n_warp = best_n;
  } else {
    ICHECK(0) << "Unknown GemmWarpPolicy";
  }

  ICHECK(m_warp * n_warp <= num_warps)
      << "m_warp * n_warp must be less than num_warps, m_warp: " << m_warp
      << ", n_warp: " << n_warp << ", num_warps: " << num_warps;
  ICHECK(M % (m_warp * kMPerWarp) == 0 && N % (n_warp * k_n_per_warp) == 0)
      << "PPU T.gemm warp partition would truncate output tile: M=" << M
      << ", N=" << N << ", m_warp=" << m_warp << ", n_warp=" << n_warp
      << ", num_warps=" << num_warps << ", policy="
      << (policy.IsFullRow()   ? "FullRow"
          : policy.IsFullCol() ? "FullCol"
                               : "Square")
      << ". M must be divisible by " << m_warp * kMPerWarp << " and N by "
      << n_warp * k_n_per_warp
      << "; adjust `threads`, the block tile shape, or the warp policy.";
  policy.m_warp = m_warp;
  policy.n_warp = n_warp;
  return {m_warp, n_warp};
}

} // namespace

struct Gemm {
  static String SelectInst(const GemmNode &op, int block_size, Target target) {
    if (op.isWgmma_) {
      FatalUnsupportedPpuArchGemm(op, target, "T.wgmma_gemm()");
    }
    if (op.isTcgen05_) {
      FatalUnsupportedPpuArchGemm(op, target, "T.tcgen05_gemm()");
    }
    // On an arch-less target the Python MMA emitter resolved the generation
    // from the real device (hgcc.get_target_arch) and applied its own gate, so
    // only the dtype-combination gate is left here.
    int arch = std::numeric_limits<int>::max();
    if (auto arch_attr = target->GetAttr<ffi::String>("arch")) {
      arch = GetPPUArchInt(target);
    }
    if (!AllowPPUMMA(op, arch)) {
      FatalMmaSyncUnavailable(op, target);
    }
    if (op.transB_ && IsFragmentBuffer(op.a_) && IsSharedBuffer(op.b_) &&
        op.a_->dtype.bits() == 16 && op.b_->dtype.bits() == 16 &&
        op.c_->dtype.bits() == 32 && op.annotations_.count(kAFromGemmC) &&
        GetPPUArchInt(target) == 10) {
      return kPpuMMAChainedRS;
    }
    return kPpuMMA;
  }

  static std::pair<int, int>
  ComputeWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, String gemm_inst) {
    int num_warps = block_size / TargetPPUGetWarpSize(target);
    // PPU MMA uses m16n16 tiles, so each warp needs at least 16 columns.
    constexpr int kNPerWarp = 16;
    return ComputeDefaultWarpPartition(policy, M, N, num_warps, kNPerWarp);
  }

  static bool ReuseExistingSharedLayout(String gemm_inst, const tvm::tl::Gemm &,
                                        const Buffer &, const Layout &,
                                        const Target &) {
    return gemm_inst == kPpuMMA || gemm_inst == kPpuMMABlockScaled;
  }

  static String InstructionKind(String gemm_inst) {
    if (gemm_inst == kPpuMMA) {
      return "mma";
    }
    return "unknown";
  }
};

} // namespace ppu

namespace {

bool MatchPpuGemmTarget(Target target) { return TargetIsPPU(target); }

bool RegisterPpuGemm() {
  RegisterGemmImpl(GemmImpl{
      "ppu.Gemm",
      MatchPpuGemmTarget,
      ppu::Gemm::SelectInst,
      ppu::Gemm::ComputeWarpPartition,
      ppu::Gemm::ReuseExistingSharedLayout,
  });
  return true;
}

const bool ppu_gemm_registered = RegisterPpuGemm();

} // namespace

} // namespace tl
} // namespace tvm
