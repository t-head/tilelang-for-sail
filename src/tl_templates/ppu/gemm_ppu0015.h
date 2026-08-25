#pragma once

// PPU: PPU0015 entry keeps the PPU MMA template independent from HGGC templates.
#include <cute/arch/mma_ppu0015.hpp>
#include <tl_templates/ppu/hggc_fp8.h>

#include "gemm_mma.h"

namespace tl {
using tl_mma::gemm_rs;
using tl_mma::gemm_sr;
using tl_mma::gemm_ss;
} // namespace tl
