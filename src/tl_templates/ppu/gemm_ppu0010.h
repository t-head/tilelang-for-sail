#pragma once

// PPU: use the local PPU MMA template so Acompute dispatch can live outside
#include "gemm_mma.h"

namespace tl {
using tl_mma::gemm_rs;
using tl_mma::gemm_sr;
using tl_mma::gemm_ss;
} // namespace tl
