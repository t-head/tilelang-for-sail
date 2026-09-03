/*!
 * \file ppu/layout/ppu_gemm_layouts.h
 * \brief PPU-specific shared-memory layout variants for cutlass-ppu Actlize
 * layouts.
 */

#ifndef TL_PPU_LAYOUT_PPU_GEMM_LAYOUTS_H_
#define TL_PPU_LAYOUT_PPU_GEMM_LAYOUTS_H_

#include "layout/layout.h"  // Layout, Buffer, etc.

namespace tvm {
namespace tl {

// PPU: padded/shared-memory layout variants for cutlass-ppu Actlize layouts.
Layout MakeGemmBLayoutPaddedPPU(int stride, int continuous, int element_size,
                                bool k_inner = true);

Layout MakeGemmABLayoutPPU(int mat_stride, int mat_continuous, int continuity,
                           int element_size, bool k_inner = true,
                           bool is_gemm_rs = false);

// PPU: Buffer-based wrapper around MakeGemmABLayoutPPU.
Layout MakePPUSwizzledLayout(const tirx::Buffer &buffer, bool k_inner = true,
                             bool allow_pad = true,
                             bool is_gemm_rs = false);

}  // namespace tl
}  // namespace tvm

#endif  // TL_PPU_LAYOUT_PPU_GEMM_LAYOUTS_H_
