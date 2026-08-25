#pragma once

#if (defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 150))
#include "gemm_ppu0015.h"
#elif (defined(__HGGC_ARCH__) && (__HGGC_ARCH__ >= 100))
#include "gemm_ppu0010.h"
#else
// No matching PPU architecture found.
#endif
