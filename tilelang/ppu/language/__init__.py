"""PPU language dialect: common TileLang plus PPU extensions."""

from __future__ import annotations

from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL
from tilelang.language.annotations import annotate_l2_hit_ratio  # noqa: F401
from tilelang.language.builtin import (  # noqa: F401
    __ffs,
    __fns,
    __ldg,
    disable_warp_group_reg_alloc,
    get_lane_idx,
    get_warp_group_idx,
    get_warp_idx,
    get_warp_idx_sync,
    ldg128,
    ldg256,
    ldg32,
    ldg64,
    lds128,
    lds32,
    lds64,
    match_all_sync,
    match_any_sync,
    named_barrier_arrive,
    no_set_max_nreg,
    shuffle_elect,
    stg128,
    stg256,
    stg32,
    stg64,
    sts128,
    sts32,
    sts64,
)
from tilelang.ppu.intrinsics import get_swizzle_layout, make_mma_swizzle_layout  # noqa: F401

# The PPU dialect shadows a handful of common constructs with versions that
# expose PPU's knobs as typed keywords: T.copy (disable_tma, eviction_policy,
# prefer_instruction), T.gemm / T.gemm_blockscaled (packed MXFP4 scales),
# T.Parallel (prefer_async), T.unroll (unroll_factor) and the reduction
# operators (nan_propagate). Semantics match the common versions; the extra
# keywords are recorded on the op and consumed by the PPU pipeline.
from .kernel import Kernel  # noqa: F401
from .reduce_op import reduce_absmax, reduce_max, reduce_min  # noqa: F401
from tilelang.language.math_intrinsics import (  # noqa: F401
    __cos,
    __exp,
    __exp10,
    __log,
    __log2,
    __log10,
    __sin,
    __tan,
    fast_rcp,
    ieee_add,
    ieee_fdiv,
    ieee_fmaf,
    ieee_frcp,
    ieee_frsqrt,
    ieee_fsqrt,
    ieee_mul,
    ieee_sub,
)
from .copy_op import copy  # noqa: F401
from .gemm_op import gemm, gemm_blockscaled  # noqa: F401
from .loop import Parallel, Unroll, unroll  # noqa: F401
from .pdl import *  # noqa: F401,F403
from .pdl import __all__ as _PDL_ALL
from .print import *  # noqa: F401,F403
from .print import __all__ as _PRINT_ALL
from .random import *  # noqa: F401,F403
from .random import __all__ as _RANDOM_ALL
from .tir import *  # noqa: F401,F403
from .tir import __all__ as _TIR_ALL

_PPU_API_ALL = (
    "Kernel",
    "Parallel",
    "Unroll",
    "__cos",
    "__exp",
    "__exp10",
    "__log",
    "__log2",
    "__log10",
    "__sin",
    "__tan",
    "annotate_l2_hit_ratio",
    "copy",
    "fast_rcp",
    "gemm",
    "gemm_blockscaled",
    "disable_warp_group_reg_alloc",
    "get_lane_idx",
    "get_swizzle_layout",
    "get_warp_group_idx",
    "get_warp_idx",
    "get_warp_idx_sync",
    "__ffs",
    "__fns",
    "__ldg",
    "ieee_add",
    "ieee_fdiv",
    "ieee_fmaf",
    "ieee_frcp",
    "ieee_frsqrt",
    "ieee_fsqrt",
    "ieee_mul",
    "ieee_sub",
    "ldg128",
    "ldg256",
    "ldg32",
    "ldg64",
    "lds128",
    "lds32",
    "lds64",
    "make_mma_swizzle_layout",
    "match_all_sync",
    "match_any_sync",
    "named_barrier_arrive",
    "no_set_max_nreg",
    "reduce_absmax",
    "reduce_max",
    "reduce_min",
    "shuffle_elect",
    "stg128",
    "stg256",
    "stg32",
    "stg64",
    "sts128",
    "sts32",
    "sts64",
    "unroll",
)

__tilelang_dialect__ = "ppu"
__all__ = tuple(
    dict.fromkeys(
        (
            *_COMMON_ALL,
            *_PPU_API_ALL,
            *_PDL_ALL,
            *_PRINT_ALL,
            *_RANDOM_ALL,
            *_TIR_ALL,
        )
    )
)

del _COMMON_ALL, _PPU_API_ALL, _PDL_ALL, _PRINT_ALL, _RANDOM_ALL, _TIR_ALL
