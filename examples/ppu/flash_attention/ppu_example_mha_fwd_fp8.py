"""V20: hybrid self-authored TileLang FP8 prefill FlashAttention for the
T-Head PPU ZW-M890P (campaign: flash_attention_fp8, production).

Implements the captured Qwen3.7-Plus PAI-vLLM contract
(`flash_attn_3::fwd`, `vllm.attention.utils.fa_utils.flash_attn_varlen_func`):
varlen causal prefill attention over a paged KV cache with GQA 8:1 and
independent per-request/per-KV-head fp32 descales.

  q, k, v : float8_e4m3fn
  out     : bfloat16 (destination-passing; mutated in place and returned)

Math (per request r, matching reference.py):
  scores = (q * q_descale[r]) @ (k * k_descale[r])^T * softmax_scale
  prob   = softmax(scores, causal mask: key_pos <= query_pos + kv_len - q_len)
  out    = (prob @ (v * v_descale[r]))  -> bfloat16

The descales are runtime tensor values read inside the kernel; no scale is
ever assumed to be one.

Architecture: a hybrid of the two self-authored route designs measured on
this wheel. The shape-invariant dispatch selects exactly one route per
request geometry; the two routes run DIFFERENT tile bodies because each
keeps the recipe it was measured with:

* Dense route — the champion single-pass kernel, byte-identical to the
  incumbent. One TileLang kernel per (total_q, q_len, reverse_bx) geometry;
  `kv_len` is a runtime scalar (`T.int32`) kernel argument, so every shape
  with the same query geometry shares one compiled kernel. Grid =
  (q_heads, ceil(q_len / BM)); each block owns BM query rows of one query
  head and streams the causal KV window tile by tile. Its numerics are the
  FA3-oracle-safe FP8 recipe and the only known one that keeps the kv==q
  causal shape inside the allclose envelope on this wheel: BN=32 half-page
  tiles, DESCENDING tile order, Max_offset=8 (+8.0 exp2 offset, P quantized
  on the [0, 256] grid), Check_inf-guarded running maxima, rescale-before-PV
  ordering, 4-stage KV pipeline. BM/threads are chosen per static query
  geometry by `_block_geometry` (wave-starved tail q_len <= 640 uses
  BM=128/256 threads, large shapes BM=256/512).

* Split-KV route — the recovery split design that a same-session paired gate
  measured 1.207-1.289x faster per shape than the incumbent split route on
  every shape that exercises it, while passing the full correctness envelope.
  `_build_fa_split_prim_func` adds a split axis to the grid: each
  (head, q-block, split) block owns one balanced slice of the KV tile
  sequence intersected with the causal window and walks it ASCENDING with
  BN = one full 64-token cache page per online-softmax step, staging Q and
  P in quarter-bank swizzled shared layouts and running the fold/scale-first
  softmax selected per measured specialization at compile time. Geometries
  without query-tail rows (q_len % BM == 0) plus the audited partial-q
  specializations in `_TWO_REGION_TAIL_KEYS` run the two-region walk: a
  mask-free pipelined prefix (score accumulator zero-filled) over the
  fully-visible tiles plus the intact masked loop over the few boundary
  tiles crossing the causal edge — the FA-SAIL-oracle mainloop structure,
  bit-identical arithmetic — which removes the per-tile causal/bounds
  predicate evaluation from ~99% of the tiles on the pacing vector pipe.
  The partial-q specializations are safe under it because masked query-tail
  rows never need correct values: the split epilogue predicates both the
  partial-output and LSE writes on the q_len row test, the combine kernel's
  discarded tail lanes are never stored, and the softmax is row-local, so
  tail-row garbage cannot corrupt any stored row; only the causal-edge/
  kv-tail tiles of each q-block keep the full predicate. Partial-q
  geometries NOT in `_TWO_REGION_TAIL_KEYS` keep the single masked-init
  loop until audited and measured. The running-output rescale issues in
  its measured
  per-specialization placement (immediately before the PV gemm — vendor PPU
  mainloop order — for every split key except the measured early-placement
  set), and the epilogue writes fp32 partials: the
  normalized partial output `acc_o / logsum` (v_descale NOT applied yet)
  and the base-2 logsumexp. Rows whose slice holds no visible token keep
  logsum == 0 and emit partial 0 / LSE -inf, so empty slices drop out of the
  merge. `_build_fa_combine_prim_func` (identical to the incumbent's) then
  merges the partials per (query, head):
  `out = v_descale * sum_s exp2(lse_s - lse_total) * partial_s`, applying
  v_descale exactly once. The split slices partition the KV tile sequence
  disjointly and completely, so every row's causal window is covered exactly
  once.

Shared facts of both routes:

- GQA: the single KV head is shared by all 8 query heads via the grid's head
  axis (kv_head = head_idx // (q_heads // kv_heads)); K/V are never expanded.
- QK^T and P@V both run as float8_e4m3fn tensor-cell MMA with fp32
  accumulation. P@V keeps V K-major by staging V^T tiles in shared memory
  (fp8 MMA requires K-major B), i.e. `gemm(P, V^T_shared, acc_o,
  transpose_B)`.
- Global V pre-transpose staging (both routes): a per-request TileLang
  kernel (`_build_fa_vtranspose_prim_func`) materializes the logically-
  indexed transposed buffer `Vt[p, kvh, d, j] = Vc[BT[req_idx, p], j, kvh, d]`
  for the request's active pages, so every per-tile V stage in the mainloops
  is a contiguous-to-contiguous copy that lowers to the async cp_async class
  — replacing the in-mainloop scalar transposed staging (1-byte global
  load + strided 1-byte smem store per element, measured ~35% of the dense
  per-thread per-tile instructions and synchronous on the pacing chain).
  The V tiles are indexed by LOGICAL page (dense: the half-page tile's page
  floordiv(k, 2); split: tile k == page at BN == page_size) while K keeps
  the physical block-table gather; the buffer extent is bucketed so the
  contract shares one compiled transpose. Pure data movement — the staged
  bytes and every downstream operand are unchanged, so results remain
  bitwise identical (verified elementwise at unit and non-identity descales
  and under a permuted block table).
- Descales are folded exactly once. Dense: `qk_scale = (softmax_scale *
  log2(e)) * (q_descale * k_descale)` (the FA3 oracle's product order),
  fused into the base-2 online softmax's exp2 arguments plus the vendor
  Max_offset=8, whose constant LSE shift cancels in the normalization.
  Split: `qk_scale = q_descale * k_descale * softmax_scale * log2(e)`; the
  folded specializations fuse it into the exp2 arguments with raw-logit
  maxima (the scale enters the slice LSE once), the scale-first
  specializations scale the logits before the reductions. BOTH split
  formulations quantize P with the same vendor Max_offset=8 as the dense
  route (every P-producing exp2 argument carries +8.0, so probabilities land
  on the [0, 256] grid instead of [0, 1] before the e4m3 quantization, which
  matches the FA3 oracle's envelope); the constant +8 shifts every slice's
  logsum/LSE by the same amount, so it cancels in the partial normalization
  (acc_o and logsum carry it identically) and in the combine LSE differences.
  In both routes the running maxima track raw (positive-scaled) logits and
  `v_descale` scales the final normalized accumulator exactly once (dense
  epilogue / combine kernel).
- Online (flash) softmax in base 2 with running max/logsum; causal masking
  is applied per tile and the KV loop only visits tiles inside the causal
  window.
- Partial query tails (q_len % BM != 0) are handled with explicit per-row
  predication on the Q load and the Out store; partial KV tail pages are
  masked by the causal/bounds predicate (their garbage slots are inside the
  allocated page, so every load stays in bounds).

Dispatch (host-side, keyed only on shape invariants — never tensor values or
pointers). The two routes key their tables on DIFFERENT tile counts because
their tile widths differ: the split tables use BN=64 (page) tile counts and
the dense LPT table uses BN=32 (half-page) tile counts; base_blocks is
q_heads * ceil(q_len / 256) for both. The split decision, in order:

1. If the KV window holds fewer than 4 page tiles, stay dense.
2. `_SPLIT_OVERRIDES` (recovery measured per-shape routes at BN=64 tile
   counts, including dense s=1 for the kv==q causal grid): shapes 1-3 run
   s=2/3/4 fold+LPT-reversed, shapes 4-5 s=3 fold, shape 6 s=6 scale-first,
   shape 7 s=5 fold; the kv==q shape 0 stays dense.
3. Otherwise the wave model (39-slot residency, max s = 8, at least 32 tiles
   per split, >= 3% predicted wall-time gain) — routing shapes 8 (s=6) and
   9/10 (s=7, scale-first) on the current contract.
4. Otherwise the wave-saturated long-KV efficiency fallback (unmeasured
   geometries only).

The split `fold` posture comes from `_FOLD_SPLIT_KEYS`, the split-route
LPT q-block reversal from `_LPT_SPLIT_KEYS`, the split-route acc_o
rescale placement from `_RESCALE_EARLY_SPLIT_KEYS`, and the split-route
two-region walk extension to audited partial-query-tail geometries from
`_TWO_REGION_TAIL_KEYS` (all keyed at BN=64 tile counts); the walk
structure otherwise resolves from the static q_len invariant
`q_len % BM == 0`. All are compile-time cache-key fields so compile
sharing can never select the wrong variant. The dense
route resolves its own LPT reversal from `_LPT_DENSE_KEYS` keyed on BN=32
tile counts (the kv==q shape 0 is reversed). LPT reversal is a bijection of
the q-block launch axis (longest-chain-first); it only reassigns which
block computes which rows, so outputs are unchanged.

Compile budget: kernels are cached per static key (lru_cache); `kv_len`
stays runtime so same-route shapes share one compiled kernel. The 11-shape
contract compiles 1 dense + 8 split + 7 combine = 16 kernels per process,
which the episode-16 paired gate completed inside the gateway's 600 s
combined-gate limit.

Every call recomputes from its arguments: no value-, pointer-, or
output-caching; split workspaces are freshly allocated per call from the
current shapes. The vendor pass-config flags (fast math / AIU-lowering
eligibility) are retained from the incumbent's measured provenance.
"""

# NOTE: no `from __future__ import annotations` here — the TileLang TIR script
# parser reads the @T.prim_func parameter annotations as live AST expressions;
# PEP 563 string annotations would break shape resolution.

from functools import lru_cache

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T
from tilelang.layout import make_quarter_bank_swizzled_layout

_FP8 = "float8_e4m3fn"
_LOG2E = 1.4426950408889634
# Vendor PPU compile stack (matches every in-tree TileLang PPU flash-attention
# example and the measured provenance of this kernel): fast math reaches the
# device compiler (--use_fast_math), which is also how the FA3 oracle kernel
# is built — keeping the softmax exp2/div codegen on the same approximation
# path matters for matching the oracle inside the atol/rtol envelope. The
# disable-flags are cleared so the PPU AIU bulk-copy and swizzled-ldmatrix
# lowerings stay eligible. Raw string keys keep the dict valid across TileLang
# builds; the ldmatrix flag is only added when the installed TileLang exposes
# it (an environment invariant detected once at import — never keyed on inputs
# or pointers).
_PASS_CONFIGS = {
    "tl.enable_fast_math": True,
    "tl.disable_aiu_lower": False,
}
try:
    from tilelang.transform import PassConfigKey as _PassConfigKey

    if hasattr(_PassConfigKey, "TL_DISABLE_LDMAT_SWZL"):
        _PASS_CONFIGS["tl.disable_ldmat_swzl"] = False
except ImportError:  # pragma: no cover - tilelang is hard dependency
    pass
_BLOCK_M = 256  # query rows per attention block (dense route)
_THREADS = 512
# KV-streaming pipeline depth. 4 stages keep the next three tiles' K/V copies in
# flight while the current tile computes, hiding global-load latency behind the
# MMA + softmax of the long serial KV walk (a scheduling-only change: it alters
# when loads issue, not what is computed, so results stay bit-identical). The
# fp8 AIU bulk-copy path is gated off on this backend, so per-copy latency is
# high and the longer 1200-4096-tile chains amortize the deeper prefetch window.
_NUM_STAGES = 4
# Wave-starved tail geometries (q_len <= 640, i.e. the q_len=508 contract shape)
# on the DENSE route: BM=256 launches only ceil(508/256)*8 = 16 blocks on the
# 39-resident-slot device, leaving > half the CUs idle while every block streams
# the ~full KV window as a long serial tile chain. BM=128 doubles the grid to 32
# blocks — still a single residency wave — and halves the per-tile compute,
# cutting the wave-starved makespan. Keyed only on the static q_len compile
# invariant (never on runtime values or pointers); the large geometries keep the
# measured 256-row/512-thread geometry bit-for-bit.
_TAIL_Q_LEN_THRESHOLD = 640
_TAIL_BLOCK_M = 128
_TAIL_THREADS = 256

# Split-KV route geometry: the recovery split body always runs the
# 256-row/512-thread geometry. Splitting multiplies the block count, so even
# the q_len=508 tail shapes fill the device at BM=256 (16 base blocks x s
# splits), and the longer rows amortize the warp-local FullRow reductions the
# 256-row geometry resolves. The KV pipeline depth is the recovery split
# body's measured 2 stages (NOT the dense route's 4): the split route was
# measured with exactly this body/pipeline combination.
_SPLIT_BLOCK_M = 256
_SPLIT_THREADS = 512
_SPLIT_STAGES = 2
# Two-CTA concurrency probe geometry for the split route. Halving the block
# (BM 256->128, threads 512->256) halves the per-CTA register and shared
# footprint; at the 1-stage KV pipeline the split body nominally allocates
# ~104 KiB smem/CTA (Q 32 KiB + K 16 KiB + V^T 16 KiB + P 8 KiB + the
# two-region boundary band K/V 32 KiB), so two CTAs co-reside per CU inside
# the 256 KiB cap. A second pipeline stage (+32 KiB of multi-buffered K/V,
# ~136 KiB/CTA) would NOT fit two CTAs per CU, which is why this geometry
# keeps stages=1. Total warps/CU is UNCHANGED (2x8 == 1x16), so the hypothesis
# is two-CTA concurrency (independent barrier/pipeline phases, halved serial
# q-block chains, finer wave quantization), NOT added warps. Only reached
# through the explicit `_SPLIT_GEOM_PROBES` dispatch opt-in below; every
# non-opted-in shape keeps `_SPLIT_BLOCK_M/_SPLIT_THREADS/_SPLIT_STAGES`.
_SPLIT_BLOCK_M_SMALL = 128
_SPLIT_THREADS_SMALL = 256
_SPLIT_STAGES_SMALL = 1
# The split-KV combine kernel merges per-row partials only (no K/V staging to
# amortize), so it keeps the recovery-measured smaller geometry.
_COMBINE_BLOCK_M = 64
_COMBINE_THREADS = 256

# Split-KV dispatch constants (shape-derived, device-specific to ZW-M890P).
_SPLIT_GAIN_THRESHOLD = 0.03  # minimum wave-model gain before the split path pays
_MIN_TILES_PER_SPLIT = 32  # per-split KV work floor amortizing split/combine overhead
# At the 256-row/512-thread attention geometry one block occupies a whole CU
# (~180 KB pipelined shared memory; 512 threads x the measured register
# envelope), so residency is 1 block per CU.
_RESIDENT_SLOTS = 39  # 39 CUs x 1 resident block/CU (measured block limit)
_SPLIT_MAX_S = 8  # largest split count considered by the wave model
# Efficiency-regime split for wave-saturated long-KV dense grids: when the
# wave-model gain gate does not fire and the grid is large enough, the
# dispatch runs the split path at _EFFICIENCY_SPLIT_S even though the wave
# model predicts no wave-quantization gain (unmeasured-geometry fallback).
_EFFICIENCY_SPLIT_MIN_BLOCKS = 960  # measured wave-saturated grid of this contract
_EFFICIENCY_SPLIT_MIN_TILES = 1200  # smallest measured win; <= 960 tiles regresses
_EFFICIENCY_SPLIT_S = 4  # fallback for unmeasured wave-saturated long-KV grids
# Measured per-shape split-count overrides at the 256-row/512-thread geometry
# with the Q+P quarter-bank swizzles, keyed on (base_blocks, total_tiles) at
# BN=64 (page) tile counts, consulted BEFORE the wave model: the measured-best
# route is shape-specific — including dense (s=1) for the 240-tile kv==q
# causal grid, where the split/combine overhead of the short chains outweighs
# the wave gain. Grids not specified here fall through to the wave model,
# then the efficiency rule above. NOTE the tile-count convention differs from
# the dense-route `_LPT_DENSE_KEYS`, which is keyed at BN=32 tile counts.
_SPLIT_OVERRIDES = {
    (480, 240): 1,  # kv==q causal grid: dense measured-best (chains too short)
    (480, 480): 3,  # episode-22 remeasure under two-region: s=3 beats s=2 (-0.4%), s=4 regresses
    (480, 720): 3,
    (480, 960): 4,
    (480, 1200): 3,
    (480, 1440): 3,
    (480, 1680): 6,
    (480, 1920): 5,
    (240, 2040): 6,  # episode-22 s-sweep reconfirmed (s=4/s=8 regress)
    (40, 500): 5,
    (24, 512): 8,
    (200, 2020): 7,
    (32, 2035): 6,
}
# Split specializations whose tile body runs the folded softmax formulation
# (qk_scale fused into the base-2 exp2 arguments; the split epilogue scales
# the LSE once). Keyed on the dispatch invariants
# (base_blocks, total_tiles, num_split) at BN=64 tile counts. Fold wins the
# measured set below and LOSES the remaining split shapes (divisibility alone
# is not the criterion at BM=256), so the allowlist is exactly the measured
# fold-positive set. Unmeasured geometries keep the scale-first body.
# The (480,1680,6) key is an episode-22 re-measurement under the two-region
# walk: shape 6 was set to scale-first when measured with the single
# masked-init loop, and re-testing showed fold wins there once the mask-free
# prefix removed the per-tile predicate work (-0.9..-1.1% latency, reproduced
# in shape-bench and in every full-gate candidate run). Shape 8's bench-level
# fold gain did NOT reproduce in the full gate (fast-pod-matched candidate vs
# incumbent: neutral), so shape 8 keeps the scale-first body. The shape-1 key
# moves with its episode-22 split-count remeasure (s=2 -> s=3): fold+s=3
# together beat the incumbent fold+s=2 by -1.3..-1.9% across all gate runs;
# the stale (480,480,2) key is dropped because dispatch no longer selects
# s=2 there. The q_len=508 single-loop shapes stay scale-first (fold flat).
_FOLD_SPLIT_KEYS = {
    (480, 480, 3),
    (40, 500, 5),
    (24, 512, 8),
    (480, 720, 3),
    (480, 960, 4),
    (480, 1200, 3),
    (480, 1440, 3),
    (480, 1920, 5),
    (480, 1680, 6),
    (32, 2035, 6),
}
# (base_blocks, total_tiles) split specializations whose q-block launch axis
# is reversed to longest-chain-first (LPT) order, at BN=64 tile counts. The
# causal window makes each q-block's KV chain grow with its index; this device
# dispatches blocks in ascending linear id over 39 CUs x 1 resident block, so
# ascending order runs the short chains first and every residency wave pays
# its longest chain (wave-quantization imbalance). Reversing the bijective
# q-block axis schedules the longest chains first and levels the wave maxima.
# The gain exists only while the causal window is strongly triangular
# (kv/q <= ~4); the allowlist is exactly the measured reversal-positive split
# key set; every unmeasured geometry keeps the natural ascending order. The
# permutation only reassigns which block computes which rows, so outputs are
# unchanged. (The kv==q dense grid's reversal lives in `_LPT_DENSE_KEYS`.)
_LPT_SPLIT_KEYS = {
    (480, 480),   # s=2 split
    (480, 720),   # s=3 split
    (480, 960),   # s=4 split
}
# (base_blocks, total_tiles) split specializations whose tile body keeps the
# EARLY acc_o rescale placement (right after scores_scale is computed, before
# the exp2 pass). The v19 paired sweep measured the LATE placement
# (immediately before the PV gemm, vendor PPU mainloop order) faster on every
# contract split key EXCEPT (480, 1680), which regressed +7.8% under it and
# keeps the early form; every unlisted key runs the late placement. Both
# placements are scheduling variants of the same operands with the same FP
# results — the rescale is independent of the acc_s exp2/reduce/P-staging
# chain and both complete before the PV gemm accumulates. Keyed at BN=64
# (page) tile counts like the other split tables; resolved host-side from
# shape invariants and baked into the compile key (compile-time constant,
# never a runtime select).
_RESCALE_EARLY_SPLIT_KEYS = {
    (480, 1680),  # s=6 scale-first: measured late-placement regression
}
# (base_blocks, total_tiles) PARTIAL-QUERY-TAIL split specializations
# (q_len % BM != 0) that run the two-region walk even though the last
# q-block carries masked query-tail rows. Keyed at BN=64 (page) tile counts
# like the other split tables; resolved host-side from shape invariants and
# baked into the compile key. The recovery-era gate kept EVERY partial-q
# geometry on the single masked-init loop on the belief that masked tail rows
# make all tiles boundary tiles; that is only true of the q_len predicate
# term, and query-tail rows never need correct values: the split epilogue
# predicates BOTH the Out_partial and Glse writes on `bx*BM+i < q_len`, the
# combine kernel's discarded tail lanes are never stored, and the softmax is
# row-local, so tail-row garbage cannot corrupt any stored row. Dropping the
# per-tile causal/bounds predicate from the fully-visible majority is
# therefore numerics-neutral for stored rows. Only the audited + measured
# contract specialization is enabled here (host region audit GREEN plus the
# episode-27 paired gate); other partial-q geometries keep the single loop
# until they are audited and measured. The key is the dispatch identity
# (base_blocks, tiles) like the other split tables, so the single entry below
# is exactly the exercised q=508 specialization's identity; it necessarily
# also covers the sibling q_len in (256, 512] geometries with 2048 KV tiles,
# which share the same region partition (host region audit GREEN) and the same
# confinement argument.
_TWO_REGION_TAIL_KEYS = {
    (16, 2048),  # q=508, kv=131068, s=7 tail shapes 9/10
}
# Two-CTA concurrency probe opt-in (split route ONLY). Keyed on the block-
# geometry-invariant shape identity (q_len, split_total_tiles) at BN=64 tile
# counts so the lookup never depends on the chosen block_m. A listed shape
# bypasses the override/wave/efficiency split-count selection AND the
# fold/LPT/rescale/two-region tables, and instead runs the explicit full spec
# below (geometry + posture) so the probe can hold the incumbent's measured
# posture constant while varying ONLY the block geometry. Every shape NOT
# listed resolves exactly the incumbent route/posture. An empty dict disables
# the probe entirely (the incumbent dispatch is then byte-for-byte unchanged).
# Spec fields: block_m, threads, stages (the geometry under test) plus
# num_split, fold, reverse_bx, rescale_late, two_region (the posture, held to
# the incumbent's measured values for this shape so geometry is the sole
# variable).
_SPLIT_GEOM_PROBES = {
    # shape 7: q=15360 kv=122880 (1920 page tiles). Incumbent posture at
    # BM=256 is s=5, fold=True, reverse_bx=False, rescale_late=True,
    # two_region=True; hold all of it and vary geometry to the two-CTA body.
    # MEASURED: -23.3% paired (4/4 rounds), 2 CTAs/CU, zero spill, oracle
    # margin improved 0.0054 -> 0.0011.
    (15360, 1920): dict(
        block_m=_SPLIT_BLOCK_M_SMALL,
        threads=_SPLIT_THREADS_SMALL,
        stages=_SPLIT_STAGES_SMALL,
        num_split=5,
        fold=True,
        reverse_bx=False,
        rescale_late=True,
        two_region=True,
    ),
    # Sibling q_len=15360 split shapes, each holding its incumbent measured
    # posture (s / fold / LPT / rescale placement / two-region) at BM=256 so
    # geometry is the sole variable; measured pairwise one by one below.
    (15360, 480): dict(  # shape 1: s=3 fold LPT late two-region
        block_m=_SPLIT_BLOCK_M_SMALL,
        threads=_SPLIT_THREADS_SMALL,
        stages=_SPLIT_STAGES_SMALL,
        num_split=3,
        fold=True,
        reverse_bx=True,
        rescale_late=True,
        two_region=True,
    ),
    (15360, 720): dict(  # shape 2: s=3 fold LPT late two-region
        block_m=_SPLIT_BLOCK_M_SMALL,
        threads=_SPLIT_THREADS_SMALL,
        stages=_SPLIT_STAGES_SMALL,
        num_split=3,
        fold=True,
        reverse_bx=True,
        rescale_late=True,
        two_region=True,
    ),
    (15360, 960): dict(  # shape 3: s=4 fold LPT late two-region
        block_m=_SPLIT_BLOCK_M_SMALL,
        threads=_SPLIT_THREADS_SMALL,
        stages=_SPLIT_STAGES_SMALL,
        num_split=4,
        fold=True,
        reverse_bx=True,
        rescale_late=True,
        two_region=True,
    ),
    (15360, 1200): dict(  # shape 4: s=3 fold late two-region
        block_m=_SPLIT_BLOCK_M_SMALL,
        threads=_SPLIT_THREADS_SMALL,
        stages=_SPLIT_STAGES_SMALL,
        num_split=3,
        fold=True,
        reverse_bx=False,
        rescale_late=True,
        two_region=True,
    ),
    (15360, 1440): dict(  # shape 5: s=3 fold late two-region
        block_m=_SPLIT_BLOCK_M_SMALL,
        threads=_SPLIT_THREADS_SMALL,
        stages=_SPLIT_STAGES_SMALL,
        num_split=3,
        fold=True,
        reverse_bx=False,
        rescale_late=True,
        two_region=True,
    ),
    (15360, 1680): dict(  # shape 6: s=6 fold EARLY rescale two-region
        block_m=_SPLIT_BLOCK_M_SMALL,
        threads=_SPLIT_THREADS_SMALL,
        stages=_SPLIT_STAGES_SMALL,
        num_split=6,
        fold=True,
        reverse_bx=False,
        rescale_late=False,
        two_region=True,
    ),
    # shape 8: q=7680 kv=130560 (2040 page tiles). Scale-first (fold=False)
    # body, s=6, no LPT, late rescale, two-region; hold posture, vary geometry.
    (7680, 2040): dict(
        block_m=_SPLIT_BLOCK_M_SMALL,
        threads=_SPLIT_THREADS_SMALL,
        stages=_SPLIT_STAGES_SMALL,
        num_split=6,
        fold=False,
        reverse_bx=False,
        rescale_late=True,
        two_region=True,
    ),
    # shapes 9/10: q=508 kv=131068 (2048 page tiles), both contract shapes
    # share this identity. Hold the incumbent route posture exactly (wave-model
    # s=7, scale-first fold=False, no LPT, late rescale, two-region) and vary
    # ONLY the block geometry to the two-CTA body, the measured -14.2..-26.7%
    # mechanism on shapes 1-8. Partial-q at block_m=128 (508 % 128 = 124, the
    # last q-block carries 4 masked tail rows) keeps the two-region walk
    # under the confinement argument: softmax is row-local, the split epilogue
    # predicates BOTH the Out_partial and Glse writes on bx*BM+i < q_len, the
    # combine kernel's discarded tail lanes are never stored, the region
    # bounds come from row 0's causal edge (a stored row of every q-block),
    # and the masked band is at most ceil(128/64)+1 = 3 tiles per q-block —
    # the same structure audited for this specialization at BM=256.
    (508, 2048): dict(
        block_m=_SPLIT_BLOCK_M_SMALL,
        threads=_SPLIT_THREADS_SMALL,
        stages=_SPLIT_STAGES_SMALL,
        num_split=7,
        fold=False,
        reverse_bx=False,
        rescale_late=True,
        two_region=True,
    ),
}
# Finite running-max sentinel for the split kernel: large enough in magnitude
# to never beat a real logit, finite so all-masked rows stay NaN-free.
_NEG_SENTINEL = -1.0e30

# Global V pre-transpose staging. The mainloop V operand is consumed K-major
# (V^T tiles in shared, fp8 MMA requires a K-major B operand), but staging it
# transposed directly from the natural page layout lowers to a scalar 1-byte
# load/store loop per element (measured ~35% of the shape-0 dense per-thread
# per-tile instructions, synchronous on the pacing chain), while the
# contiguous-to-contiguous K-style copy lowers to vectorized async
# cp_async_gs<16>. The transpose kernel materializes one logically-indexed
# transposed V buffer per request,
#     Vt[p, kvh, d, j] = Vc[BT[req_idx, p], j, kvh, d]   (p < pages_used),
# so every per-tile V stage in the mainloop becomes a contiguous-to-
# contiguous T.copy (the proven cp_async class) indexed by the LOGICAL page
# (dense: floordiv(k, 2) of the half-page tile k; split: tile k itself); the
# K staging keeps the physical block-table gather. Pure data movement: the
# staged bytes and every downstream operand are unchanged, so results stay
# bitwise identical. The Vt extent is bucketed so the contract shares ONE
# hot-path compile: `_VTRANS_MIN_PAGES` covers the largest contract KV cache
# (kv_len 131068 -> 2048 pages); larger out-of-contract extents compile
# their own variant. Blocks at or beyond pages_used exit on a runtime guard.
_VTRANS_MIN_PAGES = 2048
_VTRANS_THREADS = 256

# Dense-route (base_blocks, total_tiles) specializations whose q-block launch
# axis is reversed to longest-chain-first (LPT) order, keyed at BN=32
# (half-page) tile counts — the dense route's tile width, NOT the split
# tables' BN=64 convention — and applied only when dispatch selects the dense
# route (num_split == 1). The causal window makes each q-block's KV chain grow
# with its index; this device dispatches blocks in ascending linear id over
# 39 CUs x 1 resident block, so natural order runs the short chains first and
# every residency wave pays its longest chain (wave-quantization imbalance).
# Reversing the bijective q-block axis schedules the longest chains first and
# levels the wave maxima. The incumbent measured -5.8% (kv==q), -1.6% (kv/q=2),
# -0.6% (kv/q=3), -2.2% (kv/q=4) for these keys on the current wheel;
# every other geometry keeps natural ascending order. The permutation only
# reassigns which block computes which rows, so outputs are unchanged. The
# flag is resolved at compile time (never a runtime select) and threaded
# through the dense kernel's lru_cache key.
_LPT_DENSE_KEYS = {
    (480, 480),   # shape 0: kv==q causal grid, dense
    (480, 960),   # shape 1: kv/q=2, dense
    (480, 1440),  # shape 2: kv/q=3, dense (probe)
    (480, 1920),  # shape 3: kv/q=4, dense (probe)
}


def _block_geometry(q_len: int) -> tuple[int, int]:
    """Dense-route (BM, threads) for one static query geometry."""
    if q_len <= _TAIL_Q_LEN_THRESHOLD:
        return _TAIL_BLOCK_M, _TAIL_THREADS
    return _BLOCK_M, _THREADS


def _build_fa_prim_func(
    total_q: int,
    q_start: int,
    q_len: int,
    req_idx: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    page_size: int,
    num_pages: int,
    bt_width: int,
    batch: int,
    softmax_scale: float,
    reverse_bx: bool,
    vt_pages: int,
):
    """One static-query-geometry dense FA kernel; `kv_len` is a runtime scalar arg.

    `reverse_bx` (compile-time) enables the longest-chain-first (LPT) q-block
    dispatch order for the measured reversal-positive dense keys; the physical
    launch axis is unchanged and the logical q-block index is the bijection
    `NB-1-bx`, so outputs are identical and there is no runtime select.

    `Vt` is the request's logically-indexed transposed V buffer produced by
    the V transpose kernel (`Vt[p, kvh, d, j] = Vc[BT[req_idx, p], j, kvh, d]`
    for p < pages_used): the per-tile V stage is a contiguous-to-contiguous
    T.copy indexed by the LOGICAL half-page tile's page, while the K stage
    keeps the physical block-table gather. Same staged bytes, same operands.
    """
    BM, threads = _block_geometry(q_len)
    # One online-softmax tile == 32 KV tokens == half a cache page. This must
    # stay aligned with the FA3 oracle's kBlockN=32 (see the module docstring):
    # the running-max/quantization trajectory of FP8 flash attention is
    # tile-boundary-sensitive on sharp causal rows.
    BN = page_size // 2
    NB = (q_len + BM - 1) // BM  # q-block count (static)

    @T.prim_func
    def main(
        Q: T.Tensor((total_q, q_heads, head_dim), _FP8),
        Kc: T.Tensor((num_pages, page_size, kv_heads, head_dim), _FP8),
        Vt: T.Tensor((vt_pages, kv_heads, head_dim, page_size), _FP8),
        BT: T.Tensor((batch, bt_width), "int32"),
        QD: T.Tensor((batch, kv_heads), "float32"),
        KD: T.Tensor((batch, kv_heads), "float32"),
        VD: T.Tensor((batch, kv_heads), "float32"),
        Out: T.Tensor((total_q, q_heads, head_dim), "bfloat16"),
        kv_len: T.int32,
    ):
        # Grid rasterization: the head axis is blockIdx.x (launch-fastest), so
        # the 8 GQA query heads that share one KV head run adjacently and reuse
        # the same KV pages from L2 instead of re-streaming them per head sweep.
        with T.Kernel(q_heads, NB, threads=threads) as (bh, bx):
            # Longest-chain-first (LPT) q-block order for the measured
            # reversal-positive dense keys only (see _LPT_DENSE_KEYS). The
            # physical launch axis is the ascending bx; reversing the bijective
            # q-block mapping schedules the longest causal chains first and
            # levels the residency-wave maxima. `reverse_bx` is a compile-time
            # constant, so this lowers to a static index remap with no runtime
            # select; unmeasured geometries keep the natural ascending order.
            if reverse_bx:
                bx = NB - 1 - bx
            Q_shared = T.alloc_shared([BM, head_dim], _FP8)
            K_shared = T.alloc_shared([BN, head_dim], _FP8)
            V_shared_t = T.alloc_shared([head_dim, BN], _FP8)
            P_shared = T.alloc_shared([BM, BN], _FP8)
            acc_s = T.alloc_fragment([BM, BN], "float32")
            acc_o = T.alloc_fragment([BM, head_dim], "float32")
            scores_max = T.alloc_fragment([BM], "float32")
            scores_max_prev = T.alloc_fragment([BM], "float32")
            scores_scale = T.alloc_fragment([BM], "float32")
            scores_sum = T.alloc_fragment([BM], "float32")
            logsum = T.alloc_fragment([BM], "float32")

            kvh = bh // (q_heads // kv_heads)
            # FA3 product order: (softmax_scale * log2e) * (q_descale * k_descale).
            qk_scale = softmax_scale * _LOG2E * (QD[req_idx, kvh] * KD[req_idx, kvh])
            v_scale = VD[req_idx, kvh]

            # Load this block's query rows (predicated tail rows -> zero).
            for i, d in T.Parallel(BM, head_dim):
                if bx * BM + i < q_len:
                    Q_shared[i, d] = Q[q_start + bx * BM + i, bh, d]
                else:
                    Q_shared[i, d] = T.cast(0.0, _FP8)

            T.fill(acc_o, 0.0)
            T.fill(logsum, 0.0)
            T.fill(scores_max, -T.infinity("float32"))
            T.fill(scores_max_prev, -T.infinity("float32"))

            # Causal window: the farthest KV position any row of this block
            # may attend is min((bx+1)*BM, q_len) - 1 + (kv_len - q_len).
            loop_range = T.min(
                T.ceildiv(kv_len, BN),
                T.ceildiv(T.min((bx + 1) * BM, q_len) + kv_len - q_len, BN),
            )
            # Tiles fully visible for every row of this block: the causal edge
            # is tightest for row 0, and the final KV page may be partial. Such
            # tiles need no per-element mask, only a zeroed accumulator.
            full_range = T.min(
                loop_range,
                T.min(
                    T.floordiv(bx * BM + kv_len - q_len + 1, BN),
                    T.floordiv(kv_len, BN),
                ),
            )

            # The FA3 oracle walks the KV window TOP-DOWN: from the tile at the
            # causal edge down to token 0, masked causal/seqlen region first.
            # The running-max trajectory — and hence the e4m3 grid each tile's
            # probabilities are quantized on — depends on that order, so it is
            # reproduced exactly.
            n_masked = loop_range - full_range
            for kk in T.Pipelined(n_masked, num_stages=_NUM_STAGES):
                k = loop_range - 1 - kk
                pid = BT[req_idx, T.floordiv(k, 2)]
                kv_start = (k % 2) * BN
                # Gather one 32-token KV tile (half a cache page) through the
                # block table.
                T.copy(Kc[pid, kv_start:kv_start + BN, kvh, 0:head_dim], K_shared)
                # V stages from the logically-indexed transposed buffer: the
                # tile's page is floordiv(k, 2), so the copy is contiguous on
                # both sides and lowers to the async cp_async class. Same
                # staged bytes as the physical gather.
                T.copy(
                    Vt[T.floordiv(k, 2), kvh, 0:head_dim, kv_start:kv_start + BN],
                    V_shared_t,
                )

                # Causal + seqlen mask; -inf lanes drop out of the softmax.
                for i, j in T.Parallel(BM, BN):
                    acc_s[i, j] = T.if_then_else(
                        (bx * BM + i < q_len)
                        & (k * BN + j < kv_len)
                        & (k * BN + j <= bx * BM + i + kv_len - q_len),
                        0.0,
                        -T.infinity("float32"),
                    )
                T.gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                # Online softmax (base 2) update, FA3 form: qk_scale folded
                # into the exp2 arguments plus the vendor Max_offset=8 (+8.0,
                # P quantized on the [0, 256] grid). A -inf running max (rows
                # with no visible token yet) is mapped to 0 exactly like the
                # vendor Check_inf path. scores_max_prev is reused to carry
                # max*qk_scale - 8 into the elementwise loop (no extra
                # fragment).
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(BM):
                    m_cur = T.if_then_else(
                        scores_max[i] == -T.infinity("float32"), 0.0, scores_max[i]
                    )
                    scores_scale[i] = T.exp2((scores_max_prev[i] - m_cur) * qk_scale)
                    scores_max_prev[i] = m_cur * qk_scale - 8.0
                for i, j in T.Parallel(BM, BN):
                    acc_s[i, j] = T.exp2(
                        acc_s[i, j] * qk_scale - scores_max_prev[i]
                    )
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(BM):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                # P @ V with fp8 operands: P staged in shared (bridges the QK
                # accumulator layout to the PV A-operand layout), V staged
                # K-major as V^T. The acc_o rescale is issued immediately
                # before the PV gemm (vendor PPU mainloop order) so the
                # 65536-mul vector pass sits adjacent to the tensor op and the
                # scheduler can interleave them instead of stalling on the
                # serialized softmax chain. Same operands, same FP results.
                T.copy(acc_s, P_shared)
                for i, d in T.Parallel(BM, head_dim):
                    acc_o[i, d] = acc_o[i, d] * scores_scale[i]
                T.gemm(
                    P_shared,
                    V_shared_t,
                    acc_o,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

            # Fully-visible region, still top-down: every tile here is a
            # complete 32-token tile inside the causal window and kv_len, so
            # no per-element mask is needed, only a zeroed accumulator.
            for kk in T.Pipelined(full_range, num_stages=_NUM_STAGES):
                k = full_range - 1 - kk
                pid = BT[req_idx, T.floordiv(k, 2)]
                kv_start = (k % 2) * BN
                T.copy(Kc[pid, kv_start:kv_start + BN, kvh, 0:head_dim], K_shared)
                # V stages from the logically-indexed transposed buffer (see
                # the masked-band copy above); contiguous on both sides.
                T.copy(
                    Vt[T.floordiv(k, 2), kvh, 0:head_dim, kv_start:kv_start + BN],
                    V_shared_t,
                )

                T.fill(acc_s, 0.0)
                T.gemm(
                    Q_shared,
                    K_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

                # Online softmax (base 2) update. Every tile in the fully-
                # visible region is complete for every row of this block, so
                # the running maximum in scores_max is always finite once the
                # reduce_max has folded the tile in (rows enter this region
                # having already seen >= 1 token, and the reduce over a mask-
                # free tile cannot return -inf): the FA3 Check_inf (-inf -> 0)
                # guard is dead here and is dropped, removing one per-row
                # select from the pacing vector chain. qk_scale and
                # Max_offset=8 stay folded into the exp2 arguments and
                # scores_max_prev is reused to carry max*qk_scale - 8. The
                # acc_o rescale is issued immediately before the PV gemm
                # (vendor PPU mainloop order), same operands and same FP
                # results as rescaling earlier.
                T.copy(scores_max, scores_max_prev)
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(BM):
                    scores_scale[i] = T.exp2(
                        (scores_max_prev[i] - scores_max[i]) * qk_scale
                    )
                    scores_max_prev[i] = scores_max[i] * qk_scale - 8.0
                for i, j in T.Parallel(BM, BN):
                    acc_s[i, j] = T.exp2(
                        acc_s[i, j] * qk_scale - scores_max_prev[i]
                    )
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(BM):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                T.copy(acc_s, P_shared)
                for i, d in T.Parallel(BM, head_dim):
                    acc_o[i, d] = acc_o[i, d] * scores_scale[i]
                T.gemm(
                    P_shared,
                    V_shared_t,
                    acc_o,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )

            # Normalize, apply the V descale, and write back in place.
            for i, d in T.Parallel(BM, head_dim):
                if bx * BM + i < q_len:
                    Out[q_start + bx * BM + i, bh, d] = T.cast(
                        acc_o[i, d] / logsum[i] * v_scale, "bfloat16"
                    )

    return main


def _build_fa_split_prim_func(
    total_q: int,
    q_start: int,
    q_len: int,
    req_idx: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    page_size: int,
    num_pages: int,
    bt_width: int,
    batch: int,
    softmax_scale: float,
    num_split: int,
    fold: bool,
    reverse_bx: bool,
    rescale_late: bool,
    two_region: bool,
    block_m: int,
    threads: int,
    stages: int,
    vt_pages: int,
):
    """Split-KV attention kernel: one balanced KV-tile slice per grid.z block.

    Recovery split body (the measured-faster split design): one KV tile == one
    full cache page (BN = page_size), ASCENDING walk, Q+P quarter-bank
    swizzled staging, and the softmax formulation selected at compile time.
    `fold` selects the softmax formulation of the tile body: True uses the
    scale-fold (qk_scale fused into the base-2 exp2 arguments, raw-logit
    maxima, scaled LSE in the epilogue); False keeps the scale-first form
    (scale the logits, reduce the scaled maxima). The dispatch enables the
    fold only for measured fold-positive specializations (`_FOLD_SPLIT_KEYS`);
    every other split shape keeps the scale-first body.

    `rescale_late` selects the placement of the per-tile acc_o rescale, a
    measured per-specialization scheduling variant: True issues it immediately
    before the PV gemm (vendor PPU mainloop order, the measured-best on every
    contract split key except the `_RESCALE_EARLY_SPLIT_KEYS` set); False keeps
    the early placement right after `scores_scale` is computed. Both are the
    same operands and the same FP results (the rescale is independent of the
    acc_s exp2/reduce/P-staging chain; both complete before the PV gemm).

    `two_region` selects the walk structure, a measured per-tile vector-work
    cut: True peels the causal/bounds boundary tiles off the main walk. The
    slice then runs as a mask-free pipelined prefix (acc_s zero-filled, no
    predicate) over the fully-visible tiles plus a short masked loop over the
    boundary band `[full_range, loop_range)` carrying the exact full
    predicate — the FA-SAIL-oracle mainloop structure. Only the few
    causal-edge / kv-tail tiles per q-block land in the masked band, so the
    3-compare/2-AND/1-select mask-init is removed from ~99% of the tiles on
    the pacing vector pipe; the arithmetic is bit-identical for every stored
    row (the removed predicate is everywhere true on the mask-free tiles).
    False keeps the single masked-init loop: every tile initializes acc_s
    with the causal + bounds predicate before the QK^T gemm. Dispatch selects
    True for every `q_len % BM == 0` geometry plus the audited partial-q
    specializations in `_TWO_REGION_TAIL_KEYS`. The partial-q case is safe
    because a masked query-tail row never needs a correct value: the
    epilogue below predicates both the Out_partial and Glse writes on
    `bx*BM+i < q_len`, the combine kernel's discarded tail lanes are never
    stored, and the softmax is row-local — so a tail row's mask-free garbage
    (finite, Q_shared zero-filled) cannot corrupt any stored row, and its
    causal/bounds visibility is immaterial. The region bounds themselves come
    from row 0's causal edge, which is a stored row of every q-block
    (NB = ceil(q_len/BM)). Unlisted partial-q geometries keep the single
    masked-init loop until audited and measured.

    `kv_len` is a runtime scalar argument so all shapes sharing the static
    key reuse one compiled kernel. Rows whose slice holds no visible token
    keep logsum == 0 and emit partial 0 / LSE -inf; the combine kernel drops
    them out of the merge. v_descale is applied once in the combine.

    `block_m`/`threads`/`stages` are the compile-time block geometry (query
    rows per CTA, threads per CTA, KV pipeline depth). The incumbent default
    is 256/512/2; the two-CTA concurrency probe selects 128/256/1 through the
    `_SPLIT_GEOM_PROBES` dispatch opt-in. The tile-body arithmetic is
    independent of `block_m` (the online softmax is row-local and each row's
    KV walk order is unchanged), so a geometry change is numerics-neutral per
    stored row; only the CTA-to-row assignment and the per-CTA region bounds
    move.
    """
    BM = block_m
    BN = page_size  # one KV tile == one cache page
    NB = (q_len + BM - 1) // BM  # q-block count (static)

    @T.prim_func
    def main(
        Q: T.Tensor((total_q, q_heads, head_dim), _FP8),
        Kc: T.Tensor((num_pages, page_size, kv_heads, head_dim), _FP8),
        Vt: T.Tensor((vt_pages, kv_heads, head_dim, page_size), _FP8),
        BT: T.Tensor((batch, bt_width), "int32"),
        QD: T.Tensor((batch, kv_heads), "float32"),
        KD: T.Tensor((batch, kv_heads), "float32"),
        Out_partial: T.Tensor((num_split, total_q, q_heads, head_dim), "float32"),
        Glse: T.Tensor((num_split, total_q, q_heads), "float32"),
        kv_len: T.int32,
    ):
        with T.Kernel(q_heads, NB, num_split, threads=threads) as (
            bh,
            bx,
            bz,
        ):
            # LPT q-block order for the measured reversal-positive
            # specializations only (compile-time; see _LPT_SPLIT_KEYS).
            if reverse_bx:
                bx = NB - 1 - bx
            Q_shared = T.alloc_shared([BM, head_dim], _FP8)
            K_shared = T.alloc_shared([BN, head_dim], _FP8)
            V_shared_t = T.alloc_shared([head_dim, BN], _FP8)
            P_shared = T.alloc_shared([BM, BN], _FP8)
            acc_s = T.alloc_fragment([BM, BN], "float32")
            acc_o = T.alloc_fragment([BM, head_dim], "float32")
            scores_max = T.alloc_fragment([BM], "float32")
            scores_max_prev = T.alloc_fragment([BM], "float32")
            scores_scale = T.alloc_fragment([BM], "float32")
            scores_sum = T.alloc_fragment([BM], "float32")
            logsum = T.alloc_fragment([BM], "float32")

            # Quarter-bank swizzled layout for the Q tile (A operand of QK^T,
            # staged once per block, ldmatrix-read on every tile): values
            # unchanged; the XOR permutation simplifies the lowered addressing
            # and cuts the per-read issue cost. P_shared carries the same
            # annotation: the PV A-operand ldmatrix reads get the same
            # addressing simplification. K alone raised TSM conflicts on this
            # backend and the 64B-contiguous V^T buffer accepts no swizzle
            # scheme, so Q and P carry the annotation.
            T.annotate_layout(
                {
                    Q_shared: make_quarter_bank_swizzled_layout(Q_shared),
                    P_shared: make_quarter_bank_swizzled_layout(P_shared),
                }
            )

            kvh = bh // (q_heads // kv_heads)
            qk_scale = QD[req_idx, kvh] * KD[req_idx, kvh] * softmax_scale * _LOG2E

            # Load this block's query rows (predicated tail rows -> zero).
            for i, d in T.Parallel(BM, head_dim):
                if bx * BM + i < q_len:
                    Q_shared[i, d] = Q[q_start + bx * BM + i, bh, d]
                else:
                    Q_shared[i, d] = T.cast(0.0, _FP8)

            T.fill(acc_o, 0.0)
            T.fill(logsum, 0.0)
            T.fill(scores_max, _NEG_SENTINEL)
            T.fill(scores_max_prev, _NEG_SENTINEL)

            # Causal window bounds (identical to the dense route).
            loop_range = T.min(
                T.ceildiv(kv_len, BN),
                T.ceildiv(T.min((bx + 1) * BM, q_len) + kv_len - q_len, BN),
            )

            # Balanced KV-tile partition across the split axis, intersected
            # with this q-block's causal window. The slices over bz partition
            # [0, total_tiles) disjointly and completely, so the intersected
            # pieces cover [0, loop_range) exactly once per q-block.
            total_tiles = T.ceildiv(kv_len, BN)
            tiles_per_split = T.floordiv(total_tiles, num_split)
            remaining = T.floormod(total_tiles, num_split)
            split_len = tiles_per_split + T.if_then_else(bz < remaining, 1, 0)
            split_begin = tiles_per_split * bz + T.min(bz, remaining)
            eff_end = T.min(split_begin + split_len, loop_range)
            n_slice = T.max(eff_end - split_begin, 0)

            if two_region:
                # Two-region walk (the FA-SAIL oracle's mainloop structure):
                # tiles fully visible for every row of this block need no
                # per-element mask, only a zeroed accumulator. full_range is
                # the dense-route formula: below it every tile is completely
                # inside the causal window and the kv length, so only the
                # boundary band [full_range, loop_range) — the few tiles
                # crossing row 0's causal edge, plus the kv tail tile when
                # kv_len % BN != 0 — carries the full predicate. The removed
                # predicate is everywhere true on the mask-free tiles, so the
                # arithmetic is bit-identical to the single masked-init loop;
                # the band stays its own intact masked loop. Region bounds are
                # clamped so empty regions lower to zero-trip loops.
                full_range = T.min(
                    loop_range,
                    T.min(
                        T.floordiv(bx * BM + kv_len - q_len + 1, BN),
                        T.floordiv(kv_len, BN),
                    ),
                )
                unmasked_end = T.min(eff_end, full_range)
                masked_begin = T.max(split_begin, full_range)
                unmasked_start = T.min(split_begin, unmasked_end)
                n_unmasked = T.max(unmasked_end - unmasked_start, 0)

                # Dedicated single-buffer K/V staging for the boundary band
                # below: the pipelined prefix multi-buffers K_shared /
                # V_shared_t, and a non-pipelined consumer of those tiles
                # mismatches their lowered multi-buffer shape. Separate single
                # buffers let the band stay serial and out of the pipeliner
                # entirely (a large compile-time saving for a band of at most
                # ceil(BM/BN)+1 tiles). +2 pages of smem keeps the block
                # inside the 256 KiB cap; occupancy is already 1 block/CU.
                K_band_shared = T.alloc_shared([BN, head_dim], _FP8)
                V_band_shared_t = T.alloc_shared([head_dim, BN], _FP8)

                # Mask-free prefix: fully-visible tiles, zeroed accumulator.
                # Iterated in the 1-arg extent form (k = start + kk) matching
                # the single-loop pipeline structure that compiles fastest on
                # this wheel.
                for kk in T.Pipelined(n_unmasked, num_stages=stages):
                    k = unmasked_start + kk
                    pid = BT[req_idx, k]
                    # Gather one KV page through the block table
                    # (BN == page_size).
                    T.copy(Kc[pid, 0:page_size, kvh, 0:head_dim], K_shared)
                    # V stages from the logically-indexed transposed buffer:
                    # at BN == page_size the tile index k IS the logical page,
                    # so the copy is contiguous on both sides (cp_async class).
                    T.copy(Vt[k, kvh, 0:head_dim, 0:page_size], V_shared_t)

                    T.fill(acc_s, 0.0)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    if fold:
                        # Folded online softmax (base 2): qk_scale fused into
                        # the exp2 arguments (one contractible multiply-add
                        # per score element instead of a standalone scale
                        # pass plus subtract), maxima tracked on the raw
                        # logits (valid because this contract's descales and
                        # softmax_scale are all positive), and
                        # scores_max_prev reused to carry -max*qk_scale.
                        T.copy(scores_max, scores_max_prev)
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(BM):
                            scores_scale[i] = T.exp2(
                                (scores_max_prev[i] - scores_max[i]) * qk_scale
                            )
                            scores_max_prev[i] = -scores_max[i] * qk_scale
                        if not rescale_late:
                            # Early rescale placement (measured-best for the
                            # `_RESCALE_EARLY_SPLIT_KEYS` specializations);
                            # the late placement lives in the P@V section.
                            for i, d in T.Parallel(BM, head_dim):
                                acc_o[i, d] = acc_o[i, d] * scores_scale[i]
                        for i, j in T.Parallel(BM, BN):
                            # Vendor Max_offset=8: the +8 produces P on the
                            # [0, 256] grid instead of [0, 1] before the e4m3
                            # quantization (matching the FA3 oracle's
                            # envelope); 2^8 shifts logsum and acc_o
                            # identically, so the offset cancels in the
                            # partial normalization, and in the combine
                            # differences as a constant LSE shift.
                            acc_s[i, j] = T.exp2(
                                acc_s[i, j] * qk_scale + scores_max_prev[i] + 8.0
                            )
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(BM):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    else:
                        # Scale-first online softmax (base 2): scale the
                        # logits, reduce the scaled maxima, exp2 of the
                        # scaled difference.
                        for i, j in T.Parallel(BM, BN):
                            acc_s[i, j] = acc_s[i, j] * qk_scale

                        T.copy(scores_max, scores_max_prev)
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(BM):
                            scores_scale[i] = T.exp2(
                                scores_max_prev[i] - scores_max[i]
                            )
                        if not rescale_late:
                            # Early rescale placement (see the fold branch).
                            for i, d in T.Parallel(BM, head_dim):
                                acc_o[i, d] = acc_o[i, d] * scores_scale[i]
                        for i, j in T.Parallel(BM, BN):
                            # Vendor Max_offset=8 (see the fold branch): P on
                            # the [0, 256] grid; the constant shift cancels
                            # in the partial normalization and the combine
                            # differences.
                            acc_s[i, j] = T.exp2(acc_s[i, j] - scores_max[i] + 8.0)
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(BM):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                    # P @ V with fp8 operands: P staged in shared (bridges the
                    # QK accumulator layout to the PV A-operand layout), V
                    # staged K-major as V^T. The acc_o rescale placement is a
                    # measured per-specialization compile-time choice
                    # (`rescale_late`): the late placement issues the
                    # 65536-mul vector pass immediately before the PV gemm
                    # (vendor PPU mainloop order) so the scheduler can
                    # interleave it with the tensor op instead of serializing
                    # it ahead of the exp2/reduce/P-staging chain —
                    # measured-best on every contract split key except the
                    # `_RESCALE_EARLY_SPLIT_KEYS` set, which keeps the early
                    # form. Same operands, same FP results either way.
                    T.copy(acc_s, P_shared)
                    if rescale_late:
                        for i, d in T.Parallel(BM, head_dim):
                            acc_o[i, d] = acc_o[i, d] * scores_scale[i]
                    T.gemm(
                        P_shared,
                        V_shared_t,
                        acc_o,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                # Masked boundary band: the causal-edge / kv-tail tiles keep
                # the full causal + bounds predicate; -inf lanes drop out of
                # the softmax. Identical body and predicate to the
                # single-loop form. The band holds at most ceil(BM/BN)+1
                # tiles, so it walks serially through the dedicated
                # single-buffer K/V staging: serial keeps the band out of the
                # pipeliner (a large compile-time saving), its own buffers
                # avoid the pipelined prefix's multi-buffer shape, and the
                # band is the tail of the ascending walk with nothing after
                # it to overlap with anyway. Q_shared and P_shared are
                # single-buffer and safely reused.
                n_masked = T.max(T.max(masked_begin, eff_end) - masked_begin, 0)
                for kk in T.serial(n_masked):
                    k = masked_begin + kk
                    pid = BT[req_idx, k]
                    T.copy(Kc[pid, 0:page_size, kvh, 0:head_dim], K_band_shared)
                    # V from the logical transposed buffer (tile k == page k).
                    T.copy(Vt[k, kvh, 0:head_dim, 0:page_size], V_band_shared_t)

                    for i, j in T.Parallel(BM, BN):
                        acc_s[i, j] = T.if_then_else(
                            (bx * BM + i < q_len)
                            & (k * BN + j < kv_len)
                            & (k * BN + j <= bx * BM + i + kv_len - q_len),
                            0.0,
                            -T.infinity("float32"),
                        )
                    T.gemm(
                        Q_shared,
                        K_band_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    if fold:
                        # Folded online softmax (base 2) — bit-identical to
                        # the mask-free prefix body above.
                        T.copy(scores_max, scores_max_prev)
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(BM):
                            scores_scale[i] = T.exp2(
                                (scores_max_prev[i] - scores_max[i]) * qk_scale
                            )
                            scores_max_prev[i] = -scores_max[i] * qk_scale
                        if not rescale_late:
                            # Early rescale placement (see the prefix body).
                            for i, d in T.Parallel(BM, head_dim):
                                acc_o[i, d] = acc_o[i, d] * scores_scale[i]
                        for i, j in T.Parallel(BM, BN):
                            # Vendor Max_offset=8 (see the prefix body).
                            acc_s[i, j] = T.exp2(
                                acc_s[i, j] * qk_scale + scores_max_prev[i] + 8.0
                            )
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(BM):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    else:
                        # Scale-first online softmax (base 2) — bit-identical
                        # to the mask-free prefix body above.
                        for i, j in T.Parallel(BM, BN):
                            acc_s[i, j] = acc_s[i, j] * qk_scale

                        T.copy(scores_max, scores_max_prev)
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(BM):
                            scores_scale[i] = T.exp2(
                                scores_max_prev[i] - scores_max[i]
                            )
                        if not rescale_late:
                            # Early rescale placement (see the prefix body).
                            for i, d in T.Parallel(BM, head_dim):
                                acc_o[i, d] = acc_o[i, d] * scores_scale[i]
                        for i, j in T.Parallel(BM, BN):
                            # Vendor Max_offset=8 (see the prefix body).
                            acc_s[i, j] = T.exp2(acc_s[i, j] - scores_max[i] + 8.0)
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(BM):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                    T.copy(acc_s, P_shared)
                    if rescale_late:
                        for i, d in T.Parallel(BM, head_dim):
                            acc_o[i, d] = acc_o[i, d] * scores_scale[i]
                    T.gemm(
                        P_shared,
                        V_band_shared_t,
                        acc_o,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
            else:
                # Single masked-init walk for the q_len % BM != 0
                # specializations: masked query-tail rows make every tile a
                # boundary tile, so the predicate stays on all tiles.
                for kk in T.Pipelined(n_slice, num_stages=stages):
                    k = split_begin + kk
                    pid = BT[req_idx, k]
                    # Gather one KV page through the block table
                    # (BN == page_size).
                    T.copy(Kc[pid, 0:page_size, kvh, 0:head_dim], K_shared)
                    # V stages from the logically-indexed transposed buffer:
                    # at BN == page_size the tile index k IS the logical page,
                    # so the copy is contiguous on both sides (cp_async class).
                    T.copy(Vt[k, kvh, 0:head_dim, 0:page_size], V_shared_t)

                    # Causal + bounds mask; -inf lanes drop out of the
                    # softmax. Fully-visible tiles carry an everywhere-zero
                    # predicate, so this init is arithmetically the zero-fill
                    # on their lanes.
                    for i, j in T.Parallel(BM, BN):
                        acc_s[i, j] = T.if_then_else(
                            (bx * BM + i < q_len)
                            & (k * BN + j < kv_len)
                            & (k * BN + j <= bx * BM + i + kv_len - q_len),
                            0.0,
                            -T.infinity("float32"),
                        )
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    if fold:
                        # Folded online softmax (base 2): qk_scale fused into
                        # the exp2 arguments (one contractible multiply-add
                        # per score element instead of a standalone scale
                        # pass plus subtract), maxima tracked on the raw
                        # logits (valid because this contract's descales and
                        # softmax_scale are all positive), and
                        # scores_max_prev reused to carry -max*qk_scale.
                        T.copy(scores_max, scores_max_prev)
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(BM):
                            scores_scale[i] = T.exp2(
                                (scores_max_prev[i] - scores_max[i]) * qk_scale
                            )
                            scores_max_prev[i] = -scores_max[i] * qk_scale
                        if not rescale_late:
                            # Early rescale placement (measured-best for the
                            # `_RESCALE_EARLY_SPLIT_KEYS` specializations);
                            # the late placement lives in the P@V section.
                            for i, d in T.Parallel(BM, head_dim):
                                acc_o[i, d] = acc_o[i, d] * scores_scale[i]
                        for i, j in T.Parallel(BM, BN):
                            # Vendor Max_offset=8: the +8 produces P on the
                            # [0, 256] grid instead of [0, 1] before the e4m3
                            # quantization (matching the FA3 oracle's
                            # envelope); 2^8 shifts logsum and acc_o
                            # identically, so the offset cancels in the
                            # partial normalization, and in the combine
                            # differences as a constant LSE shift.
                            acc_s[i, j] = T.exp2(
                                acc_s[i, j] * qk_scale + scores_max_prev[i] + 8.0
                            )
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(BM):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    else:
                        # Scale-first online softmax (base 2): scale the
                        # logits, reduce the scaled maxima, exp2 of the
                        # scaled difference.
                        for i, j in T.Parallel(BM, BN):
                            acc_s[i, j] = acc_s[i, j] * qk_scale

                        T.copy(scores_max, scores_max_prev)
                        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                        for i in T.Parallel(BM):
                            scores_scale[i] = T.exp2(scores_max_prev[i] - scores_max[i])
                        if not rescale_late:
                            # Early rescale placement (see the fold branch).
                            for i, d in T.Parallel(BM, head_dim):
                                acc_o[i, d] = acc_o[i, d] * scores_scale[i]
                        for i, j in T.Parallel(BM, BN):
                            # Vendor Max_offset=8 (see the fold branch): P on
                            # the [0, 256] grid; the constant shift cancels
                            # in the partial normalization and the combine
                            # differences.
                            acc_s[i, j] = T.exp2(acc_s[i, j] - scores_max[i] + 8.0)
                        T.reduce_sum(acc_s, scores_sum, dim=1)
                        for i in T.Parallel(BM):
                            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                    # P @ V with fp8 operands: P staged in shared (bridges the
                    # QK accumulator layout to the PV A-operand layout), V
                    # staged K-major as V^T. The acc_o rescale placement is a
                    # measured per-specialization compile-time choice
                    # (`rescale_late`): the late placement issues the
                    # 65536-mul vector pass immediately before the PV gemm
                    # (vendor PPU mainloop order) so the scheduler can
                    # interleave it with the tensor op instead of serializing
                    # it ahead of the exp2/reduce/P-staging chain —
                    # measured-best on every contract split key except the
                    # `_RESCALE_EARLY_SPLIT_KEYS` set, which keeps the early
                    # form. Same operands, same FP results either way.
                    T.copy(acc_s, P_shared)
                    if rescale_late:
                        for i, d in T.Parallel(BM, head_dim):
                            acc_o[i, d] = acc_o[i, d] * scores_scale[i]
                    T.gemm(
                        P_shared,
                        V_shared_t,
                        acc_o,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

            # Epilogue: unscaled normalized fp32 partial + base-2 LSE. Rows
            # that saw no tile in this slice keep logsum == 0 and are written
            # as (0, -inf) so the merge ignores them. v_descale is applied
            # once in the combine kernel.
            for i, d in T.Parallel(BM, head_dim):
                if bx * BM + i < q_len:
                    Out_partial[bz, q_start + bx * BM + i, bh, d] = T.if_then_else(
                        logsum[i] > 0.0, acc_o[i, d] / logsum[i], 0.0
                    )
            for i in T.Parallel(BM):
                if bx * BM + i < q_len:
                    # Base-2 LSE of this slice. The fold tracks raw-logit
                    # maxima, so the scale enters here; the scale-first form's
                    # maxima are already scaled logits.
                    Glse[bz, q_start + bx * BM + i, bh] = T.if_then_else(
                        logsum[i] > 0.0,
                        T.log2(logsum[i])
                        + (scores_max[i] * qk_scale if fold else scores_max[i]),
                        -T.infinity("float32"),
                    )

    return main


def _build_fa_combine_prim_func(
    total_q: int,
    q_start: int,
    q_len: int,
    req_idx: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    batch: int,
    num_split: int,
):
    """Merge the split-KV partials per (query, head) with base-2 LSE weights."""
    BM = _COMBINE_BLOCK_M

    @T.prim_func
    def main(
        Out_partial: T.Tensor((num_split, total_q, q_heads, head_dim), "float32"),
        Glse: T.Tensor((num_split, total_q, q_heads), "float32"),
        VD: T.Tensor((batch, kv_heads), "float32"),
        Out: T.Tensor((total_q, q_heads, head_dim), "bfloat16"),
    ):
        with T.Kernel(q_heads, T.ceildiv(q_len, BM), threads=_COMBINE_THREADS) as (bh, bx):
            lse_max = T.alloc_fragment([BM], "float32")
            denom = T.alloc_fragment([BM], "float32")
            total_lse = T.alloc_fragment([BM], "float32")
            o_acc = T.alloc_fragment([BM, head_dim], "float32")

            kvh = bh // (q_heads // kv_heads)
            v_scale = VD[req_idx, kvh]

            T.fill(lse_max, -T.infinity("float32"))
            T.fill(denom, 0.0)
            T.fill(o_acc, 0.0)

            # Tail rows beyond q_len clamp their reads to the last valid row;
            # their results are discarded by the store predicate below. Empty
            # slices carry LSE -inf and partial 0, so exp2(-inf) == 0 drops
            # them out of both the normalization and the merge.
            for k in T.serial(num_split):
                for i in T.Parallel(BM):
                    row = T.min(q_start + bx * BM + i, total_q - 1)
                    lse_max[i] = T.max(lse_max[i], Glse[k, row, bh])
            for k in T.serial(num_split):
                for i in T.Parallel(BM):
                    row = T.min(q_start + bx * BM + i, total_q - 1)
                    denom[i] = denom[i] + T.exp2(Glse[k, row, bh] - lse_max[i])
            for i in T.Parallel(BM):
                total_lse[i] = lse_max[i] + T.log2(denom[i])
            for k in T.serial(num_split):
                for i, d in T.Parallel(BM, head_dim):
                    row = T.min(q_start + bx * BM + i, total_q - 1)
                    o_acc[i, d] = o_acc[i, d] + T.exp2(
                        Glse[k, row, bh] - total_lse[i]
                    ) * Out_partial[k, row, bh, d]

            # The V descale is applied exactly once here, mirroring the dense
            # route's single final scale.
            for i, d in T.Parallel(BM, head_dim):
                if bx * BM + i < q_len:
                    Out[q_start + bx * BM + i, bh, d] = T.cast(
                        o_acc[i, d] * v_scale, "bfloat16"
                    )

    return main


def _build_fa_vtranspose_prim_func(
    vt_pages: int,
    num_pages: int,
    page_size: int,
    kv_heads: int,
    head_dim: int,
    bt_width: int,
    batch: int,
    req_idx: int,
):
    """Per-request V page transpose: logical `Vt` from the physical page cache.

    Writes `Vt[p, kvh, d, j] = Vc[BT[req_idx, p], j, kvh, d]` for every
    logical page `p < pages_used = ceil(kv_len / page_size)`; blocks at or
    beyond `pages_used` exit on the runtime guard. The gather through the
    block table keeps the logical indexing valid for any page permutation and
    never assumes `BT[p] == p`. Pure data movement: the staged bytes are
    unchanged, so every downstream mainloop operand stays bitwise identical.
    The grid is the static bucket `vt_pages` so the contract shares one
    compiled kernel; `kv_len` is a runtime scalar argument.
    """

    @T.prim_func
    def main(
        Vc: T.Tensor((num_pages, page_size, kv_heads, head_dim), _FP8),
        BT: T.Tensor((batch, bt_width), "int32"),
        Vt: T.Tensor((vt_pages, kv_heads, head_dim, page_size), _FP8),
        kv_len: T.int32,
    ):
        with T.Kernel(vt_pages, kv_heads, threads=_VTRANS_THREADS) as (p, kvh):
            S = T.alloc_shared([page_size, head_dim], _FP8)
            pages = T.ceildiv(kv_len, page_size)
            if p < pages:
                pid = BT[req_idx, p]
                # Contiguous-to-contiguous page load (the K-staging copy
                # class), then transposed write of 64-byte global rows.
                T.copy(Vc[pid, 0:page_size, kvh, 0:head_dim], S)
                for d, j in T.Parallel(head_dim, page_size):
                    Vt[p, kvh, d, j] = S[j, d]

    return main


@lru_cache(maxsize=None)
def _get_fa_kernel(
    total_q: int,
    q_start: int,
    q_len: int,
    req_idx: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    page_size: int,
    num_pages: int,
    bt_width: int,
    batch: int,
    softmax_scale: float,
    reverse_bx: bool,
    vt_pages: int,
):
    """Compile (once per static query-geometry key) and cache the dense FA kernel.

    The key intentionally excludes kv_len: it is a runtime scalar argument, so
    all shapes sharing (total_q, q_len) reuse one compiled kernel. The key does
    include reverse_bx: the LPT-reversed dense keys compile their own kernel so
    unmeasured geometries sharing (total_q, q_len) keep the natural order.
    `vt_pages` fixes the transposed-V buffer extent baked into the signature.
    """
    prim_func = _build_fa_prim_func(
        total_q,
        q_start,
        q_len,
        req_idx,
        q_heads,
        kv_heads,
        head_dim,
        page_size,
        num_pages,
        bt_width,
        batch,
        softmax_scale,
        reverse_bx,
        vt_pages,
    )
    return tilelang.compile(prim_func, out_idx=None, pass_configs=dict(_PASS_CONFIGS))


@lru_cache(maxsize=None)
def _get_fa_split_kernel(
    total_q: int,
    q_start: int,
    q_len: int,
    req_idx: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    page_size: int,
    num_pages: int,
    bt_width: int,
    batch: int,
    softmax_scale: float,
    num_split: int,
    fold: bool,
    reverse_bx: bool,
    rescale_late: bool,
    two_region: bool,
    block_m: int,
    threads: int,
    stages: int,
    vt_pages: int,
):
    """Compile (once per static key) and cache the split-KV attention kernel.

    kv_len is a runtime scalar argument and stays out of the key; num_split
    (grid z), the fold posture, the LPT posture, the rescale placement, the
    two-region walk structure, the block geometry (block_m/threads/stages),
    and the transposed-V buffer extent are compile-time key fields, so compile
    sharing can never select the wrong route variant.
    """
    prim_func = _build_fa_split_prim_func(
        total_q,
        q_start,
        q_len,
        req_idx,
        q_heads,
        kv_heads,
        head_dim,
        page_size,
        num_pages,
        bt_width,
        batch,
        softmax_scale,
        num_split,
        fold,
        reverse_bx,
        rescale_late,
        two_region,
        block_m,
        threads,
        stages,
        vt_pages,
    )
    return tilelang.compile(prim_func, out_idx=None, pass_configs=dict(_PASS_CONFIGS))


@lru_cache(maxsize=None)
def _get_fa_combine_kernel(
    total_q: int,
    q_start: int,
    q_len: int,
    req_idx: int,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    batch: int,
    num_split: int,
):
    """Compile (once per static key) and cache the split-KV combine kernel."""
    prim_func = _build_fa_combine_prim_func(
        total_q,
        q_start,
        q_len,
        req_idx,
        q_heads,
        kv_heads,
        head_dim,
        batch,
        num_split,
    )
    return tilelang.compile(prim_func, out_idx=None, pass_configs=dict(_PASS_CONFIGS))


@lru_cache(maxsize=None)
def _get_fa_vtranspose_kernel(
    vt_pages: int,
    num_pages: int,
    page_size: int,
    kv_heads: int,
    head_dim: int,
    bt_width: int,
    batch: int,
    req_idx: int,
):
    """Compile (once per static geometry bucket) and cache the V transpose.

    The key fields are the static layout invariants; `kv_len` is a runtime
    scalar argument, so every shape sharing the `vt_pages` bucket reuses one
    compiled kernel.
    """
    prim_func = _build_fa_vtranspose_prim_func(
        vt_pages,
        num_pages,
        page_size,
        kv_heads,
        head_dim,
        bt_width,
        batch,
        req_idx,
    )
    return tilelang.compile(prim_func, out_idx=None, pass_configs=dict(_PASS_CONFIGS))


def _pick_num_split(base_blocks: int, total_tiles: int) -> tuple[int, float]:
    """Wave-model split count and its predicted wall-time gain.

    Residency is capped at `_RESIDENT_SLOTS` blocks; splitting the KV axis by
    s multiplies the block count by s and divides each block's serial tile
    chain by s, so idealized wall time scales as ceil(base_blocks*s/slots)/s.
    Pick the s in [2, min(_SPLIT_MAX_S, total_tiles)] minimizing that wave
    factor, restricted to splits that keep at least
    `_MIN_TILES_PER_SPLIT` KV tiles of work per block so the split/combine
    overhead stays amortized. Among wave-factor ties the LARGEST s wins (equal
    wave cost, shorter per-block chains). Return `(best_s, gain)` where gain is
    the predicted relative wall-time reduction versus the unsplit wave factor
    (0.0 when no s improves on it). Fallback for geometries the staged
    override table does not cover.
    """
    max_s = min(_SPLIT_MAX_S, total_tiles)
    best_s = 1
    best_waves = (base_blocks + _RESIDENT_SLOTS - 1) // _RESIDENT_SLOTS
    for s in range(2, max_s + 1):
        if total_tiles // s < _MIN_TILES_PER_SPLIT:
            continue
        blocks = base_blocks * s
        waves = (blocks + _RESIDENT_SLOTS - 1) // _RESIDENT_SLOTS
        if waves * best_s <= best_waves * s:  # waves/s <= best_waves/best_s
            best_waves = waves
            best_s = s
    unsplit_waves = (base_blocks + _RESIDENT_SLOTS - 1) // _RESIDENT_SLOTS
    gain = 1.0 - (best_waves / best_s) / unsplit_waves
    return best_s, gain


class Model(nn.Module):
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        seqused_k: torch.Tensor,
        max_seqlen_k: int,
        softmax_scale: float,
        causal: bool,
        window_size: tuple[int, int],
        block_table: torch.Tensor,
        scheduler_metadata: object | None,
        fa_version: int,
        q_descale: torch.Tensor,
        k_descale: torch.Tensor,
        v_descale: torch.Tensor,
        num_splits: int,
    ) -> torch.Tensor:
        del max_seqlen_q, max_seqlen_k, scheduler_metadata, num_splits
        if not causal or window_size != (-1, -1) or fa_version != 3:
            raise NotImplementedError("captured PPU contract is causal FA3 without a window")

        total_q, q_heads, head_dim = q.shape
        num_pages, page_size, kv_heads, cache_dim = k.shape
        batch, bt_width = block_table.shape
        if cache_dim != head_dim or v.shape != k.shape:
            raise ValueError("k/v cache layout must match q head_dim")

        # Fresh per-call reads of the varlen metadata (control flow only:
        # request bounds and lengths; values/pointers are never cached).
        bounds = [int(item) for item in cu_seqlens_q.tolist()]
        kv_lens = [int(item) for item in seqused_k.tolist()]

        q_c = q.contiguous()
        k_c = k.contiguous()
        v_c = v.contiguous()
        bt_c = block_table.contiguous()
        qd_c = q_descale.contiguous()
        kd_c = k_descale.contiguous()
        vd_c = v_descale.contiguous()
        out_c = out.contiguous()

        # The two routes key their dispatch tables on DIFFERENT tile counts:
        # the split tables count BN=64 (page) tiles, the dense LPT table
        # counts BN=32 (half-page) tiles. Keep the two computations separate.
        split_tile_n = page_size       # one split-route softmax tile
        dense_tile_n = page_size // 2  # one dense-route softmax tile
        for req_idx in range(len(bounds) - 1):
            q_start = bounds[req_idx]
            q_len = bounds[req_idx + 1] - q_start
            kv_len = kv_lens[req_idx]
            if q_len <= 0 or kv_len <= 0:
                continue
            # Shape-keyed dispatch (shape invariants only — never values or
            # pointers): the measured override table first, then the wave
            # model, then the wave-saturated efficiency fallback, else the
            # dense single-pass kernel.
            q_blocks = (q_len + _SPLIT_BLOCK_M - 1) // _SPLIT_BLOCK_M
            base_blocks = q_heads * q_blocks
            split_total_tiles = (kv_len + split_tile_n - 1) // split_tile_n
            # Two-CTA concurrency probe opt-in (shape invariants only). A
            # listed shape runs the explicit geometry+posture spec and skips
            # the split-count / fold / LPT / rescale / two-region tables, so
            # geometry is the sole variable vs the incumbent. An empty probe
            # table leaves every shape on the incumbent path below.
            geom_probe = _SPLIT_GEOM_PROBES.get((q_len, split_total_tiles))
            num_split = 1
            if geom_probe is not None:
                block_m = geom_probe["block_m"]
                threads = geom_probe["threads"]
                stages = geom_probe["stages"]
                num_split = geom_probe["num_split"]
            else:
                block_m = _SPLIT_BLOCK_M
                threads = _SPLIT_THREADS
                stages = _SPLIT_STAGES
            if geom_probe is None and split_total_tiles >= 4:
                override_s = _SPLIT_OVERRIDES.get(
                    (base_blocks, split_total_tiles)
                )
                if override_s is not None:
                    num_split = override_s
                else:
                    best_s, split_gain = _pick_num_split(base_blocks, split_total_tiles)
                    if split_gain >= _SPLIT_GAIN_THRESHOLD:
                        num_split = best_s
                    elif (
                        base_blocks >= _EFFICIENCY_SPLIT_MIN_BLOCKS
                        and split_total_tiles >= _EFFICIENCY_SPLIT_MIN_TILES
                    ):
                        num_split = _EFFICIENCY_SPLIT_S
            # Global V pre-transpose (both routes): materialize the
            # logically-indexed transposed V buffer for THIS request (fresh
            # allocation from the current shapes, never keyed on pointers or
            # prior calls) so every mainloop V stage lowers to the async
            # cp_async copy class instead of the scalar transposed staging.
            # The extent is bucketed so the contract shares one compiled
            # transpose kernel.
            pages_used = (kv_len + page_size - 1) // page_size
            vt_pages = max(_VTRANS_MIN_PAGES, pages_used)
            vt_c = torch.empty(
                (vt_pages, kv_heads, head_dim, page_size),
                dtype=q_c.dtype,
                device=q.device,
            )
            vtranspose_kernel = _get_fa_vtranspose_kernel(
                vt_pages,
                num_pages,
                page_size,
                kv_heads,
                head_dim,
                bt_width,
                batch,
                req_idx,
            )
            vtranspose_kernel(v_c, bt_c, vt_c, kv_len)
            if num_split > 1:
                # Folded softmax only for the measured fold-positive split
                # specializations; every other split keeps the scale-first
                # body (see _FOLD_SPLIT_KEYS). LPT q-block reversal only for
                # the measured reversal-positive split keys (see
                # _LPT_SPLIT_KEYS). The acc_o rescale runs in the measured
                # late placement (before the PV gemm) except for the
                # measured early-placement keys (see
                # _RESCALE_EARLY_SPLIT_KEYS). All resolved host-side from
                # shape invariants and baked into the compile key
                # (compile-time constants, never runtime selects).
                fold = (base_blocks, split_total_tiles, num_split) in _FOLD_SPLIT_KEYS
                reverse_bx = (base_blocks, split_total_tiles) in _LPT_SPLIT_KEYS
                rescale_late = (
                    base_blocks,
                    split_total_tiles,
                ) not in _RESCALE_EARLY_SPLIT_KEYS
                # Two-region walk (mask-free prefix + masked boundary band).
                # Selected for every divisible geometry (q_len % BM == 0: no
                # tile carries masked query rows) PLUS the audited partial-q
                # specializations in `_TWO_REGION_TAIL_KEYS`: masked
                # query-tail rows never need correct values because the split
                # epilogue predicates both the Out_partial and Glse writes on
                # `bx*BM+i < q_len`, the combine kernel's discarded tail lanes
                # are never stored, and the softmax is row-local — so the
                # fully-visible majority of their tiles is mask-free eligible
                # and only the few causal-edge/kv-tail tiles per q-block keep
                # the predicate (lane-local arithmetic confines tail-row
                # garbage; no stored row is affected). Resolved host-side from
                # shape invariants and baked into the compile key.
                two_region = (q_len % block_m == 0) or (
                    (base_blocks, split_total_tiles) in _TWO_REGION_TAIL_KEYS
                )
                if geom_probe is not None:
                    # Geometry probe: hold the incumbent measured posture
                    # exactly (fold/LPT/rescale/two-region) so the block
                    # geometry is the sole variable; the spec comes from the
                    # shape-invariant `_SPLIT_GEOM_PROBES` entry, never from
                    # the BM=256-keyed tables above.
                    fold = geom_probe["fold"]
                    reverse_bx = geom_probe["reverse_bx"]
                    rescale_late = geom_probe["rescale_late"]
                    two_region = geom_probe["two_region"]
                split_kernel = _get_fa_split_kernel(
                    total_q,
                    q_start,
                    q_len,
                    req_idx,
                    q_heads,
                    kv_heads,
                    head_dim,
                    page_size,
                    num_pages,
                    bt_width,
                    batch,
                    float(softmax_scale),
                    num_split,
                    fold,
                    reverse_bx,
                    rescale_late,
                    two_region,
                    block_m,
                    threads,
                    stages,
                    vt_pages,
                )
                combine_kernel = _get_fa_combine_kernel(
                    total_q,
                    q_start,
                    q_len,
                    req_idx,
                    q_heads,
                    kv_heads,
                    head_dim,
                    batch,
                    num_split,
                )
                # Fresh per-call fp32 workspaces allocated from the current
                # shapes (never keyed on pointers or prior calls).
                out_partial = torch.empty(
                    (num_split, total_q, q_heads, head_dim),
                    dtype=torch.float32,
                    device=q.device,
                )
                glse = torch.empty(
                    (num_split, total_q, q_heads),
                    dtype=torch.float32,
                    device=q.device,
                )
                split_kernel(
                    q_c, k_c, vt_c, bt_c, qd_c, kd_c, out_partial, glse, kv_len
                )
                combine_kernel(out_partial, glse, vd_c, out_c)
            else:
                # Dense route: LPT q-block reversal only for the measured
                # reversal-positive dense keys (shape/dtype/layout invariants,
                # never values or pointers), keyed at BN=32 tile counts;
                # resolved at dispatch and baked into the compile key.
                dense_total_tiles = (kv_len + dense_tile_n - 1) // dense_tile_n
                reverse_bx = (base_blocks, dense_total_tiles) in _LPT_DENSE_KEYS
                kernel = _get_fa_kernel(
                    total_q,
                    q_start,
                    q_len,
                    req_idx,
                    q_heads,
                    kv_heads,
                    head_dim,
                    page_size,
                    num_pages,
                    bt_width,
                    batch,
                    float(softmax_scale),
                    reverse_bx,
                    vt_pages,
                )
                kernel(q_c, k_c, vt_c, bt_c, qd_c, kd_c, vd_c, out_c, kv_len)
        if out_c is not out:
            # `out` was non-contiguous: the kernels wrote the contiguous
            # staging copy, so move the result back into the caller's tensor
            # (the ABI returns the same tensor passed as `out`).
            out.copy_(out_c)
        return out
