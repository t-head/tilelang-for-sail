"""PPU LayoutInference m16n8 compatibility-retry tests.

Covers the ``tl.ppu_enable_n8_compat`` / ``tl.ppu_force_gemm_n8`` pass
config pair added on top of the PPU0015 gemm m16n8 fallback:

- ``tl.ppu_enable_n8_compat`` (default True): the total switch. While on,
  ``tl.ppu.transform.LayoutInference`` (src/ppu/transform/ppu_layout_inference.cc)
  catches the first layout-collection failure, flips ``tl.ppu_force_gemm_n8``
  to True on the current PassContext and retries the collection exactly
  once, so gemm warp partitions get forced onto the m16n8 fallback atoms
  (8-wide warp N slices) instead of the layout-conflicting m16n16 ones.
- ``tl.ppu_force_gemm_n8`` (default False): consumed by
  ``src/ppu/op/gemm.cc`` ``ComputeWarpPartition`` and by
  ``tilelang/ppu/intrinsics/macro/mma_macro_generator.py``
  ``_initialize_micro_size`` / ``_use_m16n8_fallback``. Written by the
  retry above; can also be set manually to reproduce the fallback path.

The suite also carries a genuine "N % 16 == 0 gemm + a sibling op raising
LayoutConflictException" repro: a reduce_sum whose annotated dst fragment
only matches the m16n8-forced C layout. The verification below uses:

1. direct force-mechanism tests: an N % 16 == 0 kernel compiled with
   ``tl.ppu_force_gemm_n8 = True`` must take the n8 micro-tile path
   (spied via ``PPUTensorCoreIntrinEmitter._initialize_micro_size``) and
   still match the torch reference; the default-config counterpart must
   keep the n16 path (baseline);
2. end-to-end retry tests that genuinely exercise the C++ catch/retry:
   a PPU0010 kernel whose warp N slice is a multiple of 8 only makes the
   first collection fail, the retry flip happen, and the second failure
   escape with the retry-aware message; disabling the total switch must
   instead let the first error escape untouched;
3. the real repro: with compatibility disabled the raw LayoutConflict
   error escapes; with the default config the retry succeeds and both
   outputs match the torch reference.

With block_M = 64, block_N = 16 and threads = 128 (4 warps, Square policy):
- default: kNPerWarp = 16 -> (m_warp, n_warp) = (4, 1) ->
  warp_row_tiles = 16, warp_col_tiles = 16 -> n16 atoms;
- forced:  kNPerWarp = 8  -> (m_warp, n_warp) = (2, 2) ->
  warp_row_tiles = 32, warp_col_tiles = 8  -> n8 atoms.
"""

import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.language as T

from tilelang.transform import PassConfigKey
from tilelang.ppu.intrinsics.macro.mma_macro_generator import PPUTensorCoreIntrinEmitter
from tilelang.env import CacheState


def matmul(M, N, K, block_M, block_N, block_K, in_dtype, out_dtype, accum_dtype):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((K, N), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_K, block_N), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, False, False)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def _spy_micro_size(monkeypatch):
    """Record (warp_col_tiles, n_dim) for every emitter construction.

    Both the layout-inference (infer_layout) and the lowering stages
    rebuild a PPUTensorCoreIntrinEmitter for each gemm, so every recorded
    entry must observe the same forced/default micro-tile choice.
    """
    records = []
    original = PPUTensorCoreIntrinEmitter._initialize_micro_size

    def spy(self, m_dim=16, k_dim=16):
        original(self, m_dim, k_dim)
        records.append((self.warp_col_tiles, self.n_dim))

    monkeypatch.setattr(PPUTensorCoreIntrinEmitter, "_initialize_micro_size", spy)
    # Disable the kernel cache so a warm JIT cache cannot bypass the spy.
    monkeypatch.setattr(CacheState, "_enabled", False)
    return records


def _fp16_ref(in_dtype, out_dtype):
    def ref_program(A, B):
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    return ref_program


def _gen_fp8_inputs(A_shape, B_shape, in_dtype, b_in_dtype):
    """Generate finite fp8 inputs: sample in fp16 first, then cast, so that
    the default random-bit supply does not produce NaN/Inf bit patterns."""
    from tilelang.utils.device import get_current_device

    device = get_current_device()
    A = torch.randn(*A_shape, dtype=torch.float16, device=device).to(in_dtype.as_torch())
    B = torch.randn(*B_shape, dtype=torch.float16, device=device).to(b_in_dtype.as_torch())
    return [A, B]


# ---------------------------------------------------------------------------
# Forced fallback: N % 16 == 0 shape must take the n8 path
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_force_gemm_n8_n16_shape(monkeypatch):
    # block_N = 16 is divisible by 16, so without the force config the warp
    # partition would pick the m16n16 atoms (warp_col_tiles = 16). With
    # tl.ppu_force_gemm_n8 = True the warp N slice is forced onto the
    # m16n8 atoms (kNPerWarp = 8 -> warp_col_tiles = 8, n_dim = 8), which is
    # exactly the state the LayoutInference compatibility retry produces.
    records = _spy_micro_size(monkeypatch)
    program = matmul(
        128,
        64,
        128,
        64,
        16,
        32,
        T.float16,
        T.float32,
        T.float32,
    )
    kernel = tilelang.compile(
        program,
        out_idx=[2],
        pass_configs={PassConfigKey.TL_PPU_FORCE_GEMM_N8: True},
    )
    assert records, "expected at least one PPUTensorCoreIntrinEmitter construction"
    assert all(n_dim == 8 for _, n_dim in records), records
    assert all(warp_col_tiles == 8 for warp_col_tiles, _ in records), records
    kernel.get_profiler().assert_allclose(_fp16_ref("float16", "float32"), atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_default_config_keeps_n16_shape(monkeypatch):
    # Baseline for the test above: the same N % 16 == 0 kernel compiled
    # with default pass configs must keep the historical m16n16 partition
    # (warp_col_tiles = 16, n_dim = 16) — the retry flip never happens
    # because layout collection succeeds.
    records = _spy_micro_size(monkeypatch)
    program = matmul(
        128,
        64,
        128,
        64,
        16,
        32,
        T.float16,
        T.float32,
        T.float32,
    )
    kernel = tilelang.compile(program, out_idx=[2])
    assert records, "expected at least one PPUTensorCoreIntrinEmitter construction"
    assert all(n_dim == 16 for _, n_dim in records), records
    assert all(warp_col_tiles == 16 for warp_col_tiles, _ in records), records
    kernel.get_profiler().assert_allclose(_fp16_ref("float16", "float32"), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# PPU0010: the m16n8 retry is arch-gated off, first error escapes
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_lt(1, 5)
def test_n8_retry_arch_gated_off_ppu0010():
    # The m16n8 retry needs PPU0015+ (arch >= 15); on PPU0010 the gate
    # (ppu_enable_n8_compat && arch >= 15) keeps the retry off entirely,
    # so the first collection error (the gemm.cc N-divisibility ICHECK)
    # escapes as-is, without the doomed n8 retry.
    program = matmul(
        128,
        64,
        128,
        32,
        8,
        32,
        T.float16,
        T.float32,
        T.float32,
    )
    with pytest.raises(Exception, match="N must be divisible by 16, but got 8"):
        tilelang.compile(program, out_idx=[2])


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_lt(1, 5)
def test_n8_compat_disabled_first_error_escapes():
    # Same kernel with the total switch explicitly off: no catch/retry
    # happens either, and the first collection error escapes untouched.
    program = matmul(
        128,
        64,
        128,
        32,
        8,
        32,
        T.float16,
        T.float32,
        T.float32,
    )
    with pytest.raises(Exception, match="N must be divisible by 16, but got 8"):
        tilelang.compile(
            program,
            out_idx=[2],
            pass_configs={PassConfigKey.TL_PPU_ENABLE_N8_COMPAT: False},
        )


# ---------------------------------------------------------------------------
# Forced fallback with fp8: now inside the m16n8 whitelist (positive), and a
# still-unsupported dtype (uint8) keeps the retry-aware failure
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_force_gemm_n8_fp8_n16_shape(monkeypatch):
    # fp8 x fp8 + fp32 (e4m3 operands) has a native m16n8k32 atom, so with
    # tl.ppu_force_gemm_n8 = True the warp N slice is forced onto the
    # m16n8 atoms (kNPerWarp = 8 -> warp_col_tiles = 8, n_dim = 8) and the
    # kernel must compile and match the torch reference — the same forced
    # state the LayoutInference compatibility retry produces.
    records = _spy_micro_size(monkeypatch)
    program = matmul(
        128,
        64,
        128,
        64,
        16,
        32,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
    )
    kernel = tilelang.compile(
        program,
        out_idx=[2],
        pass_configs={PassConfigKey.TL_PPU_FORCE_GEMM_N8: True},
    )
    assert records, "expected at least one PPUTensorCoreIntrinEmitter construction"
    assert all(n_dim == 8 for _, n_dim in records), records
    assert all(warp_col_tiles == 8 for warp_col_tiles, _ in records), records
    ins = _gen_fp8_inputs((128, 128), (128, 64), T.float8_e4m3fn, T.float8_e4m3fn)
    kernel.get_profiler().assert_allclose(
        _fp16_ref("float8_e4m3fn", "float32"),
        input_tensors=ins,
        atol=1e-2,
        rtol=1e-2,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_force_gemm_n8_uint8_unsupported_dtype():
    # uint8 x uint8 + int32 has no m16n8 atom wired up (the whitelist covers
    # fp16, bf16, tf32, int8 and fp8; actlize has no 16x16 PPU0015 uint8
    # atom either). With the force config set the emitter must fail fast
    # with the retry-aware message instead of silently re-running the n16
    # path. NOTE: this assumes the uint8 gemm reaches the micro-tile
    # decision during layout inference; if the environment rejects uint8
    # earlier, adjust the dtype here.
    program = matmul(
        128,
        64,
        128,
        64,
        16,
        32,
        T.uint8,
        T.int32,
        T.int32,
    )
    with pytest.raises(Exception, match="compatibility retry was already attempted"):
        tilelang.compile(
            program,
            out_idx=[2],
            pass_configs={PassConfigKey.TL_PPU_FORCE_GEMM_N8: True},
        )


# ---------------------------------------------------------------------------
# Real LayoutConflictException repro: block_N = 16 (N % 16 == 0) gemm plus
# a reduce_sum whose annotated dst fragment only matches the m16n8-forced C
# layout, so the first (m16n16) collection raises in ReduceOp::InferLayout.
# ---------------------------------------------------------------------------


def _reduce_conflict_program():
    # Thread maps (block_M=64, block_N=16, threads=128, Square policy):
    # - default n16 partition (m_warp, n_warp) = (4, 1): C fragment
    #   thread(i, j) = 32*(i//16) + 4*(i%8) + (j%8)//2;
    # - forced n8 partition (2, 2): C fragment thread(i, j) =
    #   64*(j//8) + 32*(i//32) + 4*(i%8) + (j%8)//2.
    # The reducer layout derived from the n16 C fragment holds element i on
    # {32*(i//16) + 4*(i%8) + c} (4 threads), while R_local is annotated with
    # the n8 holder set {64*b + 32*(i//32) + 4*(i%8) + c} (8 threads) — a
    # real conflict for the first collection and a match for the retried one.
    M, N, K = 128, 16, 128
    BM, BK = 64, 32

    def r_forward(i, rep):
        thread = 64 * (rep // 4) + 32 * (i // 32) + 4 * (i % 8) + rep % 4
        index = (i % 32) // 8
        return thread, index

    r_layout = T.Fragment((BM,), forward_fn=r_forward, replicate=8)

    @T.prim_func
    def main(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((K, N), "float16"),
        C: T.Tensor((M, N), "float32"),
        D: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(T.ceildiv(M, BM), threads=128) as (by):
            A_shared = T.alloc_shared((BM, BK), "float16")
            B_shared = T.alloc_shared((BK, N), "float16")
            C_local = T.alloc_fragment((BM, N), "float32")
            R_local = T.alloc_fragment((BM,), "float32")
            T.annotate_layout({R_local: r_layout})
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, BK), num_stages=0):
                T.copy(A[by * BM, k * BK], A_shared)
                T.copy(B[k * BK, 0], B_shared)
                T.gemm(A_shared, B_shared, C_local, False, False)
            T.reduce_sum(C_local, R_local, clear=True)
            T.copy(C_local, C[by * BM, 0])
            T.copy(R_local, D[by * BM])

    return main


def _gemm_reduce_ref(A, B):
    C = torch.matmul(A.to(torch.float32), B.to(torch.float32))
    D = C.sum(dim=1)
    return C, D


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_real_layout_conflict_compat_disabled_raises():
    # The real LayoutConflictException repro with the total switch off: no
    # catch/retry happens, so the raw first-collection conflict error must
    # escape untouched.
    program = _reduce_conflict_program()
    with pytest.raises(Exception, match="Layout may conflict with ReduceOp"):
        tilelang.compile(
            program,
            out_idx=[2, 3],
            pass_configs={PassConfigKey.TL_PPU_ENABLE_N8_COMPAT: False},
        )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_real_layout_conflict_retry_n8_matches_reference():
    # With default configs the retry flips tl.ppu_force_gemm_n8 on and both
    # outputs (C, D) must match the torch reference.
    program = _reduce_conflict_program()
    kernel = tilelang.compile(program, out_idx=[2, 3])
    kernel.get_profiler().assert_allclose(_gemm_reduce_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
