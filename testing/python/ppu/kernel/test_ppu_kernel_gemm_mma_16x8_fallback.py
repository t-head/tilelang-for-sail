"""PPU0015 GEMM MMA 16x8 fallback regression tests.

Reproduces the T.gemm failure when a warp's N slice is not divisible by 16.
On PPU0015, ``tilelang/ppu/intrinsics/macro/mma_macro_generator.py``
(``PPUTensorCoreIntrinEmitter::_initialize_micro_size``) originally asserted
``warp_col_tiles % 16 == 0`` and raised for N slices such as 8 or 24.

The fallback selects the MMA atom (actlize-native PPU0015 m16n8 atoms) by
the warp N slice:
- ``warp_col_tiles % 16 == 0`` -> m16n16k* atoms (no fallback)
- ``warp_col_tiles % 8 == 0``  -> m16n8* atoms:
  - fp16 x fp16 (fp16 or fp32 accum): m16n8k16
  - bf16 x bf16 (fp32 accum):          m16n8k16
  - tf32 x tf32 (fp32 accum):          m16n8k8
  - int8 x int8 (int32 accum):         m16n8k32
  - fp8 x fp8 over the e4m3/e5m2 operand mixes, A != B included
    (fp32 accum):                       m16n8k32

Plain fp32 and fp4 cases keep the 16x16 atoms (no fallback). fp8 with fp16
accumulation, fp8 k16 (unreachable: the fp8 k_dim is always 32), uint8
mixes and fp4 are not covered here. PPU0010 does not support the fallback
(it errors out on n8 tiles directly), so no PPU0010 cases are included.

All cases must match the torch reference. ``warp_col_tiles = block_N //
n_warp``; with block_M = 32 and the default Square warp policy all cases
below get n_warp = 1, so warp_col_tiles equals block_N.

The ``_nt`` cases feed a transposed, N-major B operand (global shape
``(N, K)``, ``T.gemm(..., False, True)``) to cover the fallback's N-major
B shared-memory mappings in mma_macro_generator.py.
"""

import tilelang
import tilelang.testing
import tilelang.language as T


def _is_fp8_dtype(dtype) -> bool:
    return "float8" in str(dtype)


def _gen_fp8_inputs(A_shape, B_shape, in_dtype, b_in_dtype):
    """Generate finite fp8 inputs: sample in fp16 first, then cast, so that
    the default random-bit supply does not produce NaN/Inf bit patterns.

    A and B are cast separately so mixed combos (e.g. e4m3 A x e5m2 B) do
    not share one cast."""
    import torch

    from tilelang.utils.device import get_current_device

    device = get_current_device()
    A = torch.randn(*A_shape, dtype=torch.float16, device=device).to(in_dtype.as_torch())
    B = torch.randn(*B_shape, dtype=torch.float16, device=device).to(b_in_dtype.as_torch())
    return [A, B]


def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
    b_in_dtype=None,
    trans_A=False,
    trans_B=False,
):
    if b_in_dtype is None:
        b_in_dtype = in_dtype
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, b_in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, b_in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages=0,
    num_threads=128,
    b_in_dtype=None,
    trans_A=False,
    trans_B=False,
):
    program = matmul(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        in_dtype,
        out_dtype,
        accum_dtype,
        num_stages,
        num_threads,
        b_in_dtype,
        trans_A,
        trans_B,
    )

    kernel = tilelang.compile(program, out_idx=[2])
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        if in_dtype in (T.tfloat32, T.float32):
            # Convert float32 to tfloat32 because tfloat32 mma cannot truncate
            # float32 automatically, -0x1000 means
            A = (A.view(torch.int32) - 0x1000).view(torch.float32)
            B = (B.view(torch.int32) - 0x1000).view(torch.float32)
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    B_shape = (N, K) if trans_B else (K, N)
    _b = b_in_dtype if b_in_dtype is not None else in_dtype
    if _is_fp8_dtype(in_dtype) or _is_fp8_dtype(_b):
        # fp8 tolerance follows the main-path fp8 gemm cases
        # (testing/python/ppu/kernel/test_ppu_kernel_gemm.py).
        ins = _gen_fp8_inputs((M, K), B_shape, in_dtype, _b)
        profiler.assert_allclose(ref_program, input_tensors=ins, atol=1e-2, rtol=1e-2)
    else:
        profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# fp16 x fp16, fp32 accumulation (m16n8k16)
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_f16f16f32_n8():
    # block_N = 8 -> the single warp owns an 8-wide N slice
    # (warp_col_tiles = 8), not divisible by 16 but divisible by 8:
    # must fall back to the m16n8k16 atom.
    run_gemm(
        128,
        64,
        128,
        32,
        8,
        32,
        T.float16,
        T.float32,
        T.float32,
        0,
        128,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_f16f16f32_n24():
    # block_N = 24 -> warp_col_tiles = 24 (16 + 8 mixed), not divisible by 16
    # but divisible by 8: must fall back to the m16n8k16 atom.
    run_gemm(
        128,
        48,
        128,
        32,
        24,
        32,
        T.float16,
        T.float32,
        T.float32,
        0,
        128,
    )


# ---------------------------------------------------------------------------
# fp16 x fp16, fp16 accumulation (m16n8k16)
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_f16f16f16_n8():
    # warp_col_tiles = 8 -> m16n8k16 atom with fp16 accumulator.
    run_gemm(
        128,
        64,
        128,
        32,
        8,
        32,
        T.float16,
        T.float16,
        T.float16,
        0,
        128,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_f16f16f16_n24():
    # warp_col_tiles = 24 (16 + 8 mixed) -> m16n8k16 atom with fp16 accumulator.
    run_gemm(
        128,
        48,
        128,
        32,
        24,
        32,
        T.float16,
        T.float16,
        T.float16,
        0,
        128,
    )


# ---------------------------------------------------------------------------
# bf16 x bf16, fp32 accumulation (m16n8k16)
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_bf16bf16f32_n8():
    # warp_col_tiles = 8 -> m16n8k16 atom with bf16 operands.
    run_gemm(
        128,
        64,
        128,
        32,
        8,
        32,
        T.bfloat16,
        T.float32,
        T.float32,
        0,
        128,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_bf16bf16f32_n24():
    # warp_col_tiles = 24 (16 + 8 mixed) -> m16n8k16 atom with bf16 operands.
    run_gemm(
        128,
        48,
        128,
        32,
        24,
        32,
        T.bfloat16,
        T.float32,
        T.float32,
        0,
        128,
    )


# ---------------------------------------------------------------------------
# tf32 x tf32, fp32 accumulation (m16n8k8)
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_tf32f32f32_n8():
    # warp_col_tiles = 8 -> m16n8k8 atom (tf32 k_dim = 8, block_K = 32 is a
    # multiple of 8).
    run_gemm(
        128,
        64,
        128,
        32,
        8,
        32,
        T.tfloat32,
        T.float32,
        T.float32,
        0,
        128,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_tf32f32f32_n24():
    # warp_col_tiles = 24 (16 + 8 mixed) -> m16n8k8 atom.
    run_gemm(
        128,
        48,
        128,
        32,
        24,
        32,
        T.tfloat32,
        T.float32,
        T.float32,
        0,
        128,
    )


# ---------------------------------------------------------------------------
# int8 x int8, int32 accumulation (m16n8k32)
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_i8i8i32_n8():
    # warp_col_tiles = 8 -> m16n8k32 atom (int8 k_dim = 32, block_K = 64 is a
    # multiple of 32).
    run_gemm(
        128,
        64,
        128,
        32,
        8,
        64,
        T.int8,
        T.int32,
        T.int32,
        0,
        128,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_i8i8i32_n24():
    # warp_col_tiles = 24 (16 + 8 mixed) -> m16n8k32 atom.
    run_gemm(
        128,
        48,
        128,
        32,
        24,
        64,
        T.int8,
        T.int32,
        T.int32,
        0,
        128,
    )


# ---------------------------------------------------------------------------
# fp8 x fp8 (e4m3/e5m2 mixes), fp32 accumulation (m16n8k32)
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_e4m3e4m3f32_n8():
    # warp_col_tiles = 8 -> m16n8k32 fp8 atom (fp8 k_dim = 32, block_K = 32
    # is exactly one k slice).
    run_gemm(
        128,
        64,
        128,
        32,
        8,
        32,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        0,
        128,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_e4m3e4m3f32_n24():
    # warp_col_tiles = 24 (16 + 8 mixed) -> m16n8k32 fp8 atom.
    run_gemm(
        128,
        48,
        128,
        32,
        24,
        32,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        0,
        128,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_e4m3e5m2f32_n8():
    # warp_col_tiles = 8 -> m16n8k32 fp8 atom with mixed operands: e4m3 A
    # x e5m2 B (A and B are cast separately in _gen_fp8_inputs).
    run_gemm(
        128,
        64,
        128,
        32,
        8,
        32,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        0,
        128,
        b_in_dtype=T.float8_e5m2,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_e4m3e5m2f32_n24():
    # warp_col_tiles = 24 (16 + 8 mixed) -> m16n8k32 fp8 atom, e4m3 A x
    # e5m2 B.
    run_gemm(
        128,
        48,
        128,
        32,
        24,
        32,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        0,
        128,
        b_in_dtype=T.float8_e5m2,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_e5m2e4m3f32_n8():
    # warp_col_tiles = 8 -> m16n8k32 fp8 atom, e5m2 A x e4m3 B.
    run_gemm(
        128,
        64,
        128,
        32,
        8,
        32,
        T.float8_e5m2,
        T.float32,
        T.float32,
        0,
        128,
        b_in_dtype=T.float8_e4m3fn,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_e5m2e4m3f32_n24():
    # warp_col_tiles = 24 (16 + 8 mixed) -> m16n8k32 fp8 atom, e5m2 A x
    # e4m3 B.
    run_gemm(
        128,
        48,
        128,
        32,
        24,
        32,
        T.float8_e5m2,
        T.float32,
        T.float32,
        0,
        128,
        b_in_dtype=T.float8_e4m3fn,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_e5m2e5m2f32_n8():
    # warp_col_tiles = 8 -> m16n8k32 fp8 atom with e5m2 operands.
    run_gemm(
        128,
        64,
        128,
        32,
        8,
        32,
        T.float8_e5m2,
        T.float32,
        T.float32,
        0,
        128,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_e5m2e5m2f32_n24():
    # warp_col_tiles = 24 (16 + 8 mixed) -> m16n8k32 fp8 atom with e5m2
    # operands.
    run_gemm(
        128,
        48,
        128,
        32,
        24,
        32,
        T.float8_e5m2,
        T.float32,
        T.float32,
        0,
        128,
    )


# ---------------------------------------------------------------------------
# N-major B (trans_B=True) cases: cover the fallback's N-major B
# shared-memory mappings in mma_macro_generator.py
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_f16f16f32_nt_n8():
    # trans_B = True, warp_col_tiles = 8 -> m16n8k16 atom with the N-major B
    # mapping ldmatrix_32x4_to_shared_8x16_layout_b_fp16.
    run_gemm(
        128,
        64,
        128,
        32,
        8,
        32,
        T.float16,
        T.float32,
        T.float32,
        0,
        128,
        trans_B=True,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_f16f16f32_nt_n24():
    # trans_B = True, warp_col_tiles = 24 (16 + 8 mixed) -> m16n8k16 atom
    # with the N-major B mapping ldmatrix_32x4_to_shared_8x16_layout_b_fp16.
    run_gemm(
        128,
        48,
        128,
        32,
        24,
        32,
        T.float16,
        T.float32,
        T.float32,
        0,
        128,
        trans_B=True,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_bf16bf16f32_nt_n8():
    # trans_B = True, warp_col_tiles = 8 -> m16n8k16 atom with bf16 operands
    # reusing the fp16 N-major B mapping
    # ldmatrix_32x4_to_shared_8x16_layout_b_fp16.
    run_gemm(
        128,
        64,
        128,
        32,
        8,
        32,
        T.bfloat16,
        T.float32,
        T.float32,
        0,
        128,
        trans_B=True,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_tf32f32f32_nt_n8():
    # trans_B = True, warp_col_tiles = 8 -> m16n8k8 atom with the tf32
    # N-major B mapping ldmatrix_32x2_to_shared_8x8_layout_b_fp32.
    run_gemm(
        128,
        64,
        128,
        32,
        8,
        32,
        T.tfloat32,
        T.float32,
        T.float32,
        0,
        128,
        trans_B=True,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_tf32f32f32_nt_n24():
    # trans_B = True, warp_col_tiles = 24 (16 + 8 mixed) -> m16n8k8 atom
    # with the tf32 N-major B mapping
    # ldmatrix_32x2_to_shared_8x8_layout_b_fp32.
    run_gemm(
        128,
        48,
        128,
        32,
        24,
        32,
        T.tfloat32,
        T.float32,
        T.float32,
        0,
        128,
        trans_B=True,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_f16f16f16_nt_n8():
    # trans_B = True, warp_col_tiles = 8 -> m16n8k16 atom with an fp16
    # accumulator and the N-major B mapping
    # ldmatrix_32x4_to_shared_8x16_layout_b_fp16.
    run_gemm(
        128,
        64,
        128,
        32,
        8,
        32,
        T.float16,
        T.float16,
        T.float16,
        0,
        128,
        trans_B=True,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_i8i8i32_nt_n8():
    # trans_B = True, warp_col_tiles = 8 -> m16n8k32 atom with the int8
    # N-major B mapping ldmatrix_32x8_to_shared_8x32_layout_b.
    run_gemm(
        128,
        64,
        128,
        32,
        8,
        64,
        T.int8,
        T.int32,
        T.int32,
        0,
        128,
        trans_B=True,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_i8i8i32_nt_n24():
    # trans_B = True, warp_col_tiles = 24 (16 + 8 mixed) -> m16n8k32 atom
    # with the int8 N-major B mapping ldmatrix_32x8_to_shared_8x32_layout_b.
    run_gemm(
        128,
        48,
        128,
        32,
        24,
        64,
        T.int8,
        T.int32,
        T.int32,
        0,
        128,
        trans_B=True,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_e4m3e4m3f32_nt_n8():
    # trans_B = True, warp_col_tiles = 8 -> m16n8k32 fp8 atom with the fp8
    # N-major B mapping ldmatrix_32x8_to_shared_8x32_layout_b.
    run_gemm(
        128,
        64,
        128,
        32,
        8,
        32,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        0,
        128,
        trans_B=True,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_e4m3e4m3f32_nt_n24():
    # trans_B = True, warp_col_tiles = 24 (16 + 8 mixed) -> m16n8k32 fp8
    # atom with the N-major B mapping
    # ldmatrix_32x8_to_shared_8x32_layout_b.
    run_gemm(
        128,
        48,
        128,
        32,
        24,
        32,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        0,
        128,
        trans_B=True,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_mma_16x8_fallback_e4m3e5m2f32_nt_n8():
    # trans_B = True, warp_col_tiles = 8 -> m16n8k32 fp8 atom, e4m3 A x
    # e5m2 B, exercising the mixed-dtype N-major B cast path.
    run_gemm(
        128,
        64,
        128,
        32,
        8,
        32,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        0,
        128,
        b_in_dtype=T.float8_e5m2,
        trans_B=True,
    )


if __name__ == "__main__":
    tilelang.testing.main()
