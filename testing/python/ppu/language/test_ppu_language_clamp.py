# PPU variant of testing/python/language/test_tilelang_language_clamp.py.
# Plain T.clamp follows the upstream (post-revert) semantics: it expands to
# min/max and does not propagate NaN. NaN-propagating symmetric clipping is
# provided by the PPU dialect's T.clamp_symmetric, mirroring the CUDA
# backend's clamp_symmetric (#3605), including 16-bit packed x2 vectors.
import pytest

import tilelang
import tilelang.testing
from tilelang.ppu import language as T


def clamp_within_bounds(
    N,
    block_N,
    dtype,
    min_val=None,
    max_val=None,
):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=block_N) as bx:
            A_shared = T.alloc_shared([block_N], dtype)
            T.copy(A[bx * block_N], A_shared)
            for i in T.Parallel(block_N):
                A_shared[i] = T.clamp(A_shared[i], min_val=min_val, max_val=max_val)
            T.copy(A_shared, B[bx * block_N])

    return main


def run_clamp(
    N,
    block_N,
    dtype,
    min=None,
    max=None,
):
    program = clamp_within_bounds(N, block_N, dtype, min, max)

    kernel = tilelang.compile(program, out_idx=[1], target="ppu")
    profiler = kernel.get_profiler()

    def ref_program(A):
        import torch

        output = torch.clamp(A, min, max)
        return output

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def clamp_value_range(
    N,
    block_N,
    dtype,
):
    @T.prim_func
    def main(
        A: T.Tensor((1, N), dtype),
        B: T.Tensor((1, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=block_N) as bx:
            A_frag = T.alloc_fragment([1, block_N], dtype=dtype)
            min_frag = T.alloc_fragment([1], dtype=dtype)
            max_frag = T.alloc_fragment([1], dtype=dtype)
            T.copy(A[0, bx * block_N], A_frag)
            T.reduce_min(A_frag, min_frag, dim=1)
            T.reduce_max(A_frag, max_frag, dim=1)
            for i in T.Parallel(block_N):
                A_frag[0, i] = T.clamp(A_frag[0, i], min_frag[0] * 0.5, max_frag[0] * 0.5)
            T.copy(A_frag, B[0, bx * block_N])

    return main


def run_clamp_value_range(
    N,
    block_N,
    dtype,
):
    program = clamp_value_range(
        N,
        block_N,
        dtype,
    )
    kernel = tilelang.compile(program, out_idx=[1], target="ppu")

    import torch

    torch_dtype = dtype.as_torch()

    def ref_program(A):
        min_val = torch.min(A) * 0.5
        max_val = torch.max(A) * 0.5
        output = torch.clamp(A, min_val, max_val)
        return output

    A = torch.randint(-5, 5, (1, N)).cuda().to(dtype=torch_dtype)
    B = kernel(A)
    ref_b = ref_program(A)
    torch.testing.assert_close(B, ref_b)


@tilelang.testing.requires_ppu
def test_clamp():
    # clamp tests for float16 and float32
    run_clamp(1024, 128, T.float16, -0.05, 0.05)
    run_clamp(1024, 128, T.float32, -0.06, 0.05)
    run_clamp_value_range(1024, 128, T.float16)
    run_clamp_value_range(1024, 128, T.float32)


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("threads", [32, 64, 128])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_clamp_symmetric(threads, dtype):
    import torch

    @T.prim_func
    def main(A: T.Tensor((128,), dtype), B: T.Tensor((128,), dtype), C: T.Tensor((128,), dtype)):
        with T.Kernel(1, threads=threads):
            for i in T.Parallel(128):
                C[i] = T.clamp_symmetric(A[i], B[i])

    kernel = tilelang.compile(main, target="ppu")
    torch_dtype = getattr(torch, dtype)
    subnormal = torch.finfo(torch_dtype).tiny / 2
    values = [0.0, -0.0, 0.5, -0.5, 4.0, -4.0, float("inf"), -float("inf"), float("nan"), -float("nan"), subnormal, -subnormal]
    a = torch.tensor(values, device="cuda", dtype=torch_dtype).repeat(11)[:128].contiguous()
    int_dtype = torch.int32 if dtype == "float32" else torch.int16
    sign_mask = torch.iinfo(int_dtype).min
    # Construct negative NaNs by bits: narrowing may canonicalize their sign.
    a.view(int_dtype)[9::12].bitwise_or_(sign_mask)
    c = torch.empty_like(a)
    for limit in [2.0, subnormal, float("inf")]:
        b = torch.full_like(a, limit)
        # Input-only, limit-only, and both-NaN lanes, mixed with numeric lanes
        # in the same packed x2 call. Include both NaN signs in each operand.
        b[2::12] = float("nan")
        b[5::12] = float("nan")
        b.view(int_dtype)[5::12].bitwise_or_(sign_mask)
        b[8::24] = float("nan")
        b[9::24] = float("nan")
        kernel(a, b, c)
        ref = torch.clamp(a.float(), -b.float(), b.float())
        torch.testing.assert_close(c.float(), ref, equal_nan=True, atol=0, rtol=0)
        numeric = ~torch.isnan(ref)
        assert torch.equal(torch.signbit(c[numeric]), torch.signbit(ref[numeric]))


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("threads", [32, 64, 128])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_clamp_symmetric_codegen(threads, dtype):
    @T.prim_func
    def main(A: T.Tensor((128,), dtype), C: T.Tensor((128,), dtype)):
        with T.Kernel(1, threads=threads):
            for i in T.Parallel(128):
                C[i] = T.clamp_symmetric(A[i], 1.0)

    kernel = tilelang.compile(main, target="ppu")
    source = kernel.get_kernel_source()
    helper = "tl::clamp_symmetric2(" if dtype != "float32" and threads < 128 else "tl::clamp_symmetric("
    assert helper in source
    assert "tl::clamp(" not in source


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
def test_clamp_symmetric_types(dtype):
    x = T.Var("x", dtype + "x4")
    result = T.clamp_symmetric(x, 1.0)
    assert result.dtype == dtype + "x4"
    assert all(arg.dtype == dtype + "x4" for arg in result.args)
    with pytest.raises(TypeError, match="float32"):
        T.clamp_symmetric(T.Var("double_x", "float64"), 1.0)
    with pytest.raises(TypeError, match="matching operand dtypes"):
        T.clamp_symmetric(x, T.Var("limit", "float64"))
    with pytest.raises(ValueError, match="matching lanes"):
        T.clamp_symmetric(x, T.Var("limit", dtype + "x2"))


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("dtype", [T.float16, T.bfloat16, T.float32, T.float64])
def test_clamp_vectorized_operands(dtype):
    import torch

    # Plain min/max clamp: no NaN inputs, since the target's min/max
    # operations decide NaN handling and do not propagate it.
    N = 128

    @T.prim_func
    def main(A: T.Tensor((N,), dtype), Lo: T.Tensor((N,), dtype), Hi: T.Tensor((N,), dtype), C: T.Tensor((N,), dtype)):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(N):
                C[i] = T.clamp(A[i], Lo[i], Hi[i])

    kernel = tilelang.compile(main, target="ppu")
    a = torch.linspace(-2.0, 2.0, N, dtype=torch.float64, device="cuda")
    lo = torch.full_like(a, -1.0)
    hi = torch.full_like(a, 1.0)
    lo[4:8] = 5.0  # Reversed bounds: min(max(x, lo), hi) still selects hi.
    lo[8:12], hi[8:12] = 0.5, 0.5
    a[12], a[13] = float("inf"), -float("inf")
    a, lo, hi = (value.to(dtype.as_torch()) for value in (a, lo, hi))
    c = torch.empty_like(a)
    kernel(a, lo, hi, c)
    ref = torch.clamp(a.double(), lo.double(), hi.double())
    torch.testing.assert_close(c.double(), ref, atol=0, rtol=0)


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("dtype", [T.float16, T.bfloat16, T.float32, T.float64])
def test_clamp_vector_input_scalar_bounds(dtype):
    import torch

    @T.prim_func
    def main(A: T.Tensor((128,), dtype), C: T.Tensor((128,), dtype)):
        with T.Kernel(1, threads=32):
            idx = T.Ramp(T.get_thread_binding() * 4, 1, 4)
            C[idx] = T.clamp(A[idx], T.cast(-1.0, dtype), T.cast(1.0, dtype))

    a = torch.tensor([-3.0, -2.0, 0.5, 2.0], device="cuda").repeat(32).to(dtype.as_torch())
    c = torch.empty_like(a)
    tilelang.compile(main, target="ppu")(a, c)
    torch.testing.assert_close(c.float(), a.float().clamp(-1, 1), atol=0, rtol=0)


FP8_OPS = {
    "max": lambda a, b, dtype: T.max(a, b),
    "min": lambda a, b, dtype: T.min(a, b),
    "clamp": lambda a, b, dtype: T.clamp(a, T.cast(-1.0, dtype), T.cast(1.0, dtype)),
}


def fp8_operand_kernel(N, block_N, dtype, op):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=block_N) as bx:
            for i in T.Parallel(block_N):
                C[bx * block_N + i] = FP8_OPS[op](A[bx * block_N + i], B[bx * block_N + i], dtype)

    return main


def run_fp8_min_max_clamp(N, block_N, dtype, op):
    import torch

    kernel = tilelang.compile(fp8_operand_kernel(N, block_N, dtype, op), target="ppu")
    torch_dtype = dtype.as_torch()

    # Integers are exactly representable in both float8 formats, so the
    # reference comparison can be exact.
    a = torch.randint(-4, 5, (N,), device="cuda").to(torch.float32)
    b = torch.randint(-4, 5, (N,), device="cuda").to(torch.float32)
    if op == "max":
        ref = torch.maximum(a, b)
    elif op == "min":
        ref = torch.minimum(a, b)
    else:
        ref = a.clamp(-1.0, 1.0)

    # Allocate the output here instead of using out_idx: rebuilding a torch
    # tensor from an fp8 DLPack tensor is not supported on every torch version.
    C = torch.empty(N, device="cuda", dtype=torch_dtype)
    kernel(a.to(torch_dtype), b.to(torch_dtype), C)
    torch.testing.assert_close(C.float(), ref.to(torch_dtype).float(), atol=0, rtol=0)


def fp8_operand_vectorized_kernel(N, block_N, dtype, op, vec=4):
    @T.prim_func
    def main(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=block_N // vec) as bx:
            for i in T.Parallel(block_N // vec):
                for v in T.vectorized(vec):
                    idx = bx * block_N + i * vec + v
                    C[idx] = FP8_OPS[op](A[idx], B[idx], dtype)

    return main


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("dtype", [T.float8_e4m3, T.float8_e5m2])
@pytest.mark.parametrize("op", list(FP8_OPS))
def test_fp8_min_max_clamp(dtype, op):
    run_fp8_min_max_clamp(1024, 128, dtype, op)


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("op", list(FP8_OPS))
def test_fp8_vectorized_min_max(op):
    # Vectorizing reaches the per-lane fallback, a separate overload-resolution
    # context from the scalar expression.
    tilelang.compile(fp8_operand_vectorized_kernel(1024, 128, T.float8_e4m3, op), out_idx=[2], target="ppu")


@tilelang.testing.requires_ppu
def test_fp8_e4m3fn_min_max():
    # float8_e4m3fn shares the emitted device type with float8_e4m3; clamp
    # covers both min and max.
    tilelang.compile(fp8_operand_kernel(1024, 128, T.float8_e4m3fn, "clamp"), out_idx=[2], target="ppu")


if __name__ == "__main__":
    tilelang.testing.main()
