import pytest

import tilelang
import tilelang.ppu.language as T
import tilelang.testing


def _kernel_body(source):
    return source[source.rindex("__global__") :]


def _half_math_kernel(dtype):
    if dtype == T.float16:

        @T.prim_func
        def main(
            A: T.Tensor((32,), dtype),
            C: T.Tensor((32,), dtype),
            B: T.Tensor((8, 32), dtype),
        ):
            with T.Kernel(1, threads=32):
                i = T.get_thread_binding()
                B[0, i] = T.sinh(A[i])
                B[1, i] = T.cosh(A[i])
                B[2, i] = T.tanh(A[i])
                B[3, i] = T.atan(A[i])
                B[4, i] = T.erf(A[i])
                B[5, i] = T.nearbyint(A[i])
                B[6, i] = T.fmod(A[i], C[i])
                B[7, i] = T.pow(A[i], C[i])

    else:

        @T.prim_func
        def main(
            A: T.Tensor((32,), dtype),
            C: T.Tensor((32,), dtype),
            B: T.Tensor((7, 32), dtype),
        ):
            with T.Kernel(1, threads=32):
                i = T.get_thread_binding()
                B[0, i] = T.sinh(A[i])
                B[1, i] = T.cosh(A[i])
                B[2, i] = T.tanh(A[i])
                B[3, i] = T.atan(A[i])
                B[4, i] = T.erf(A[i])
                B[5, i] = T.nearbyint(A[i])
                B[6, i] = T.fmod(A[i], C[i])

    return main


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("dtype", [T.float16, T.bfloat16])
def test_ppu_half_math_bridges_compile(dtype):
    kernel = tilelang.compile(_half_math_kernel(dtype), target="ppu")
    body = _kernel_body(kernel.get_kernel_source())
    expected = ["hsinh", "hcosh", "htanh", "hatan", "herf", "hnearbyint", "hfmod"]
    if dtype == T.float16:
        expected.append("hpow")
    assert all(f"{name}(" in body for name in expected)


def _extended_half_math_kernel(dtype):
    @T.prim_func
    def main(A: T.Tensor((32,), dtype), B: T.Tensor((12, 32), dtype)):
        with T.Kernel(1, threads=32):
            i = T.get_thread_binding()
            B[0, i] = T.tan(A[i])
            B[1, i] = T.exp(A[i])
            B[2, i] = T.exp2(A[i])
            B[3, i] = T.exp10(A[i])
            B[4, i] = T.log(A[i])
            B[5, i] = T.log2(A[i])
            B[6, i] = T.log10(A[i])
            B[7, i] = T.sin(A[i])
            B[8, i] = T.cos(A[i])
            B[9, i] = T.floor(A[i])
            B[10, i] = T.ceil(A[i])
            B[11, i] = T.trunc(A[i])

    return main


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("dtype", [T.float16, T.bfloat16])
def test_ppu_extended_half_math_bridges_compile(dtype):
    kernel = tilelang.compile(_extended_half_math_kernel(dtype), target="ppu")
    body = _kernel_body(kernel.get_kernel_source())
    expected = ["htan", "hexp", "hexp2", "hexp10", "hlog", "hlog2", "hlog10", "hsin", "hcos", "hfloor", "hceil", "htrunc"]
    assert all(f"{name}(" in body for name in expected)
