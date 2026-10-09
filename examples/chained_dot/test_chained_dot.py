"""Pytest for chained_dot examples.

Tests 4 variants by importing kernels directly from example scripts:
  1. chained_dot: a@b, (a@b)@b, ((a@b)@b)@b
  2. chained_dot2_transB: a@b, (a@b)@b.T
  3. chained_dot3_transB: a@b, (a@b)@b, ((a@b)@b)@b.T
  4. modify_chained_dot: a@b, (a@b)@b, ((a@b)@b)@b, v@b (independent branch)
"""

import torch

from chained_dot import matmul as matmul_chain3
from chained_dot2_transB import matmul as matmul_chain2_transB
from chained_dot3_transB import matmul as matmul_chain3_transB
from modify_chained_dot import matmul as matmul_modify


SHAPE = 64
BLOCK_SIZE = 64


def test_chained_dot(shape=SHAPE):
    """Verify 3-stage chain: C=a@b, D=C@b, E=D@b"""
    a = torch.randn(shape, shape, device="cuda", dtype=torch.float16)
    b = torch.randn(shape, shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, shape, device="cuda", dtype=torch.float16)

    kernel = matmul_chain3(shape, shape, shape, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    c, d, e = kernel(a, b, v)

    ref_c = a @ b
    ref_d = ref_c @ b
    ref_e = ref_d @ b

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(d, ref_d, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(e, ref_e, rtol=1e-2, atol=1e-2)


def test_chained_dot2_transB(shape=SHAPE):
    """Verify 2-stage chain with transpose_B on 2nd gemm: C=a@b, D=C@b^T"""
    a = torch.randn(shape, shape, device="cuda", dtype=torch.float16)
    b = torch.randn(shape, shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, shape, device="cuda", dtype=torch.float16)

    kernel = matmul_chain2_transB(shape, shape, shape, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    c, d, _e = kernel(a, b, v)

    ref_c = a @ b
    ref_d = ref_c @ b.T

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(d, ref_d, rtol=1e-2, atol=1e-2)


def test_chained_dot3_transB(shape=SHAPE):
    """Verify 3-stage chain with transpose_B on 3rd gemm: C=a@b, D=C@b, E=D@b^T"""
    a = torch.randn(shape, shape, device="cuda", dtype=torch.float16)
    b = torch.randn(shape, shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, shape, device="cuda", dtype=torch.float16)

    kernel = matmul_chain3_transB(shape, shape, shape, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    c, d, e = kernel(a, b, v)

    ref_c = a @ b
    ref_d = ref_c @ b
    ref_e = ref_d @ b.T

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(d, ref_d, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(e, ref_e, rtol=1e-2, atol=1e-2)


def test_modify_chained_dot(shape=SHAPE):
    """Verify 3-stage chain + independent branch: C=a@b, D=C@b, E=D@b, F=v@b"""
    a = torch.randn(shape, shape, device="cuda", dtype=torch.float16)
    b = torch.randn(shape, shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, shape, device="cuda", dtype=torch.float16)

    kernel = matmul_modify(shape, shape, shape, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    c, d, e, f = kernel(a, b, v)

    ref_c = a @ b
    ref_d = ref_c @ b
    ref_e = ref_d @ b
    ref_f = v @ b

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(d, ref_d, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(e, ref_e, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(f, ref_f, rtol=1e-2, atol=1e-2)
