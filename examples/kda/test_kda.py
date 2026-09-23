"""Pytest for kda examples.

Each test calls the main() function of the corresponding example script,
which internally runs the tilelang kernel, compares against the FLA reference,
and prints accuracy metrics.  A test passes if the kernel compiles and runs
without exceptions.
"""

import os

import pytest

os.environ.setdefault("TILELANG_KDA_DISABLE_AUTOTUNE", "1")

from chunk_o import main as chunk_o_main
from chunk_delta_h_fwd import main as chunk_delta_h_fwd_main
from chunk_intra_token_parallel import main as chunk_intra_token_parallel_main
from chunk_inter_solve_fused import main as chunk_inter_solve_fused_main
from chunk_bwd_dv import main as chunk_bwd_dv_main
from chunk_bwd_dqkwg import main as chunk_bwd_dqkwg_main
from chunk_bwd_gla_dA import main as chunk_bwd_gla_dA_main
from chunk_bwd_intra import main as chunk_bwd_intra_main
from chunk_delta_bwd import main as chunk_delta_bwd_main
from wy_fast import main as wy_fast_main
from wy_fast_bwd import main as wy_fast_bwd_main


def test_chunk_o():
    """chunk forward O: inter-chunk hidden + intra-chunk attention"""
    chunk_o_main()


def test_chunk_delta_h_fwd():
    """chunk gated delta rule forward H computation"""
    chunk_delta_h_fwd_main()


def test_chunk_intra_token_parallel():
    """chunk intra token-parallel Aqk/Akk computation"""
    chunk_intra_token_parallel_main()


@pytest.mark.skip(reason="temporarily skipped per user request")
def test_chunk_inter_solve_fused():
    """chunk inter solve fused Aqk/Akk"""
    chunk_inter_solve_fused_main()


def test_chunk_bwd_dv():
    """chunk backward dV local"""
    chunk_bwd_dv_main()


def test_chunk_bwd_dqkwg():
    """chunk backward dQ/dK/dW/dG"""
    chunk_bwd_dqkwg_main()


def test_chunk_bwd_gla_dA():
    """chunk backward GLA dA"""
    chunk_bwd_gla_dA_main()


def test_chunk_bwd_intra():
    """chunk backward intra"""
    chunk_bwd_intra_main()


def test_chunk_delta_bwd():
    """chunk delta backward"""
    chunk_delta_bwd_main()


def test_wy_fast():
    """recompute W and U forward"""
    wy_fast_main()


def test_wy_fast_bwd():
    """recompute W and U backward"""
    wy_fast_bwd_main()
