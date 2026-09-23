"""PPU round-mode codegen and numerical tests.

Regression guard for the T.round lowering on the PPU backend.  The PPU
intrinsic rules (src/ppu/codegen/intrin_rule_ppu.cc) are a hand-maintained
clone of the CUDA rules; a missing special case used to collapse both
rounding modes into the same device function (roundf):

- ties-to-even (default) must lower to nearbyintf, matching the TVM constant
  folding semantics (std::nearbyint) and the CUDA backend.
- ties-away-from-zero must lower to roundf.

Exact .5 tie values are also validated numerically, so a device toolchain
(hgcc) with non-conforming nearbyint/round semantics is caught as well.
"""

import math
import re

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


M, N = 32, 32
BLOCK_M, BLOCK_N = 32, 32

# Exact .5 ties (where the two rounding modes differ) plus non-tie values to
# catch off-by-one and sign errors.
TIE_VALUES = [
    0.5, 1.5, 2.5, 3.5, 4.5, -0.5, -1.5, -2.5, -3.5, -4.5,
    0.4, 0.6, -0.4, -0.6, 2.0, -2.0, 0.0, 1.0,
]


def make_round_kernel(rounding_mode):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), T.float32),
        B: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), T.ceildiv(M, BLOCK_M), threads=128) as (bx, by):
            A_local = T.alloc_fragment((BLOCK_M, BLOCK_N), T.float32)
            B_local = T.alloc_fragment((BLOCK_M, BLOCK_N), T.float32)
            T.copy(A[by * BLOCK_M, bx * BLOCK_N], A_local)
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                B_local[i, j] = T.round(A_local[i, j], rounding_mode)
            T.copy(B_local, B[by * BLOCK_M, bx * BLOCK_N])

    return main


def _count_calls(source, func_name):
    """Count (fastmath, plain) occurrences of func_name in the kernel source."""
    fastmath = re.findall(rf"__{func_name}\b", source)
    plain = re.findall(rf"(?<!__){func_name}\b", source)
    return len(fastmath), len(plain)


@tilelang.testing.requires_ppu
@pytest.mark.parametrize(
    ("rounding_mode", "expected_func", "collapsed_func"),
    [
        ("ties-to-even", "nearbyintf", "roundf"),
        ("ties-away-from-zero", "roundf", "nearbyintf"),
    ],
    ids=["ties-to-even", "ties-away-from-zero"],
)
def test_round_modes(rounding_mode, expected_func, collapsed_func):
    kernel = tilelang.compile(make_round_kernel(rounding_mode), out_idx=[1])
    source = kernel.get_kernel_source()

    # Source-level check: the expected non-fastmath function must be emitted.
    fastmath, plain = _count_calls(source, expected_func)
    assert plain > 0, (
        f"round[{rounding_mode}]: expected {expected_func} in PPU kernel source:\n{source}"
    )
    assert fastmath == 0, (
        f"round[{rounding_mode}]: unexpected fastmath __{expected_func} in PPU kernel source:\n{source}"
    )

    # Before the round->nearbyint special case was added to PPUMath, both
    # rounding modes collapsed to roundf; make sure the other rounding
    # function is absent (the pattern has no lookbehind, so it also catches
    # the fastmath __-prefixed variant).
    collapsed = re.findall(rf"{collapsed_func}\b", source)
    assert not collapsed, (
        f"round[{rounding_mode}]: found {collapsed_func} in PPU kernel source "
        f"(rounding modes collapsed?):\n{source}"
    )

    # Numerical check on exact .5 ties: both modes must follow their own
    # semantics on the device (guards against a non-conforming toolchain).
    a = torch.tensor(TIE_VALUES, dtype=torch.float32)
    repeats = math.ceil(M * N / len(TIE_VALUES))
    a = a.repeat(repeats)[: M * N].reshape(M, N).cuda()

    if rounding_mode == "ties-to-even":
        expected = torch.round(a.cpu())
    else:  # ties-away-from-zero
        a_cpu = a.cpu()
        expected = torch.sign(a_cpu) * torch.floor(torch.abs(a_cpu) + 0.5)

    out = kernel(a).cpu()
    torch.testing.assert_close(out, expected, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    tilelang.testing.main()
