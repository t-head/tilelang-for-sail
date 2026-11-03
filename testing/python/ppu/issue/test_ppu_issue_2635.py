# PPU variant of testing/python/issue/test_tilelang_issue_2635.py:
# ties-away-from-zero rounding on 16-bit / fp8 buffers (issue #2635).
import tilelang
import tilelang.testing
import tilelang.ppu.language as T


def _make_round_kernel(dtype):
    @T.prim_func
    def main(A: T.Tensor((16,), dtype), B: T.Tensor((16,), dtype)):
        with T.Kernel(1, threads=16):
            for i in T.Parallel(16):
                B[i] = T.round(A[i], "ties-away-from-zero")

    return main


@tilelang.testing.requires_ppu
def test_round_ties_away_from_zero_compiles_for_bfloat16():
    tilelang.compile(_make_round_kernel("bfloat16"), target="ppu")


@tilelang.testing.requires_ppu
def test_round_ties_away_from_zero_compiles_for_float8():
    for dtype in ("float8_e4m3", "float8_e5m2"):
        tilelang.compile(_make_round_kernel(dtype), target="ppu")


if __name__ == "__main__":
    tilelang.testing.main()
