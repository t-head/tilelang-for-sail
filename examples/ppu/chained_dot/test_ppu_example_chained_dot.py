import tilelang.testing
import ppu_chained_dot
import ppu_chained_dot2_transB
import ppu_chained_dot3_transB
import ppu_modify_chained_dot


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_chained_dot():
    ppu_chained_dot.main()


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_chained_dot2_transB():
    ppu_chained_dot2_transB.main()


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_chained_dot3_transB():
    ppu_chained_dot3_transB.main()


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_modify_chained_dot():
    ppu_modify_chained_dot.main()


if __name__ == "__main__":
    tilelang.testing.main()
