import tilelang.testing
import ppu_example_gemm
import ppu_example_gemm_autotune
import ppu_example_gemm_intrinsics
import ppu_example_gemm_persistent


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_example_gemm():
    ppu_example_gemm.main()


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_example_gemm_autotune():
    ppu_example_gemm_autotune.main()


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_example_gemm_intrinsics():
    ppu_example_gemm_intrinsics.main()


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_example_gemm_persistent():
    ppu_example_gemm_persistent.main()


if __name__ == "__main__":
    tilelang.testing.main()
