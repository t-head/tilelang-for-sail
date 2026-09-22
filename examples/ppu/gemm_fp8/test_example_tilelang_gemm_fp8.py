import tilelang.testing
import example_tilelang_gemm_fp8


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_example_gemm_fp8():
    example_tilelang_gemm_fp8.main()


if __name__ == "__main__":
    tilelang.testing.main()
