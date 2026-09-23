import tilelang.testing

import example_gemm_sp


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_eq(1, 0)
def test_example_gemm_sp():
    example_gemm_sp.main()


if __name__ == "__main__":
    tilelang.testing.main()
