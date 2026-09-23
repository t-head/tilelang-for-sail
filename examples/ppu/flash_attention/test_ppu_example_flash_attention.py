import tilelang.testing
import ppu_example_gqa_bwd
import ppu_example_gqa_fwd_bshd
import ppu_example_gqa_fwd_varlen
import ppu_example_mha_bwd_bhsd
import ppu_example_mha_bwd_bshd
import ppu_example_mha_fwd_bhsd
import ppu_example_mha_fwd_bshd
import ppu_example_mha_fwd_varlen


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_example_gqa_bwd():
    ppu_example_gqa_bwd.main()


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_example_gqa_fwd_bshd():
    ppu_example_gqa_fwd_bshd.main()


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_example_gqa_fwd_varlen():
    ppu_example_gqa_fwd_varlen.main()


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_example_mha_bwd_bhsd():
    ppu_example_mha_bwd_bhsd.main()


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_example_mha_bwd_bshd():
    ppu_example_mha_bwd_bshd.main()


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_example_mha_fwd_bhsd():
    ppu_example_mha_fwd_bhsd.main()


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_example_mha_fwd_bshd():
    ppu_example_mha_fwd_bshd.main()


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_example_mha_fwd_varlen():
    ppu_example_mha_fwd_varlen.main()


if __name__ == "__main__":
    tilelang.testing.main()
