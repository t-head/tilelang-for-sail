import tilelang.testing
import torch
import ppu_example_gqa_bwd
import ppu_example_gqa_fwd_bshd
import ppu_example_gqa_fwd_varlen
import ppu_example_mha_bwd_bhsd
import ppu_example_mha_bwd_bshd
import ppu_example_mha_fwd_bhsd
import ppu_example_mha_fwd_bshd
import ppu_example_mha_fwd_fp8
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


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_example_mha_fwd_fp8(monkeypatch):
    q_lengths = (67, 65, 0, 3)
    kv_lengths = (64, 321, 0, 0)
    q_heads, kv_heads, head_dim = 16, 2, 128
    bounds = torch.tensor((0, 67, 132, 132, 135), dtype=torch.int32)
    block_table = torch.tensor([[3, 0, 0, 0, 0, 0], [6, 0, 5, 2, 1, 4], [0] * 6, [0] * 6], dtype=torch.int32)
    generator = torch.Generator().manual_seed(0)
    q = (torch.randn((sum(q_lengths), q_heads, head_dim), generator=generator) * 0.25).to(torch.float8_e4m3fn)
    cache_shape = (7, 64, kv_heads, head_dim)
    k = (torch.randn(cache_shape, generator=generator) * 0.25).to(torch.float8_e4m3fn)
    v = (torch.randn(cache_shape, generator=generator) * 0.25).to(torch.float8_e4m3fn)
    scales = torch.linspace(0.6, 1.4, 24).reshape(3, len(q_lengths), kv_heads)
    softmax_scale = head_dim**-0.5

    # Use the dense route for the one-page request and exercise split-KV
    # on the six-page request without a long-KV input.
    monkeypatch.setitem(ppu_example_mha_fwd_fp8._SPLIT_OVERRIDES, (q_heads, 6), 2)
    out = torch.full((q_heads, len(q), head_dim), 9.0, dtype=torch.bfloat16, device="cuda").transpose(0, 1)
    result = ppu_example_mha_fwd_fp8.Model()(
        q.cuda(),
        k.cuda(),
        v.cuda(),
        out,
        bounds.cuda(),
        max(q_lengths),
        torch.tensor(kv_lengths, dtype=torch.int32, device="cuda"),
        max(kv_lengths),
        softmax_scale,
        True,
        (-1, -1),
        block_table.cuda(),
        None,
        3,
        *scales.cuda().unbind(),
        0,
    )
    assert result is out

    q_float, k_float, v_float = q.float(), k.float(), v.float()
    expected = torch.zeros_like(q_float)
    for req, (q_len, kv_len) in enumerate(zip(q_lengths, kv_lengths)):
        if q_len == 0 or kv_len == 0:
            continue
        start, end = bounds[req].item(), bounds[req + 1].item()
        pages = block_table[req, : (kv_len + 63) // 64]
        q_scale, k_scale, v_scale = scales[:, req].repeat_interleave(8, dim=1).reshape(3, q_heads, 1, 1)
        keys = k_float[pages].reshape(-1, kv_heads, head_dim)[:kv_len].repeat_interleave(8, dim=1).transpose(0, 1)
        values = v_float[pages].reshape(-1, kv_heads, head_dim)[:kv_len].repeat_interleave(8, dim=1).transpose(0, 1)
        scores = (q_float[start:end].transpose(0, 1) * q_scale) @ (keys * k_scale).transpose(-2, -1)
        allowed = torch.arange(kv_len)[None, :] <= torch.arange(q_len)[:, None] + kv_len - q_len
        weights = (scores * softmax_scale).masked_fill(~allowed, -torch.inf).softmax(-1).masked_fill(~allowed, 0)
        expected[start:end] = (weights @ (values * v_scale)).transpose(0, 1)
    actual = out.cpu()
    torch.testing.assert_close(actual, expected.to(torch.bfloat16), atol=0.01, rtol=0.02)
    assert torch.count_nonzero(actual[:3]) == 0
    assert torch.count_nonzero(actual[-3:]) == 0


if __name__ == "__main__":
    tilelang.testing.main()
