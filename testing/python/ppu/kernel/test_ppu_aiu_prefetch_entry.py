"""AIU loop entry must not rely on a pending group surviving the loop tail."""

import re

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.testing.requires_ppu_compute_version(1, 5)
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("iterations", [2, 4])
def test_aiu_prefetch_first_consumer(head_dim, iterations):
    m, n = 128, 32

    @T.prim_func
    def main(
        Q: T.Tensor((m, head_dim), "bfloat16"),
        K: T.Tensor((iterations, n, head_dim), "bfloat16"),
        V: T.Tensor((iterations, n, head_dim), "bfloat16"),
        O: T.Tensor((m, n), "float32"),
    ):
        with T.Kernel(1, threads=256):
            q = T.alloc_shared((m, head_dim), "bfloat16")
            k = T.alloc_shared((n, head_dim), "bfloat16")
            v = T.alloc_shared((n, head_dim), "bfloat16")
            acc = T.alloc_fragment((m, n), "float32")
            T.copy(Q, q, prefer_instruction="aiu")
            T.copy(K[0, 0:n, 0:head_dim], k, prefer_instruction="aiu")
            T.clear(acc)
            for step in T.serial(iterations):
                T.copy(V[step, 0:n, 0:head_dim], v, prefer_instruction="aiu")
                T.gemm(q, k, acc, transpose_B=True)
                if step < iterations - 1:
                    T.copy(K[step + 1, 0:n, 0:head_dim], k, prefer_instruction="aiu")
                # Waiting for V also drains the K prefetch. The pass therefore
                # has no pending loop-tail group from which to infer a wait
                # before Q/K's first consumer in the initial iteration.
                T.gemm(q, v, acc, transpose_B=True)
            T.copy(acc, O)

    kernel = tilelang.compile(main, out_idx=[3], pass_configs={"tl.disable_aiu_lower": False})
    source = kernel.get_kernel_source()
    loop = re.search(r"for \(int step\b", source)
    assert loop, source
    torch.manual_seed(45)
    q = torch.randn((m, head_dim), device="cuda", dtype=torch.bfloat16) * 0.1
    k = torch.randn((iterations, n, head_dim), device="cuda", dtype=torch.bfloat16) * 0.1
    v = torch.randn_like(k) * 0.1
    expected = torch.zeros((m, n), device="cuda", dtype=torch.float32)
    for i in range(iterations):
        expected += q.float() @ k[i].float().T
        expected += q.float() @ v[i].float().T
    for _ in range(10):
        torch.testing.assert_close(kernel(q, k, v), expected, atol=0.02, rtol=0.02)
    assert "tl::cp_async_wait<0>()" in source[: loop.start()], source
    assert "__syncthreads()" in source[source.index("tl::cp_async_wait<0>()") : loop.start()], source


if __name__ == "__main__":
    tilelang.testing.main()
