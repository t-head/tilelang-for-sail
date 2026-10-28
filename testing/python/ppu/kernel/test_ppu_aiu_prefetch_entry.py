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


@tilelang.testing.requires_ppu_compute_version(1, 5)
@pytest.mark.parametrize("mode,work", [("simple", 0), ("prefetch", 0), ("prefetch", 4), ("explicit", 0), ("explicit", 4)])
def test_aiu_prefetch_existing_wait_covers_entry(mode, work):
    m, k, steps = 64, 128, 4

    @T.prim_func
    def main(
        A: T.Tensor((m, k), "bfloat16"),
        B: T.Tensor((steps, m, k), "bfloat16"),
        O: T.Tensor((m, m), "float32"),
        D: T.Tensor((m, m), "float32"),
    ):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((m, k), "bfloat16")
            b = T.alloc_shared((m, k), "bfloat16")
            acc = T.alloc_fragment((m, m), "float32")
            independent = T.alloc_fragment((m, m), "float32")
            T.copy(A, a, prefer_instruction="aiu")
            if mode != "simple":
                T.ptx_wait_group(0)
                T.sync_threads()
            T.copy(B[0, :, :], b, prefer_instruction="aiu")
            T.clear(acc)
            T.clear(independent)
            for step in T.serial(steps):
                for _repeat in T.serial(work):
                    T.gemm(a, a, independent, transpose_B=True)
                if mode == "explicit":
                    T.ptx_wait_group(0)
                    T.sync_threads()
                T.gemm(a, b, acc, transpose_B=True)
                if step < steps - 1:
                    T.copy(B[step + 1, :, :], b, prefer_instruction="aiu")
            T.copy(acc, O)
            T.copy(independent, D)

    kernel = tilelang.compile(main, out_idx=[2, 3], pass_configs={"tl.disable_aiu_lower": False})
    source = kernel.get_kernel_source()
    loop = re.search(r"for \(int step\b", source)
    assert loop, source
    prefix = source[: loop.start()]
    # The loop's existing wait (explicit or inferred from its backedge) must
    # protect B without an extra wait/barrier ahead of the independent work.
    last_commit = prefix.rindex("tl::cp_async_commit();")
    assert "tl::cp_async_wait" not in prefix[last_commit:], source
    torch.manual_seed(48)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.05
    b = torch.randn((steps, m, k), device="cuda", dtype=torch.bfloat16) * 0.05
    expected = sum(a.float() @ b[i].float().T for i in range(steps))
    independent = (a.float() @ a.float().T) * (steps * work)
    for _ in range(10):
        out, extra = kernel(a, b)
        torch.testing.assert_close(out, expected, atol=0.02, rtol=0.02)
        torch.testing.assert_close(extra, independent, atol=0.02, rtol=0.02)


@tilelang.testing.requires_ppu_compute_version(1, 5)
@pytest.mark.parametrize("steps", [1, 4])
@pytest.mark.parametrize("drain_tail", [False, True])
def test_aiu_prefetch_loop_exit_consumer(steps, drain_tail):
    m, k = 64, 128

    @T.prim_func
    def main(
        A: T.Tensor((m, k), "bfloat16"),
        B: T.Tensor((steps + 1, m, k), "bfloat16"),
        O: T.Tensor((m, m), "float32"),
    ):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((m, k), "bfloat16")
            b = T.alloc_shared((m, k), "bfloat16")
            acc = T.alloc_fragment((m, m), "float32")
            T.copy(A, a, prefer_instruction="aiu")
            T.copy(B[0, :, :], b, prefer_instruction="aiu")
            T.ptx_wait_group(0)
            T.sync_threads()
            T.clear(acc)
            for step in T.serial(steps):
                T.gemm(a, b, acc, transpose_B=True)
                T.copy(B[step + 1, :, :], b, prefer_instruction="aiu")
                if drain_tail:
                    T.ptx_wait_group(0)
                    T.sync_threads()
            T.gemm(a, b, acc, transpose_B=True)
            T.copy(acc, O)

    kernel = tilelang.compile(main, out_idx=[2], pass_configs={"tl.disable_aiu_lower": False})
    for seed in (45, 48):
        torch.manual_seed(seed)
        a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.05
        b = torch.randn((steps + 1, m, k), device="cuda", dtype=torch.bfloat16) * 0.05
        expected = sum(a.float() @ b[i].float().T for i in range(steps + 1))
        for _ in range(10):
            torch.testing.assert_close(kernel(a, b), expected, atol=0.02, rtol=0.02)


if __name__ == "__main__":
    tilelang.testing.main()
