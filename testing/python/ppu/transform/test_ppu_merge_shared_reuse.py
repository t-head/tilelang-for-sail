"""Regression tests for shared-memory lifetime reuse in MergeSharedMemoryAllocations.

Covers the two geometries handled by the region-based kill reorder of
liveness events:

1. ``test_ppu_gemm_shared_reuse_after_loop``: a pipelined gemm whose
   C_shared staging buffer lives entirely after the mainloop must reuse
   the A/B pipeline region (~72KB arena instead of ~136KB) while still
   computing correct results. This pins the PPU gemm shared-memory
   reduction. The PPU backend materializes the merged arena as dynamic
   shared memory (``buf_dyn_shmem`` with per-buffer byte offsets).

2. ``test_ppu_shared_reuse_no_cross_iteration_overwrite``: a buffer
   loaded before a loop and read in every iteration (the Q_shared shape
   of sparse_mla_bwd) must NOT share its region with a buffer first
   written inside the same loop (the acc_dkv_shared shape). Reusing
   there overwrites the loop-invariant buffer across the back edge and
   produces garbage — the geometry of upstream issue tile-ai#2199.
"""

import re

import tilelang
import tilelang.language as T
import tilelang.testing


def _merge_shared_pass_configs():
    return {
        # Merge the static ("shared"-scope) allocations into one arena.
        "tirx.merge_static_smem": True,
        # Attribute every touch to its deepest scope so in-loop events are
        # visible to the liveness analysis.
        "tl.enable_aggressive_shared_memory_merge": True,
    }


def _shared_arena_offsets(src):
    """Map each merged shared buffer to its byte offset in the arena.

    The PPU backend materializes the merged allocations as dynamic shared
    memory: ``void* A_shared = ((void*)((char*)buf_dyn_shmem + 0));``.
    """
    return {
        name: int(off)
        for name, off in re.findall(
            r"void\* (\w+) = \(\(void\*\)\(\(char\*\)buf_dyn_shmem \+ (\d+)\)\)", src
        )
    }


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_shared_reuse_after_loop():
    M, N, K = 256, 128, 64
    block_M, block_N, block_K = 256, 128, 32
    num_stages = 3

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            C_local = T.alloc_fragment((block_M, block_N), T.float16)
            # Staging buffer whose lifetime lies entirely after the mainloop.
            C_shared = T.alloc_shared((block_M, block_N), T.float16)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    kernel = tilelang.compile(
        main,
        out_idx=[2],
        pass_configs=_merge_shared_pass_configs(),
    )

    # The C_shared staging buffer lives entirely after the mainloop, so it
    # must reuse the A/B pipeline region: C lands at offset 0 (sharing A's
    # region) and the arena stays at the ~72KB pipeline footprint instead
    # of growing to ~136KB with a separate C region.
    src = kernel.get_kernel_source()
    offsets = _shared_arena_offsets(src)
    assert {"A_shared", "B_shared", "C_shared"} <= set(offsets), (
        f"merged dynamic shared arena (buf_dyn_shmem) not found; "
        f"buffers seen: {sorted(offsets)}"
    )
    assert offsets["C_shared"] == offsets["A_shared"] == 0, (
        f"C_shared must reuse the A/B pipeline region; offsets: {offsets}"
    )
    # A's 3-stage region is 48KB; B follows it, so a reused arena tops out
    # at 49152 + 24KB = 72KB. A non-reused layout would push C (or B) past
    # the 72KB mark (the ~136KB layout).
    assert offsets["B_shared"] == 49152, (
        f"expected B_shared right after A's 3-stage region; offsets: {offsets}"
    )
    assert max(offsets.values()) < 49152 + 24576, (
        f"arena grew past the 72KB pipeline footprint (non-reused layout "
        f"is ~136KB); offsets: {offsets}"
    )

    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        return torch.matmul(A.to(torch.float), B.to(torch.float)).to(torch.float16)

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_ppu
def test_ppu_shared_reuse_no_cross_iteration_overwrite():
    M, D, D2 = 64, 64, 64
    BLOCK_M = 64
    N_ITER = 4

    @T.prim_func
    def main(
        A: T.Tensor((M, D), T.float16),
        B: T.Tensor((D, D2), T.float16),
        C: T.Tensor((M, D2), T.float16),
    ):
        with T.Kernel(T.ceildiv(M, BLOCK_M), threads=128) as bx:
            # Q geometry: loaded once before the loop, read every iteration.
            A_shared = T.alloc_shared((BLOCK_M, D), T.float16)
            B_shared = T.alloc_shared((D, D2), T.float16)
            # acc_dkv geometry: first written inside the loop.
            acc_shared = T.alloc_shared((BLOCK_M, D2), T.float16)
            acc_local = T.alloc_fragment((BLOCK_M, D2), T.float32)
            T.copy(A[bx * BLOCK_M, 0], A_shared)
            T.copy(B[0, 0], B_shared)
            T.clear(acc_local)
            for i in T.Pipelined(N_ITER, num_stages=1):
                T.gemm(A_shared, B_shared, acc_local)
                T.copy(acc_local, acc_shared)
            T.copy(acc_shared, C[bx * BLOCK_M, 0])

    kernel = tilelang.compile(
        main,
        out_idx=[2],
        pass_configs=_merge_shared_pass_configs(),
    )

    # Structural check alongside the numerical one: the packer must judge
    # A_shared (live across the back edge) as overlapping acc_shared
    # (first written inside the loop), so all three 8KB shared buffers get
    # disjoint regions in the arena.
    src = kernel.get_kernel_source()
    offsets = _shared_arena_offsets(src)
    assert sorted(offsets.values()) == [0, 8192, 16384], (
        f"expected three disjoint 8KB regions (no reuse across the back "
        f"edge); got {offsets}"
    )

    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        acc = torch.matmul(A.to(torch.float), B.to(torch.float)) * N_ITER
        return acc.to(torch.float16)

    # If the packer wrongly treats A_shared as dead before acc_shared's first
    # write, acc_shared reuses A_shared's region: the second iteration then
    # reads the overwritten values and the result diverges from the reference.
    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
