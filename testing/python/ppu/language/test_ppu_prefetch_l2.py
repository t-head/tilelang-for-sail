"""Address-based L2 prefetch (tl.prefetch_l2) on the PPU backend.

Ported from the CUDA front end. PPU has a single hardware prefetch instruction
for the global L2 form; eviction policies and the L1 variants have no distinct
PPU encoding, so only the policy-less global form is exported and every other
form is rejected at the front end or in codegen.
"""

import pytest
import torch

import tilelang
import tilelang.ppu.language as T
import tilelang.testing
from tvm.error import TVMError
from tvm.ir import Op
from tvm.target import Target

PPU_TARGET = "ppu"
SIZE = 4096
OFFSET = 512


def _kernel(enable_prefetch):
    @T.prim_func
    def kernel(data: T.Tensor((SIZE,), "float32"), out: T.Tensor((SIZE,), "float32")):
        with T.Kernel(1, threads=256):
            for i in T.Parallel(SIZE):
                if enable_prefetch:
                    T.prefetch_l2(data[(i + OFFSET) % SIZE])
                out[i] = data[i] * 2.0

    return kernel


def _raw_kernel(op_name, is_global, priority=-1):
    @T.prim_func
    def kernel(data: T.Tensor((SIZE,), "float32"), out: T.Tensor((SIZE,), "float32")):
        with T.Kernel(1, threads=256):
            for i in T.Parallel(SIZE):
                if op_name == "tl.prefetchu_l1":
                    args = [T.address_of(data[i])]
                elif op_name == "tl.prefetch_l1":
                    args = [T.address_of(data[i]), is_global]
                else:
                    args = [T.address_of(data[i]), priority, is_global]
                T.evaluate(T.call_intrin("void", Op.get(op_name), *args))
                out[i] = data[i] * 2.0

    return kernel


@tilelang.testing.requires_ppu
def test_prefetch_export_surface():
    assert T.prefetch_l2.__module__ == "tilelang.ppu.language.prefetch"
    assert "prefetch_l2" in T.__all__
    for name in ("prefetch_l1", "prefetchu_l1"):
        assert name not in T.__all__
        assert not hasattr(T, name)


def _shared_prefetch_kernel(policy):
    @T.prim_func
    def kernel(out: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((128,), "float32")
            for i in T.Parallel(128):
                shared[i] = 1.0
                T.prefetch_l2(shared[i], eviction_policy=policy)
                out[i] = shared[i]

    return kernel


def _lower(kernel):
    with Target(PPU_TARGET):
        return tilelang.lower(kernel, target=PPU_TARGET, enable_device_compile=False).kernel_source


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("policy", ["evict_normal", "evict_last", "evict_first"])
def test_prefetch_l2_rejects_eviction_policy(policy):
    with pytest.raises(ValueError, match="does not support eviction_policy"):
        _shared_prefetch_kernel(policy)


@tilelang.testing.requires_ppu
def test_prefetch_l2_rejects_non_global():
    with pytest.raises(ValueError, match="requires global memory"):
        _shared_prefetch_kernel(None)


@tilelang.testing.requires_ppu
def test_prefetch_l2_codegen():
    source = _lower(_kernel(True))
    # T.Parallel lowers to a thread-strided loop, so the prefetch appears once per loop nest level.
    assert source.count("tl::prefetch_l2<-1, true>(") >= 1
    control = _lower(_kernel(False))
    assert "prefetch_l2" not in control


@tilelang.testing.requires_ppu
def test_prefetch_l2_raw_ir_rejects_policy_and_local():
    with pytest.raises(TVMError, match="policy-less global form"):
        _lower(_raw_kernel("tl.prefetch_l2", True, priority=0))
    with pytest.raises(TVMError, match="policy-less global form"):
        _lower(_raw_kernel("tl.prefetch_l2", False))


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("op_name", ["tl.prefetch_l1", "tl.prefetchu_l1"])
def test_prefetch_l1_not_supported(op_name):
    with pytest.raises(TVMError, match="no distinct L1 prefetch encoding"):
        _lower(_raw_kernel(op_name, True))


@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_prefetch_l2_runtime():
    # Prefetch is a performance hint: results must match both the reference and
    # the same kernel compiled without the prefetch.
    with_prefetch = tilelang.compile(_kernel(True), target=PPU_TARGET, execution_backend="tvm_ffi")
    without_prefetch = tilelang.compile(_kernel(False), target=PPU_TARGET, execution_backend="tvm_ffi")
    data = torch.randn(SIZE, device="cuda", dtype=torch.float32)
    expected = data * 2.0
    for compiled in (with_prefetch, without_prefetch):
        out = torch.empty_like(data)
        compiled(data, out)
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected)


if __name__ == "__main__":
    tilelang.testing.main()
