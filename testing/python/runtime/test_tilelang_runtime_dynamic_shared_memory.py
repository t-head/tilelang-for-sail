import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.jit
def dynamic_smem_kernel():
    # Symbolic length to drive dynamic shared memory allocation
    length = T.dynamic("len", dtype=T.int32)  # noqa: F821

    @T.prim_func
    def main(global_tensor: T.Tensor[(length,), T.int32]):  # noqa: F821
        # Launch a simple kernel that copies from global memory into shared memory
        # using a dynamically-sized allocation. No writes back to global_tensor.
        with T.Kernel(1, threads=32) as _:
            buffer_shared = T.alloc_shared((length,), dtype=T.int32)  # noqa: F821
            T.copy(buffer_shared, global_tensor)

    return main


def _require_cuda_tensor(shape, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    try:
        return torch.randint(0, 100, shape, dtype=dtype, device="cuda")
    except RuntimeError as err:
        pytest.skip(f"CUDA runtime unavailable: {err}")


def _run_and_check(kernel, n):
    a = _require_cuda_tensor((n,), torch.int32)
    kernel(a)
    torch.cuda.synchronize()


def test_dynamic_shared_memory_varies_across_calls():
    kernel = dynamic_smem_kernel()

    # Run with different dynamic shared memory sizes across invocations
    _run_and_check(kernel, 100)
    _run_and_check(kernel, 200)
    # Repeat sizes to exercise attribute caching path
    _run_and_check(kernel, 200)
    _run_and_check(kernel, 100)


@tilelang.testing.requires_ppu
def test_ppu_dynamic_shared_memory_uses_hggc_setter():
    length = T.dynamic("len", dtype=T.int32)

    @T.prim_func
    def main(global_tensor: T.Tensor[(length,), T.int32]):
        with T.Kernel(1, threads=32):
            buffer_shared = T.alloc_shared((length,), dtype=T.int32)
            T.copy(buffer_shared, global_tensor)

    kernel = tilelang.compile(main, target="ppu", execution_backend="cython")
    host_source = kernel.get_host_source()
    assert "static int main_set_dynamic_smem()" in host_source
    assert "hggcGetDevice(&device)" in host_source
    assert "hggcFuncSetAttribute" in host_source
    assert "main_set_dynamic_smem() != 0" in host_source

    # Exercise both first-time setup and the per-device cached path.
    _run_and_check(kernel, 100)
    _run_and_check(kernel, 100)


if __name__ == "__main__":
    tilelang.testing.main()
