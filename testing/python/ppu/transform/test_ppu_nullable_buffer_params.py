# PPU variant of testing/python/transform/test_nullable_buffer_params.py:
# nullable buffer params + T.print under the PPU target.
import torch
import tilelang
import tilelang.testing
from tilelang.ppu import language as T


@tilelang.testing.requires_ppu
def test_nullable_shared_shape():
    """Test that buffers sharing a shape variable can be nullable."""

    @tilelang.jit(target="ppu")
    def get_kernel():
        m = T.dynamic("m")

        @T.prim_func
        def test_kernel(
            a: T.Tensor[(m,), T.int32],
            b: T.Tensor[(m,), T.int32],
            c: T.Tensor[(m,), T.int32],
        ):
            with T.Kernel(1, threads=64):
                tx = T.get_thread_binding()
                if tx == 0:
                    T.print(m)

        return test_kernel

    m = 200
    kernel = get_kernel()

    tensor_a = torch.randn((m,), device="cuda", dtype=torch.float32).to(torch.int32)
    tensor_b = torch.randn((m,), device="cuda", dtype=torch.float32).to(torch.int32)
    tensor_c = torch.randn((m,), device="cuda", dtype=torch.float32).to(torch.int32)

    kernel(tensor_a, tensor_b, tensor_c)
    kernel(tensor_a, None, None)
    kernel(None, tensor_b, None)
    kernel(None, None, tensor_c)
    kernel(tensor_a, None, tensor_c)

    try:
        kernel(None, None, None)
        raise AssertionError("all-None inputs should have raised an error")
    except RuntimeError as e:
        assert "at least one non-null buffer" in str(e), f"wrong error message: {e}"


@tilelang.testing.requires_ppu
def test_nullable_single_source_shape():
    """A single buffer with a symbolic shape var must be non-null (no segfault
    when binding m from x.shape[0] with x == None)."""

    @tilelang.jit(target="ppu")
    def get_kernel():
        m = T.dynamic("m")

        @T.prim_func
        def sample_kernel(x: T.Tensor[(m,), T.int32]):
            with T.Kernel(1, threads=1):
                tx = T.get_thread_binding()
                if tx == 0:
                    T.print(m)

        return sample_kernel

    m = 16
    kernel = get_kernel()

    x = torch.randn((m,), device="cuda", dtype=torch.float32).to(torch.int32)
    kernel(x)
    kernel(None)


@tilelang.testing.requires_ppu
def test_nullable_shared_shape_with_no_source_buffers_but_other_tensor_present():
    """Unused buffers sharing a symbolic shape var can both be None when another
    (non-null) tensor argument exists."""

    @tilelang.jit(execution_backend="tvm_ffi", target="ppu")
    def get_kernel():
        m = T.dynamic("m")

        @T.prim_func
        def test_kernel(
            a: T.Tensor[(m,), T.float16],
            b: T.Tensor[(m,), T.float16],
            out: T.Tensor[(1,), T.float16],
        ):
            with T.Kernel(1, threads=32):
                fragment = T.alloc_fragment((1,), T.float32)
                T.copy(out[0], fragment)
                T.copy(fragment, out[0])

        return test_kernel

    kernel = get_kernel()

    out = torch.randn((1,), device="cuda", dtype=torch.float16)
    out_ref = out.clone()

    kernel(None, None, out)
    torch.testing.assert_close(out, out_ref)


if __name__ == "__main__":
    tilelang.testing.main()
