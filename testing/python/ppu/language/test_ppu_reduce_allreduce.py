"""Cross-thread AllReduce on the PPU backend (tl.reduce / tl.finalize_reducer).

Follows the CUDA-side structure of #3692 (full-warp fast path through
tl::warp_reduce) and #3715 (cross-warp levels collapsed into a single
shared-memory round) in src/tl_templates/ppu/reduce.h.

Differences from CUDA that are deliberate:
- No f32x2 packed reduction is added on top of what the PPU backend already
  had; float/float16/bfloat16 keep the shuffle chain.
- The integer reducers go through `__reduce_*_sync`, which hgcc emits as the
  hardware warp-reduce (`v.red.add.u32`, `v.red.max.i32`). The ISA has no
  float or 16-bit warp-reduce, so only integers get this path.
- A thread range that does not start at thread 0 is not supported yet, see
  test_allreduce_layout_runtime_thread_offset.
"""

import re

import pytest
import torch

import tilelang
import tilelang.language as language
import tilelang.testing
from tvm.target import Target

PPU_TARGET = "ppu"


_REDUCERS = {
    "sum": language.reduce_sum,
    "max": language.reduce_max,
    "min": language.reduce_min,
    "bitand": language.reduce_bitand,
    "bitor": language.reduce_bitor,
    "bitxor": language.reduce_bitxor,
}


def _make_kernel(operation, dtype, width, *, scale=1, batch=1, clear=True, thread_offset=0):
    rows = 4
    threads = max(32, width * scale + thread_offset)
    reduce_fn = _REDUCERS[operation]

    @language.prim_func
    def kernel(
        inputs: language.Tensor((rows, width, scale), dtype),
        outputs: language.Tensor((rows, scale), dtype),
    ):
        with language.Kernel(1, threads=threads):
            source = language.alloc_fragment((rows, width, scale), dtype)
            result = language.alloc_fragment((rows, scale), dtype)
            language.annotate_layout(
                {
                    source: language.Fragment(
                        source.shape,
                        forward_fn=lambda row, column, subgroup: (thread_offset + column * scale + subgroup, row),
                    ),
                    result: language.Fragment(
                        result.shape,
                        forward_fn=lambda row, subgroup, replica: (thread_offset + replica * scale + subgroup, row),
                        replicate=width,
                    ),
                }
            )
            language.copy(inputs, source)
            if not clear:
                language.fill(result, 1)
            reduce_fn(source, result, dim=1, clear=clear, batch=batch)
            language.copy(result, outputs)

    return kernel


def _reference(inputs, operation, clear):
    values = inputs.cpu().to(torch.int64)
    if operation == "sum":
        result = values.sum(dim=1)
    elif operation == "max":
        result = values.amax(dim=1)
    elif operation == "min":
        result = values.amin(dim=1)
    else:
        combine = {
            "bitand": torch.bitwise_and,
            "bitor": torch.bitwise_or,
            "bitxor": torch.bitwise_xor,
        }[operation]
        result = values[:, 0].clone()
        for column in range(1, values.shape[1]):
            result = combine(result, values[:, column])
    if not clear:
        initial = torch.ones_like(result)
        combine = {
            "sum": torch.add,
            "max": torch.maximum,
            "min": torch.minimum,
            "bitand": torch.bitwise_and,
            "bitor": torch.bitwise_or,
            "bitxor": torch.bitwise_xor,
        }[operation]
        result = combine(result, initial)
    return result.to(inputs.dtype).to(torch.int64)


def _compile(program):
    return tilelang.compile(program, out_idx=-1, target=PPU_TARGET, execution_backend="tvm_ffi")


@tilelang.testing.requires_ppu
@pytest.mark.parametrize(
    ("operation", "dtype"),
    [(operation, dtype) for operation in _REDUCERS for dtype in ("int32", "uint32")] + [("sum", "int16"), ("max", "uint16")],
)
@pytest.mark.parametrize(
    ("width", "batch", "clear"),
    [(32, 1, True), (128, 2, True), (64, 1, False), (64, 2, False)],
)
def test_allreduce_integer_runtime(operation, dtype, width, batch, clear):
    generator = torch.Generator().manual_seed(42)
    dtype_info = torch.iinfo(getattr(torch, dtype))
    lower, upper = dtype_info.min, dtype_info.max + 1
    inputs = torch.randint(lower, upper, (4, width, 1), generator=generator, dtype=torch.int64)
    inputs[:, 0, 0] = lower
    inputs[:, 1, 0] = upper - 1
    inputs = inputs.to(getattr(torch, dtype)).cuda()
    actual = _compile(_make_kernel(operation, dtype, width, batch=batch, clear=clear))(inputs)
    torch.testing.assert_close(actual.cpu().to(torch.int64), _reference(inputs, operation, clear), rtol=0, atol=0)


@tilelang.testing.requires_ppu
@pytest.mark.parametrize(
    ("width", "scale", "thread_offset"),
    [(16, 1, 0), (32, 2, 0), (256, 1, 0), (128, 2, 0), (64, 4, 0), (32, 8, 0), (16, 16, 0), (8, 32, 0)],
)
def test_allreduce_layout_runtime(width, scale, thread_offset):
    inputs = torch.arange(4 * width * scale, dtype=torch.int32).reshape(4, width, scale).cuda()
    kernel = _make_kernel("sum", "int32", width, scale=scale, thread_offset=thread_offset)
    actual = _compile(kernel)(inputs)
    torch.testing.assert_close(actual.cpu().to(torch.int64), _reference(inputs, "sum", True), rtol=0, atol=0)


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("operation", ["max", "min"])
@pytest.mark.parametrize(
    ("dtype", "width"),
    [("float32", 32), ("float32", 128), ("float16", 32), ("bfloat16", 128)],
)
def test_allreduce_float_runtime(operation, dtype, width):
    # float/float16/bfloat16 always take the shuffle chain, with and without a
    # cross-warp shared-memory round.
    torch_dtype = getattr(torch, dtype)
    generator = torch.Generator().manual_seed(7)
    inputs = torch.randn((4, width, 1), generator=generator, dtype=torch.float32).to(dtype=torch_dtype).cuda()
    kernel = _make_kernel(operation, dtype, width)
    actual = _compile(kernel)(inputs).cpu()
    values = inputs.cpu().to(torch.float32)
    expected = values.amax(dim=1) if operation == "max" else values.amin(dim=1)
    torch.testing.assert_close(actual.to(torch.float32), expected, rtol=0, atol=0)


@tilelang.testing.requires_ppu
def test_allreduce_emits_warp_aligned_groups():
    # A cross-warp reduce is a single AllReduce over the whole block (one
    # shared-memory round inside); a 32-thread reduce needs no barrier and no
    # workspace.
    with Target(PPU_TARGET):
        wide = tilelang.lower(_make_kernel("max", "float32", 64, batch=2), target=PPU_TARGET, enable_device_compile=False).kernel_source
        narrow = tilelang.lower(_make_kernel("sum", "int32", 32), target=PPU_TARGET, enable_device_compile=False).kernel_source
    # float32 rows are packed two per lane on the PPU backend (MaxOp_f32x2).
    assert re.search(r"tl::AllReduce<tl::MaxOp(_f32x2)?, 64, 1, 0, tl::SyncThreadsBarrier, \d+, \d+>::run_batch", wide)
    assert re.search(r"tl::AllReduce<tl::SumOp, 32, 1, 0>::run\(", narrow)


# Kept last: the illegal memory access poisons the device context for every
# kernel launched afterwards in the same process.
@tilelang.testing.requires_ppu
@pytest.mark.xfail(
    strict=True,
    reason="Pre-existing PPU limitation, unchanged by the #3692/#3715 port: a thread "
    "range that does not start at thread 0 is emitted as AllReduce<..., 0> (the "
    "resolved thread range needs a named barrier, which only CUDA sm80+ uses here) "
    "with the offset folded into `if (32 <= threadIdx.x)`, and the kernel faults "
    "with an illegal memory access.",
)
def test_allreduce_layout_runtime_thread_offset():
    inputs = torch.arange(4 * 64, dtype=torch.int32).reshape(4, 64, 1).cuda()
    kernel = _make_kernel("sum", "int32", 64, thread_offset=32)
    actual = _compile(kernel)(inputs)
    torch.testing.assert_close(actual.cpu().to(torch.int64), _reference(inputs, "sum", True), rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
