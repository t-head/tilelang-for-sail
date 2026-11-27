"""PPU copy of the region-atomic diagnostics added by upstream #3465.

The PPU ``tl.atomic_add`` lowering keeps a fork-local SIMT loop builder, so the
shared region validation (``backend/common/op/atomic_simt.h``) is not inherited
through ``atomic_reduce.h`` the way ``tl.atomic_max`` / ``tl.atomic_min`` are.
Mismatched src/dst regions used to lower silently (different extents) or fail
with a bare ``IndexError`` (different ranks).
"""

import pytest
import torch

import tilelang
import tilelang.ppu.language as T
import tilelang.testing
from tvm.target import Target


PPU_TARGET = "ppu"
OPERATIONS = ["add", "max", "min"]


def region_atomic_program(operation, source_shape, destination_shape):
    @T.prim_func
    def main(
        A: T.Tensor(source_shape, "float32"),
        B: T.Tensor(destination_shape, "float32"),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared(source_shape, "float32")
            B_shared = T.alloc_shared(destination_shape, "float32")
            T.clear(A_shared)
            T.clear(B_shared)
            source_region = tuple(slice(None) for _ in source_shape)
            destination_region = tuple(slice(None) for _ in destination_shape)
            getattr(T, f"atomic_{operation}")(B_shared[destination_region], A_shared[source_region])

    return main


def lower_ppu(program):
    with Target(PPU_TARGET):
        return tilelang.lower(program, target=PPU_TARGET, enable_device_compile=False).kernel_source


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(
    ("source_shape", "destination_shape"),
    [
        ((4, 8, 16), (8, 16)),
        ((2, 4, 8, 16), (8, 16)),
        ((8, 16), (4, 8, 16)),
    ],
)
def test_region_atomic_rank_mismatch_has_diagnostic(operation, source_shape, destination_shape):
    with pytest.raises(ValueError, match="Cannot map atomic region") as exc_info:
        lower_ppu(region_atomic_program(operation, source_shape, destination_shape))
    message = str(exc_info.value)
    for context in (f"tl.atomic_{operation}_elem_op", "src=A_shared", "dst=B_shared", "src_range=", "dst_range="):
        assert context in message


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(
    ("source_shape", "destination_shape"),
    [
        ((4, 16), (8, 8)),
        ((8, 8), (4, 16)),
        ((8, 4), (8, 8)),
        ((8, 16), (8, 8)),
        ((1, 4, 1, 16), (8, 1, 8)),
        ((8, 1, 16), (1, 8, 8, 1)),
        ((8,), (1,)),
    ],
)
def test_region_atomic_extent_mismatch_has_diagnostic(operation, source_shape, destination_shape):
    with pytest.raises(ValueError, match="Cannot map atomic region") as exc_info:
        lower_ppu(region_atomic_program(operation, source_shape, destination_shape))
    message = str(exc_info.value)
    for context in (f"tl.atomic_{operation}_elem_op", "extent", "src=A_shared", "dst=B_shared", "src_range=", "dst_range="):
        assert context in message


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(
    ("source_shape", "destination_shape"),
    [
        ((8, 16), (8, 16)),
        ((1, 8, 16), (8, 16)),
        ((8, 1, 16), (1, 8, 16, 1)),
        ((1, 1), (1,)),
    ],
)
def test_region_atomic_matching_region_is_accepted(operation, source_shape, destination_shape):
    source = lower_ppu(region_atomic_program(operation, source_shape, destination_shape))
    assert f"Atomic{operation.title()}" in source


def region_atomic_offset_program(operation):
    @T.prim_func
    def main(A: T.Tensor((10, 12), "float32"), B: T.Tensor((12, 16), "float32")):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((10, 12), "float32")
            T.copy(A, A_shared)
            getattr(T, f"atomic_{operation}")(B[3:11, 4:12], A_shared[1:9, 2:10])

    return main


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("operation", OPERATIONS)
def test_region_atomic_matching_extents_with_offsets(operation):
    kernel = tilelang.compile(region_atomic_offset_program(operation), target=PPU_TARGET, execution_backend="tvm_ffi")
    source = torch.arange(120, dtype=torch.float32, device="cuda").reshape(10, 12)
    destination = torch.full((12, 16), 200.0 if operation == "min" else 1.0, device="cuda")
    expected = destination.clone()
    expected[3:11, 4:12] = source[1:9, 2:10] + (1.0 if operation == "add" else 0.0)

    kernel(source, destination)
    torch.testing.assert_close(destination, expected, rtol=0, atol=0)


def region_atomic_scalar_source_program(operation, source_form, source_index=1):
    @T.prim_func
    def main(A: T.Tensor((4,), "float32"), B: T.Tensor((20,), "float32")):
        with T.Kernel(1, threads=128):
            atomic_op = getattr(T, f"atomic_{operation}")
            if source_form == "load":
                atomic_op(B[2:18], A[source_index])
            elif source_form == "region":
                atomic_op(B[2:18], A[source_index : source_index + 1])
            else:
                atomic_op(B[2:18], A[source_index] + 1.0)

    return main


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("source_form", ["load", "region", "expression"])
@pytest.mark.parametrize("source_index", [0, 1])
def test_region_atomic_scalar_source(operation, source_form, source_index):
    kernel = tilelang.compile(
        region_atomic_scalar_source_program(operation, source_form, source_index), target=PPU_TARGET, execution_backend="tvm_ffi"
    )
    source = torch.tensor([99.0, 3.0, -77.0, 55.0], device="cuda")
    value = (99.0 if source_index == 0 else 3.0) + (1.0 if source_form == "expression" else 0.0)
    destination = torch.full((20,), value + 4.0 if operation == "min" else 1.0, device="cuda")
    expected = destination.clone()
    expected[2:18] = 1.0 + value if operation == "add" else value

    kernel(source, destination)
    torch.testing.assert_close(destination, expected, rtol=0, atol=0)
