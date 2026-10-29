"""PPU-specific low-level TIR language operators."""

from __future__ import annotations

import tvm.tirx.op as _tvm_op
from tvm import tirx

from tilelang.language.tir.ir import _dtype_forward

__all__ = [
    "ppu_aiu_load",
    "ptx_ldmatrix",
    "ptx_mma",
    "ptx_mma_scaled",
    "ptx_mma_sp",
    "tix_ldmatrix_swzl",
]

# Unchanged upstream PTX operators are rebound directly in the PPU dialect.
ptx_mma = _dtype_forward(_tvm_op.ptx_mma)


def ptx_ldmatrix(trans, num, src_access_ptr, dst_access_ptr):
    """TileLang intrinsic for ptx load matrix from shared memory

    Uses `tl.ptx_ldmatrix` which expects access pointers created via
    `T.access_ptr` (i.e. `tl.access_ptr` wrapping a `BufferLoad`).

    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix

    Parameters
    ----------
    trans : bool
        The matrix is loaded in column-major format.

    num : IntImm
        The number of matrices (2 or 4).

    src_access_ptr : PrimExpr
        A `tl.access_ptr` pointing to the source (shared memory) buffer.

    dst_access_ptr : PrimExpr
        A `tl.access_ptr` pointing to the destination (local/register) buffer.

    Returns
    -------
    call : PrimExpr
        The call expression (handle-typed).
    """
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.ptx_ldmatrix"),
        trans,
        num,
        src_access_ptr,
        dst_access_ptr,
    )


@_dtype_forward
def ptx_mma_sp(
    dtype,
    shape,
    A_layout,
    B_layout,
    A_dtype,
    B_dtype,
    C_dtype,
    multiplicand_a,
    a_index,
    multiplicand_b,
    b_index,
    accumulator,
    c_index,
    metadata,
    meta_index,
    sparse_selector,
    saturate,
):
    """TVM intrinsic for sparse tensor core ptx instructions
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-sparse-mma

    Parameters
    ----------
    dtype : str
        The data type of the result.

    shape : str
        The shape of mma fragment.

    A_layout : Literal["row", "col"]
        The layout of multiplicand fragment A.

    B_layout : Literal["row", "col"]
        The layout of multiplicand fragment B.

    A_dtype : str
        The data type of multiplicand fragment A.

    B_dtype : str
        The data type of multiplicand fragment B.

    C_dtype : str
        The data type of accumulator fragment C.

    multiplicand_a : Var
        The multiplicand fragment A variable.

    a_index : Expr
        The index of multiplicand fragment A.

    multiplicand_b : Var
        The multiplicand fragment B variable.

    b_index : Expr
        The index of multiplicand fragment B.

    accumulator : Var
        The accumulator fragment C variable.

    c_index : Expr
        The index of accumulator fragment C.

    metadata : Expr
        The metadata of operand.

    meta_index : Expr
        The metadata index of operand.

    sparse_selector : Expr
        The sparse selector indicating the thread that stores the metadata.

    saturate : bool
        The optional saturation at the output.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.ptx_mma_sp(
        dtype,
        shape,
        A_layout,
        B_layout,
        A_dtype,
        B_dtype,
        C_dtype,
        multiplicand_a,
        a_index,
        multiplicand_b,
        b_index,
        accumulator,
        c_index,
        metadata,
        meta_index,
        sparse_selector,
        saturate,
    )


def tix_ldmatrix_swzl(trans, num, src_access_ptr, dst_access_ptr, swzl_mode, trans_block=False):
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tix_ldmatrix_swzl"),
        trans,
        num,
        src_access_ptr,
        dst_access_ptr,
        swzl_mode,
        trans_block,
    )


def ppu_aiu_load(smem_ptr, gmem_ptr, dim_c, dim_w, cube_c, cube_w, stride_w_bytes, start_c, start_w, swzl_mode, is_swizzle=1):
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.ppu_aiu_load"),
        smem_ptr,
        gmem_ptr,
        dim_c,
        dim_w,
        cube_c,
        cube_w,
        stride_w_bytes,
        start_c,
        start_w,
        swzl_mode,
        is_swizzle,
        annotations={"no_implicit_async_commit_wait": tirx.const(1, "int32")},
    )


@_dtype_forward
def ptx_mma_scaled(
    dtype,
    shape,
    A_layout,
    B_layout,
    A_dtype,
    B_dtype,
    C_dtype,
    multiplicand_a,
    a_index,
    multiplicand_b,
    b_index,
    accumulator,
    c_index,
    saturate,
    scale_a,
    scale_b,
    scale_a_selector,
    scale_b_selector,
):
    return tirx.call_intrin(
        dtype,
        "tirx.ptx_mma",
        shape,
        A_layout,
        B_layout,
        A_dtype,
        B_dtype,
        C_dtype,
        multiplicand_a,
        a_index,
        multiplicand_b,
        b_index,
        accumulator,
        c_index,
        saturate,
        scale_a,
        scale_b,
        scale_a_selector,
        scale_b_selector,
    )
