"""PPU-specific floating-point arithmetic intrinsics."""

from __future__ import annotations

from tvm import DataType, tirx
from tvm.tirx import PrimExpr


def clamp_symmetric(x: PrimExpr, limit: PrimExpr) -> PrimExpr:
    """Symmetric clipping with NaN propagation: ``clamp(x, -limit, limit)``.

    Mirrors the CUDA ``T.clamp_symmetric`` semantics: for non-NaN operands,
    requires ``limit > 0`` and returns ``copysign(min(abs(x), limit), x)``,
    preserving the sign of zero. PPU has no ``min.NaN.xorsign.abs``
    instruction, so the device helper composes the operation from the
    NaN-propagating packed min/max primitives instead of a single PTX
    instruction; no SM-version gate applies.
    Supports float16, bfloat16, and float32 with matching operand dtypes.
    Python numeric limits are converted to the input's element dtype;
    a scalar limit is broadcast to vector inputs.
    The caller supplies a positive limit for clipping semantics. No runtime
    check is emitted. Even-width float16/bfloat16 vectors use packed x2
    helpers, one per pair; float32 vectors use one call per lane.

    If either operand is NaN, the result is a NaN. Nonpositive numeric limits
    are outside this API's contract.
    Subnormals are preserved. Each operand is evaluated once.
    """
    x = tirx.convert(x)
    limit = tirx.const(limit, DataType(x.dtype).with_lanes(1)) if isinstance(limit, (int, float)) else tirx.convert(limit)
    x_dtype, limit_dtype = DataType(x.dtype), DataType(limit.dtype)
    element_dtype = x_dtype.with_lanes(1)
    if element_dtype not in (DataType("float16"), DataType("bfloat16"), DataType("float32")):
        raise TypeError("T.clamp_symmetric requires float16, bfloat16, or float32 operands")
    if limit_dtype.with_lanes(1) != element_dtype:
        raise TypeError("T.clamp_symmetric requires matching operand dtypes")
    if limit_dtype.lanes == 1 and x_dtype.lanes != 1:
        limit = tirx.Broadcast(limit, x_dtype.lanes)
    elif limit_dtype.lanes != x_dtype.lanes:
        raise ValueError("T.clamp_symmetric requires matching lanes or a scalar limit")
    return tirx.call_intrin(x.dtype, tirx.op.Op.get("tl.clamp_symmetric"), x, limit)


__all__ = ["clamp_symmetric"]
