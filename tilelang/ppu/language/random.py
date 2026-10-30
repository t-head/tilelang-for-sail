"""PPU random-number language operators."""

from __future__ import annotations

from tvm import tirx

from tilelang.language import common as _common

__all__ = ["rng_init", "rng_rand", "rng_rand_float"]


def rng_init(seed, seq=None, off=0, generator="acrandStatePhilox4_32_10_t"):
    if generator not in {"acrandStateMRG32k3a_t", "acrandStatePhilox4_32_10_t", "acrandStateXORWOW_t"}:
        raise ValueError(f"Unsupported PPU random generator: {generator}")
    if seq is None:
        seq = _common.get_thread_binding() + _common.get_block_binding() * _common.get_thread_extent()
    return tirx.call_intrin("void", tirx.op.Op.get("tl.rng_init"), tirx.convert(seed), tirx.convert(seq), tirx.convert(off), generator)


def rng_rand():
    return tirx.call_intrin("uint32", tirx.op.Op.get("tl.rng_rand"))


def rng_rand_float(bit=32, dist="uniform"):
    if bit not in (32, 64) or dist not in ("uniform", "normal"):
        raise ValueError(f"Unsupported PPU random distribution: bit={bit}, dist={dist}")
    return tirx.call_intrin(f"float{bit}", tirx.op.Op.get("tl.rng_rand_float"), dist)
