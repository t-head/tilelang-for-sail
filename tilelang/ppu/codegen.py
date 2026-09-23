from __future__ import annotations

from tvm.target import Target

from tilelang.backend.device_codegen import DeviceCodegen, global_func_device_codegen, register_device_codegen


def _is_ppu_target(target: Target) -> bool:
    return target.kind.name == "ppu"


register_device_codegen(
    "ppu",
    DeviceCodegen(
        "ppu",
        build=global_func_device_codegen("target.build.tilelang_ppu"),
        build_without_compile=global_func_device_codegen("target.build.tilelang_ppu_without_compile"),
        supports_target=_is_ppu_target,
    ),
)
