from __future__ import annotations

from tvm.target import Target

from tilelang.backend.execution_backend import ExecutionBackendSpec, register_execution_backend


def _is_ppu_target(target: Target) -> bool:
    return target.kind.name == "ppu"


register_execution_backend(
    "ppu",
    ExecutionBackendSpec(
        "tvm_ffi",
        supports_target=_is_ppu_target,
        enable_host_codegen=True,
        enable_device_compile=True,
    ),
)
register_execution_backend(
    "ppu",
    ExecutionBackendSpec("cython", supports_target=_is_ppu_target),
)
