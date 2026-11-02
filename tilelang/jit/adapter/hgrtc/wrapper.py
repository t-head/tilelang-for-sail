"""Hgrtc Source Wrapper for TileLang.

Generates Python runtime code for launching PPU kernels compiled via
hgcc (the Hgrtc runtime-compilation backend).

Why this exists:
- Hgrtc compiles kernels at runtime, needs Python launch code (not C++)
- The generated launcher talks to the HGGC driver (libhggc.so) through
  ctypes, so no cuda-python or any CUDA dependency is involved

Key design:
- Walk the host IR to extract kernel call sites, then generate one
  pure-Python ``call()`` dispatcher that launches each kernel with
  hgLaunchKernel
- TMA descriptors, the L2 persisting-cache window and PDL launch
  attributes are CUDA-only host-side APIs; the PPU launcher never emits
  them
"""

from __future__ import annotations
from typing import Any, ClassVar

from tvm import IRModule
from tvm.target import Target
from tvm.tirx.stmt_functor import post_order_visit

from tilelang import tvm as tvm
from tilelang.jit.adapter.wrapper import TLCUDASourceWrapper
from tilelang.jit.adapter.utils import match_declare_kernel, pythonic_expr, parse_function_call_args

PREDEF_HOST_FUNC_PY = '''
import ctypes
import os


def _load_hg_driver():
    """Load the HGGC driver library and perform one-time initialization.

    Unlike CUDA (cudaSetDevice implicitly cuInit's), the HGGC driver must be
    initialized explicitly with hgInit(0) before any module/launch API call.
    """
    _sdk = os.environ.get("PPU_SDK", "")
    _candidates = [
        os.path.join(_sdk, "lib", "libhggc.so"),
        os.path.join(_sdk, "targets", "x86_64-linux", "lib", "libhggc.so"),
        "libhggc.so",
    ]
    _lib = None
    for _cand in _candidates:
        try:
            _lib = ctypes.CDLL(_cand)
            break
        except OSError:
            continue
    if _lib is None:
        raise RuntimeError(
            "Failed to load libhggc.so; please source the PPU SDK envsetup.sh "
            f"(PPU_SDK={{_sdk!r}})"
        )
    _lib.hgInit.argtypes = [ctypes.c_uint]
    _lib.hgInit.restype = ctypes.c_int
    _res = _lib.hgInit(0)
    if _res != 0:
        raise RuntimeError(f"hgInit(0) failed: HGresult={{_res}}")
    _lib.hgFuncSetAttribute.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    _lib.hgFuncSetAttribute.restype = ctypes.c_int
    _lib.hgLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    _lib.hgLaunchKernel.restype = ctypes.c_int
    return _lib


_hg = _load_hg_driver()

_function_names = {}

def call({}):
    {}
'''

# HG_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES == 8 (hggc_v2/hggc.h)
SET_SMEM_ATTR_PY = """
    _ppu_res = _hg.hgFuncSetAttribute(kernels["{0}"], 8, {1})
    if _ppu_res != 0:
        raise RuntimeError(f"Failed to set max dynamic shared memory size to {1} for kernel {0}: HGresult={{_ppu_res}}")
"""

KERNEL_LAUNCH_FUNC_PY = """
    _ppu_args_{0} = [{1}]
    _ppu_params_{0} = (ctypes.c_void_p * len(_ppu_args_{0}))(*[ctypes.addressof(_a) for _a in _ppu_args_{0}])
    _ppu_res = _hg.hgLaunchKernel(kernels["{0}"], {2}, {3}, {4}, {5}, {6}, {7}, {8}, ctypes.c_void_p(stream), _ppu_params_{0}, None)
    if _ppu_res != 0:
        raise RuntimeError(f"Failed to launch kernel {0}: HGresult={{_ppu_res}}")
"""


class TLHgrtcSourceWrapper(TLCUDASourceWrapper):
    """Hgrtc backend wrapper: generates Python kernel launch code for PPU.

    Same architecture as the NVRTC wrapper (a pure-Python launcher driving
    a vendor driver API), but the generated launcher talks to the HGGC
    driver (libhggc.so) through ctypes instead of cuda-python: kernels
    compiled to .hgbin by hgcc are launched with hgLaunchKernel.

    Data flow:
        IRModule → collect kernel metadata → generate Python code →
        executable function
    """

    _TYPE_MAP: ClassVar[dict[str, str]] = {
        "float32": "ctypes.c_float",
        "float16": "ctypes.c_uint16",
        "bfloat16": "ctypes.c_uint16",
        "float8_e4m3": "ctypes.c_uint8",
        "float8_e4m3fn": "ctypes.c_uint8",
        "float8_e5m2": "ctypes.c_uint8",
        "float64": "ctypes.c_double",
        "int64": "ctypes.c_int64",
        "uint64": "ctypes.c_uint64",
        "int32": "ctypes.c_int32",
        "uint32": "ctypes.c_uint32",
        "bool": "ctypes.c_bool",
        "int8": "ctypes.c_int8",
        "uint8": "ctypes.c_uint8",
        "int16": "ctypes.c_int16",
        "uint16": "ctypes.c_uint16",
        "uchar": "ctypes.c_uint8",
    }

    _generated_host_func: str | None = None

    def __init__(
        self,
        scheduled_ir_module: IRModule,
        source: str,
        target: Target,
        device_mod: IRModule | None = None,
        host_mod: IRModule | None = None,
        pass_configs: dict[str, Any] | None = None,
    ):
        """Initialize Hgrtc wrapper with compiled IR modules.

        Args:
            scheduled_ir_module: TVM IR after scheduling passes
            source: Generated PPU device source code
            target: Compilation target (must be PPU)
            device_mod: Device-side IR module (kernel functions)
            host_mod: Host-side IR module (launch logic)
            pass_configs: Optional compiler pass configurations
        """
        super().__init__(scheduled_ir_module, source, target, device_mod, host_mod, pass_configs)

    @property
    def host_func(self):
        """Override parent's host_func to return generated Python code."""
        if self._generated_host_func is not None:
            return self._generated_host_func
        return super().host_func

    @host_func.setter
    def host_func(self, value):
        """Allow setting generated host function code."""
        self._generated_host_func = value

    def parse_source_information(self):
        super().parse_source_information()
        # cuTensorMapEncode* (TMA), the L2 persisting-cache window and PDL
        # launch attributes are CUDA-only host-side APIs; the PPU Python
        # launcher does not support them.
        self.tma_descriptor_args = None
        self.l2_persistent_map = {}
        self.pdl_sync_map = {}

    def _pythonic_expr(self, expr: tvm.tirx.PrimExpr) -> str:
        """Convert TVM expression to Python string, ignoring casts.

        Casts are noise in generated Python code - Python is dynamically typed.
        """
        return pythonic_expr(expr, self._TYPE_MAP, ignore_cast=True, floor_div_op="//")

    def create_dispatch_func(self, code, function_informations):
        """Generate Python dispatch function that launches PPU kernels.

        Args:
            code: Device source containing kernel declarations
            function_informations: Dict mapping kernel names to metadata
                (grid/block dims, params, shared memory size)

        Returns:
            Python source code defining a call() function that launches
            each kernel with hgLaunchKernel
        """
        # Extract the set of dynamic symbolic names used in the primary function
        dynamic_symbolic_set = self.get_dynamic_symbolic_set(self.prim_func)

        function_args = [{"name": "kernels", "type": "dict"}]
        # Collect function arguments based on primary function's parameters and buffer mappings
        for param in self.prim_func.params:
            if param in self.prim_func.buffer_map:
                buffer = self.prim_func.buffer_map[param]
                function_args.append(
                    {
                        "name": buffer.data.name,
                        "type": "ctypes.c_void_p",
                    }
                )
            elif isinstance(param, tvm.tirx.Var):
                function_args.append({"name": param.name, "type": self._lookup_type(param.dtype)})
            else:
                raise ValueError(f"Parameter {param} is not in the buffer map of the primary function.")
        # Add dynamic symbols as integer arguments
        for dyn_sym, dyn_sym_dtype in dynamic_symbolic_set:
            if dyn_sym not in [arg["name"] for arg in function_args]:
                function_args.append({"name": dyn_sym, "type": self._lookup_type(dyn_sym_dtype)})

        function_args.append(self.get_stream_type())

        # Format the function arguments for declaration
        def_args = ", ".join([f"{arg['name']}" for arg in function_args])

        # Collect kernel launch metadata, then generate the launch code
        kernel_launch_code = """"""
        for function_name, function_info in function_informations.items():
            block_info = function_info["block_info"]
            grid_info = function_info["grid_info"]
            dynamic_smem_buf = function_info["dynamic_smem_buf"]
            function_params = function_info["function_params"]

            # Find the location of the global kernel function in the code
            index = match_declare_kernel(code, function_name + "(")

            # Analyze the function declaration to prepare for argument extraction
            declaration = code[index:].split(";")[0]

            # Materialize pointer arguments from the tensors' data_ptr()
            def transform_hgrtc_arg(name: str, arg_type: str):
                if arg_type == "ctypes.c_void_p":
                    return (f"{name}.data_ptr()", arg_type)
                return (name, arg_type)

            call_args = parse_function_call_args(declaration, function_args, function_params, transform_arg=transform_hgrtc_arg)

            smem_str = 0 if dynamic_smem_buf is None else dynamic_smem_buf

            kernel_launch_code += self._render_kernel_launch(
                function_name,
                (
                    self._pythonic_expr(grid_info[0]),
                    self._pythonic_expr(grid_info[1]),
                    self._pythonic_expr(grid_info[2]),
                ),
                (
                    self._pythonic_expr(block_info[0]),
                    self._pythonic_expr(block_info[1]),
                    self._pythonic_expr(block_info[2]),
                ),
                smem_str,
                call_args,
            )

        host_func = PREDEF_HOST_FUNC_PY.format(repr(list(function_informations.keys())), def_args, kernel_launch_code)
        return host_func

    def _render_kernel_launch(
        self,
        function_name: str,
        grid_strs: tuple[str, str, str],
        block_strs: tuple[str, str, str],
        smem_str,
        call_args: list,
    ) -> str:
        """Render one kernel launch inside the generated Python dispatcher.

        The HGGC driver takes kernel parameters as an array of pointers to
        the actual argument values (void**), so each argument is materialized
        as a ctypes object whose address goes into the params array.
        """
        constructions = []
        for arg_name, arg_type in call_args:
            if arg_type == "ctypes.c_void_p":
                constructions.append(f"ctypes.c_void_p({arg_name})")
            else:
                constructions.append(f"{arg_type}({arg_name})")
        args_str = ", ".join(constructions)

        launch_code = ""
        if smem_str != 0:
            launch_code += SET_SMEM_ATTR_PY.format(function_name, smem_str)
        launch_code += KERNEL_LAUNCH_FUNC_PY.format(
            function_name,
            args_str,
            grid_strs[0],
            grid_strs[1],
            grid_strs[2],
            block_strs[0],
            block_strs[1],
            block_strs[2],
            smem_str,
        )
        return launch_code

    def update_lib_code(self, code: str):
        """Update library code and generate host dispatch function.

        Entry point for code generation. Walks the host IR to extract kernel
        call sites, matches them with device kernels, then generates Python
        dispatch code via create_dispatch_func().

        Args:
            code: Device source code containing compiled kernels

        Returns:
            The same code string (stored in self.lib_code). Side effect:
            sets self.host_func to generated Python dispatcher.
        """
        # Update the library code with the given code string
        self.lib_code = code

        # Organize function information for code generation
        function_informations = {}
        for function_name in self.function_names:
            # Do not update function with dispatch host function
            if (function_name not in self.block_info) or (function_name not in self.grid_info):
                continue

            assert function_name in self.device_mod, f"Function {function_name} not found in device module"
            device_func = self.device_mod[function_name]
            kernel_params_cnt = len(device_func.params)
            function_params: list[str] | None = None

            def visitor(node, fn=function_name, param_cnt=kernel_params_cnt):
                nonlocal function_params
                if isinstance(node, tvm.tirx.Call):
                    if not (hasattr(node, "op") and node.op == tvm.ir.Op.get("tirx.tvm_call_packed")):
                        return
                    args = node.args
                    if not args or args[0] != fn:
                        return
                    if len(args) < 1 + param_cnt:
                        raise AssertionError("tvm_call_packed should have at least 1 argument and match device function parameters")
                    function_params = args[1 : 1 + param_cnt]

            post_order_visit(self.host_func.body, visitor)
            assert function_params is not None, "function_params should not be None"

            function_informations[function_name] = {
                "function_name": function_name,
                "block_info": self.block_info[function_name],
                "grid_info": self.grid_info[function_name],
                "dynamic_smem_buf": self.dynamic_smem_buf[function_name],
                "function_params": function_params,
            }

        # Create the host function wrapper for the PPU kernel
        self.host_func = self.create_dispatch_func(code, function_informations)
        return self.lib_code

    def get_stream_type(self) -> dict[str, str]:
        """Return stream parameter spec for Python signature.

        Hgrtc backend uses raw int for stream handle. Default to 0
        (default stream) for convenience.
        """
        return {"name": "stream=0", "type": "int"}
