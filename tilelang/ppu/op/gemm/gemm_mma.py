from __future__ import annotations

from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.layout import make_ppu_swizzled_layout, make_swizzled_layout
from tilelang.ppu.intrinsics.macro.mma_macro_generator import (
    PPUTensorCoreIntrinEmitter,
)
from tilelang.utils.language import is_shared, is_fragment, is_full_region
from tilelang import tvm as tvm
from tvm.target import Target
from tvm.ir import Range
from tvm import tirx, DataType
from tilelang import language as T
from tilelang.transform.simplify import _Simplify

GEMM_INST_MMA_PPU = "ppu.mma"

# Annotation key: maps source_buffer -> override_buffer for layout operations.
GEMM_BUFFER_LAYOUT_OVERRIDES = "gemm_buffer_layout_overrides"


class PPUGemmMMA(GemmBase):
    intrin_emitter_cls = PPUTensorCoreIntrinEmitter

    def _make_mma_emitter(self, target: Target, thread_nums: int, thread_var: tirx.Var | None = None):
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GEMM_INST_MMA_PPU)
        warp_row_tiles = int(self.M // m_warp)
        warp_col_tiles = int(self.N // n_warp)
        emitter = self.intrin_emitter_cls(
            a_dtype=self.a_dtype,
            b_dtype=self.b_dtype,
            accum_dtype=self.accum_dtype,
            a_transposed=self.trans_A,
            b_transposed=self.trans_B,
            block_row_warps=m_warp,
            block_col_warps=n_warp,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=self.chunk,
            thread_var=thread_var,
        )
        arch = str(target.attrs["arch"]) if target and "arch" in target.attrs else "ppu_10"
        try:
            emitter.ppu_arch = int(arch.split("_")[-1].rstrip("af"))
        except ValueError:
            emitter.ppu_arch = 10
        return emitter

    @staticmethod
    def _has_layout_for_buffer(layout_map, buffer):
        """Check if buffer already has a layout in the map (mirrors C++ HasLayoutForBuffer)."""
        if layout_map is None:
            return False
        for buf in layout_map:
            if buf is buffer:
                return True
            if hasattr(buf, 'data') and hasattr(buffer, 'data') and buf.data.same_as(buffer.data):
                return True
            if hasattr(buf, 'name') and hasattr(buffer, 'name') and str(buf.name) == str(buffer.name):
                return True
        return False

    def _is_chained_rs_gemm(self, ppu_arch: int, layout_map) -> bool:
        """Detect chained RS GEMM condition (A sourced from previous GEMM's C output)."""
        return (
            self.is_gemm_rs()
            and ppu_arch == 10
            and DataType(self.a_dtype).bits == 16
            and DataType(self.b_dtype).bits == 16
            and DataType(self.accum_dtype).bits == 32
            and self._has_layout_for_buffer(layout_map, self.A)
        )

    def infer_layout(self, target: Target, thread_nums: int):
        mma_emitter = self._make_mma_emitter(target, thread_nums)
        if self.is_gemm_ss():
            return {
                self.A: make_swizzled_layout(self.A),
                self.B: make_swizzled_layout(self.B),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
        elif self.is_gemm_sr():
            return {
                self.A: make_swizzled_layout(self.A),
                self.B: mma_emitter.make_mma_load_layout(self.B, matrix="B"),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
        elif self.is_gemm_rs():
            if (
                mma_emitter.ppu_arch == 10
                and DataType(self.a_dtype).bits == 16
                and DataType(self.b_dtype).bits == 16
                and DataType(self.accum_dtype).bits == 32
            ):
                # Always return extra B original-layout buffers for PPU0010 RS GEMM
                extra_layout = make_ppu_swizzled_layout(
                    self.B,
                    k_major=self.trans_B,
                    is_gemm_rs=False,
                )
                b_name = str(self.B.name)
                unique_id = "_" + str(id(self.gemm_node))
                common_buf = tirx.decl_buffer(
                    self.B.shape, self.B.dtype, b_name + "_original_layout")
                unique_buf = tirx.decl_buffer(
                    self.B.shape, self.B.dtype, b_name + unique_id + "_original_layout")
                return {
                    self.A: mma_emitter.make_mma_load_layout(self.A, matrix="A", a_from_gemm_c=True),
                    self.B: make_ppu_swizzled_layout(
                        self.B,
                        k_major=self.trans_B,
                        is_gemm_rs=True,
                    ),
                    self.C: mma_emitter.make_mma_store_layout(self.C),
                    common_buf: extra_layout,
                    unique_buf: extra_layout,
                }
            return {
                self.A: mma_emitter.make_mma_load_layout(self.A, matrix="A"),
                self.B: make_swizzled_layout(self.B),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
        elif self.is_gemm_rr():
            return {
                self.A: mma_emitter.make_mma_load_layout(self.A, matrix="A"),
                self.B: mma_emitter.make_mma_load_layout(self.B, matrix="B"),
                self.C: mma_emitter.make_mma_store_layout(self.C),
            }
        else:
            raise ValueError(f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_var: tirx.Var,
        mbar_phase_expr: tirx.PrimExpr | None = None,
    ):
        thread_nums = thread_bounds.extent
        # Emitter lane/warp math uses zero-based ids within the current thread bounds.
        local_thread_var = thread_var - thread_bounds.min
        # Detect chained RS GEMM for this lower call
        arch = str(target.attrs["arch"]) if target and "arch" in target.attrs else "ppu_10"
        try:
            ppu_arch = int(arch.split("_")[-1].rstrip("af"))
        except ValueError:
            ppu_arch = 10
        a_from_gemm_c = self._is_chained_rs_gemm(ppu_arch, layout_map)
        mma_emitter = self._make_mma_emitter(target, thread_nums, thread_var=local_thread_var)

        a_dtype = self.a_dtype
        b_dtype = self.b_dtype
        warp_rows = mma_emitter.warp_rows
        warp_cols = mma_emitter.warp_cols
        local_size_a = mma_emitter.local_size_a
        local_size_b = mma_emitter.local_size_b
        block_K = mma_emitter.chunk
        micro_size_k = mma_emitter.micro_size_k
        # We use region for memory input to support strided gemm
        # T.gemm(A_shared[0:128, :], B_shared, C_local)
        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion

        A_buf = A_region.buffer
        B_buf = B_region.buffer
        C_buf = C_region.buffer

        clear_accum = self.clear_accum

        assert block_K >= micro_size_k, f"block_K ({block_K}) must be >= micro_size_k ({micro_size_k})"

        assert is_full_region(C_region), "Fragment output C must be a full region"

        if self.is_gemm_ss():

            @T.prim_func
            def _gemm_ssr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Tensor Core mma ops,
                accumulating into C_local.
                """
                A_local = T.alloc_local((warp_rows * local_size_a), a_dtype)
                B_local = T.alloc_local((warp_cols * local_size_b), b_dtype)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, (block_K // micro_size_k)):
                    # Load A into fragment
                    mma_emitter.ldmatrix_a(
                        A_local,
                        A_region,
                        ki,
                    )

                    # Load B into fragment
                    mma_emitter.ldmatrix_b(
                        B_local,
                        B_region,
                        ki,
                    )

                    # Perform Matrix Multiplication
                    mma_emitter.mma(A_local, B_local, C_buf, ki)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            return _Simplify(_gemm_ssr, inline_let=True)
        elif self.is_gemm_sr():
            assert is_full_region(B_region), "Fragment input B must be a full region"

            @T.prim_func
            def _gemm_srr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Tensor Core mma ops,
                accumulating into C_local.
                """
                A_local = T.alloc_local((warp_rows * local_size_a), a_dtype)

                for ki in T.serial(0, (block_K // micro_size_k)):
                    if clear_accum:
                        T.clear(C_buf)
                    # Load A into fragment
                    mma_emitter.ldmatrix_a(
                        A_local,
                        A_region,
                        ki,
                    )

                    # Perform Matrix Multiplication
                    mma_emitter.mma(A_local, B_buf, C_buf, ki)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            # alloc_buffers body
            # insert into parent block
            return _Simplify(_gemm_srr, inline_let=True)
        elif self.is_gemm_rs():
            # PPU: RS with A sourced from a previous GEMM's C fragment needs a
            # dedicated lowering that reads A directly as a fragment buffer
            # and only loads B via ldmatrix.

            assert is_full_region(A_region), "Fragment input A must be a full region"

            @T.prim_func
            def _gemm_rsr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Tensor Core mma ops,
                accumulating into C_local.
                """
                B_local = T.alloc_local((warp_cols * local_size_b), b_dtype)
                if clear_accum:
                    T.clear(C_buf)
                for ki in T.serial(0, (block_K // micro_size_k)):
                    # Load B into fragment
                    mma_emitter.ldmatrix_b(
                        B_local,
                        B_region,
                        ki,
                        a_from_gemm_c=a_from_gemm_c,
                    )

                    # Perform Matrix Multiplication
                    mma_emitter.mma(A_buf, B_local, C_buf, ki)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            func = _Simplify(_gemm_rsr, inline_let=True)
            if a_from_gemm_c:
                # Directly set gemm_buffer_layout_overrides annotation on the SBlock
                override_buf = self._find_override_buffer(layout_map, B_buf)
                if override_buf is not None:
                    overrides = tvm.runtime.convert({B_buf: override_buf})
                    body = func.body  # SBlockRealize
                    assert isinstance(body, tirx.SBlockRealize), (
                        f"Expected SBlockRealize as PrimFunc body, got {type(body)}")
                    block = body.block  # SBlock
                    new_annotations = dict(block.annotations)
                    new_annotations[GEMM_BUFFER_LAYOUT_OVERRIDES] = overrides
                    new_block = tirx.SBlock(
                        iter_vars=block.iter_vars,
                        reads=block.reads,
                        writes=block.writes,
                        name_hint=block.name_hint,
                        body=block.body,
                        init=block.init,
                        alloc_buffers=block.alloc_buffers,
                        match_buffers=block.match_buffers,
                        annotations=new_annotations,
                    )
                    new_body = tirx.SBlockRealize(
                        body.iter_values, body.predicate, new_block)
                    func = func.with_body(new_body)
            return func
        elif self.is_gemm_rr():
            assert is_full_region(A_region), "Fragment input A must be a full region"
            assert is_full_region(B_region), "Fragment input B must be a full region"

            @T.prim_func
            def _gemm_rrr() -> None:
                """
                The inner macro that loads data from shared buffers A_shared and
                B_shared into local fragments, then issues Tensor Core mma ops,
                accumulating into C_local.
                """

                for ki in T.serial(0, (block_K // micro_size_k)):
                    # Perform Matrix Multiplication
                    mma_emitter.mma(A_buf, B_buf, C_buf, ki)

            # Simplify to optimize the index computing
            # Must inline let statements to simplify the analysis
            return _Simplify(_gemm_rrr, inline_let=True)
        else:
            raise ValueError(f"Unsupported gemm combination, A: {self.A.scope()}, B: {self.B.scope()}")

    @staticmethod
    def _find_override_buffer(layout_map, b_buf):
        """Find the B_original_layout buffer in layout_map for scoped buffer overrides."""
        if layout_map is None:
            return None
        original_name = str(b_buf.name) + "_original_layout"
        for buf in layout_map:
            if str(buf.name) == original_name:
                return buf
        return None

    def is_gemm_ss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B)

    def is_gemm_sr(self) -> bool:
        return is_shared(self.A) and is_fragment(self.B)

    def is_gemm_rs(self) -> bool:
        return is_fragment(self.A) and is_shared(self.B)

    def is_gemm_rr(self) -> bool:
        return is_fragment(self.A) and is_fragment(self.B)
