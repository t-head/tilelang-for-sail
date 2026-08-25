import tilelang.testing
import ppu_example_gemm
import ppu_example_gemm_intrinsics


def regression_example_gemm_intrinsics():
    tilelang.testing.process_func(ppu_example_gemm_intrinsics.run_regression_perf, M=1024, N=1024, K=1024)


def regression_example_gemm():
    tilelang.testing.process_func(ppu_example_gemm.run_regression_perf)


if __name__ == "__main__":
    tilelang.testing.regression()
