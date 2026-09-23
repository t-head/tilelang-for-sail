import itertools
import pytest
import torch
import re
import ast
import os
import gc
from utils import run_fa_cycle_on_device, run_tilelang_cycle_on_device, format_ratio
from tabulate import tabulate

from kernels.example_mha_fwd_bshd import main as tilelang_mha_fwd_bshd_main
from kernels.example_mha_fwd_bhsd import main as tilelang_mha_fwd_bhsd_main
from kernels.example_mha_bwd_bshd import main as tilelang_mha_bwd_bshd_main
from kernels.example_mha_bwd_bhsd import main as tilelang_mha_bwd_bhsd_main
from kernels.example_gqa_fwd_bshd import main as tilelang_gqa_fwd_bshd_main
from kernels.example_gqa_bwd import main as tilelang_gqa_bwd_bshd_main

try:
    from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func
    from flash_attn.flash_attn_interface import flash_attn_func
    HAS_FLASH = True
except BaseException:
    HAS_FLASH = False
    raise ValueError("No flash-2 found")

from tilelang.profiler import do_bench


def bench_flash_attention(batch, heads, seq_len, head_dim, groups, causal, algo, mode, Q, K, V, dO, device="cuda"):
    """Run flash-2 benchmark and return latency and TFlops"""
    mode = mode.split('_')[0]
    assert mode in ["fwd", "bwd"]
    dtype = torch.float16

    if algo == "mha":
        qkv = torch.stack([Q,K,V], dim=2)
        fn = lambda: flash_attn_qkvpacked_func(qkv, causal=causal)
    elif algo == "gqa":
        head_kv = heads // groups
        fn = lambda: flash_attn_func(Q, K, V, causal=causal)

    if mode == "bwd":
        o = fn()
        fn = lambda: o.backward(dO, retain_graph=True)

    latency_ms = do_bench(fn, warmup=10, rep=100)

    flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * head_dim
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    tflops = total_flops * 1e-12 / (latency_ms * 1e-3)

    if mode == "fwd":
        return latency_ms, tflops, {"O": fn()}
    elif mode == "bwd":
        # run only once for grad accuracy
        Q.grad = None
        K.grad = None
        V.grad = None
        o.backward(dO)
        dQ, dK, dV = Q.grad.clone(), K.grad.clone(), V.grad.clone()
        return latency_ms, tflops, {"O": o, "dQ": dQ, "dK": dK, "dV": dV}


def run_tilelang_benchmark(batch, heads, seq_len, head_dim, groups, causal, algo, mode, Q, K, V, dO):
    """Run tilelang benchmark and extract latency from output"""
    import io
    import sys

    fn = algo + "_" + mode

    try:
        if fn == "mha_fwd_bshd":
            tilelang_output, *perf_results = tilelang_mha_fwd_bshd_main(batch, heads, seq_len, head_dim, causal, True, Q, K, V)
        elif fn == "mha_fwd_bhsd":
            seq_q = seq_kv = seq_len
            tilelang_output, *perf_results = tilelang_mha_fwd_bhsd_main(batch, heads, seq_q, seq_kv, head_dim, causal, True, Q, K, V)
        elif fn == "gqa_fwd_bshd":
            tilelang_output, *perf_results = tilelang_gqa_fwd_bshd_main(batch, heads, seq_len, head_dim, causal, groups, True, Q, K, V)
        elif fn == "mha_bwd_bshd":
            tilelang_output, *perf_results = tilelang_mha_bwd_bshd_main(batch, heads, seq_len, head_dim, causal, True, Q, K, V, dO)
        elif fn == "mha_bwd_bhsd":
            tilelang_output, *perf_results = tilelang_mha_bwd_bhsd_main(batch, heads, seq_len, head_dim, causal, True, Q, K, V, dO)
        elif fn == "gqa_bwd_bshd":
            # let qkv use same head_dim, because Flash-2 only support such config
            d_head_qk = head_dim
            d_head_v = head_dim
            tilelang_output, *perf_results = tilelang_gqa_bwd_bshd_main(batch, heads, seq_len, d_head_qk, d_head_v, groups, causal, True, Q, K, V, dO)
        
        if "bhsd" in mode:
            tilelang_output = tilelang_output.transpose(1, 2)
        
        if "fwd" in mode:
            tilelang_output = {"O": tilelang_output}
        elif "bwd" in mode:
            tilelang_output = {"O": tilelang_output, "dQ": Q.grad.clone(), "dK": K.grad.clone(), "dV": V.grad.clone()}
        
        best_latency = perf_results[0]
        best_tflops = perf_results[1]
        best_config = perf_results[2]

        return best_latency, best_tflops, best_config, tilelang_output

    except Exception as e:
        print(f"Tilelang error: {e}")
        return None, None, None, None


def run_comparison(batch, heads, seq_len, head_dim, groups, causal, algo="mha", mode="fwd_bshd"):
    """Run both tilelang and flash-2 benchmarks and return results"""
    causal_str = "TRUE" if causal else "FALSE"
    if algo == "mha":
        config_str = f"B{batch}_H{heads}_D{head_dim}_L{seq_len}_causal_{causal_str}_{algo.upper()}_{mode.upper()}"
    elif algo == "gqa":
        config_str = f"B{batch}_H{heads}_D{head_dim}_L{seq_len}_G{groups}_causal_{causal_str}_{algo.upper()}_{mode.upper()}"

    print(f"\n{'='*70}")
    print(f"Running: {config_str}")
    print(f"{'='*70}")

    results = {
        "config": config_str,
        "batch": batch,
        "heads": heads,
        "dim": head_dim,
        "seq_len": seq_len,
        "groups": groups,
        "causal": causal_str,
        "algo": algo,
        "mode": mode,
    }

    Q = (
        torch.empty(batch, seq_len, heads, head_dim, dtype=torch.half,
                    device="cuda").normal_().requires_grad_())

    head_kv = heads // groups
    K = (
        torch.empty(batch, seq_len, head_kv, head_dim, dtype=torch.half,
                    device="cuda").normal_().requires_grad_())
    V = (
        torch.empty(batch, seq_len, head_kv, head_dim, dtype=torch.half,
                    device="cuda").normal_().requires_grad_())
    dO = torch.randn_like(Q)

    # Run tilelang
    tilelang_latency, tilelang_tflops, tilelang_best_config, tilelang_output = run_tilelang_benchmark(
        batch, heads, seq_len, head_dim, groups, causal, algo, mode, Q, K, V, dO
    )
    results["tilelang_latency_ms"] = tilelang_latency
    results["tilelang_tflops"] = tilelang_tflops

    if "fwd" in mode:
        tilelang_output_cpu = {
            "O": tilelang_output["O"].detach().cpu(),
        }
        del tilelang_output["O"]
    else:
        tilelang_output_cpu = {
            "O": tilelang_output["O"].detach().cpu(),
            "dQ": tilelang_output["dQ"].detach().cpu(),
            "dK": tilelang_output["dK"].detach().cpu(),
            "dV": tilelang_output["dV"].detach().cpu(),
        }
        del tilelang_output["O"], tilelang_output["dQ"], tilelang_output["dK"], tilelang_output["dV"]
    gc.collect()
    torch.cuda.empty_cache()

    # Run flash-2
    if HAS_FLASH:
        try:
            flash_latency, flash_tflops, flash_output = bench_flash_attention(
                batch, heads, seq_len, head_dim, groups, causal, algo, mode, Q, K, V, dO
            )
            results["flash_latency_ms"] = flash_latency
            results["flash_tflops"] = flash_tflops
        except Exception as e:
            results["flash_latency_ms"] = "ERROR"
            results["flash_tflops"] = "ERROR"
            print(f"Flash-2 error: {e}")
    else:
        results["flash_latency_ms"] = "NO_FLASH"
        results["flash_tflops"] = "NO_FLASH"

    # check result
    if "bwd" in mode:
        torch.testing.assert_close(tilelang_output_cpu["O"], flash_output["O"].detach().cpu(), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(tilelang_output_cpu["dQ"], flash_output["dQ"].detach().cpu(), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(tilelang_output_cpu["dK"], flash_output["dK"].detach().cpu(), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(tilelang_output_cpu["dV"], flash_output["dV"].detach().cpu(), rtol=1e-2, atol=1e-2)
    elif "fwd" in mode:
        torch.testing.assert_close(tilelang_output_cpu["O"], flash_output["O"].detach().cpu(), rtol=1e-2, atol=1e-2)
    print("All check passed.")

    device_name = torch.cuda.get_device_name().lower()
    USE_PPU = ("ppu" in device_name) or ("zw" in device_name)

    dev = "gpu"
    if USE_PPU:
        os.environ['HGGC_RESET_CACHE'] = '1'
        os.environ['ALIPPU_RESET_CE_MASK'] = '1'
        dev = "ppu"
    
    del Q, K, V, dO
    gc.collect()
    torch.cuda.empty_cache()
    fa_cycle, fa_tc = run_fa_cycle_on_device(batch, heads, seq_len, head_dim, groups, causal, algo, mode,
                           "./cycle.log", dev=dev)
    gc.collect()
    torch.cuda.empty_cache()
    tilelang_cycle, tilelang_tc = run_tilelang_cycle_on_device(batch, heads, seq_len, head_dim, groups, causal, algo, mode,
                           tilelang_best_config, "./cycle.log", dev=dev)

    # Print summary
    print(f"\nResults for {config_str}:")
    print("-" * 50)

    print(f"tilelang_best_config: {tilelang_best_config}")

    if (results.get("tilelang_latency_ms") and results.get("flash_latency_ms") and
        isinstance(results["tilelang_latency_ms"], (int, float)) and
        isinstance(results["flash_latency_ms"], (int, float))):
        table_data = [
            ["Metric", "Tilelang", "Flash-2", "Ratio (T/Flash)"],
            ["Latency (ms)", f"{results['tilelang_latency_ms']:.4f}", f"{results['flash_latency_ms']:.4f}",
             format_ratio(results["tilelang_latency_ms"], results["flash_latency_ms"])],
            ["TFlops", f"{results['tilelang_tflops']:.2f}", f"{results['flash_tflops']:.2f}",
             format_ratio(results["tilelang_tflops"], results["flash_tflops"])],
            ["cycles", f"{tilelang_cycle:,.0f}", f"{fa_cycle:,.0f}",
             format_ratio(tilelang_cycle, fa_cycle)],
            ["tc", f"{tilelang_tc}", f"{fa_tc}",
             format_ratio(tilelang_tc, fa_tc)]
        ]
        print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
    else:
        print("Some benchmarks failed:")
        for k, v in results.items():
            if k not in ["batch", "heads", "dim", "seq_len", "causal", "algo", "mode", "config"]:
                print(f"  {k}: {v}")

    return results


# ==================== Pytest 参数化配置 ====================

# 生成测试配置 - 默认使用 FULL_CONFIG，可通过环境变量调整
BENCHMARK_CONFIG_NAME = os.environ.get("BENCHMARK_CONFIG", "QUICK").upper()

# 配置空间 - 修改这里来控制测试范围
BENCHMARK_CONFIGS = []

# 单卡 smoke 配置：覆盖每个 benchmark kernel 各一个小规模 case。
SINGLE_CONFIG = {
    "batch": [1],
    "heads": [4],
    "dims": [64],
    "n_ctx": [256],
    "groups": [1, 2],
    "causal": [False],
    "algos": ["mha", "gqa"],
    "mha_modes": ["fwd", "bwd"],
    "gqa_modes": ["fwd", "bwd"],
    "shapes": ["bshd", "bhsd"],
}

# Daily测试配置
# mha: mha 的 fwd/bwd
# gqa: gqa 的 fwd
DAILY_CONFIG = {
    "batch": [4, 8, 16],
    "heads": [8, 16, 32, 64],
    "dims": [64, 128],
    "n_ctx": [1024, 2048, 4096, 8192, 16384],
    "groups": [1, 4, 8],
    "causal": [False, True],
    "algos": ["gqa", "mha"],
    "mha_modes": ["fwd", "bwd"],
    "gqa_modes": ["fwd"],
    "shapes": ["bshd"],
}

# 精简配置：DAILY 的 1/10，保留全部类别
# batch 取中位数 8；heads 取小(16)+大(64)；n_ctx 取短(1024)+中(4096)+长(16384)
QUICK_CONFIG = {
    "batch": [8],
    "heads": [16, 64],
    "dims": [64, 128],
    "n_ctx": [1024, 4096, 16384],
    "groups": [1, 4, 8],
    "causal": [False, True],
    "algos": ["gqa", "mha"],
    "mha_modes": ["fwd", "bwd"],
    "gqa_modes": ["fwd"],
    "shapes": ["bshd"],
}

# 完整配置
FULL_CONFIG = {
    "batch": [4, 8, 16],
    "heads": [8, 16, 32, 64],
    "dims": [64, 128],
    "n_ctx": [1024, 2048, 4096, 8192, 16384],
    "groups": [1, 4, 8],
    "causal": [False, True],
    "algos": ["gqa", "mha"],
    "modes": ["fwd", "bwd"],
    "shapes": ["bhsd", "bshd"],
}


def generate_configs(config_dict):
    """根据配置字典生成所有组合"""
    configs = []
    # 支持两种配置格式: 新格式(mha_modes/gqa_modes分离)和旧格式(modes统一)
    is_new_format = "mha_modes" in config_dict
    mha_modes = config_dict.get("mha_modes", config_dict.get("modes", []))
    gqa_modes = config_dict.get("gqa_modes", config_dict.get("modes", []))

    for batch, heads, dim, n_ctx, groups, causal, algo, shape in itertools.product(
        config_dict["batch"],
        config_dict["heads"],
        config_dict["dims"],
        config_dict["n_ctx"],
        config_dict["groups"],
        config_dict["causal"],
        config_dict["algos"],
        config_dict["shapes"],
    ):
        # 过滤group
        if algo == "mha" and groups != 1:
            continue

        if algo == "gqa" and groups == 1:
            continue

        if algo == "gqa" and shape == "bhsd":
            continue

        # 根据 algo 选择对应的 modes
        modes = mha_modes if algo == "mha" else gqa_modes
        for mode in modes:
            mode_full = mode + "_" + shape
            # 构建测试 ID 字符串
            causal_str = "Causal" if causal else "NonCausal"
            if algo == "mha":
                test_id = f"B{batch}_H{heads}_D{dim}_L{n_ctx}_{causal_str}_{algo.upper()}_{mode_full.upper()}"
            elif algo == "gqa":
                test_id = f"B{batch}_H{heads}_D{dim}_L{n_ctx}_G{groups}_{causal_str}_{algo.upper()}_{mode_full.upper()}"
            configs.append(pytest.param(batch, heads, n_ctx, dim, groups, causal, algo, mode_full, test_id, id=test_id))
    return configs


if BENCHMARK_CONFIG_NAME == "SINGLE":
    BENCHMARK_CONFIGS = generate_configs(SINGLE_CONFIG)
elif BENCHMARK_CONFIG_NAME == "DAILY":
    BENCHMARK_CONFIGS = generate_configs(DAILY_CONFIG)
elif BENCHMARK_CONFIG_NAME == "QUICK":
    BENCHMARK_CONFIGS = generate_configs(QUICK_CONFIG)
elif BENCHMARK_CONFIG_NAME == "FULL":
    BENCHMARK_CONFIGS = generate_configs(FULL_CONFIG)
else:
    raise ValueError(f"Unknown BENCHMARK_CONFIG: {BENCHMARK_CONFIG_NAME}. Use SINGLE, QUICK, DAILY, or FULL.")


@pytest.mark.parametrize("batch, heads, seq_len, head_dim, groups, causal, algo, mode, test_id", BENCHMARK_CONFIGS)
def test_bench_comparison(batch, heads, seq_len, head_dim, groups, causal, algo, mode, test_id):
    """Benchmark comparison between tilelang (mha_autotune) and flash-2"""
    result = run_comparison(batch, heads, seq_len, head_dim, groups, causal, algo, mode)

    # 断言两个实现都成功运行
    assert isinstance(result["tilelang_latency_ms"], (int, float)), \
        f"Tilelang benchmark failed for {test_id}: {result['tilelang_latency_ms']}"
    assert isinstance(result["flash_latency_ms"], (int, float)), \
        f"Flash-2 benchmark failed for {test_id}: {result['flash_latency_ms']}"

    # 可选：检查性能差异在合理范围内（例如 5 倍以内）
    # speedup = result["flash_latency_ms"] / result["tilelang_latency_ms"]
    # assert 0.2 <= speedup <= 5.0, f"Performance difference too large for {test_id}: {speedup}x"


if __name__ == "__main__":
    # 当直接运行时，使用小规模配置进行测试
    print("Running with SINGLE_CONFIG")
    print("Use pytest to run with different configurations\n")

    all_results = []
    config = SINGLE_CONFIG
    # 支持新格式(mha_modes/gqa_modes)和旧格式(modes)
    mha_modes = config.get("mha_modes", config.get("modes", []))
    gqa_modes = config.get("gqa_modes", config.get("modes", []))

    for batch, heads, dim, n_ctx, groups, causal, algo, shape in itertools.product(
        config["batch"], config["heads"], config["dims"], config["n_ctx"], config["groups"], config["causal"], config["algos"], config["shapes"]
    ):
        # 过滤group
        if algo == "mha" and groups != 1:
            continue

        if algo == "gqa" and groups == 1:
            continue

        if algo == "gqa" and shape == "bhsd":
            continue

        # 根据 algo 选择对应的 modes
        modes = mha_modes if algo == "mha" else gqa_modes
        for mode in modes:
            mode_full = mode + "_" + shape
            result = run_comparison(batch, heads, n_ctx, dim, groups, causal, algo, mode_full)
            all_results.append(result)

    # 汇总表格
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")

    summary_data = []
    success_count = 0
    for r in all_results:
        if (isinstance(r.get("tilelang_latency_ms"), (int, float)) and
            isinstance(r.get("flash_latency_ms"), (int, float))):
            summary_data.append([
                r["config"],
                f"{r['tilelang_latency_ms']:.4f}",
                f"{r['flash_latency_ms']:.4f}",
                f"{r['tilelang_tflops']:.2f}",
                f"{r['flash_tflops']:.2f}",
                format_ratio(r["flash_latency_ms"], r["tilelang_latency_ms"]),
            ])
            success_count += 1

    print(f"Successfully completed: {success_count}/{len(all_results)} configurations\n")

    print(tabulate(
        summary_data,
        headers=["Config", "Tilelang (ms)", "Flash-2 (ms)", "Tilelang (TF)", "Flash-2 (TF)", "Speedup"],
        tablefmt="grid"
    ))

    # 保存到文件
    output_file = "comparison_results.txt"
    with open(output_file, "w") as f:
        f.write("Comparison Benchmark Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total configs: {len(all_results)}, Successful: {success_count}\n\n")
        for r in all_results:
            f.write(f"Config: {r['config']}\n")
            if isinstance(r.get("tilelang_latency_ms"), (int, float)):
                f.write(f"  Tilelang: {r['tilelang_latency_ms']:.4f} ms, {r['tilelang_tflops']:.2f} TFlops\n")
            else:
                f.write(f"  Tilelang: {r.get('tilelang_latency_ms')}\n")
            if isinstance(r.get("flash_latency_ms"), (int, float)):
                f.write(f"  Flash-2:  {r['flash_latency_ms']:.4f} ms, {r['flash_tflops']:.2f} TFlops\n")
            else:
                f.write(f"  Flash-2:  {r.get('flash_latency_ms')}\n")
            f.write("\n")

    print(f"\nDetailed results saved to: {output_file}")
