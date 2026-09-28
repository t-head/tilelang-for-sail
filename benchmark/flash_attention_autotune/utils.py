import os
import subprocess
import re


def format_ratio(numerator, denominator):
    """Format a TileLang/Flash-2 ratio string, e.g. ``"0.854x"``.

    Profiling bugs (e.g. acu reporting 0 cycles/TC) turn the ratio into a
    0/0 division.  Return ``"Invalid"`` in that case instead of crashing or
    printing a misleading number.
    """
    if denominator:
        return f"{numerator / denominator:.3f}x"
    return "Invalid"


def run_cmd(cmd: str, timeout=None, stdout=subprocess.PIPE, stderr=subprocess.PIPE):
    print(f"Run command: {cmd}, timeout: {timeout}")
    ret = subprocess.run(args=cmd, timeout=timeout, shell=True, stdout=stdout, stderr=stderr, encoding="utf-8")
    if stdout:
        for line in ret.stdout.splitlines() + ret.stderr.splitlines():
            print(line)
    if ret.returncode != 0:
        print("Run command failed!")
    else:
        print("Run command succeed!")
    return ret


def inject_pass_configs_from_env(kernel_func):
    """Inject pass_configs from TILELANG_PASS_CONFIGS env var onto jit_impl.

    The autotune skip path (all tunable params provided) calls jit_compile()
    with no arguments, so per-config pass_configs is lost.  This injects the
    best config's pass_configs directly onto jit_impl before the kernel is
    called in acu/ncu profiling scripts.
    """
    import json

    pc_str = os.environ.get("TILELANG_PASS_CONFIGS", "")
    if pc_str:
        pc = json.loads(pc_str)
        kernel_func.jit_impl.pass_configs = pc


# devices = {
#     "name": ["cycle", "tensor core efficiency", "waves"],
#     "gpu":  ["sm__cycles_active.max", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", "launch__waves_per_multiprocessor"],
#     "ppu":  ["ce__cycles_active.max", "cu__inst_executed_pipe_tensor_fp16.avg.pct_of_peak_sustained_active", "launch__waves_per_cu"],
# }
def read_cycle_from_nculog(filename, mode, framework):
    kernel_pattern = r"(.*)kernel(.*)Device(.*)"
    cycles_pattern = "__cycles_active.max"
    tc_pattern = "pct_of_peak_sustained_active"
    kernel_list = []
    cycles_list = []
    tc_list = []
    with open(filename, newline="") as log_file:
        for line in log_file.read().split("\n"):
            if re.search(kernel_pattern, line):
                kernel_list.append(line.strip())
            if re.search(cycles_pattern, line):
                cycles_list.append(int(line.strip().split()[-1]))
            if re.search(tc_pattern, line):
                tc_list.append(float(line.strip().split()[-1]))
    # print(kernel_list)
    # print(cycles_list)
    # print(tc_list)
    assert len(kernel_list) == len(cycles_list)
    assert len(kernel_list) == len(tc_list)
    op_cycles = dict()
    cycle_sum = 0
    tc_sum = 0
    for i in range(len(kernel_list)):
        op = kernel_list[i]
        cycle = cycles_list[i]
        op_cycles[op] = cycle
        # if "fwd" in op.lower() or "mla" in op.lower(): # flashmla ppu / triton / flashinfer
        #     fwd_cycle_sum += cycle
        #     fwd_tc_sum += tc_list[i]
        if framework == "flash-2":
            if "fwd" in mode and "flash_fwd" in op.lower() or "bwd" in mode and "flash_bwd" in op.lower():
                cycle_sum += cycle
                tc_sum += tc_list[i]
        elif framework == "tilelang":
            if "fwd" in mode and "main_kernel" in op.lower():
                cycle_sum += cycle
                tc_sum += tc_list[i]
            elif "bwd" in mode and "flash_bwd_kernel" in op.lower():
                cycle_sum = cycle
                tc_sum = tc_list[i]
    # calculate statistics data
    if cycle_sum != 0:
        # fwd unit case
        return cycle_sum, tc_sum, op_cycles
    else:
        print("Not valid CSV file!")
        return 0, 0, []
        # exit(-1)


def run_fa_cycle_on_device(batch, heads, seq_len, head_dim, groups, causal, algo, mode, output_file, dev="gpu", run_local=False):
    """Run flash attention cycle benchmark on specified device.

    Args:
        batch: Batch size
        heads: Number of heads
        seq_len: Sequence length
        head_dim: Head dimension
        groups: Number of groups (for GQA)
        causal: Whether to use causal attention
        algo: "mha" or "gqa"
        mode: "fwd" or "bwd"
        output_file: Output file path
        dev: Device to run on ("gpu" or "ppu")
        run_local: Whether to save results to output_file
    """
    output_lines = list()

    # Build case name
    causal_str = "causal" if causal else "noncausal"
    case_name = f"B{batch}_H{heads}_D{head_dim}_L{seq_len}_G{groups}_{causal_str}_{algo}_{mode}"

    log_file = "./gpu_cycles_single_case.log"
    cmd = "rm -f " + log_file
    run_cmd(cmd)

    # Get metrics based on device
    metrics_string = (
        "sm__cycles_active.max,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
        if dev == "gpu"
        else "ce__cycles_active.max,cu__inst_executed_pipe_tensor_fp16.avg.pct_of_peak_sustained_active"
    )
    mode = mode.split("_")[0]

    # Build the command to run run_flash.py
    cmd = '{} --clock-control none --metrics="{}" \
          --page=details python ./run_flash.py \
          --batch {} --heads {} --seq_len {} --head_dim {} \
          --groups {} --{} --algo {} --mode {} \
          2>&1 | tee -a {}'.format(
        "ncu" if dev == "gpu" else "acu",
        metrics_string,
        batch,
        heads,
        seq_len,
        head_dim,
        groups,
        "causal True" if causal else 'causal ""',
        algo,
        mode,
        log_file,
    )

    run_cmd(cmd)
    cycle, tc, detail = read_cycle_from_nculog(log_file, mode, "flash-2")
    output_lines.append([case_name.replace(",", "_"), str(cycle), str(tc), str(cmd), str(detail)])

    dirname = os.path.dirname(output_file)
    cmd = f"mkdir -p {dirname}"
    run_cmd(cmd)

    return cycle, tc

    # if len(output_lines) == 1:
    #     with open("local.log", "w") as f:
    #         writer = csv.writer(f)
    #         for row in output_lines:
    #             writer.writerow(row)
    #         print("write result to local.log succeed")

    # if run_local:
    #     with open(output_file, "w") as f:
    #         writer = csv.writer(f)
    #         for row in output_lines:
    #             writer.writerow(row)
    #         print("write result succeed")
    # else:
    #     if not os.path.exists(output_file):
    #         with open(output_file, "w", newline="") as f:
    #             writer = csv.writer(f)
    #             writer.writerow(headers)
    #     with open(output_file, "a") as f:
    #         writer = csv.writer(f)
    #         for row in output_lines:
    #             writer.writerow(row)
    #     print("write result succeed")


def run_tilelang_cycle_on_device(
    batch, heads, seq_len, head_dim, groups, causal, algo, mode, best_config, output_file, dev="gpu", run_local=False
):
    """Run flash attention cycle benchmark on specified device.

    Args:
        batch: Batch size
        heads: Number of heads
        seq_len: Sequence length
        head_dim: Head dimension
        groups: Number of groups (for GQA)
        causal: Whether to use causal attention
        algo: "mha" or "gqa"
        mode: "fwd" or "bwd"
        best_config: Dictionary containing block_M, block_N, num_stages, threads
        output_file: Output file path
        dev: Device to run on ("gpu" or "ppu")
        run_local: Whether to save results to output_file
    """
    output_lines = list()

    # Build case name
    causal_str = "causal" if causal else "noncausal"
    case_name = f"B{batch}_H{heads}_D{head_dim}_L{seq_len}_G{groups}_{causal_str}_{algo}_{mode}"

    log_file = "./gpu_cycles_single_case.log"
    cmd = "rm -f " + log_file
    run_cmd(cmd)

    # Inject pass_configs from best_config into env for acu/ncu profiling.
    # The autotune skip path loses per-config pass_configs; kernel scripts
    # read TILELANG_PASS_CONFIGS and set jit_impl.pass_configs before kernel call.
    import json
    from enum import Enum

    pass_configs = best_config.get("pass_configs")
    if pass_configs:
        safe_pc = {k.value if isinstance(k, Enum) else k: v for k, v in pass_configs.items()}
        os.environ["TILELANG_PASS_CONFIGS"] = json.dumps(safe_pc)
    else:
        os.environ.pop("TILELANG_PASS_CONFIGS", None)

    # Get metrics based on device
    metrics_string = (
        "sm__cycles_active.max,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"
        if dev == "gpu"
        else "ce__cycles_active.max,cu__inst_executed_pipe_tensor_fp16.avg.pct_of_peak_sustained_active"
    )
    fn = algo + "_" + mode
    filename = "example_" + fn + ".py"

    if fn == "mha_fwd_bshd":
        filename = "example_mha_fwd_bshd.py"
        cmd = '{} --clock-control none --metrics="{}" \
              --page=details python ./kernels/{} \
              --batch {} --heads {} --seq_len {} --dim {} \
              --{} --block_M {} --block_N {} --num_stages {} --threads {} 2>&1 | tee -a {}'.format(
            "ncu" if dev == "gpu" else "acu",
            metrics_string,
            filename,
            batch,
            heads,
            seq_len,
            head_dim,
            "causal True" if causal else 'causal ""',
            best_config["block_M"],
            best_config["block_N"],
            best_config["num_stages"],
            best_config["threads"],
            log_file,
        )
    elif fn == "mha_fwd_bhsd":
        filename = "example_mha_fwd_bhsd.py"
        cmd = '{} --clock-control none --metrics="{}" \
              --page=details python ./kernels/{} \
              --batch {} --heads {} --seq_q {} --seq_kv {} --dim {} \
              --{} --block_M {} --block_N {} --num_stages {} --threads {} 2>&1 | tee -a {}'.format(
            "ncu" if dev == "gpu" else "acu",
            metrics_string,
            filename,
            batch,
            heads,
            seq_len,
            seq_len,
            head_dim,
            "causal True" if causal else 'causal ""',
            best_config["block_M"],
            best_config["block_N"],
            best_config["num_stages"],
            best_config["threads"],
            log_file,
        )
    elif fn == "gqa_fwd_bshd":
        filename = "example_gqa_fwd_bshd.py"
        cmd = '{} --clock-control none --metrics="{}" \
              --page=details python ./kernels/{} \
              --batch {} --heads {} --seq_len {} --dim {} \
              --{} --groups {} --block_M {} --block_N {} --num_stages {} --threads {} 2>&1 | tee -a {}'.format(
            "ncu" if dev == "gpu" else "acu",
            metrics_string,
            filename,
            batch,
            heads,
            seq_len,
            head_dim,
            "causal True" if causal else 'causal ""',
            groups,
            best_config["block_M"],
            best_config["block_N"],
            best_config["num_stages"],
            best_config["threads"],
            log_file,
        )
    elif fn == "mha_bwd_bshd":
        filename = "example_mha_bwd_bshd.py"
        cmd = '{} --clock-control none --metrics="{}" \
              --page=details python ./kernels/{} \
              --batch {} --h {} --n_ctx {} --d_head {} \
              --{} --block_M {} --block_N {} --num_stages {} --threads {} 2>&1 | tee -a {}'.format(
            "ncu" if dev == "gpu" else "acu",
            metrics_string,
            filename,
            batch,
            heads,
            seq_len,
            head_dim,
            "causal True" if causal else 'causal ""',
            best_config["block_M"],
            best_config["block_N"],
            best_config["num_stages"],
            best_config["threads"],
            log_file,
        )
    elif fn == "mha_bwd_bhsd":
        filename = "example_mha_bwd_bhsd.py"
        cmd = '{} --clock-control none --metrics="{}" \
              --page=details python ./kernels/{} \
              --batch {} --h {} --n_ctx {} --d_head {} \
              --{} --block_M {} --block_N {} --num_stages {} --threads {} 2>&1 | tee -a {}'.format(
            "ncu" if dev == "gpu" else "acu",
            metrics_string,
            filename,
            batch,
            heads,
            seq_len,
            head_dim,
            "causal True" if causal else 'causal ""',
            best_config["block_M"],
            best_config["block_N"],
            best_config["num_stages"],
            best_config["threads"],
            log_file,
        )
    elif fn == "gqa_bwd_bshd":
        filename = "example_gqa_bwd.py"
        cmd = '{} --clock-control none --metrics="{}" \
              --page=details python ./kernels/{} \
              --batch {} --h {} --n_ctx {} --d_head_qk {} --d_head_v {} \
              --{} --groups {} --block_M {} --block_N {} --num_stages {} --threads {} 2>&1 | tee -a {}'.format(
            "ncu" if dev == "gpu" else "acu",
            metrics_string,
            filename,
            batch,
            heads,
            seq_len,
            head_dim,
            head_dim,
            "causal True" if causal else 'causal ""',
            groups,
            best_config["block_M"],
            best_config["block_N"],
            best_config["num_stages"],
            best_config["threads"],
            log_file,
        )

    run_cmd(cmd)
    cycle, tc, detail = read_cycle_from_nculog(log_file, mode, "tilelang")
    output_lines.append([case_name.replace(",", "_"), str(cycle), str(tc), str(cmd), str(detail)])

    dirname = os.path.dirname(output_file)
    cmd = f"mkdir -p {dirname}"
    run_cmd(cmd)

    return cycle, tc
    # if len(output_lines) == 1:
    #     with open("local.log", "w") as f:
    #         writer = csv.writer(f)
    #         for row in output_lines:
    #             writer.writerow(row)
    #         print("write result to local.log succeed")

    # if run_local:
    #     with open(output_file, "w") as f:
    #         writer = csv.writer(f)
    #         for row in output_lines:
    #             writer.writerow(row)
    #         print("write result succeed")
    # else:
    #     if not os.path.exists(output_file):
    #         with open(output_file, "w", newline="") as f:
    #             writer = csv.writer(f)
    #             writer.writerow(headers)
    #     with open(output_file, "a") as f:
    #         writer = csv.writer(f)
    #         for row in output_lines:
    #             writer.writerow(row)
    #     print("write result succeed")
