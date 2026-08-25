---
name: tilelang-ppu
description: Pre-commit verification suite for the TileLang PPU backend (36 example cases across gemm, chained_dot, flash_attention, plus PPU1.5 examples/ppu/). Use when the user asks to commit changes to the PPU backend — first ask whether to run this case list for validation before committing.
---

# TileLang PPU Full Case Verification

Minimal baseline test suite for the TileLang PPU backend. When triggered (typically at commit time), the agent should first ask the user whether to run this full case list for pre-commit validation.

## Prerequisites

- Repository built with `-DUSE_PPU=ON -DUSE_CUDA=ON` (otherwise all cases fail
  with `Target kind "ppu" is not defined`)
- `python`, `bash`, GNU `timeout`

## Run

```bash
bash <repo>/.agents/skills/tilelang-ppu/scripts/run_all_cases.sh            # fresh run
bash <repo>/.agents/skills/tilelang-ppu/scripts/run_all_cases.sh --resume   # continue interrupted run
```

Optional env vars: `REPO_DIR` (default auto-detected), `PYTHON` (default
`python`), `CASE_TIMEOUT` (default 300s, timeout = FAIL), `TILELANG_CACHE_DIR`
(if set, cleared before every case).

## Test list (36 cases)

gemm (4):

- examples/gemm/example_gemm.py
- examples/gemm/example_gemm_persistent.py
- examples/gemm/example_gemm_autotune.py
- examples/gemm/example_gemm_advanced_autotune.py

chained_dot (4):

- examples/chained_dot/chained_dot.py
- examples/chained_dot/chained_dot2_transB.py
- examples/chained_dot/chained_dot3_transB.py
- examples/chained_dot/modify_chained_dot.py

flash_attention (9):

- examples/flash_attention/example_gqa_bwd.py
- examples/flash_attention/example_gqa_fwd_bshd.py
- examples/flash_attention/example_gqa_fwd_varlen.py
- examples/flash_attention/example_mha_bwd_bhsd.py
- examples/flash_attention/example_mha_bwd_bshd.py
- examples/flash_attention/example_mha_fwd_bhsd.py
- examples/flash_attention/example_mha_fwd_bshd.py
- examples/flash_attention/example_mha_fwd_varlen.py
- examples/flash_attention/regression_example_flash_attention.py

`bert_padding.py` and `varlen_utils.py` are import-only helpers, never run directly.

### PPU1.5 (examples/ppu/) — 890 only

ppu/gemm (6):

- examples/ppu/gemm/ppu_example_gemm.py
- examples/ppu/gemm/ppu_example_gemm_intrinsics.py
- examples/ppu/gemm/ppu_example_gemm_persistent.py
- examples/ppu/gemm/ppu_regression_example_gemm.py
- examples/ppu/gemm/ppu_example_gemm_autotune.py
- examples/ppu/gemm/example_gemm_advanced_autotune.py

ppu/chained_dot (4):

- examples/ppu/chained_dot/ppu_chained_dot.py
- examples/ppu/chained_dot/ppu_chained_dot2_transB.py
- examples/ppu/chained_dot/ppu_chained_dot3_transB.py
- examples/ppu/chained_dot/ppu_modify_chained_dot.py

ppu/flash_attention (9):

- examples/ppu/flash_attention/ppu_example_gqa_bwd.py
- examples/ppu/flash_attention/ppu_example_gqa_fwd_bshd.py
- examples/ppu/flash_attention/ppu_example_gqa_fwd_varlen.py
- examples/ppu/flash_attention/ppu_example_mha_bwd_bhsd.py
- examples/ppu/flash_attention/ppu_example_mha_bwd_bshd.py
- examples/ppu/flash_attention/ppu_example_mha_fwd_bhsd.py
- examples/ppu/flash_attention/ppu_example_mha_fwd_bshd.py
- examples/ppu/flash_attention/ppu_example_mha_fwd_varlen.py
- examples/ppu/flash_attention/ppu_regression_example_flash_attention.py

`ppu_bert_padding.py` and `ppu_varlen_utils.py` are import-only helpers, never run directly.

## Outputs

- `case_logs/<case_path_with_underscores>.log` — per-case stdout+stderr
- `case_logs/summary.log` — `CASE_RESULT` / `CASE_FAIL` lines +
  final marker `VERIFY_ALLCASES_DONE=<0|1>`; script exit code equals the marker

## Success criteria (ALL must hold)

1. `VERIFY_ALLCASES_DONE=0` (script exits 0)
2. Every `CASE_RESULT` shows `exit=0`
3. No `CASE_FAIL` lines
4. No timeouts (exit=124 = FAIL)

## On failure

Inspect `CASE_FAIL` lines and matching `case_logs/<name>.log`. To re-run only
failed cases: delete their `CASE_RESULT` lines from summary.log, run with `--resume`.
