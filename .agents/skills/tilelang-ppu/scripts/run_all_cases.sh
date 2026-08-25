#!/usr/bin/env bash
# TileLang PPU full example test-list runner. See SKILL.md for details.
#
# Usage: run_all_cases.sh [--resume]
# Env:   REPO_DIR PYTHON CASE_TIMEOUT TILELANG_CACHE_DIR (all optional)
# Exit:  0 = all required cases passed (marker VERIFY_ALLCASES_DONE=0)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}"
PYTHON="${PYTHON:-$(command -v python3 || command -v python || echo python)}"
CASE_TIMEOUT="${CASE_TIMEOUT:-300}"

cd "$REPO_DIR" || exit 1
mkdir -p case_logs
if [ "${1:-}" != "--resume" ]; then
  rm -f case_logs/summary.log
fi

# ---- Test list ---------------------------------------------------------------
PY_CASES=(
  examples/gemm/example_gemm.py
  examples/gemm/example_gemm_persistent.py
  examples/gemm/example_gemm_autotune.py
  examples/gemm/example_gemm_advanced_autotune.py
  examples/chained_dot/chained_dot.py
  examples/chained_dot/chained_dot2_transB.py
  examples/chained_dot/chained_dot3_transB.py
  examples/chained_dot/modify_chained_dot.py
  examples/flash_attention/example_gqa_bwd.py
  examples/flash_attention/example_gqa_fwd_bshd.py
  examples/flash_attention/example_gqa_fwd_varlen.py
  examples/flash_attention/example_mha_bwd_bhsd.py
  examples/flash_attention/example_mha_bwd_bshd.py
  examples/flash_attention/example_mha_fwd_bhsd.py
  examples/flash_attention/example_mha_fwd_bshd.py
  examples/flash_attention/example_mha_fwd_varlen.py
  examples/flash_attention/regression_example_flash_attention.py
  # PPU1.5 examples (890 only)
  examples/ppu/gemm/ppu_example_gemm.py
  examples/ppu/gemm/ppu_example_gemm_intrinsics.py
  examples/ppu/gemm/ppu_example_gemm_persistent.py
  examples/ppu/gemm/ppu_regression_example_gemm.py
  examples/ppu/gemm/ppu_example_gemm_autotune.py
  examples/ppu/gemm/ppu_example_gemm_advanced_autotune.py
  examples/ppu/chained_dot/ppu_chained_dot.py
  examples/ppu/chained_dot/ppu_chained_dot2_transB.py
  examples/ppu/chained_dot/ppu_chained_dot3_transB.py
  examples/ppu/chained_dot/ppu_modify_chained_dot.py
  examples/ppu/flash_attention/ppu_example_gqa_bwd.py
  examples/ppu/flash_attention/ppu_example_gqa_fwd_bshd.py
  examples/ppu/flash_attention/ppu_example_gqa_fwd_varlen.py
  examples/ppu/flash_attention/ppu_example_mha_bwd_bhsd.py
  examples/ppu/flash_attention/ppu_example_mha_bwd_bshd.py
  examples/ppu/flash_attention/ppu_example_mha_fwd_bhsd.py
  examples/ppu/flash_attention/ppu_example_mha_fwd_bshd.py
  examples/ppu/flash_attention/ppu_example_mha_fwd_varlen.py
  examples/ppu/flash_attention/ppu_regression_example_flash_attention.py
)

run_one() {
  local c="$1"
  if grep -qF "CASE_RESULT $c " case_logs/summary.log 2>/dev/null; then
    return
  fi
  local name
  name=$(echo "$c" | tr '/' '_' | sed 's/\.py$//')
  if [ -n "${TILELANG_CACHE_DIR:-}" ]; then
    rm -rf "${TILELANG_CACHE_DIR:?}"/* 2>/dev/null
  fi
  local start end
  start=$(date +%s)
  timeout -k 10 "$CASE_TIMEOUT" "$PYTHON" "$c" > "case_logs/${name}.log" 2>&1
  local code=$?
  end=$(date +%s)
  echo "CASE_RESULT $c exit=$code time=$((end-start))s" >> case_logs/summary.log
}

for c in "${PY_CASES[@]}"; do
  run_one "$c"
done

# ---- Verdict ------------------------------------------------------------------
# Remove stale verdict lines from a previous run (resume scenario)
if [ -f case_logs/summary.log ]; then
  grep -v -e '^CASE_FAIL ' -e '^VERIFY_ALLCASES_DONE=' case_logs/summary.log > case_logs/summary.tmp \
    && mv case_logs/summary.tmp case_logs/summary.log
fi

fail=0
while read -r _ c kv _; do
  code="${kv#exit=}"
  [ "$code" = "0" ] && continue
  echo "CASE_FAIL $c exit=$code" >> case_logs/summary.log
  fail=1
done < <(grep '^CASE_RESULT ' case_logs/summary.log)

echo "VERIFY_ALLCASES_DONE=$fail" >> case_logs/summary.log
exit "$fail"
