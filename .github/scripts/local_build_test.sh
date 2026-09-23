#!/usr/bin/env bash
#
# local_build_test.sh —— 本地/单机验证脚本，等价于 CI 中 ppu-tests job 的 pod 命令块。
#
# 用途：
#   在真实 890P 机器（例如 ssh chenyang.li@30.21.206.25）上，直接跑通
#   "build tilelang wheel -> 装 PPU 依赖 -> 装 skip json -> 跑 test_tilelang.py"
#   的完整流程，用于在提交 CI 之前做本地验证。
#
#   本脚本镜像 .github/workflows/ci.yml 里 ppu-tests 的 ppu-scheduler-action
#   command 脚本块（build wheel、装 PPU 依赖、sed conftest、pip install
#   dist/*.whl、拷 skip json、BOARD_TYPE -> CI_TEST_PLATFORM 映射、跑测试）。
#
# 用法示例：
#   # 从源码根目录（默认当前目录）跑，每个目录组只保留前 5 个用例
#   bash .github/scripts/local_build_test.sh --board ZW-M890P --limit 5
#
#   # 指定源码根目录与并发/超时
#   bash .github/scripts/local_build_test.sh \
#     --src /workspace/source --board OAM-810E --limit 3 --timeout 600 --workers 4
#
#   # 全量（不截断）
#   bash .github/scripts/local_build_test.sh --limit 0
#
# 参数：
#   --board     BOARD_TYPE，默认 ZW-M890P（映射 CI_TEST_PLATFORM：
#               OAM-810E -> ppu0010, ZW-M890P -> ppu0015, 其它 -> ppu0015）
#   --limit     每个目录组保留的最大用例数，默认 5（0=不限制）
#   --src       tilelang 源码根目录，默认当前目录（脚本会 cd 过去）
#   --timeout   单个用例最长运行时间（秒），默认 600
#   --workers   并行 worker 数，默认 4
#
# 设计约束：
#   - 不引入第三方依赖；仅使用 bash + pip + python + 常见命令。
#   - 顶部只 set -o pipefail，不做全局 set -e（以免破坏容错行）；
#     关键步骤（build/安装）用显式 if 判断 + [FATAL] + exit 1 做 fail-fast，
#     清理类命令保留 `|| true` 容错。

set -o pipefail

# ---------------------------------------------------------------------------
# 默认参数
# ---------------------------------------------------------------------------
BOARD="ZW-M890P"
LIMIT=5
SRC="$(pwd)"
TIMEOUT=600
WORKERS=4

# ---------------------------------------------------------------------------
# 参数解析（简单的 while-case）
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --board)   BOARD="$2";   shift 2 ;;
    --limit)   LIMIT="$2";   shift 2 ;;
    --src)     SRC="$2";     shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    -h|--help)
      grep -E '^#( |$)' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *)
      echo "[FATAL] 未知参数: $1"
      exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# 打印将要执行的关键参数，便于验证
# ---------------------------------------------------------------------------
echo "============================================================"
echo " local_build_test.sh —— 等价于 CI ppu-tests pod 命令"
echo "   board   = ${BOARD}"
echo "   limit   = ${LIMIT}"
echo "   src     = ${SRC}"
echo "   timeout = ${TIMEOUT}"
echo "   workers = ${WORKERS}"
echo "============================================================"

# ---------------------------------------------------------------------------
# 切换到源码根目录（后续相对路径均以 --src 为基准）
# ---------------------------------------------------------------------------
if [[ ! -d "${SRC}" ]]; then
  echo "[FATAL] 源码根目录不存在: ${SRC}"
  exit 1
fi
cd "${SRC}" || { echo "[FATAL] 无法进入源码根目录: ${SRC}"; exit 1; }
echo "[INFO] 当前工作目录: $(pwd)"

# BOARD_TYPE -> CI_TEST_PLATFORM 映射（与 ci.yml 一致）
case "${BOARD}" in
  "OAM-810E")  export CI_TEST_PLATFORM=ppu0010 ;;
  "ZW-M890P")  export CI_TEST_PLATFORM=ppu0015 ;;
  *)           export CI_TEST_PLATFORM=ppu0015 ;;
esac
export BOARD_TYPE="${BOARD}"
echo "[INFO] Board: ${BOARD_TYPE} -> Platform: ${CI_TEST_PLATFORM}"

# ---------------------------------------------------------------------------
# 环境准备：source PPU SDK envsetup
# ---------------------------------------------------------------------------
unset PIP_INDEX_URL
git config --global --unset-all url.https://ai-gerrit.eng.t-head.cn/Public/Github/.insteadOf 2>/dev/null || true

if [[ -f /usr/local/PPU_SDK/envsetup.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/PPU_SDK/envsetup.sh
  echo "[INFO] 已 source /usr/local/PPU_SDK/envsetup.sh"
else
  echo "[WARN] 未找到 /usr/local/PPU_SDK/envsetup.sh，跳过 SDK 环境加载"
fi

# ---------------------------------------------------------------------------
# Build tilelang wheel
# ---------------------------------------------------------------------------
echo "[STEP] 安装构建依赖 ..."
pip install wheel build setuptools cython packaging 'setuptools==68.1.2' || {
  echo "[FATAL] 构建依赖安装失败"; exit 1; }

echo "[STEP] build tilelang wheel ..."
NO_VERSION_LABEL=OFF NO_GIT_VERSION=1 python -m build --wheel || {
  echo "[FATAL] tilelang wheel build 失败"; exit 1; }
echo "[INFO] wheel build 完成"
# python -m build 的产物校验：dist 下必须存在 wheel
if ! ls dist/*.whl >/dev/null 2>&1; then
  echo "[FATAL] 未在 dist/ 下找到构建产物 (*.whl)"; exit 1
fi

# ---------------------------------------------------------------------------
# Install PPU-specific dependencies
# ---------------------------------------------------------------------------
echo "[STEP] 安装测试与 PPU 依赖 ..."
pip install pytest==9.0.0 pyyaml==6.0.3 junit-xml || {
  echo "[FATAL] pytest/pyyaml/junit-xml 安装失败"; exit 1; }
pip install einops==0.8.1 ninja==1.13.0 mkl==2021.4.0 cuda-bindings==13.0 || {
  echo "[FATAL] einops/ninja/mkl/cuda-bindings 安装失败"; exit 1; }
pip install 'z3-solver>=4.13.0,<4.15.5' tabulate || {
  echo "[FATAL] z3-solver/tabulate 安装失败"; exit 1; }
yes | pip uninstall nvidia-cuda-nvrtc 2>/dev/null || true

# Use image-preinstalled torch/triton (hggcrt3 variant, compatible with PPU SDK libacblas).
echo "[INFO] 使用镜像预装的 torch/triton ($(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo '?'))"
# flash-attn is not bundled; install if available but do not fail.
pip install flash-attn 2>/dev/null || echo "[WARN] flash-attn not available, skipping"
cd "${SRC}"

# Install system deps（容错，失败不阻断）
apt-get update && apt-get upgrade -y && apt-get install -y libopenmpi-dev 2>/dev/null || true

# ---------------------------------------------------------------------------
# Comment out conftest.py lines 10-13 (perf marker skip that conflicts with PPU tests)
# ---------------------------------------------------------------------------
if [[ -f testing/conftest.py ]]; then
  sed -i '10,13 s/^/# /' testing/conftest.py
  echo "[INFO] 已注释 testing/conftest.py 第 10-13 行"
else
  echo "[WARN] 未找到 testing/conftest.py，跳过 sed"
fi

# ---------------------------------------------------------------------------
# Install built tilelang wheel
# ---------------------------------------------------------------------------
echo "[STEP] 安装构建好的 tilelang wheel ..."
pip install dist/*.whl || { echo "[FATAL] tilelang wheel 安装失败"; exit 1; }

# ---------------------------------------------------------------------------
# Setup test environment：拷 skip json 到 CWD
# ---------------------------------------------------------------------------
# ci.yml 里是在 /workspace 下 ln -sf source tilelang 并拷 skip json；
# 本地单机验证直接在源码根目录运行，故把 skip json 拷到当前目录即可。
if ls .github/scripts/*_skip.json >/dev/null 2>&1; then
  cp .github/scripts/*_skip.json . || echo "[WARN] 拷贝 *_skip.json 失败，继续"
  echo "[INFO] 已拷贝 skip json: $(ls *_skip.json 2>/dev/null | tr '\n' ' ')"
else
  echo "[WARN] 未找到 .github/scripts/*_skip.json"
fi

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
echo "[STEP] 运行 test_tilelang.py ..."
python .github/scripts/test_tilelang.py \
  -o test_result.xml -v \
  --timeout "${TIMEOUT}" \
  --maxfail 3 \
  --workers "${WORKERS}" \
  --limit "${LIMIT}" || TEST_RC=$?
TEST_RC=${TEST_RC:-0}

echo "============================================================"
echo " 测试完成，退出码: ${TEST_RC}"
echo " 结果文件: $(pwd)/test_result.xml"
echo "============================================================"

exit ${TEST_RC}
