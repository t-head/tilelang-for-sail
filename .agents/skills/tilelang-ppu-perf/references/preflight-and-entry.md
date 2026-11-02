# 预检与入口规范

目的：用 5 分钟把会让整批数据作废的前置条件钉死。跳过预检的典型代价是跑了 40 分钟才发现设备被占、版本不对、或入口跑的是另一套配置。

## 1. 先向用户问清楚（不要猜）

| 项 | 为什么要问 | 备注 |
|---|---|---|
| 主机 / 端口 / 认证方式 | 没有它就什么都做不了 | 认证方式决定能否批量非交互执行 |
| 容器名与进入方式 | 工具与驱动通常只在容器内可见 | 需要确认是 `docker exec` 还是别的调度 |
| 代码仓路径 | 决定测试入口的相对根 | 与本地仓路径常常不同 |
| 数据/归档挂载路径 | 决定产物落在哪、能否持久 | 容器内临时目录会被清理 |
| 可用设备编号 | 抢设备会让他人任务变慢或 OOM | 问"哪些卡当前空闲"而不是自己猜 |
| 登录 shell 类型 | tcsh/fish 等会撕碎引号与 `set -u` | 复杂命令改投递脚本执行 |
| 目标 arch / 芯片型号 | 决定预期指令集与容量数值 | 容量绝对值是后续判断的前提 |
| 工具链版本 | flag 名、section 清单、save-temps 开关都随版本变 | 不做版本核对就会照抄过时命令 |

## 2. 容器内环境自检（逐项跑一次，输出存档）

**先调用对应单端 skill**（PPU 侧 `acu-report-skill`，NVIDIA 侧 `ncu-report-skill`）确认该工具版本下的采集命令与 section 要求；本节只负责"把版本与帮助信息落成证据"，不裁定具体参数。

```bash
# 1) profiler（例：acu / ncu）
which <profiler>
<profiler> --version 2>&1 | head -3
<profiler> --help > <task_archive>/logs/help_profiler.txt 2>&1
<profiler> <list-sets-flag>   > <task_archive>/logs/list_sets.txt 2>&1    # 按 help 找实际写法
<profiler> <list-sections-flag> > <task_archive>/logs/list_sections.txt 2>&1

# 2) device 编译器（例：hgcc / nvcc）
which <device-compiler>
<device-compiler> --version 2>&1 | tail -3
<device-compiler> --help > <task_archive>/logs/help_device_compiler.txt 2>&1

# 3) host 编译器（gcc / clang 等）
which <host-compiler> && <host-compiler> --version 2>&1 | head -2
<host-compiler> --help > <task_archive>/logs/help_host_compiler.txt 2>&1

# 4) DSL / JIT 的 CLI（若有；版本与缓存/编译开关都以它的 help 为准）
which <dsl-cli> && <dsl-cli> --help > <task_archive>/logs/help_dsl_cli.txt 2>&1
python -c "import <dsl> as t; print('<dsl>', t.__version__, t.__file__)"
python -c "import torch; print('torch', torch.__version__)"

# 设备可见性与空闲状态
env | grep -E '<VISIBLE_DEVICES_ENV>'   # 例：CUDA_VISIBLE_DEVICES / PPU_VISIBLE_DEVICES
<device-smi>                            # 例：nvidia-smi / ppu-smi
```

要求：

- **四类 CLI 的 `--help` 必须现场跑一次并落盘**，不要凭记忆或历史会话写命令。落盘内容就是"为什么用这个参数名"的证据。
- **参数名一律按 help 实测确定**：`--help` 里没有的写法不要用；help 输出里语义相近但名字不同的（`--keep` / `-save-temps` / `--keep-dir` 之类）必须逐个确认。
- help 里找不到某能力（如列出 set/section）就记一条"该版本不提供"，并走单端 skill 给的替代路径，不要自造参数。
- 自检结论写进 `<task_archive>/logs/env_snapshot.txt`，并在 log 头部标注：工具版本、编译器版本、DSL 版本与路径、arch、设备编号、仓 commit。

## 3. 入口规范（按项目实际情况条件化）

**核心原则：只用项目约定的测试入口，不要用仓库里可能存在的"直跑"脚本**。直跑脚本通常不处理入口里的环境变量分支（跳过 reference、配置集筛选、计时器归零），走错入口会静默得到属于另一套配置的数字。

### 若项目使用 pytest 作为入口

```bash
python -m pytest <test_file>::<test_fn>[<params>] -m <marker> --run-benchmark -q
```

- 参数化 id 顺序陷阱：`@pytest.mark.parametrize` 的 id 顺序与函数签名顺序经常相反（尤其多层 parametrize）。选 case 前必须先用 `--collect-only -q` 打印实际 node id，不要凭签名顺序手拼。
- **marker 与 `--run-benchmark` 一类开关都是插件/配置相关**：`--run-benchmark` 需要 `pytest-benchmark` 或项目自定义的 benchmark 插件注册该标记才可用。没有注册时报 `unrecognized arguments` / `unknown marker` 属预期——先去 `pytest.ini`/`pyproject.toml`/`conftest.py` 确认实际可用的开关名，不要硬加参数。
- 若项目把 benchmark 标记换成别的名字（如 `-m perf`、`--bench`），以项目配置为准。

### 若项目使用其他入口（自定义 runner、Makefile 目标、脚本 + 参数等）

遵循同样的原则，逐项确认后再跑：

1. **入口即配置**：确认这个入口会不会自己改配置集、跳过 reference、重置计时器；把这些副作用写进 README。
2. **单 case 可寻址**：确认能只跑目标 case（等价于 node id 的定位方式），不要靠"跑全量再筛日志"。
3. **可列举**：确认能列出用例清单（等价于 `--collect-only`），用于核对参数顺序与实际 id。
4. **环境变量可控**：确认哪些环境变量影响行为（跳过对比、设备选择、shape 覆盖），并在 `CMD:` 行里记录它们的取值。

## 4. 两段式 smoke（正式批量前必做）

| 阶段 | 命令 | 通过标准 |
|---|---|---|
| S1 裸跑 | 项目入口 + benchmark 相关开关，**不开 profiler** | 正常退出；能打印出 bench 记录（含耗时字段）；无 reference 报错 |
| S2 单次采集 | 开 profiler，filter 限定单个 kernel，launch 控制放到最小 | 生成报告文件且体积合理（KB 级起步）；导出 CSV 能拿到关键列 |

S1 失败不要往下走：先解决编译/环境/OOM 问题。S2 失败先看 raw log 尾部的 profiler 段，常见是 kernel filter 没命中。

**功能验证（reference 对比）与 S2 必须分开跑**：混在一次运行里既污染时长，也让 launch 计数不可控。reference 是可选顺手项，S1 里做最省事。

## 5. 失败模式对照

| 现象 | 常见根因 | 处置 |
|---|---|---|
| 入口整体非零退出但 bench 有数字 | 计时器返回 0 导致派生指标除零（如带宽 = bytes/time） | 该 case 的计算指标作废，只用报告里的 duration；或修计时逻辑后重跑 |
| `node id not found`（pytest） | 参数化 id 顺序猜错 / marker 未注册 | `--collect-only` 打印真实 id；核对插件与 marker |
| `unrecognized arguments`（pytest） | `--run-benchmark` 一类开关的插件没装或未注册 | 查配置确认开关名；不要硬加参数 |
| profiler 报告 0 个 kernel | filter 不匹配实际符号名 | 从裸跑 log 或 codegen 产物里抄真实符号名 |
| 命令在远端被拆碎、变量为空 | 登录 shell 非 bash | 把命令写成脚本投递（`ssh host 'bash -s' < script.sh`），不要拼超长单行 |
| 连接偶发 `Bad file descriptor` / 拒绝 | 短连接过密触发 sshd 限流 | 复用长连接（ControlMaster/ControlPersist），失败后隔几秒重试一次 |
| 采集中途进程被杀、报告缺后半段 | profiler 出错时默认终止被 profile 进程 | 按 help 与单端 skill 找到"出错不杀进程"的开关（常见形如 `--kill no`） |
| 照抄文档里的参数报 unknown option | 工具版本变了，flag 改名或已移除 | 回 §2 查 `--help`，按当次版本重写命令 |
