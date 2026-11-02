---
name: tilelang-ppu-perf
description: TileLang（或其他 DSL/JIT）kernel 在 PPU 上做性能采集、导出、归档、分析路由与跨端对比的方法论。当需要 profile 一个 kernel、从 .acurep 报告导出指标表、核对采集/导出口径、判断报告是否可信、决定该走单端还是跨端分析、或对比 PPU(acu) 与 NVIDIA(ncu) 两端报告找出差距根因时使用。覆盖采集流程与数据纪律，以及跨端对账方法论；单端指标解读交给 acu-report-skill，NVIDIA 侧交给 ncu-report-skill。
---

# TileLang kernel 在 PPU 上的性能采集、归档与跨端分析

## 定位与边界

本 skill 负责**"数据怎么拿、怎么存、怎么保证可信、交给谁分析、两端怎么对账"**。单端指标怎么解读不属于本 skill（交给对应单端 skill），但"两份单端结论放在一起怎么对账"属于本 skill。

| 需求 | 前置条件 | 去处 |
|---|---|---|
| profile 一个 kernel、导出/核对/归档数据 | 采集命令本身由对应单端 skill 定（见下两行） | 本 skill（流程与数据纪律） |
| 运行 acu / 解析 PPU 报告 | **必须先调用 `acu-report-skill`** | `acu-report-skill` |
| 运行 ncu / 解析 NVIDIA 报告 | **必须先调用 `ncu-report-skill`** | `ncu-report-skill` |
| 两端都有报告，问"差在哪" | **两侧各自已完成单端分析**（指标解读 + 停等分解 + 端内饱和判断）；"两份文件存在"不是充分条件 | 本 skill：[references/cross-end-analysis.md](references/cross-end-analysis.md)（可比性 → 绝对量对账 → 指令结构 → 停等 → 交易效率 → 饱和 → 闭环 → 归因） |

路由细节、交接包内容与跨端门槛见 [references/analysis-routing.md](references/analysis-routing.md)。

**连接信息一律向用户询问**：主机/端口/认证方式、容器名、代码路径、挂载路径、空闲设备、登录 shell 类型、目标 arch。不猜、不沿用历史会话里的值——即使同一用户，也需确认仍然有效。

## Golden Rule

**先把口径钉死，再采集；先落地 codegen，再分析；先双写归档，再下结论。**

顺序颠倒的代价：口径错 → 整批数据作废；codegen 丢了 → 事后无法核对指令数与地址模式；没归档 → 结论不可复现。

## 六段式工作流

```
0 预检 → 1 选取 → 2 保全 → 3 采集+导出 → 4 归档 → 5 路由分析 → 6 跨端对比
```

| 段 | 目标 | 产出 | 详见 |
|---|---|---|---|
| 0 预检 | 把会让整批作废的前置条件钉死 | 环境/版本/CLI help 指纹 log、smoke 记录 | [references/preflight-and-entry.md](references/preflight-and-entry.md) |
| 1 选取 | 一次采集只覆盖想看的东西 | 确定的入口 + kernel filter + launch 控制 | [references/kernel-and-launch-selection.md](references/kernel-and-launch-selection.md) |
| 2 保全 | 拿到编译产物第一手证据 | JIT 缓存目录 / save-temps 产物 / compile_cmds.log | [references/codegen-and-save-temps.md](references/codegen-and-save-temps.md) |
| 3 采集+导出 | 报告 + 指标表，各自口径正确 | 原始报告 + 宽表 / 逐指令导出 | [references/collection-and-export.md](references/collection-and-export.md) |
| 4 归档 | 数据与复现口径同时留存 | manifest（校验和 + 结构化元数据）+ README + 双写副本 | [references/archive-and-manifest.md](references/archive-and-manifest.md) |
| 5 路由 | 交给对的解读器 | 结论/报告 | [references/analysis-routing.md](references/analysis-routing.md) |
| 6 跨端对比 | 两端差距定量归因（仅在两侧单端分析已完成后进入） | 按贡献度排序的归因 + 证据强度 | [references/cross-end-analysis.md](references/cross-end-analysis.md) |

## 铁律（违反任一条，数据就不可用）

1. **入口只用项目约定的测试入口**（如 pytest 的 `<file>::<test_fn>[<params>] -m <marker>`、项目自定义 runner、Makefile 目标等）。仓库里常见的"直跑"脚本通常不处理测试入口里的环境变量分支（跳过 reference、配置集筛选、计时器归零），走错入口会静默跑出不属于本批的配置。
2. **采集 set 与导出 page 是两件事**。set 决定"报告里有哪些指标"，page 决定"CSV 排成什么形状"。用 detailed + 显式指标列表采的报告，无论如何导出都不可能有 full set 的列——这不是导出缺陷。具体 set/section 清单以工具当前版本为准。
3. **导 wide 指标表必须显式指定 raw page**。默认 page 是 section 视图（长表，一行一个 metric），只有 raw page 才是一 kernel 一行、指标一列的宽表。判据是**结构不是列数**：看到"每行都是 Metric Name / Metric Value"的长表 CSV，先怀疑 page 选错，而不是怀疑报告缺数据。
4. **列数不是契约，列名才是**。列数随采集 set 与工具版本变化，没有固定区间。取数一律按列名定位，禁止按列号下标。
5. **codegen 产物必须在采集当次落地**。JIT 缓存会被后续任务覆盖、临时目录会被清理；事后从哈希目录反查并逐份匹配的成本远高于当场打包。
6. **raw log 必须归档，不能只留终端输出**。失败 traceback、采集告警、被忽略的 flag 只存在于 raw log 里。
7. **旧数据只重命名、不删除**。口径升级后把旧导出标注归档后缀，并在 README 里写明映射关系。
8. **功能验证与 profile 分开跑**。reference 对比是可选的顺手项；一旦和 profile 混在同一次运行里，会污染时长与 launch 计数，也会让"这个 case 到底对不对"变得不可判读。
9. **单位与量纲先核对再比较**。ns / us / ms、pct-of-peak / 绝对值、`*_sum` / `*_avg` 同名不同义是最常见的错读来源；单位未知时降级声明，不按量级反推。
10. **工具参数不写死在文档里**。profiler / device-compiler / host-compiler / DSL CLI 的 flag 名与语义随版本变化，一律以当次 `--help` 输出和对应单端 skill 的要求为准。
11. **结论必须能追到"哪个文件、哪一列、什么值"**。追不到就降级为推断或假设。
12. **跨端只对账绝对量**。SOL% / 利用率% / IPC 永不跨端相减；说"饱和"前必须知道容量绝对值；codegen 产物是区分"源码差距"与"编译器差距"的唯一第一手证据。

## 常见失败信号速查

| 症状 | 先怀疑 |
|---|---|
| 报告只有 1 行，但 kernel 明显有多个阶段 | launch 控制没覆盖全部 launch（一个 prim_func 可能生成多个 device kernel） |
| 报告行数比预期多/少 | kernel filter 命中了辅助 kernel，或 JIT/graph 缓存改变了 launch 次数 |
| CSV 有值但取不到关键指标 | 导出 page 选成了 section 视图（长表结构），或采集 set 没采该 section |
| 同一报告两次导出内容不同 | 导出命令不同（page 不同）或用了 `--csv-file` 与重定向两种写法 |
| 时长与 bench 数字差一个量级 | 计时器/单位口径不同，或 profile 开启后 launch 被串行化 |
| 报告打不开 / `cat` 出乱码 | 报告是二进制格式，必须用工具 CLI 或 API 导出，不能当文本读 |
| 被 profile 的进程报计时器异常但报告完整 | 采集侧与计时器侧是两回事，先确认报告是否可用，再交给单端 skill 判断 |
| 文档里的文件名找不到 | 归档时改过名——查 README 的映射表与 manifest |

## 不做什么

- 不预置任何针对特定 kernel 的脚本：拿到本 skill 应能自行构造所需命令。遇到新 kernel 时，按 [references/kernel-and-launch-selection.md](references/kernel-and-launch-selection.md) 的方法重新推导，而不是"找一套现成脚本塞进去"。
- 不替代单端解读：单端指标怎么读（SOL、停等分类、端内饱和判断）由 `acu-report-skill` / `ncu-report-skill` 负责。跨端对比方法论在本 skill 内（[references/cross-end-analysis.md](references/cross-end-analysis.md)），但它同样要求两端先各自完成单端分析才会生效。
