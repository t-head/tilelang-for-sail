# 采集与导出

目的：让"报告里有什么"和"CSV 长什么样"两件事各自正确。这两件事由不同开关控制，混为一谈是本领域最高频的返工原因。

本文只记录**编排层的配置决策**：采什么档、限定什么、采几次、导出哪个 page、导出后怎么验。**具体命令怎么写由对应单端 skill 决定**（PPU 侧 `acu-report-skill`，NVIDIA 侧 `ncu-report-skill`）——flag 名、set 清单、section 覆盖面都随工具版本变化，不要照抄任何示例。

## 1. 前提：报告是二进制格式，不能当文本读

- `.acurep`、`.ncu-rep`（及 `.ncu-repz`）是**二进制容器**，不是文本。用编辑器打开、`cat`、`grep`、`head` 都不会得到可用信息（常常只看到魔数或压缩流）。
- 取数据的唯一途径是**对应工具的 CLI 或 API 导出**（CSV / 报告摘要 / 结构化输出）。导出后的 CSV 才是可 grep、可按列名解析的文本。
- 因此"报告打不开"几乎总是方法错，而不是文件坏了。判断文件是否损坏要看工具自身的报错与退出码。
- 报告后缀不要改（改名后工具可能拒绝识别），转换与提取一律通过导出完成。

## 2. 采集侧：先定配置决策，命令交给单端 skill

| 决策项 | 要定什么 | 为什么重要 |
|---|---|---|
| 采集档（set / section） | full？detailed + 指标列表？专题 set？ | 指标是否存在是**采集期**决定的，事后无法补 |
| kernel 过滤 | 精确符号还是正则；锚定前后缀 | 决定报告里混不混入无关 kernel |
| launch 控制 | 采哪几次 launch、是否跳过头部 | 决定行数与覆盖到哪些阶段 |
| 出错策略 | 采集异常时是否终止被 profile 进程 | 否则一次失败连带后半段数据全丢 |
| 输出 | 报告前缀、是否覆盖同名 | 报告名即归档键，重名会覆盖证据 |

**`--set full` 一类"全量档"不等于证据齐备**：

- "全量"只指**该工具版本定义的全量 section**，具体覆盖面因版本而异；新版本加/减 section 是常态。
- 有些 section 需要额外编译选项或权限才能采集，full 也不会替你打开。
- 实际覆盖面以**现场 `--list-sets` / `--list-sections` 输出**和**报告里实际存在的 section**为准，不以文档或惯例为准。
- 反过来，`detailed` + 显式指标列表采出来的报告，之后怎么导出都不会出现 full 的列——这不是导出缺陷。

采集命令的骨架（占位符必须由单端 skill / 当次 `--help` 落成实际拼写）：

```bash
<profiler> <set-or-sections-flags> \
  <kernel-filter-flag> <pattern> \
  <launch-control-flag> <n> \
  <on-error-policy-flag> <value> \
  <output-flag> <out_stem> \
  -- <被测命令，通常是项目约定的测试入口>
```

- `<被测命令>` 的 stdout/stderr 重定向到独立 log，不要和 profiler 自己的输出混在一份文件里。
- **完整命令行（含工具版本、入口、filter、launch 控制）必须原样写进 driver log 的 `CMD:` 行**。事后判断"这批数据是哪套配置"只能靠它。
- launch 控制语义见 [kernel-and-launch-selection.md](kernel-and-launch-selection.md) §3：`function` / `launch` / `replay pass` 三者不是一回事，不要用乘法估算。

## 3. 导出侧：page 决定 CSV 的形状

```bash
<profiler> -i <report> <export-flag> <page-flag> <page> > <out>.csv 2> <out>.err
```

| page | 形状 | 内容 | 用途 |
|---|---|---|---|
| `details`（默认） | **长表**（一行一个 metric） | section 视图：`ID/Process/Kernel/Section Name/Metric Name/Metric Unit/Metric Value/Rule ...`，**每行是一个指标** | 看 section 分组与 rule 结果；**不能直接当指标表用** |
| `raw` | **宽表**（一行一个 kernel 实例） | 已采集指标各占一列，**每行是一个 kernel 实例**，列名就是具体指标名 | 取数、跨报告对账、脚本解析的**唯一正确来源** |
| `source` | 逐指令 | `Range Id / Function Name / Address / Source / Instructions Executed ...` | 指令级分析：**profiler 视角的逐指令视图，带 PC 级指标列** |

两条等价关系（容易踩坑）：

- `--csv-file out.csv` 这类写法通常**等价于默认 page**（details，长表），不会给出宽表。
- 不带 page 参数的 CSV 导出同样是 details。

### source page 与 pasm/sass 不是一回事

| | source page 导出 | 纯反汇编文本（pasm / sass） |
|---|---|---|
| 来源 | profiler 对报告的再导出 | 编译器/反汇编器输出，或从二进制 dump |
| 内容 | 指令 + **PC 级指标列**（执行次数、停等采样等） | 只有指令文本（地址、opcode、操作数） |
| 能回答 | "哪条指令花了多少" | "生成了什么代码" |
| 不能回答 | 具体 opcode 语法细节 | 任何动态行为（执行了多少次） |

两者都要就各自归档、文件名用 `_source` / `_pasm` 区分，不要互相替代。

## 4. 导出后立刻做的六项校验

- [ ] `rc == 0`，stderr 单独落盘且为空（不要丢进 `/dev/null`）
- [ ] 结构与 page 匹配（**按结构特征判，不看列数**）：
  - 长表（details）特征：有 `Metric Name` / `Metric Value` 列，一行一个指标；
  - 宽表（raw）特征：没有 `Metric Name` 列，直接以具体指标名作列（`duration`、指令计数、交易计数等），一行一个 kernel 实例。
  - 判据：导出文件若呈现 `Metric Name / Metric Value …` 的长表结构 → page 选错。列数多少不作为判据：它随采集 set 与工具版本变化，没有固定区间（见 SKILL.md 铁律 4）
- [ ] 行数 == survey 结果里的 launch 实例数（多/少都要查 filter 与 launch 控制）
- [ ] 关键列存在：duration、至少一组指令计数、至少一组访存交易计数（按列名找，不按下标）
- [ ] 关键值非空、非全零（全零先查"是否命中了空跑阶段"和"该 section 是否采了"；怎么解读交给单端 skill）
- [ ] 同一报告用同一命令重复导出，结果逐字节一致（`sha256sum` 对账），证明导出过程确定性

导出命令与被导出报告名一起写进 `exports/EXPORT_CMDS.txt`：`<report> <page> -> <csv>`。README 里复述一次人读版本，并写明列名覆盖面（不写死列数）。

## 5. 单位与口径登记（只登记，不解读）

导出后要做的只是**把口径记下来**，判断"这个数值意味着什么"属于单端 skill。

- **时长**：ns / us / ms 混用极常见。宽表通常**不带单位行**，长表带 `Metric Unit`。单位来源要么能从导出行/报告正文确认，要么就写"未知"。
- **单位未知时降级声明，不按量级反推**。反推出来的单位会污染整列，且无法复核。
- **聚合口径**：`*.sum` / `*.avg` / `*.max` 语义不同，选哪一层要记下来，不同层的数字不要混算。
- **派生指标**（工具用 `derived_` 之类前缀算出来的）：与原始计数只选一层，别重复计入；选了哪层要写进 README。
- 报告里写数字必须带列名与单位来源，例如 `duration = <value> (<unit>)`，列 `<duration-column>`。

## 6. 生成逐指令视图（可选但推荐）

```bash
<profiler> -i <report> <export-flag> <page-flag> source > <out>_source.csv 2> <out>_source.err
```

- 逐指令视图体积可达 MB 级，只对需要做指令级分析的 kernel 生成。
- 与宽表同名同源归档，文件名加 `_source` 后缀；另存纯反汇编文本时用 `_pasm` / `_sass`。
- NVIDIA 侧对应命令与口径由 `ncu-report-skill` 负责，两侧 flag 不要混用。

## 7. 采集异常的分层判断

采集中出现的异常要分清层次，否则会把"工具自身问题"误判成"数据作废"：

| 层次 | 现象 | 判断与处置 |
|---|---|---|
| 采集侧 | 报告文件未生成、体积异常小、导出报错 | 真失败：查 filter、权限、驱动、是否 kill 了目标进程 |
| 报告内容 | section 缺失、指标为 0 | 先查采集档与编译要求，属"证据不全"而非"采集失败" |
| 计时器/框架侧 | 被测进程抛异常（如带宽类 `ZeroDivisionError`、计时返回 0），但报告数据完整 | **与采集侧分开看**：报告通常仍可用，先导出验证，再由单端 skill 判断是否可用 |
| 权限/编译要求 | 某些计数器需要管理员权限或特定编译选项 | 交给对应单端 skill 确认，不要自己开权限或改编译参数硬试 |

**profiler 与框架内置的 CUPTI / 性能计数器可能冲突**（同一进程内两套采集器抢硬件计数器），典型表现是被测进程抛异常甚至退出，而 `.acurep` / `.ncu-rep` 里的数据依然完整。遇到这类现象：

1. 先确认报告已生成且能正常导出（这是"数据可用"的判据）；
2. 把"计时器异常"与"采集失败"分开记录到 README 的已知缺陷；
3. 需要功能/性能计时数字时，另跑一次不开 profiler 的运行取数（不要在同一次运行里既要报告又要干净的计时）。
