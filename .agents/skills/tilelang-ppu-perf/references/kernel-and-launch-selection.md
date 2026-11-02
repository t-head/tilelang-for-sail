# kernel 与 launch 选取

目的：一次采集只覆盖你想看的东西。报告里混入无关 kernel 或漏掉某个阶段，比采不到更糟——前者会污染统计，后者会让"这个 kernel 就这么点时间"变成错误结论。

## 1. 三个选取要素

| 要素 | 作用 | 怎么定 |
|---|---|---|
| kernel filter | 决定哪些 device 符号进报告 | 用能唯一锚定目标符号的写法，见 §2；flag 形式由单端 skill 定 |
| launch 控制 | 决定采到哪些 launch、跳过多少次 | 先 survey 真实 launch 序列再回填，见 §3；**不要用公式估算** |
| case / 入口 | 决定输入形状、规模、走哪个分支 | 用项目约定的测试入口；若项目用 pytest，`--collect-only -q` 打印真实 node id |

三者必须一起记录进 driver log 的 `CMD:` 行（连同工具版本），否则事后无法判断这份报告是"哪个 case 的哪次 launch"。

## 2. kernel filter 怎么写

1. **从实际符号名出发，不要凭源码函数名猜**。DSL 里的函数名（如 `foo`）与生成后的 device 符号名（如 `_foo_kernel`）通常不同，还可能带签名哈希后缀。获取途径：裸跑 log、codegen 产物、或先跑一次不加 filter 的采集看符号清单。
2. **锚定前缀 + 可选的阶段后缀**，例如 `_<stem>_kernel` 而不是裸 `<stem>`——裸词容易命中调用链上的辅助符号。
3. **注意 regex 引擎差异**：多数 profiler 的 kernel filter 是"包含匹配"而非全匹配，写 `<stem>` 会把 `<stem>_1`、`<stem>_helper` 一并纳入。需要精确时用 `^...$` 形式，并在第一次采集后核对报告行数。
4. **一次只采一个语义单元**。想对比两个 kernel，就采两份报告，不要放在一份里靠行号区分。

## 3. 三个概念必须分清：function / launch / replay pass

| 概念 | 定义 | 计数单位 | 与报告的关系 |
|---|---|---|---|
| **function**（device 符号） | 编译产物里的一个设备端函数 | 符号 | 报告按符号分组；一个 DSL kernel 可能生成**多个**符号 |
| **launch** | 一次 host→device 的 dispatch | 次 | 报告里的一"行"/一个实例，通常对应一次 launch |
| **replay pass** | profiler 为了收多组 section 把同一批 launch **重放**一遍 | pass | 会放大 wall time 与总执行计数，但**不是**新的业务 launch；数据按样本归并 |

关键推论：

- **一个 Python 层 kernel 常常生成多个 device function**。典型模式是同一 prim_func 里有两段 kernel 段（准备阶段 + 主体），编译后变成 `<stem>_kernel` 与 `<stem>_kernel_1`。若 launch 控制只放行 1 次，报告里只会出现第一个符号，第二个"凭空消失"。
- **launch 次数 ≠ function 数 × 阶段数**。没有任何可靠公式：分支、循环边界、graph replay、缓存命中都会改变真实序列。**唯一正确做法是 survey**（见下）。
- **replay pass 不是 launch**。看到"总耗时是单次的 N 倍"先分清是 N 次 launch 还是 N 个 replay pass：后者是采集开销，不能当性能数字。
- **warmup 与重复计时同样产生真实 launch**。计时器的 warmup 次数、`repeat` 次数都会被 profiler 看到。用 launch 控制圈定范围后要核对行数，不要假设"只跑了一次"。

### survey 流程（取代一切估算）

```
1) 先给一个足够大的 launch 控制值，采一次"survey 报告"（可只采轻量档，省时间）
2) 看报告里出现几个不同符号、每个符号几行（宽表去重计数，或用 details 的 kernel 列）
3) 把每个符号的真实 launch 次数记下来 → 回填精确的 launch 控制值
4) 用回填值正式采一次，核对行数与 survey 结果一致
```

survey 报告本身也要归档（它是"为什么用这个计数"的证据），并在 README 写明推导过程。

### 若项目使用 autotune / 配置搜索

- **profile 前把配置钉死**（指定 config 或关闭搜索），否则报告里会出现几十个不属于你的 launch。
- **必须保留候选选择结果的记录**（选了哪个 config、依据是什么、有没有 fallback），只把"最终 config"写进 README 是不够的——没有选择记录，事后无法判断报告对应哪个候选。
- **钉死后要用固定配置重放一次**，确认 launch 序列与预期一致（符号数、每符号 launch 次数、grid/block）。这一步是"报告确实采自该 config"的证据链，跳过它等于把 survey 结论当假设。
- 若项目不使用 autotune，此段跳过，但在 README 里写一句"无配置搜索"。

### 其他会改变 launch 次数的因素

- **JIT 编译缓存**：首次运行含编译，不产生额外 device launch，但会拉长 wall time（也会影响"为什么这次特别慢"的判断）。
- **CUDA graph / graph replay**：一次 replay 内部的多次 launch 是否被 profiler 逐个记录，取决于工具支持；不确定时改用 eager 路径采集，并在 README 声明。
- **多进程 / 多 rank**：每个进程都有自己的 launch 序列，profiler 可能只跟一个进程；分布式场景必须写明跟的是哪个 rank。

## 4. 采集前的自检清单

- [ ] 入口已确认为项目约定的测试入口（若项目用 pytest，node id 已由 `--collect-only` 确认，参数顺序无误）
- [ ] 裸跑 / 入口正常退出，bench 记录能正常打印
- [ ] kernel 符号名来自实际产物（不是猜）
- [ ] launch 控制值来自 survey 结果（含每个符号的次数），不是估算公式
- [ ] autotune / graph / 多进程都已在本次运行中钉死、关闭或显式声明
- [ ] 设备号显式指定，避免多卡环境下落到别的卡
- [ ] 记录 `CMD:` 行（含工具版本、入口、filter、launch 控制）到 driver log

## 5. 采完立刻核对（不通过就不要继续导出）

| 检查 | 判据 |
|---|---|
| 行数 | 宽表导出的数据行数 == survey 时记下的每个符号 launch 次数之和 |
| 符号数 | 与 codegen 产物中的符号逐个对得上，无多余项、无缺失项 |
| grid / block | 与源码中该 case 的期望配置一致（用来发现"命中了别的分支"或"命中了别的 config"） |
| 关键指标非空 | duration 与至少一组访存/指令计数有非零值；全零说明匹配到了空跑阶段 |
| 时长量级 | 与裸跑 bench 的耗时不差一个量级（差一个量级先查单位与 replay pass，再查 profile 是否导致串行化） |

任一项不符，回到 §2/§3 重新推导（重做 survey），不要在错误的报告上做分析。
