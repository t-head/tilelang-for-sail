# codegen 产物保全与 save-temps

目的：拿到编译器实际吐出的第一手证据。报告里的指令数、地址模式、寄存器用量都是"编译器选择的结果"，只有对照生成源才能区分"DSL 源码写的"与"后端编译器自己选的"。

## 1. 为什么要做（不做的代价）

- JIT 缓存目录里的生成源（device 侧生成源 / host 侧 wrapper / 参数序列化文件 / `executable.so`）是**唯一能解释报告差异的静态证据**。
- 这些目录默认带哈希名、默认位置在用户缓存或 `/tmp` 下：**后续任务会覆盖同名缓存、临时目录会被清理**。采集当天不打包，事后只能靠 grid/block + 签名特征去猜哪一份是哪个 kernel，成本极高且容易错配。
- 报告解读出现分歧时（例如"为什么这条访存变成了 4 段"），优先读生成源，不要从汇编反推编译器行为。

## 2. 产物分层与用途（device 与 host 同等重要）

| 层 | 典型产物 | 用途 |
|---|---|---|
| DSL 生成源（device 侧） | `<device_kernel>.cu`、`.hg`、`.cl` 等按 DSL 而定 | 确认元素→线程映射、向量化宽度、循环展开、内存布局；解释地址模式 |
| host 侧产物 | `host_kernel.cu` / wrapper、`executable.so` | 确认 launch 配置来源、参数传递、shared_mem 申请方式；**缺失时无法证明 codegen 完整** |
| 中间表示 | `.ptx`、`.ll`（若可得） | 确认内建函数下推、地址计算形式 |
| 机器码/反汇编 | 逐指令导出（source page）、纯反汇编文本 | 指令级核对，见 [collection-and-export.md](collection-and-export.md) |

**"只拿到了 device 生成源"不算保全完成**：host 侧产物决定了 launch 参数与参数搬运方式，缺它就无法解释"grid/block 为什么是这个值"。

## 3. 编译配置基线

采集与 codegen 必须是**同一次、同一套参数**编出来的。基线要求：

- 优化级别与正式运行一致（optimized build，不要为了"方便读"降到 `-O0`——优化级别一变，指令数与向量化全变）。
- 打开调试信息以便把指令对回源码，例如 `<lineinfo-flag>`（`-lineinfo` 一类）或 `<debug-flag>`（`-G0 -g1` 一类，视工具而定）。**具体 flag 名以当次 `--help` 和对应单端 skill 的要求为准**，不同编译器版本叫法不同。
- 目标 arch、宏定义、fast-math 类开关如实记录。

编译相关的信息与权限问题（哪些 section 需要什么编译选项、是否要管理员权限）由对应单端 skill 确认，本文只要求"编译配置已记录、可复现"。

## 4. 把生成源留在手里的五种做法（可组合）

### 4.1 重定向 JIT 缓存目录（首选）

```bash
export <DSL_CACHE_ENV>=<task_archive>/codegen/jit_cache/<config_tag>
```

- 具体变量名按 DSL 查（不同 DSL/版本命名不同，例如 `TILELANG_CACHE_DIR`、`TRITON_CACHE_DIR` 等），不要照抄别的 DSL 的变量名。
- **每个 config 用独立目录**：不同 grid/block/shape 编出的符号可能同名但内容不同，共用目录会互相覆盖，事后无法判断哪份对应哪个 case。
- 缓存目录在容器内必须可见（挂载盘），否则还是落在容器临时层。

### 4.2 确保真的发生了编译（cache miss 验证）

缓存命中意味着"这次没有新产物"，你会拿到上一轮的遗留文件，却不自知。做法：

1. 每个 config 用**新建的空目录**，先确认目录为空（`ls -A` 无输出）。
2. 跑一次该 config（或直接走到 compile-only 停止点，见 §5）。
3. 采集后立刻检查该目录**从空变为有内容**（至少出现生成源与 host 产物）。
4. 目录仍为空 → 命中了别处缓存或编译被跳过：查 DSL 的缓存开关、是否设了禁用重编的环境变量、`codegen_manifest.json` 里记一笔"未触发编译"。

### 4.3 拦截编译器调用，落 save-temps

想知道"这份报告到底是哪条编译命令行产生的"时，用 PATH 前置一个包装脚本替换真实编译器。包装脚本要做的事：

```
1) 记录完整调用信息到 <task_archive>/codegen/compile_cmds.log：
   timestamp / argv（逐参数原样）/ cwd / 目标 arch / 优化级别 / 宏定义
2) 把子进程的 stdout、stderr 各自落盘，并记录 return code
3) 追加 <save-temps-flag>（等价的"保留中间文件"开关，见下）
4) exec <real-compiler> "$@"
```

`<save-temps-flag>` 必须按当次 CLI help 确定，不要硬编码：

| 编译器 | 常见形式 | 说明 |
|---|---|---|
| `hgcc` | `--keep --keep-dir=<dir>` | 以 `--help` 实际输出为准 |
| `nvcc` | `--keep --keep-dir=<dir>` | 同上 |
| 其他 / 版本不确定 | 查 `--help` 里的 "keep / save-temps / intermediate" 相关条目 | 参数名与语义随版本变化 |

**不同编译器版本的同名开关语义可能不同**（有的写 `-save-temps=obj`，有的只认 `--keep`，有的需要配套目录开关）——一律以当次 `--help` 输出为准；拿不准就先只录制不注入开关（此时靠 §4.1 的缓存目录拿产物）。

要点：

- **只加录制，不加语义改动**：任何会改变优化级别或语义的参数都会让产物与正式报告不一致。
- 包装脚本本身要归档（它是"数据怎么来的"证据）。
- `compile_cmds.log` 里必须能看到目标 arch、优化级别、宏定义、完整 `argv` 与 `cwd`——这四项是复现编译的前提。
- 编译失败时同样要记 return code 与 stderr：失败的那条命令行往往正是解释"报告为什么少一个 kernel"的关键。

### 4.4 直接向 DSL 要源码文本

部分 DSL 提供 `get_kernel_source()` / `print_source()` 之类接口，可在驱动脚本里把生成源 dump 到归档目录。优点是**不依赖缓存路径约定**；缺点是通常拿不到 host 侧与参数序列化文件。可与 §4.1 组合使用。

### 4.5 临时文件会被 callback 删掉：必须在 `finally` 清理前复制

某些 DSL（如 TileLang）在编译 callback 的收尾逻辑里（常在 `finally` 块）删除中间临时文件（`.hg` / `.hgbin` / 临时目录）。**`finally` 在 callback 返回之前执行**，所以"等 callback 返回后再复制"拿到的一定是已删除的路径——脚本不报错，归档里只有一个空目录。

- 复制必须发生在 `finally` 清理**之前**，两种可行位置：
  - callback 内部、`finally` 块执行之前完成复制；
  - 或在 `finally` 块内、清理语句（`unlink` / `rmtree` / `rm`）**之前**插入复制逻辑。
- **不允许"callback 返回后再复制"**：那时文件已不存在，只能得到空归档。
- 复制到 `codegen/` 归档目录后，要校验文件数与非零体积（与 §4.2 的"空目录 → 有内容"两步验证同一套判据）。
- 复制失败要显式报错，不要静默吞掉——"什么都没留下"和"留下了空目录"必须可区分。
- 归档目录必须是持久盘，且复制目标不要放在同一临时目录树下。

## 5. compile-only 停止点：只保全产物，不执行 kernel

有些任务只需要"编译器生成了什么"，不需要跑通。此时把流程停在**确定性的一步**：

```
device 编译完成 → host 侧产物导出完成 → 缓存/产物落盘 → 【停止】
                                                      ↑
                                            不进入 runtime launch
```

- 停止点必须在 runtime launch **之前**：这样既拿到了产物，又避免了设备被占、OOM、计时器异常等运行时噪声。
- 停止点之后要立刻做 §4.2 的 cache miss 验证 + 产物清单核对，否则不知道自己停在了哪。
- 若 DSL 只能通过"跑一次"触发编译，那就跑到首个编译完成即中断（或只跑最小 shape），并在 README 记明"这次跑过 runtime"。

**capture ≠ correctness**

compile-only 采集到的产物只证明"编译器生成了什么"，**不证明"kernel 跑得对不对"**。正确性验证是独立步骤（独立一次运行、独立判据），不能从产物存在推导出功能正确，也不能从"跑通了"推导出"这份产物就是报告对应的那份"。

## 6. 事后反查：用 launch 特征唯一定位

如果生成源没能当场保全，按下面顺序把范围缩到唯一一份：

1. **用例签名**：dtype、元素数、是否 dynamic shape、常量维度。
2. **launch 配置**：grid/block 三元组（报告里就有，且对每个 case 基本唯一）。
3. **符号名与哈希后缀**：从报告里抄 kernel 符号名，比对缓存目录内的符号定义。
4. **源码特征串**：`grep` 该 kernel 独有的常量/宏/注释片段。
5. 仍有多份时，按文件 mtime 与采集时间窗口过滤。

定位完成后把"报告 ↔ 生成源"的对应关系写进 `manifest.json` 与任务 README，避免下一轮再猜一次。

**事后反查到的产物要分清 `recovered` 与 `regenerated`**（完整定义见 §7）：若找回来的就是采集当次生成的那一份（缓存/备份里仍活着、逐份校验一致）→ 标 `recovered`，它没有被重新生成；只有在原件已不可得、必须用同 config 重新编译捕获时才标 `regenerated`。二者只有 `recovered` 能当采集当次证据使用。

## 7. 归档形态

```
codegen/
├── jit_cache/<config_tag>/          # 或按 kernel 分子目录
│   └── <hash>/                      # device 生成源 / host 产物 / 参数文件 / executable.so
├── compile_cmds.log                 # §4.3 的录制结果（含 argv/cwd/stderr/rc）
├── compiler_wrapper.sh              # §4.3 的包装脚本本体
├── codegen_manifest.json            # 结构化清单（见下）
└── codegen_manifest.sha256          # 上述文件的校验和
```

`codegen_manifest.json` 至少记录：

```
kernel: <语义名>
symbol: <device 符号名>
config: <grid/block/shared_mem>
files: [<相对路径>, ...]                       # 含 host 侧产物
build: <优化级别 + 调试信息 + arch + 关键宏>
compiler_version: <device/host 编译器版本>
provenance: original | recovered | regenerated
#   original    = 采集当次直接保全的产物
#   recovered   = 事后从缓存/备份中找回的采集当次产物（未重新生成，但保全时机晚于采集）
#   regenerated = 用同 config 重新编译/采集生成的产物（不是采集当次那一份）
provenance_note: <recovered / regenerated 时说明找回或重编的原因与时间>
compile_rc: <0 | 非 0>
```

- 打包成 `codegen_<kernel>.tar.gz` 与报告并列存放；体积大时至少保留 device 生成源 + host 侧产物 + 参数文件 + `compile_cmds.log`。
- manifest 里列全路径，事后可 `sha256sum -c` 复核（判据用 `sha256sum` 的退出码）。

## 8. 常见坑

| 坑 | 后果 | 规避 |
|---|---|---|
| 只用默认缓存目录 | 被后续任务覆盖，生成源丢失 | §4.1 重定向到归档目录 |
| 多 config 共用一个缓存目录 | 符号同名互相覆盖，无法判断归属 | 每 config 独立空目录 |
| 没验证 cache miss | 拿到上一轮遗留文件，误当归次产物 | §4.2 空目录 → 有内容 两步验证 |
| 等脚本结束才去复制产物 | `finally` 在 callback 返回前已删掉临时文件，所以"紧跟返回后复制"同样只能拿到空路径 | §4.5：在 `finally` 清理前复制（callback 内，或 `finally` 内清理语句之前） |
| 只记参数不记 stdout/stderr/rc | 编译告警与被忽略的选项全部丢失 | 各自落盘并记 return code |
| 包装脚本顺手加了 `-O0`/禁用向量化 | 产物与报告不一致，结论全废 | 只录制不改参数；调试信息用 lineinfo 而非降优化 |
| 硬编码 `-save-temps=obj` 之类写法 | 换个编译器版本直接报错或静默无效 | 以当次 `--help` 为准 |
| 只留 `executable.so` 或只留 device 生成源 | 无法解释指令选择或 launch 配置来源 | device 与 host 产物都要留 |
| 缓存目录在容器临时层 | 容器重启即丢 | 必须落在挂载盘 |
| 直接改生成文件后缀（如 `.cu` → `.txt`） | 某些工具按后缀识别，读不出 | 保持后缀，只做打包 |
| 把 compile-only 产物当"跑通了" | 正确性从未被验证 | 声明 capture ≠ correctness |
