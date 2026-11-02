# 归档与 manifest

目的：一次 profile 任务的产出 = **数据 + 可复现口径**。只有数据没有口径，三个月后连自己都读不懂；只有口径没有数据，结论无法复核；只有校验和没有结构化元数据，交接时下游仍要重新问一遍。

## 1. 双写原则

| 位置 | 角色 | 要求 |
|---|---|---|
| 远端持久盘（挂载卷） | 长期主副本 | 不放容器临时层（`/tmp`、容器 writable layer 会被清理） |
| 本地任务目录 | 工作副本 + 分析入口 | 与远端逐份 SHA-256 对账（同一套 `MANIFEST.sha256`） |

两边都要有 manifest；分析一律在本地副本上做，远端只读，避免"最后一次改动只在一端"。

## 2. 目录骨架

```
<task_archive>/<date_or_id>_<scope>/
├── README.md                  # 人读口径：环境、版本、每个报告的 launch 参数、已知缺陷、修正历史
├── manifest.json              # 结构化元数据：workload/config/版本/provenance/报告↔codegen 映射
├── reports/                   # 原始报告本体（二进制），不加工、不改后缀
├── exports/                   # 导出视图（raw 宽表 / details 长表 / source 逐指令）
│   └── EXPORT_CMDS.txt        # 每条导出命令 -> 产物 的对应关系
├── logs/                      # raw log：采集日志、被测命令主日志、env_snapshot、CLI help 输出
├── codegen/                   # JIT 缓存打包 / save-temps / compile_cmds.log / codegen_manifest
├── scripts/                   # 本次用过的驱动与解析脚本（含编译器包装脚本）
└── MANIFEST.sha256            # 全量校验和（只算校验，不承载口径）
```

`<scope>` 是任务范围（如 `profile_supplement`、`v015_full`），`<date_or_id>` 用日期或任务号皆可——两种都行，同批内保持一致即可。

单 kernel 任务的简化版：把 `reports/`、`exports/`、`logs/` 按 kernel 名分子目录（`<kernel>/<tool>/`），其余保持顶层。

## 3. 命名约定

```
<tool>_<kernel>[_<version>][_<case>][_<page>].<ext>
```

| 片段 | 取值 | 说明 |
|---|---|---|
| `<tool>` | `acu` / `ncu` 等 | 一眼看出是哪个工具产出 |
| `<kernel>` | 语义名（不是符号名） | 与源码模块同名的可读标识 |
| `<version>` | 代码/DSL 版本前缀 | 同一 kernel 跨版本对比时必须带，否则两份文件无法并存 |
| `<case>` | 形状/配置关键参数 | 参数多了用缩写（维度、长度、batch 之类），并在 README 里展开 |
| `<page>` | `raw` / `source` / `pasm` / `sass` / 无 | 导出视图类型 |

规则：

- **同一 kernel 的多组 config 必须分开命名**，绝不允许"同一个文件名装着不同 launch 配置"。
- **旧口径产物不删，加后缀归档**（如 `_section`、`_legacy`、`_v1`），并在 README 里列改名映射。
- **不改原始报告后缀**：工具靠后缀识别二进制格式，改名后可能读不出。

## 4. 必备清单（缺一项就算本批不完整）

- [ ] 原始报告（每个 case 至少一份，二进制原样）
- [ ] 宽表 raw 导出（每份报告一份）
- [ ] 逐指令导出（仅对需要指令级分析的 kernel）
- [ ] 采集日志（profiler 自身 stdout/stderr）
- [ ] 被测命令主日志（测试入口输出）
- [ ] driver log（含 `CMD:` 行：工具版本、入口、filter、launch 控制、时间戳）
- [ ] 环境与版本指纹（`env_snapshot`：profiler / device-compiler / host-compiler / DSL 的版本与 `--help` 摘要）
- [ ] codegen 打包（device 侧生成源 + host 侧产物 + `params.pkl` + `compile_cmds.log`）
- [ ] 解析/统计脚本
- [ ] `manifest.json`（结构化元数据）
- [ ] `MANIFEST.sha256`（校验和）

## 5. manifest：两件东西，不要混

`MANIFEST.sha256` 只回答"文件有没有变"，`manifest.json` 回答"这堆文件是什么"。两者都要，缺任一项都不算可交接。

### 5.1 结构化元数据最小字段

每份报告一条记录，至少包含：

```
workload: <kernel 名 + 输入形状>
config: <grid/block/shared_mem>
tool_version: <profiler 版本>
compiler_version: <编译器版本>
dsl_version: <DSL 版本 + commit>
provenance: original | recovered | regenerated
#   original    = 采集当次直接保全的产物
#   recovered   = 事后从缓存/备份中找回的采集当次产物（未重新生成，但保全时机晚于采集）
#   regenerated = 用同 config 重新编译/采集生成的产物（不是采集当次那一份）
provenance_note: <recovered / regenerated 时说明找回或重编的原因与时间；original 时留空>
```

另外必须记录**报告 → 编译产物的映射**（哪份报告对应哪个 codegen 目录），否则事后无法证明"这份指标是这份源码编出来的"：

```
report: reports/<report_file>
codegen_dir: codegen/jit_cache/<hash_or_kernel>/
compile_cmd_log: codegen/compile_cmds.log#<line>
```

三个值的区别只有一条轴：**是不是采集当次那一份产物、以及什么时候被保全下来**。

- `original`：采集当次直接保全 → 可直接当"实测证据"。
- `recovered`：事后从缓存/备份里找回的**采集当次那一份**（未重新生成，只是保全晚了）→ 仍可当实测证据，但必须在 README 点名"保全晚于采集"与找回路径。
- `regenerated`：原件已不可得，用同 config 重新编译/采集出来的新产物 → **不能**与采集当次产物并列当"实测证据"，必须点名。

`regenerated` 的典型场景：原始报告丢了、只留下 CSV，事后用同一源码重编一份补产物；或 codegen 缓存被覆盖后重新捕获。`recovered` 的典型场景：归档漏拷，但 JIT 缓存/备份里那份还在，事后按 launch 特征唯一定位后拷出（定位方法见 [codegen-and-save-temps.md](codegen-and-save-temps.md) §6）。**晚找到不等于重新生成**——反查回来的原件标 `recovered`，不标 `regenerated`。

### 5.2 生成与校验

```bash
# 生成（在归档根目录）：分步执行，避免管道吞掉上游错误
# 注意：重定向会在 find 运行前就把 tmp 建出来，所以 tmp 自身必须排除，否则把自己算进表里
find . -type f ! -name 'MANIFEST.sha256' ! -name 'MANIFEST.sha256.tmp' ! -name '.DS_Store' \
  -exec sha256sum {} + > MANIFEST.sha256.tmp
rc=$?
[ $rc -eq 0 ] || { echo "sha256sum failed with rc=$rc" >&2; exit $rc; }
sort -k2 MANIFEST.sha256.tmp > MANIFEST.sha256
rm MANIFEST.sha256.tmp

# 校验：成败只看 sha256sum 的退出码，展示过滤是另一件事
sha256sum -c MANIFEST.sha256 > /tmp/verify.log 2>&1
rc=$?
grep -v ': OK$' /tmp/verify.log || true     # 失败项展示；无输出即全部通过
exit $rc                                    # 用 sha256sum 自己的退出码判定
```

**不要把管道当错误屏障**：`find … | sort > MANIFEST.sha256` 的退出码只反映最后一个命令，上游 `sha256sum` 失败（文件消失、权限不足）会被吞掉，产出一份"看起来完整"的残缺校验表。要么先落 tmp 再排序（上方写法），要么显式加 `set -o pipefail`——分步写法更容易看出是哪一步失败。

**不要把校验和展示过滤当成败判据**：全部通过时 `grep -v` 匹配不到任何行，会返回 1，看起来像失败。

### 5.3 跨机复核（覆盖整棵归档树）

两端都用 SHA-256，不要另搞一套摘要算法：

```bash
# 在每台机器上分别生成，再逐行 diff（同样分步，避免管道吞错）
(cd <dir> && find . -type f ! -name '.DS_Store' -exec sha256sum {} + > /tmp/sums_$(hostname).raw)
rc=$?
[ $rc -eq 0 ] || { echo "sha256sum failed with rc=$rc" >&2; exit $rc; }
sort -k2 /tmp/sums_$(hostname).raw > /tmp/sums_$(hostname).txt
rm /tmp/sums_$(hostname).raw
diff /tmp/sums_A.txt /tmp/sums_B.txt   # 不隐藏 stderr，退出码非 0 就是有不一致
```

要点：

- 用 `find . -type f` 覆盖**整棵树**，不要拿顶层 `*.csv` 之类 glob 当对账范围。
- **不隐藏 stderr**（不要 `2>/dev/null`）：找不到文件、权限不足正是需要看见的失败。
- 出现不一致时不要"取新的覆盖"，先查是传输截断还是内容真的不同。

## 6. README 必含字段

| 字段 | 内容 |
|---|---|
| 环境指纹 | 工具/编译器/DSL 版本、arch、设备编号 |
| 覆盖范围 | 本次覆盖了哪些 kernel / case，哪些没覆盖及原因 |
| 每份报告的口径 | 入口、kernel filter、launch 控制、grid/block、采集时间、对应 codegen 目录 |
| 起源说明 | 哪些是 `original` / `recovered` / `regenerated`（及原因）；recovered 另注明找回路径与时间 |
| 导出命令 | 用哪个 page 导的、列名覆盖面（不写死列数） |
| 已知缺陷 | 缺指标、降级证据、被排除的 case、单位未知项 |
| 修正历史 | 第 N 版结论被什么证据推翻（不要静默改写旧结论） |
| 改名映射 | 旧文件名 → 新文件名（归档重命名时） |

## 7. 归档时机（不要攒到最后）

| 时机 | 动作 |
|---|---|
| 采集完成 | 报告 + 采集日志 + `CMD:` 行立刻双写 |
| codegen 落地 | 产物一生成就复制到 `codegen/`（临时目录随时可能被清理），并记 `provenance` |
| 导出完成 | 宽表 + `EXPORT_CMDS.txt` + 更新 `manifest.json` |
| 分析开始前 | `MANIFEST.sha256` 生成、校验通过、跨机 diff 无差异 |
| 口径变更后 | 旧产物加后缀归档 + 更新 README 映射 + 重新生成 manifest |

## 8. 常见坑

| 坑 | 后果 | 规避 |
|---|---|---|
| 产物只留在容器临时目录 | 容器重启/清理即丢，任务白跑 | 一律落挂载盘 + 本地双写 |
| 只留 CSV 不留报告 | 无法重导其他 page，也无法验证 CSV | 报告本体永远比导出视图优先级高 |
| 只用终端回显不留 log | traceback 与告警只存在于 log 里 | stdout/stderr 各自重定向到文件 |
| 重命名后没记映射 | 文档里的旧路径全部失效 | README 维护一张改名表 |
| manifest 事后补 | 中途传输截断无法发现 | 传输完成即生成并校验 |
| manifest 只有校验和 | 交接时下游仍要逐项追问 workload/版本/配置 | 同时维护 `manifest.json` |
| `regenerated` 当采集当次证据用（或晚找回的 `recovered` 不声明） | 结论建立在事后重建的产物上，或把晚找回当成原始产物 | 用 `provenance` 三值标注，并在 README 点名 |
| 把 `grep -v ': OK$'` 的返回码当校验结果 | 全部通过时被判成失败 | 用 `sha256sum` 的退出码判定 |
| 跨机对账只查顶层文件 | 子目录损坏/截断被漏掉 | `find . -type f` 覆盖整棵树 |
| 归档目录里混入 `.DS_Store` 等噪声 | manifest 每次 diff 都变 | 生成时显式排除 |
