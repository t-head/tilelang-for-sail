---
name: "MR Bug 扫描"
description: "Bug Scan from Recent Commits —— 扫描最近提交，查找潜在 bug 并提出最小修复方案"
triggers:
  merge_request:
    types:
    - "opened"
engine:
  agent: "claude"
  model: "qwen3.8-max"
mcp: []
skills: []
permissions:
  repo: "WRITE"
  network:
    allow:
    - "https://docs.nvidia.com/cuda/parallel-thread-execution/"

---

扫描该 MR 中代码的问题。

## 输入

- 合入主干的提交列表（含 SHA、作者、改动文件、diff）
- 关联 CR 描述与评论
- 关联 CI run 的失败信号（如有）

## 输出

每条建议一段：

```text
## [<风险等级 H/M/L>] <一句话风险描述>

- 提交：<SHA前7位> | CR: !<id> | 文件:行：<path>:<line>
- 风险：<具体技术分析，2-3 句>
- 证据：<diff 片段 / 相关 CI 失败 / 相似历史 bug>
- 建议修复：<最小修改描述，含目标代码位置>
```

## 规则约束

- **只使用仓库中的具体证据**（提交 SHA、CR、文件路径、diff、失败的测试、CI 信号）
- **不要臆造 bug**；如果证据不足，请说明"证据不足，跳过"，不要硬给建议。聚焦改动的代码，不对 MR 里没有改动的、既存的代码评论。
- **优先选择最小且安全的修复**；避免重构和无关清理
- 不要为同一逻辑点重复出多条建议；按最具体的位置合并
- 风险等级：
  - H：逻辑错误 / 明显空指针 / 资源泄露 / 数据丢失 / 安全风险
  - M：边界条件未覆盖：但是仅 debug 生效的防御措施 assert、unreachable 等暂时认为是正确的，暂时不要求 release 版本有充足防御。
  - L：可读性 / 防御性编程建议
- 单次输出 ≤ 10 条；超出说明建议太多，请削减 “可读性 / 防御性编程建议” 。
- 函数、变量编码应使用 camelCase（小驼峰）。
- `hg-patched start`、`hg-patched end` 不是注释代码，是为了回溯历史修改留着的，不要管它。

**相关的额外说明**
- 如果修改了`.aoneci/script/`目录内的`ppu0010_skip.json`或者`ppu0015_skip.json`，`reason`的值不能为空字符串或者未说明原因的字符串，比如`"N/A"`,`"NA"`。此种情形，风险等级为H。
