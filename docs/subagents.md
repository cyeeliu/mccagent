# 子代理（Subagents）

主代理通过 `spawn_subagent` 工具把一次性任务派给子代理：子代理在**独立上下文**中啃大量原始资料，只把精炼结论返还主代理——适合"读 100 个文件只为一个小答案"的场景，也用于并行 fan-out。

## 预设角色

| preset | 工具 | 迭代上限 | token 预算 | 超时 |
|---|---|---|---|---|
| `code_explorer` | read_file / list_dir / grep | 10 | 80k | 120s |
| `test_runner` | run_bash / read_file | 10 | 150k | 300s |
| `summarizer` | （无） | 1 | 30k | 30s |

预设模型由配置解析注入（见 [configuration.md](configuration.md#模型配置与解析)）：`subagent` 类型未配置时回落 `main`；`summarizer` 预设优先用 `summarize` 类型配置。典型用法——主对话 Opus、子代理 Haiku，大任务成本降数倍。

## 三重护栏

| 护栏 | 机制 |
|---|---|
| 超时 | `asyncio.wait_for` 强制截断，返回 `reason=timeout` 的错误结果而非异常 |
| token 预算 | `budget_tokens` 按模型定价折算为单次运行 USD 上限，复用主循环成本熔断 |
| 并发上限 | `asyncio.Semaphore`（默认 4），超出的派生排队等待 |

## 递归防护（双保险）

1. 子代理的工具注册表来自 `subset` 筛选，**天然不含 `spawn_subagent` 自身**——子代理无法再派生
2. 深度上限 `max_depth = 3`：`ToolContext.depth` 随派生递增，达到上限直接拒绝（第二道防线，防未来新增派生类工具时失效）

## 权限与上下文

- 子代理共享主代理的**权限检查器**：其内部每个工具调用照常过权限门控；deny 规则（如 `Read(.env)`）在子代理内同样生效
- 子代理没有交互确认通道，`Ask` 决策即拒绝（防无人值守滥用）
- session_id 派生为 `{parent}::{preset}`，depth = 父 + 1；pre/post-tool hooks 与 `on_subagent_stop` 生命周期钩子照常触发，HookContext 携带 depth 可区分层级

## Fan-out 用法

主代理单轮发出多个 `spawn_subagent` 调用，主循环的并行工具执行自动 fan-out。`commands/analyze-repo.json` 记录了教科书用例——4 个 `code_explorer` 并行分析仓库不同侧面（入口 / 核心模块 / I/O 层 / 测试组织），主代理收到 4 份摘要后综合成整体架构说明。

调用返回格式：

```
[subagent=code_explorer tokens=1523in/204out reason=end_turn]
主入口在 packages/mcc-cli/...，使用 click 解析参数…
```

`reason=timeout` 或失败时标记为错误结果，主代理可据此调整策略（换预设 / 拆小任务）。
