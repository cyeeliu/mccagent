# Workflow 运行时

workflow 是代码内注册的可信编排脚本（非模型自由编排），模型通过
`workflow` 工具按名调用并传 args。脚本拿到 ExecutionState 注入的
编排原语：agent() 子代理调用（journal 语义缓存支持断点续跑）、
parallel() 屏障并发、pipeline() 无屏障流水线、workflow() 单层嵌套。

## 架构概览

```
模型调用 workflow(name, args, resume_from_run_id)
 └── WorkflowTool.call
      ├── run_id 预留/校验 + run 级 asyncio.Lock 串行
      ├── WorkflowJournal（.workflows/<runId>.journal.jsonl）
      ├── ExecutionState(agent/parallel/pipeline/workflow/phase/log)
      │    └── agent() → SubagentWorkflowRunner → SubagentRunner
      │         ├── journal key 命中 → 直接返回缓存（resume 短路）
      │         └── 未命中 → 子代理执行 → schema 校验 → record
      └── 快照 <runId>.json + 输出 <runId>.output.json + last_run.txt
```

## 使用方式

### 模型侧调用

```
> 用 review-changes 审查这段代码的改动
[agent 调用 workflow]
  workflow(name="review-changes", args={"changes": "<diff>"})
  → {"launched": {"runId": "wf_review-changes_ab12cd34", ...},
     "result": {"confirmed": [...]}, "task": {...}}
```

### 注册新 workflow（代码内）

```python
from mcc_harness import register_workflow

async def my_flow(ctx, args):
    ctx.phase("Collect")
    items = await ctx.parallel([
        lambda: ctx.agent(f"fetch {x}", label=f"fetch:{x}")
        for x in args["urls"]
    ])
    return {"items": items}

register_workflow(
    {"name": "my-flow", "description": "demo", "phases": ["Collect"]},
    my_flow,
)
```

### 编排原语

| 原语 | 语义 |
|------|------|
| `ctx.agent(prompt, schema, label, phase)` | 子代理调用；schema 模式强制 JSON 输出 + 校验（失败重试一次）；journal 缓存命中则短路 |
| `ctx.parallel(thunks)` | 屏障并发，任一失败整体失败 |
| `ctx.pipeline(items, *stages)` | 无屏障流水线，item 独立穿阶段（A 在阶段 3 时 B 可在阶段 1） |
| `ctx.workflow(name, args)` | 嵌套已注册 workflow（仅一层），共享 journal/budget/limits |
| `ctx.phase(title)` / `ctx.log(msg)` | 进度事件（TUI 实时可见，重名阶段不重复广播） |

### 断点续跑

失败的 run 可用 `resume_from_run_id` 恢复：journal 中已完成 agent()
的结果按语义 key（`kind|label|prompt|schema` 哈希）命中缓存直接返回，
只重跑失败点之后的部分。resume 时 args 必须与原始 run 一致（不传则
沿用保存的 args）。

## 三重护栏

1. **Budget**：`args.budget`（或配置默认）限制 run 级 token 总量，
   超额 agent() 抛错
2. **ExecutionLimits**：`concurrency` 并发信号量 + `agent_cap` 单次
   run 子代理调用总数上限（嵌套 workflow 共享计数）
3. **Schema 校验**：结构化输出经 SimpleJsonSchema（type/required/
   properties/items/enum 子集）校验，失败自动重试一次再抛错

## 持久化产物（`.workflows/`）

| 文件 | 内容 |
|------|------|
| `<runId>.json` | run 快照（name/args/task 状态与进度） |
| `<runId>.output.json` | workflow 返回值 |
| `<runId>.journal.jsonl` | append-only agent 结果缓存（resume 依据） |
| `last_run.txt` | 最近一次 run id |

`.workflows/` 已 gitignore。

## 配置

```json
{
  "workflow": {
    "enabled": true,
    "concurrency": 4,
    "agent_cap": 20,
    "budget_tokens": null
  }
}
```

| 字段 | 默认 | 说明 |
|------|------|------|
| `enabled` | `true` | 关闭时不注册 workflow 工具 |
| `concurrency` | `4` | run 级并发信号量 |
| `agent_cap` | `20` | 单次 run 子代理调用总数上限 |
| `budget_tokens` | `null` | 默认 token 预算（args.budget 可覆盖） |

## 内置 workflow

`review-changes`：对 args.changes 按 correctness/security/performance/style
四维度流水线审查（audit → 逐发现对抗式 verify），仅保留被确认的发现，
按 severity 排序返回。

## 与后台任务 / Cron 的区别

| 特性 | Workflow | 后台任务 | Cron |
|------|----------|----------|------|
| 编排内容 | 多步 agent 调用（代码） | 单条 shell 命令 | 定时 prompt 注入 |
| 触发方 | 模型按名调用 | 模型主动发起 | 时间表驱动 |
| 断点续跑 | journal 语义缓存 | 无 | durable 持久化 |
| 护栏 | budget/并发/agent_cap | 超时/并发 | 无 |
