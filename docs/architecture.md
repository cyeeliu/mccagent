# 架构

## 包依赖

8 个 package 自底向上分层，上层只依赖下层：

```
          mcc-cli
             │
        ┌────┴────┐
        │         │
     mcc-tui   mcc-harness
                   │
       ┌───────────┼───────────┐
       │           │           │
   mcc-tools   mcc-session   mcc-llm
       │           │           │
       └───────────┴─────┬─────┘
                         │
                     mcc-core
                         │
                     mcc-config
```

| package | 职责 |
|---|---|
| `mcc-core` | 核心类型：`Message` / `ContentBlock` / `Tool` 协议 / `ToolContext` / `AgentEvent` 事件联合 / `AgentError` |
| `mcc-config` | `settings.json` 加载：home（`~/.mcc/`）与 project（`<cwd>/.mcc/`）逐字段合并 |
| `mcc-llm` | `LlmProvider` 协议 + `AnthropicClient`（SSE 流式）+ `OpenAiClient`（Chat Completions 兼容）+ `MockLlm` |
| `mcc-tools` | `ToolRegistry`（含 `subset` 筛选）+ 7 个文件/进程工具 |
| `mcc-session` | `SessionRecorder`（JSONL + meta）+ `MarkdownMemoryStore` + 记忆工具 |
| `mcc-harness` | `AgentLoop` 主循环 + `PermissionChecker` + Hook 机制 + 子代理 |
| `mcc-tui` | Textual 应用、按键读取、消息渲染 |
| `mcc-cli` | click 入口，装配全部组件 |

## 事件流

Agent 以事件（`AgentEvent`）驱动前端，前端只消费事件、不感知 Agent 内部：

```
用户输入 ──► UserEcho
              │
   ┌──────────┼──────────────────┐
   ▼          ▼                  ▼
TextDelta  ToolCallStart    PermissionRequest
(流式文本)  ToolCallEnd     (等待用户决策)
   │          │
   └──────────┤
              ▼
           TurnEnd(cost_usd)
```

`AgentLoop.run_streaming` 把事件推入 `asyncio.Queue`；`AgentEventSource`（mcc-cli/events.py）把 push 模型桥接为 pull 模型的 async generator，TUI 与 headless 共用这一接口。

## Agent 主循环

`AgentLoop.run_streaming(user_input, tx)` 每轮迭代：

1. 检查会话预算（`max_cost_usd` 超限则熔断）
2. 触发 `on_turn_start` hooks
3. 调 LLM（带指数退避重试，429/5xx/overloaded 可重试）
4. 按 stop_reason 分流：
   - `end_turn` — 录制快照、触发 `on_turn_end`、返回最终文本
   - `tool_use` — 并行执行工具调用（`asyncio.gather`），结果作为 tool 消息回填
5. 每轮录制 `TurnSnapshot`（含完整请求消息、assistant 块、工具输出、用量、成本）

多轮上下文：`history` 列表在每次运行结束时更新，TUI 连续对话天然具备上下文延续。

## 工具执行管线

每个工具调用依次经过：

```
权限检查 ──► pre-tool hooks ──► 执行 ──► post-tool hooks ──► 结果回填
   │              │(可阻止/改写输入)        │(可改写输出)
   └─ Deny: 直接返回错误
   └─ Ask:  发 PermissionRequest 事件等待用户决策
```

工具在 `ToolContext`（cwd / session_id / depth）中执行；同一轮的多个调用并行执行，单个失败不中断其余。

## 子代理

`SubagentRunner` 复用 `AgentLoop`：预设规格（system / model / 工具子集 / 护栏）→ 派生深度 +1 的上下文 → 独立消息历史 → 只把精炼结论返还主代理。详见 [subagents.md](subagents.md)。
