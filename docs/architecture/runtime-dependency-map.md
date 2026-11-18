# Runtime 依赖地图

> 基线 commit: `69ded02` (2026-11-16)
> **注意：此文档为 Phase 0 基线快照，迁移后状态见 [runtime-current-state.md](runtime-current-state.md)**

## 1. 包依赖关系（当前）

```
mcc-cli
  ├── mcc-harness (agent, runtime, teammate, subagent, team)
  ├── mcc-tui
  ├── mcc-tools (task.py: TaskStore)
  ├── mcc-session
  ├── mcc-config
  ├── mcc-observability
  └── mcc-core

mcc-harness
  ├── mcc-core
  ├── mcc-tools
  ├── mcc-session
  ├── mcc-config
  ├── mcc-observability
  ├── mcc-llm
  ├── mcc-mcp
  └── mcc-skills

mcc-tools
  └── mcc-core

mcc-core
  └── (pydantic)
```

**问题**：mcc-harness 同时依赖 mcc-tools，而 mcc-tools 包含 TaskStore。
方案要求 TaskStore 领域模型下沉到 mcc-core，工具包不成为 Orchestrator 底层依赖。

## 2. 执行类依赖图

```
AgentLoop (agent.py)
  ├── LLM Client (mcc-llm)
  ├── ToolRegistry (mcc-tools)
  ├── PermissionEngine
  ├── HookEngine
  ├── BudgetManager
  ├── ContextManager / SessionStore (mcc-session)
  ├── GoalController
  ├── TodoService
  └── SessionRecorder

SubagentRunner (subagent.py)
  └── AgentLoop (内部构造)

TeammateRuntime (teammate.py)
  ├── AgentLoop (内部构造)
  ├── MessageBus
  └── TeammateManager

InteractiveRuntime (runtime/interactive.py)
  └── AgentLoop (包装)

EphemeralRuntime (runtime/ephemeral.py)
  └── SubagentRunner (包装)

PersistentRuntime (runtime/persistent.py)
  ├── TeammateRuntime (包装)
  └── MessageBus

RuntimeFactory (runtime/factory.py)
  ├── InteractiveRuntime
  ├── EphemeralRuntime
  └── PersistentRuntime

RuntimeCoordinator (runtime/coordinator.py)
  └── AgentRuntime (Protocol)

TeamOrchestrator (team/orchestrator.py)
  ├── RuntimeCoordinator
  ├── TaskScheduler
  ├── AgentRegistry
  ├── MessageBus
  └── PersistentRuntime (通过 hasattr 适配)

TeammateManager (teammate.py)
  ├── TeammateRuntime
  ├── PersistentRuntime
  ├── RuntimeCoordinator
  └── AgentRegistry

CLI wiring.py
  ├── AgentLoopBuilder
  ├── SubagentRunner
  ├── TeammateManager
  ├── RuntimeFactory
  └── RuntimeCoordinator
```

## 3. 目标依赖关系（迁移后）

```
mcc-core/runtime/         — 领域模型（零高层依赖）
  └── (pydantic)

mcc-harness/execution/    — AgentExecutor
  ├── mcc-core/runtime/
  └── AgentLoop (内部委托)

mcc-harness/factory/      — AgentFactory / RuntimeFactory
  ├── mcc-core/runtime/
  ├── mcc-harness/execution/
  └── mcc-llm, mcc-tools, ...

mcc-harness/runtime/      — Runtime 实现
  ├── mcc-core/runtime/
  └── mcc-harness/execution/

mcc-harness/task/         — TaskDispatcher / TaskStore
  ├── mcc-core/runtime/
  └── mcc-harness/runtime/

mcc-harness/team/         — TeamOrchestrator
  ├── mcc-harness/task/
  └── mcc-harness/runtime/ (仅 Protocol)
```

## 4. 禁止依赖（目标架构）

| From | To | 原因 |
|------|----|------|
| mcc-core | mcc-harness / mcc-cli / mcc-tools | 核心层不依赖高层 |
| mcc-cli | AgentLoop / SubagentRunner / TeammateRuntime | CLI 不直接构造执行对象 |
| TeamOrchestrator | AgentLoop / LLM Provider | 编排器不执行 |
| AgentRuntime | AgentLoop (直接构造) | Runtime 通过 Executor 间接使用 |

## 5. 关键循环依赖风险

当前无循环依赖，但以下重构路径需注意：

- `RuntimeCoordinator` 注销需通知 `TeamOrchestrator` 解绑 → 由 `ApplicationService` 编排，避免直接依赖
- `TaskDispatcher` 需选 Runtime → 依赖 `RuntimeCoordinator`（单向，安全）
- `AgentFactory` 需创建 Runtime → 依赖 `RuntimeFactory`（单向，安全）
