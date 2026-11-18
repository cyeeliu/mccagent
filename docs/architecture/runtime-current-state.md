# Runtime 架构当前状态

> 最新 commit: `7429d0a` (2026-11-18)
> 迁移完成日期: 2026-11-18

## 1. 执行体系总览

迁移后存在 **统一的执行架构**，所有路径经 AgentExecutor → AgentRuntime：

| 路径 | 入口 | Executor | Runtime | 用途 |
|------|------|----------|---------|------|
| 主 Agent | `DefaultAgentFactory.build_interactive_runtime_v2()` | `AgentLoopExecutor` | `InteractiveRuntimeV2` | CLI/TUI 交互式对话 |
| 子代理 | `SubagentExecutor` | `SubagentExecutor` | `TaskRuntime` | 一次性短生命周期任务 |
| 团队成员 | `PersistentRuntimeV2.from_teammate()` | `AgentExecutor` | `PersistentRuntimeV2` | 常驻 teammate 循环执行 |

## 2. mcc-core 领域模型

```
mcc-core/src/mcc_core/runtime/
    __init__.py
    identifiers.py    — task_id / attempt_id 生成与校验
    enums.py          — RuntimeStatus / TaskStatus / RuntimeMode
    errors.py         — Runtime Error 体系
    task.py           — RuntimeTask 领域模型
    task_record.py    — TaskRecord 持久化模型
    task_store.py     — TaskStore Protocol
    handle.py         — TaskHandle
    result.py         — RuntimeResult / TokenUsage
    event.py          — RuntimeEvent 体系
    health.py         — RuntimeHealth
    capabilities.py   — RuntimeCapability
    definition.py     — AgentDefinition 统一模型
    protocols.py      — AgentRuntime Protocol
```

依赖：仅 `pydantic>=2.9`，无上层包依赖。

## 3. mcc-harness 执行层

```
mcc-harness/src/mcc_harness/execution/
    __init__.py
    protocols.py            — AgentExecutor Protocol
    agent_loop_executor.py  — AgentLoop → AgentExecutor 适配
    subagent_executor.py    — SubagentRunner → AgentExecutor 适配
    cancellation.py         — CancellationToken
    context.py              — ExecutionContext
    event_adapter.py        — EventAdapter
    factory.py              — Executor 工厂方法
```

`AgentLoopExecutor` 将 `AgentLoop` 封装为 `AgentExecutor` 接口，
Runtime 不再直接依赖 `AgentLoop` 类型。

## 4. mcc-harness Runtime 层

```
mcc-harness/src/mcc_harness/runtime/
    __init__.py
    base.py              — AgentRuntime Protocol re-export
    base_runtime.py      — BaseRuntime 公共实现
    interactive.py       — InteractiveRuntimeV2（包装 AgentLoopExecutor）
    persistent.py        — PersistentRuntimeV2（BoundedTaskQueue + WorkerPool）
    task_runtime.py      — TaskRuntime（一次性任务执行）
    bounded_task_queue.py — 有界任务队列
    worker_pool.py       — WorkerPool 并发执行
    coordinator.py       — RuntimeCoordinator 生命周期管理
    dispatcher.py        — TaskDispatcher 正式任务派发
    factory.py           — RuntimeFactory 工厂方法
    state_machine.py     — AgentStateMachine
    task_store_adapter.py — TaskStore 适配器
```

### 4.1 AgentRuntime Protocol

统一 Protocol 方法（`mcc_core/runtime/protocols.py`）：

| 方法 | 签名 | 说明 |
|------|------|------|
| `runtime_id` | `@property -> str` | 运行时唯一标识 |
| `agent_id` | `@property -> str` | Agent 标识 |
| `status` | `@property -> RuntimeStatus` | 当前生命周期状态 |
| `alive` | `@property -> bool` | 是否仍在运行 |
| `start` | `async (task, *, task_id) -> None` | 启动执行 |
| `stop` | `async (*, timeout) -> None` | 请求停止 |
| `cancel` | `() -> None` | 发送取消信号 |
| `wait_done` | `async (*, timeout) -> RuntimeResult` | 等待终态 |
| `submit` | `async (task) -> TaskHandle` | 提交任务 |
| `cancel_task` | `async (task_id, attempt_id, reason) -> None` | 取消任务 |
| `pause` / `resume` | `async () -> None` | 暂停/恢复 |
| `health_check` | `async () -> RuntimeHealth` | 健康检查 |
| `subscribe_events` | `() -> AsyncIterator[RuntimeEvent]` | 事件订阅 |

### 4.2 PersistentRuntimeV2

原生重写为 `BoundedTaskQueue + WorkerPool + AgentExecutor` 架构：
- `BoundedTaskQueue`：有界队列，背压控制
- `WorkerPool`：并发 Worker，每个持有独立 AgentExecutor
- `submit_task(task, *, task_id) -> str`：提交任务到队列
- `wait_task(task_id, *, timeout) -> RuntimeResult`：等待任务结果

通过 `from_teammate(teammate, bus)` 类方法提供向后兼容入口。

## 5. RuntimeCoordinator

Runtime 生命周期唯一管理中心：
- 事务注册：注册失败无半注册对象
- 退出监听：`on_runtime_finished` 回调
- 健康检查：`health_check_all()`
- 统一关闭：`shutdown_all(timeout)`

## 6. TaskDispatcher / TaskStore

- `TaskDispatcher`（`runtime/dispatcher.py`）：正式任务唯一派发入口
  - `dispatch(task, runtime) -> TaskHandle`
  - `wait_for_result(handle, timeout) -> RuntimeResult`
- `TaskStore`（`mcc_core/runtime/task_store.py`）：Task / Attempt / Result 唯一事实来源
  - task_id 与 attempt_id 分离
  - 重复提交幂等
  - Retry 不覆盖旧 Attempt

## 7. TeamOrchestrator

`team/orchestrator.py` 编排层：
- 依赖 Store / Scheduler / Dispatcher，不直接创建 AgentLoop
- 通过 `TaskDispatcher.dispatch()` 派发正式任务
- 无 dispatcher 时回退到 `runtime.submit_task/wait_task`（PersistentRuntimeV2 公开 API）
- 无 `hasattr` 兼容分支

## 8. TeammateManager

`teammate.py`：
- `spawn()` 创建 `TeammateRuntime` spec + `PersistentRuntimeV2.from_teammate()` 包装
- `TeammateRuntime` 作为 teammate 规格容器，执行委托给 PersistentRuntimeV2
- 消息派发使用 `kind="task_submitted"`（非 `kind="assignment"`）

## 9. CLI 装配

`mcc-cli/wiring.py` 作为 Composition Root：
- 通过 `DefaultAgentFactory` 创建 Runtime
- 通过 `RuntimeCoordinator` 注册 Runtime
- 显式构造 `AgentApplication` 并注入 `AgentEventSource`
- `AgentApplication` 作为团队操作统一入口

## 10. 已删除的旧架构

| 旧组件 | 删除 commit | 替代 |
|--------|------------|------|
| `EphemeralRuntime` | `4a0c5f0` | `TaskRuntime` |
| 旧 `InteractiveRuntime` | `f2c9125` | `InteractiveRuntimeV2` |
| 旧 `PersistentRuntime` | `f2c9125` | `PersistentRuntimeV2` |
| `hasattr(runtime, "submit_task")` 兼容分支 | `dbf120a` | `TaskDispatcher` 派发 |
| `kind="assignment"` 消息派发 | `07e342b` | `kind="task_submitted"` |
| V2 feature flag | `f2c9125` | V2 为唯一实现 |

## 11. 保留的适配层

| 适配层 | 说明 |
|--------|------|
| `AgentLoopExecutor` | 将 `AgentLoop` 封装为 `AgentExecutor`，InteractiveRuntimeV2 使用 |
| `SubagentExecutor` | 将 `SubagentRunner` 封装为 `AgentExecutor`，TaskRuntime 使用 |
| `TeammateRuntime` | Teammate 规格容器，执行委托给 PersistentRuntimeV2 |
| `SubagentRunner` | 子代理工具（`SpawnSubagentTool`）的执行引擎 |
