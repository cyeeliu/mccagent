# Runtime 架构当前状态

> 最新 commit: `e565ab6` (2026-11-20)
> 迁移完成日期: 2026-11-18
> P2 收尾日期: 2026-11-20
> Phase 7-8 深度修复: 2026-11-20（P1-11 状态副本消除 + P1-04/05/06 旧引擎拆除 + P1-12~15 生命周期 + P2-01~06 质量收尾）

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
- 状态机：`OPEN → CLOSING → CLOSED`（P1-12）
- 事务注册：`register_and_start` 锁内检查→锁外启动→锁内确认，失败回滚
- `begin_shutdown`：切换 CLOSING，禁止新注册（供 Application admission gate 调用）
- 健康监控：`RuntimeMonitorState` 去重，HEALTHY→UNHEALTHY 只触发一次故障回调（P1-13）
- 退出监听：`on_exit` 回调，Runtime 到达终态时触发
- 统一关闭：`shutdown_all(timeout)` 幂等，聚合异常，按优先级排序

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
- 不调用 `submit_task/wait_task`（Phase 6 已删除旧路径，AST 守卫验证）
- 不使用 `hasattr` 兼容分支（AST 守卫验证）
- 结果通过 `_aggregate_results` 从 TaskStore 聚合，不读 `run.results` 内存副本（P1-11）

## 8. TeammateManager

`teammate.py`：
- `spawn()` 创建 `TeammateRuntime` spec + `PersistentRuntimeV2.from_teammate()` 包装
- `TeammateRuntime` 作为 teammate 规格容器，执行委托给 PersistentRuntimeV2
- 消息派发使用 `kind="task_submitted"`（非 `kind="assignment"`）

## 9. CLI 装配

`mcc-cli/wiring.py` 作为 Composition Root：
- 通过 `DefaultAgentFactory` 创建 Runtime（P1-06：`create_runtime`/`create_executor` 从 AgentDefinition 装配）
- 通过 `RuntimeCoordinator` 注册 Runtime
- 显式构造 `AgentApplication` 并注入 `AgentEventSource`
- `AgentApplication` 作为团队操作统一入口
- Application 状态机：`CREATED → RUNNING → STOPPING → STOPPED`（P1-14）
- shutdown 8 步事务：admission gate → team manager → cancel all → stop runtimes → MCP → event source → clear terminal → report（P1-15）
- 并发 shutdown 共享同一 task，总 timeout 作为整体 deadline

## 10. 已删除的旧架构

| 旧组件 | 删除 commit | 替代 |
|--------|------------|------|
| `EphemeralRuntime` | `4a0c5f0` | `TaskRuntime` |
| 旧 `InteractiveRuntime` | `f2c9125` | `InteractiveRuntimeV2` |
| 旧 `PersistentRuntime` | `f2c9125` | `PersistentRuntimeV2` |
| `hasattr(runtime, "submit_task")` 兼容分支 | `dbf120a` | `TaskDispatcher` 派发 |
| `kind="assignment"` 消息派发 | `07e342b` | `kind="task_submitted"` |
| V2 feature flag | `f2c9125` | V2 为唯一实现 |
| `_TeammateExecutorAdapter` | Phase 7 | `PersistentRuntimeV2.from_executor` |
| `TeamOrchestrator.submit_task/wait_task` | Phase 6 | `TaskDispatcher.dispatch/wait_for_result` |
| `TeamRun.results` 独立可写状态副本 | Phase 7 (P1-11) | `_aggregate_results` 从 TaskStore 聚合 |

## 11. 保留的适配层

| 适配层 | 说明 |
|--------|------|
| `AgentLoopExecutor` | 将 `AgentLoop` 封装为 `AgentExecutor`，InteractiveRuntimeV2 使用 |
| `SubagentExecutor` | 优先使用 `agent_factory` 装配路径，`SubagentRunner.run` 仅作回退（P1-05） |
| `TeammateDefinition` / `TeammateMessageAdapter` / `TeammateExecutionService` | 从 TeammateRuntime 提取的职责分离组件（P1-04） |
| `SubagentRunner` | 子代理工具（`SpawnSubagentTool`）的执行引擎，SubagentExecutor 优先走 Factory 路径 |

## 12. P2 收尾完成状态

以下各项在 Phase 1-8 阶段已完成，文档据此校准：

| 项目 | 状态 | 完成阶段 | 说明 |
|------|------|----------|------|
| 统一 Protocol | 已完成 | Phase 1 | `AgentRuntime` / `AgentExecutor` / `AgentFactory` 统一契约 |
| Runtime 实现 | 已完成 | Phase 1 | `InteractiveRuntimeV2` / `TaskRuntime` / `PersistentRuntimeV2` |
| 旧执行引擎拆除 | 已完成 | Phase 7 | `_TeammateExecutorAdapter` 已删除，SubagentExecutor 优先走 Factory 路径（P1-04/05） |
| AgentFactory 完善 | 已完成 | Phase 7 | `create_runtime`/`create_executor` 从 AgentDefinition 装配（P1-06） |
| Attempt/Assignment 持久化 | 已完成 | Phase 1 | TaskStore 写入 Attempt 结果（P1-01） |
| TeamRun 状态恢复 | 已完成 | Phase 7 | `_aggregate_results` 从 TaskStore 聚合，不读内存副本（P1-11） |
| Dispatcher 结果闭环 | 已完成 | Phase 2 | TaskDispatcher 派发 → 执行 → 写入 Store → 返回结果（P0-03） |
| Coordinator 状态机 | 已完成 | Phase 8 | OPEN/CLOSING/CLOSED + 注册事务（P1-12） |
| 健康回调去重 | 已完成 | Phase 8 | RuntimeMonitorState 去重（P1-13） |
| Application admission gate | 已完成 | Phase 8 | STOPPING 状态 + begin_shutdown 第一步（P1-14） |
| shutdown 错误报告 | 已完成 | Phase 8 | ApplicationShutdownReport 8 步事务（P1-15） |
| RuntimeCapabilities 实例化 | 已完成 | Phase 8 | PersistentRuntimeV2 从实例配置生成 capabilities（P2-01） |
| AgentExecutor 协议统一 | 已完成 | Phase 8 | emit/cancellation 可选参数，所有 Runtime 统一传参（P2-02） |
| RuntimeTask 校验 | 已完成 | Phase 8 | timeout/时区/序列化/格式/长度/控制字符校验（P2-03） |
| RuntimeEvent 关联字段 | 已完成 | Phase 8 | correlation_id/causation_id/result_ref/schema_version/trace_id/parent_event_id（P2-04） |
| 架构 AST 规则 | 已完成 | Phase 8 | 禁止 submit_task/wait_task/hasattr 能力探测/AgentLoop 构造/反向依赖（P2-05） |
| 文档校准 | 已完成 | Phase 8 | 本文档与实际代码一致（P2-06） |

Phase 1-8 全部完成。生产 wiring 走正确路径，旧执行引擎已拆除，
状态副本已消除，生命周期状态机完整，架构 AST 守卫覆盖关键禁止模式。
