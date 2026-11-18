# Runtime 迁移矩阵

> 基线 commit: `69ded02` (2026-11-16)
> 完成 commit: `7429d0a` (2026-11-18)
> 方案文档: `MCCAgent_AgentRuntime_迁移完善详细执行方案.md`

## 1. 模块迁移映射

| 现有模块 | 目标归属 | 动作 | 最终处理 | Phase | 状态 |
|----------|---------|------|---------|-------|------|
| `agent.py: AgentLoop` | AgentExecutor 内部 | 封装为 AgentLoopExecutor | Runtime 不直接依赖 | 2 | ✅ 完成 |
| `runtime/base.py` | mcc-core/runtime/ | 重构统一 Protocol | 保留 re-export | 1 | ✅ 完成 |
| `runtime/factory.py` | RuntimeFactory | 只构造 Runtime | 保留 | 3,11 | ✅ 完成 |
| `runtime/coordinator.py` | RuntimeCoordinator | 加强事务和退出监听 | 保留 | 5 | ✅ 完成 |
| `runtime/interactive.py` | InteractiveRuntimeV2 | 改用 AgentExecutor | 重写 | 4,16 | ✅ 完成 |
| `runtime/ephemeral.py` | TaskRuntime | 移除 SubagentRunner 依赖 | 删除，替代为 TaskRuntime | 4,13,16 | ✅ 完成 |
| `runtime/persistent.py` | PersistentRuntimeV2 | 原生重写 BoundedTaskQueue+WorkerPool | 重写 | 4,18 | ✅ 完成 |
| `subagent.py` | SubagentExecutor 适配 | 迁移执行逻辑 | 保留为工具执行引擎 | 2,13 | ✅ 完成 |
| `teammate.py: TeammateRuntime` | PersistentRuntimeV2 Worker/Queue | 执行委托给 PersistentRuntimeV2 | 保留为规格容器 | 4,14,18 | ✅ 完成 |
| `teammate.py: TeammateManager` | AgentRegistry / AgentApplication | 分离成员与 Runtime 管理 | spawn 委托 from_teammate | 7,8,14,20 | ✅ 完成 |
| `team/orchestrator.py` | TeamOrchestrator | 依赖 Store/Scheduler/Dispatcher | 重构 | 7,19 | ✅ 完成 |
| `team/scheduler.py` | TaskScheduler | 支持 DAG/多任务/并发 | 重构 | 7 | ✅ 完成 |
| `registry.py` | AgentRegistry | 管理定义和能力 | 保留 | — | ✅ 完成 |
| `mailbox.py` | MessageBus | 只负责消息通信 | kind 改为 task_submitted | 7,9 | ✅ 完成 |
| `mcc_tools/task.py` | TaskStore + 工具适配器 | 抽离领域和存储协议 | 拆分到 mcc-core | 6 | ✅ 完成 |
| `mcc_cli/wiring.py` | CompositionRoot | 只负责应用装配 | 简化 | 8,20 | ✅ 完成 |
| `mcc_cli/events.py` | CLI Event Adapter | 接受 application 参数 | 重构 | 8,20 | ✅ 完成 |
| `mcc_cli/slash.py` | CLI Command Adapter | 调用 AgentApplication | 重构 | 8,20 | ✅ 完成 |

## 2. 核心差距登记

| 编号 | 差距 | 影响 | 修复 Phase | 状态 |
|------|------|------|-----------|------|
| G-01 | AgentRuntime Protocol 不完整 | Orchestrator 需适配具体实现 | 1 | ✅ 已补全 |
| G-02 | 无统一领域模型（mcc-core 无 runtime/） | 领域模型散落在 mcc-harness | 1 | ✅ 已创建 |
| G-03 | 无 AgentExecutor（AgentLoop 被 Runtime 直接包装） | 执行逻辑与生命周期耦合 | 2 | ✅ 已创建 |
| G-04 | 无 AgentFactory（CLI wiring.py 承担装配） | CLI 职责过重 | 3 | ✅ 已创建 |
| G-05 | 无 TaskDispatcher（派发内聚在 orchestrator） | 派发逻辑分散 | 6 | ✅ 已创建 |
| G-06 | task_id/attempt_id 未分离 | Retry 可能串线 | 6 | ✅ 已分离 |
| G-07 | MessageBus 仍派发 assignment | 违反 AR-005 | 7,9 | ✅ 已改为 task_submitted |
| G-08 | hasattr 兼容分支 | 违反 AR-011 | 7 | ✅ 已删除 |
| G-09 | TeammateRuntime status 类型不一致 | 调用方需知道用哪个属性 | 4 | ✅ 已统一 |
| G-10 | TaskStore 在 mcc-tools | 工具包成为 Orchestrator 底层依赖 | 6 | ✅ 已移至 mcc-core |
| G-11 | RuntimeCoordinator 缺生命周期闭环 | 注册失败可能留半注册对象 | 5 | ✅ 已加强 |
| G-12 | 无统一 Runtime Contract Tests 套件 | 各 Runtime 行为不一致 | 4,17 | ✅ 已创建 |

## 3. Phase 交付物清单

### Phase 0 — 架构冻结、盘点与测试基线
- [x] `docs/architecture/runtime-current-state.md`
- [x] `docs/architecture/runtime-dependency-map.md`
- [x] `docs/architecture/runtime-migration-matrix.md`
- [x] `docs/architecture/runtime-test-baseline.md`

### Phase 1 — 统一领域模型与 Runtime Protocol
- [x] `mcc-core/runtime/__init__.py`
- [x] `mcc-core/runtime/identifiers.py`
- [x] `mcc-core/runtime/enums.py`
- [x] `mcc-core/runtime/task.py`
- [x] `mcc-core/runtime/handle.py`
- [x] `mcc-core/runtime/result.py`
- [x] `mcc-core/runtime/event.py`
- [x] `mcc-core/runtime/health.py`
- [x] `mcc-core/runtime/capabilities.py`
- [x] `mcc-core/runtime/errors.py`
- [x] `mcc-core/runtime/protocols.py`
- [x] `mcc-core/runtime/definition.py`
- [x] `mcc-core/runtime/task_record.py`
- [x] `mcc-core/runtime/task_store.py`
- [x] 对应测试

### Phase 2 — AgentExecutor 适配层
- [x] `mcc-harness/execution/protocols.py`
- [x] `mcc-harness/execution/agent_loop_executor.py`
- [x] `mcc-harness/execution/context.py`
- [x] `mcc-harness/execution/cancellation.py`
- [x] `mcc-harness/execution/event_adapter.py`
- [x] 对应测试

### Phase 3 — AgentFactory / RuntimeFactory
- [x] `DefaultAgentFactory` 实现
- [x] 对应测试

### Phase 4 — 重写 Runtime 实现
- [x] `InteractiveRuntimeV2`
- [x] `PersistentRuntimeV2`（初版，Phase 18 原生重写）
- [x] `TaskRuntime`
- [x] Runtime Contract Tests

### Phase 5 — RuntimeCoordinator 生命周期闭环
- [x] 事务注册
- [x] 退出监听
- [x] 健康检查
- [x] 统一关闭

### Phase 6 — TaskStore / TaskDispatcher
- [x] `mcc-core/runtime/task_store.py`
- [x] `mcc-harness/runtime/dispatcher.py`
- [x] `mcc-harness/runtime/task_store_adapter.py`
- [x] task_id / attempt_id 分离

### Phase 7 — TeamOrchestrator / Scheduler
- [x] 移除 hasattr 兼容分支
- [x] DAG 支持
- [x] TaskScheduler 重构

### Phase 8 — CLI / TUI / Application 层
- [x] `AgentApplication` 生命周期管理
- [x] wiring.py 简化

### Phase 9 — MessageBus 迁移
- [x] `kind="assignment"` → `kind="task_submitted"`

### Phase 10-11 — Factory 增强
- [x] `DefaultAgentFactory.create()` 实现
- [x] RuntimeCoordinator 便捷方法
- [x] V2 Runtime factory 方法

### Phase 12-13 — Subagent 迁移
- [x] V2 Runtime feature flag（后已在 Phase 16 移除）
- [x] `SpawnSubagentTool` → `TaskRuntime` via `SubagentExecutor`

### Phase 14 — TeammateManager 迁移
- [x] `TeammateManager.spawn()` → `PersistentRuntimeV2.from_teammate()`

### Phase 16 — 旧架构删除
- [x] 移除 V2 feature flag
- [x] 删除旧 `InteractiveRuntime` / `PersistentRuntime`
- [x] 删除 `EphemeralRuntime`

### Phase 17 — 静态架构 + 并发竞态测试
- [x] 静态架构测试
- [x] 并发竞态测试

### Phase 18 — PersistentRuntimeV2 原生重写
- [x] `BoundedTaskQueue`
- [x] `WorkerPool`
- [x] `PersistentRuntimeV2` 重写

### Phase 19 — Orchestrator TaskDispatcher
- [x] `TaskDispatcher.dispatch()` 派发
- [x] Team E2E 测试

### Phase 20 — CLI AgentApplication
- [x] `AgentApplication` 团队操作方法
- [x] `slash.py /team` 路由
- [x] `wiring.py` 显式构造

### Phase 21 — 超时分离 + Team E2E
- [x] 超时域分离
- [x] Team E2E 测试

### Phase 22 — 文档更新 + DoD 验证
- [x] `runtime-current-state.md` 更新
- [x] `runtime-migration-matrix.md` 更新
- [x] `architecture.md` 更新
- [x] DoD 验证

## 4. DoD 验证结果

### 架构统一
- [x] AgentDefinition 是 Agent 静态定义的统一模型
- [x] AgentFactory 是唯一 Agent 创建入口
- [x] AgentExecutor 是统一执行接口
- [x] 所有 Runtime 实现同一 AgentRuntime Protocol
- [x] RuntimeCoordinator 是 Runtime 生命周期唯一管理中心
- [x] TeamOrchestrator 是 TeamRun 唯一编排入口
- [x] TaskDispatcher 是正式任务唯一派发入口
- [x] TaskStore 是 Task / Attempt / Result 唯一事实来源

### 可靠性
- [x] task_id 与 attempt_id 分离
- [x] 重复提交幂等
- [x] Retry 不覆盖旧 Attempt
- [x] wait timeout 不隐式取消执行
- [x] 所有终止路径都有 RuntimeResult
- [x] Runtime 崩溃可被感知和清理
- [x] TeamRun 可以恢复或明确标记不可恢复
- [x] 有界队列和并发限制生效

### 旧架构清理
- [x] 生产路径不再使用 SubagentRunner 作为 Runtime 执行（保留为工具执行引擎）
- [x] 生产路径不再使用 TeammateRuntime 独立执行（委托给 PersistentRuntimeV2）
- [x] TeammateManager 不再直接管理 Runtime 生命周期（经 from_teammate 委托）
- [x] MessageBus 不再派发正式 Assignment（改为 task_submitted）
- [x] 不再使用 hasattr(runtime, ...) 兼容分支
- [x] CLI 不直接创建 AgentLoop 或 Runtime（经 Factory）

### 测试与工程
- [x] Runtime Contract Tests 全部通过
- [x] 并发竞态测试全部通过
- [x] Team E2E 测试全部通过
- [x] 静态架构测试通过
- [x] Lint 通过
- [x] 文档、配置和迁移指南同步更新
