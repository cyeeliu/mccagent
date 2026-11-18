# Runtime 测试基线

> 基线 commit: `69ded02` (2026-11-16)
> 测试命令: `uv run pytest tests/ -x -q`

## 1. 测试基线

| 指标 | 值 |
|------|-----|
| 测试文件数 | 58 |
| 测试结果 | 2754 passed, 0 failed |
| ruff | clean |
| commit | `69ded02` |

## 2. Runtime 相关测试文件

| 文件 | 覆盖范围 | 关键测试 |
|------|---------|---------|
| `test_runtime.py` | Runtime 基础契约 | start/stop/cancel/wait_done, submit_task/wait_task |
| `test_runtime_infra.py` | Runtime 基础设施 | RuntimeFactory, RuntimeCoordinator, run_turn |
| `test_orchestrator.py` | TeamOrchestrator | 任务派发, wait_task, wait_done |
| `test_teammate.py` | TeammateRuntime/Manager | assign_task, kind="assignment", wait_task |
| `test_subagent.py` | SubagentRunner | 子代理执行 |
| `test_cancel.py` | 取消语义 | cancel 信号 |
| `test_e2e.py` | 端到端 | AgentLoopBuilder |
| `test_loop_contract.py` | AgentLoop 契约 | AgentLoop 行为 |
| `test_harness.py` | Harness 层 | AgentLoop, Runtime |
| `test_slash.py` | Slash 命令 | TeammateManager |
| `test_workflow.py` | 工作流 | SubagentRunner |

## 3. 测试覆盖矩阵（当前 Runtime 行为）

| 行为 | Interactive | Ephemeral | Persistent | 测试文件 |
|------|------------|-----------|-----------|---------|
| start → RUNNING/IDLE | ✅ | ✅ | ✅ | test_runtime.py |
| stop → 终态 | ✅ | ✅ | ✅ | test_runtime.py |
| cancel → CANCELLED | ✅ | ✅ | ✅ | test_cancel.py |
| wait_done → RuntimeResult | ✅ | ✅ | ✅ | test_runtime.py |
| submit_task | ❌ | ❌ | ✅ | test_runtime.py |
| wait_task | ❌ | ❌ | ✅ | test_orchestrator.py |
| run_turn | ✅ | ❌ | ❌ | test_runtime_infra.py |
| pause/resume | ❌ | ❌ | ✅ | test_runtime.py |
| inject_instruction | ❌ | ❌ | ✅ | test_runtime.py |
| health_check | ❌ | ❌ | ❌ | — |
| subscribe_events | ❌ | ❌ | ❌ | — |

## 4. Phase 1+2 新增测试计划

### Phase 1 测试（mcc-core/runtime/）

| 测试文件 | 覆盖 |
|---------|------|
| `tests/test_runtime_domain.py` | ID 类型、枚举、状态迁移表 |
| `tests/test_runtime_task.py` | RuntimeTask 字段校验、序列化往返 |
| `tests/test_runtime_result.py` | RuntimeResult 字段校验、终态判定 |
| `tests/test_runtime_event.py` | RuntimeEvent 字段、事件类型、序列关联 |
| `tests/test_runtime_protocol.py` | Protocol 结构校验（非运行时行为） |

### Phase 2 测试（mcc-harness/execution/）

| 测试文件 | 覆盖 |
|---------|------|
| `tests/test_agent_executor.py` | AgentLoopExecutor execute → RuntimeResult |
| `tests/test_cancellation.py` | CancellationToken 触发、传播 |
| `tests/test_event_adapter.py` | AgentEvent → RuntimeEvent 转换 |

## 5. 验收标准

### Phase 0
- [x] 4 份文档创建完成
- [x] 所有构造入口、旧类引用、派发路径有明确归属

### Phase 1
- [ ] mcc-core/runtime/ 新增 11 个文件
- [ ] 领域对象不依赖 mcc-harness / mcc-cli / mcc-tools
- [ ] 序列化往返测试通过
- [ ] 非法状态迁移测试通过
- [ ] 现有 2754 测试全绿（零回归）

### Phase 2
- [ ] mcc-harness/execution/ 新增 5 个文件
- [ ] AgentLoopExecutor.execute 返回统一 RuntimeResult
- [ ] CancellationToken 正确传播
- [ ] AgentEvent → RuntimeEvent 转换正确
- [ ] 现有 2754 测试全绿（零回归）
