# 多 Agent 团队

长驻 teammate 协作：lead 通过 `spawn_teammate` 启动持续运行的子 Agent，
通过 MessageBus 通信、Plan approval 审批、git worktree 隔离。

## 架构概览

```
lead (主 AgentLoop)
 ├── spawn_teammate("coder", prompt="...")
 │    └── TeammateRuntime (asyncio.Task)
 │         ├── WORK: consume inbox → AgentLoop.run → send_to_lead
 │         ├── IDLE: claim_task 或等待唤醒
 │         └── SHUTDOWN: lead request_shutdown
 ├── MessageBus (.mailboxes/{name}.jsonl)
 ├── PlanStore (plan approval 状态机)
 └── Worktree (.worktrees/{name}/ git 隔离)
```

## 核心概念

### 长驻 Teammate

与一次性 subagent 不同，teammate 持续运行直到 lead 请求 shutdown。
teammate 在 WORK/IDLE 两阶段间循环：

- **WORK**：consume inbox → 处理消息 → 有任务则执行 `AgentLoop.run` → 发 `work_done` 给 lead
- **IDLE**：尝试 `claim_task` 领取下一个 ready task → 有则 WORK → 无则等待唤醒

### MessageBus

文件持久化的异步邮箱，每个 recipient 拥有独立的 `.mailboxes/{name}.jsonl`：

- `send(sender, recipient, content, kind, meta)`：追加一行 JSON
- `consume(recipient)`：读取并清空（destructive read）
- `peek(recipient)`：检查是否有未读消息

`asyncio.Lock` 按 recipient 分锁，不同 recipient 互不阻塞。

### Plan Approval Protocol

teammate 改文件前须 lead 审批：

1. teammate `submit_plan(plan)` → 创建 pending PlanRequest → 通知 lead
2. lead `review_plan(request_id, approve, feedback)` → 更新状态 → 唤醒 teammate
3. teammate 收到 approved → 继续执行；rejected → 放弃或修改重提

### Worktree 隔离

为 teammate 创建独立工作目录，避免并行编辑冲突：

- git repo：`git worktree add -b task/{name}` 创建独立分支
- 非 git：降级为普通子目录
- worktree 根写入 `.task_id` 标记

### Lead Wake

teammate 发消息给 lead 时，lead 在下一轮交互前检查 inbox：
有消息则 `format_team_events` 格式化为 `[team events]` 注入为新 turn。

## Lead 工具

| 工具 | 说明 |
|------|------|
| `spawn_teammate` | 启动长驻 teammate（name, role, prompt, model?, task_id?, require_plan?） |
| `list_teammates` | 列出所有活跃 teammate 及状态 |
| `send_message` | 向 teammate 发消息 |
| `request_shutdown` | 请求 teammate 关闭 |
| `review_plan` | 审批或驳回 teammate 的 plan |
| `create_worktree` | 为 task 创建隔离工作目录 |

## Teammate 工具

| 工具 | 说明 |
|------|------|
| `send_to_lead` | 向 lead 汇报进展/结果 |
| `submit_plan` | 提交工作计划等待审批 |
| `list_tasks` | 查看可用任务 |
| `claim_task` | 领取任务 |
| `complete_task` | 完成任务 |

## 配置

```json
{
  "team": {
    "enabled": true,
    "max_teammates": 4,
    "default_model": "claude-haiku-4-5",
    "worktree_enabled": false
  }
}
```

- `enabled`：关闭时不注册团队工具、不创建 TeammateManager
- `max_teammates`：同时活跃的 teammate 上限
- `default_model`：teammate 默认模型（未显式指定时）
- `worktree_enabled`：为 teammate 开启 git worktree 隔离

## 使用示例

```
> 我需要同时写代码和测试，帮我派两个 teammate
[lead 调用 spawn_teammate]
  spawn_teammate(name="coder", role="代码编写员", prompt="实现登录接口")
  spawn_teammate(name="tester", role="测试编写员", prompt="为登录接口写测试")

[teammate 并行工作]
  coder: [WORK] AgentLoop.run → 写代码 → send_to_lead("登录接口已实现")
  tester: [WORK] AgentLoop.run → 写测试 → send_to_lead("测试已编写")

[lead wake]
  [team events]
  [WORK_DONE from coder] 登录接口已实现
  [WORK_DONE from tester] 测试已编写

> request_shutdown(teammate="coder")
> request_shutdown(teammate="tester")
```

## 运行时产物

| 路径 | 说明 |
|------|------|
| `.mailboxes/{name}.jsonl` | 消息邮箱文件 |
| `.worktrees/{name}/` | git worktree 工作目录 |
| `.tasks/` | 共享任务记录（与 task 工具集共用） |

均已加入 `.gitignore`。
