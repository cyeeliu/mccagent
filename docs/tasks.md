# Task - 持久化任务与依赖图

task 工具集在 `.tasks/` 目录下持久化任务记录，支持依赖图（`blockedBy`）与生命周期管理，适合跨会话或多 agent 协作的复杂任务编排。

## 任务模型

每个任务落盘为 `.tasks/task_{8hex}.json`：

```json
{
  "id": "task_a1b2c3d4",
  "subject": "实现 API 层",
  "description": "REST 端点 + 校验",
  "status": "pending",
  "owner": null,
  "blockedBy": ["task_e5f6a7b8"]
}
```

| 字段 | 说明 |
|---|---|
| `id` | 运行时生成，格式 `task_` + 8 位 hex |
| `subject` | 任务标题（非空） |
| `description` | 详细描述 |
| `status` | `pending` / `in_progress` / `completed` |
| `owner` | 认领者标识（claim 时设置） |
| `blockedBy` | 依赖的任务 ID 列表 |

## 生命周期

```
pending --claim_task--> in_progress --complete_task--> completed
```

- `claim_task`：仅当 status=pending 且所有依赖 completed 时可认领，设 owner + status=in_progress
- `complete_task`：仅当 status=in_progress 且 owner 匹配时可完成，设 status=completed，报告新解锁的任务

## 依赖图

`update_task` 给任务加依赖（`addBlockedBy`）：

- 只能在 pending 且 unowned 时加依赖
- 自依赖拒绝
- **环检测**：传递依赖检查防 A→B→A 循环
- 依赖必须存在
- 自动去重

```
schema (completed) → api (pending, blockedBy: [schema]) → tests (pending, blockedBy: [api])
```

`can_start` 检查所有依赖是否 completed。`complete_task` 完成时报告新解锁的任务（依赖刚被满足的 pending 任务）。

## 工具

| 工具 | 说明 |
|---|---|
| `create_task` | 创建任务，返回运行时生成的 ID |
| `update_task` | 给 pending 任务加依赖 |
| `list_tasks` | 列出所有任务（含状态/owner/依赖） |
| `get_task` | 按 ID 查单个任务（JSON） |
| `claim_task` | 认领 pending 且依赖完成的任务 |
| `complete_task` | 完成自己认领的任务，报告解锁 |

工具名 input_schema 带 `task_[0-9a-f]{8}` pattern 校验。

## system 引导

tasks 启用时，system prompt 追加：

> Use task tools to track dependencies. Create all task nodes first; after create_task returns IDs, use update_task with those exact IDs to add dependencies.

## 配置

```json
{
  "tasks": {
    "enabled": true
  }
}
```

| 字段 | 默认 | 说明 |
|---|---|---|
| `enabled` | `true` | 关闭时不注册 6 个工具、不创建 .tasks/ |

`.tasks/` 已加入 `.gitignore`（运行时状态）。

## 多 agent 协作

单 agent 场景 owner 恒为 `"agent"`。多 agent 时各子代理用自己 session_id 作 owner，`complete_task` 的 owner 校验防止误完成他人的任务。

## 路径安全

TaskStore 校验 task_id 格式（`task_` + 8 hex），文件名直接拼 `task_id.json`，无 traversal 风险。

## 参考

- [learn-claude-code s10](../../learn-claude-code/s10_task_system/) — Task System 蓝本
