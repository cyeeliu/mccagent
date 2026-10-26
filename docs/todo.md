# Todo - 会话内任务清单

`todo_write` 工具让模型自管理当前编码 session 的进度清单。替换式更新（每次提交完整列表），harness 在连续多轮未更新时注入提醒，防止模型偏离计划。

## 工具

**`todo_write`**：提交完整 todo 列表（替换，非追加）

```json
{
  "todos": [
    {"content": "读取现有代码", "status": "completed"},
    {"content": "编写测试", "status": "in_progress"},
    {"content": "运行测试", "status": "pending"}
  ]
}
```

| 字段 | 说明 |
|---|---|
| `content` | 任务描述（非空） |
| `status` | `pending` / `in_progress` / `completed` |

约束：
- 最多 20 条
- 同时只允许 1 条 `in_progress`
- 替换式：每次调用替换整个列表

返回渲染文本：
```
[x] 读取现有代码
[>] 编写测试
[ ] 运行测试

(1/3 completed)
```

## 提醒机制

AgentLoop 维护 `rounds_since_todo` 计数器：

- 模型调用 `todo_write` → 计数重置为 0
- 模型不调用 `todo_write` 的轮 → 计数 +1
- 计数 ≥ `reminder_interval`（默认 3）且 todo 非空 → 追加 `<reminder>Update your todos.</reminder>` 到消息序列，计数重置

提醒作为 user 消息注入，下一轮 LLM 调用可见，促使模型更新进度。

## system 引导

todo 启用时，system prompt 追加：

> Before starting any multi-step task, use todo_write to plan your steps. Update status as you go.

## 配置

```json
{
  "todo": {
    "enabled": true,
    "reminder_interval": 3
  }
}
```

| 字段 | 默认 | 说明 |
|---|---|---|
| `enabled` | `true` | 关闭时不注册工具、不注入 reminder |
| `reminder_interval` | `3` | 连续多少轮未更新触发提醒 |

## 与 Task 的区别

| | Todo | Task |
|---|---|---|
| 状态 | 内存态（会话级） | 持久化（`.tasks/`） |
| 粒度 | 单会话规划 | 跨会话/多 agent 协作 |
| 依赖 | 无 | 有依赖图（blockedBy） |
| 工具数 | 1（todo_write） | 6 |
| resume | 丢失 | 保留 |

简单规划用 Todo，跨会话复杂依赖用 Task。

## 参考

- [learn-claude-code s05](../../learn-claude-code/s05_todo_write/) — TodoWrite 蓝本
