# 后台任务

异步执行长耗时 shell 命令，不阻塞主循环。命令在 asyncio.Task 中通过
SandboxExecutor 执行（沙箱隔离），完成后结果注入为 user 消息，
使模型能在后续轮次感知后台任务完成。

## 架构概览

```
AgentLoop 每轮迭代首
 ├── background_manager.collect() → 已完成结果
 ├── format_background_results → <background_result> XML
 └── messages.append(Message.user(notification))

run_bash(run_in_background=True)
 └── BackgroundTaskManager.start(command, cwd)
      └── asyncio.Task → SandboxExecutor.run → result → ready queue
```

## 使用方式

### run_bash 加 run_in_background 参数

```
> 帮我跑测试，后台执行
[agent 调用 run_bash]
  run_bash(command="npm test", run_in_background=true)
  → "[Background task bg_0001 started. Result delivered on a later turn.]"

[agent 继续其他工作...]

[下一轮迭代，后台任务完成]
  <background_result>
    <task_id>bg_0001</task_id>
    <status>completed</status>
    <command>npm test</command>
    <summary>all 42 tests passed</summary>
  </background_result>
```

### 查询与取消

| 工具 | 说明 |
|------|------|
| `background_status` | 查询后台任务状态（传 task_id 查单个，不传查全部） |
| `cancel_background` | 取消运行中的后台任务 |

## 核心组件

### BackgroundTaskManager

- `start(command, cwd, timeout) -> task_id`：创建 asyncio.Task 执行命令
- `collect() -> list[BackgroundResult]`：非阻塞批量取出已完成结果
- `list_tasks() / get_status(task_id)`：查询任务状态
- `cancel(task_id) / shutdown_all()`：取消任务

复用 `SandboxExecutor`（沙箱隔离），`asyncio.Semaphore` 限并发。

### 结果注入

AgentLoop 每轮迭代首调用 `collect()`，有结果则格式化为
`<background_result>` XML 注入 messages，同时发 `Notice` 事件。

## 配置

```json
{
  "background": {
    "enabled": true,
    "default_timeout": 120,
    "max_concurrent": 8
  }
}
```

| 字段 | 默认 | 说明 |
|------|------|------|
| `enabled` | `true` | 关闭时不暴露 run_in_background 参数、不创建 manager |
| `default_timeout` | `120` | 后台命令默认超时（秒） |
| `max_concurrent` | `8` | 同时运行的后台任务上限 |

## 与多 Agent 团队的区别

| 特性 | 后台任务 | 多 Agent 团队 |
|------|----------|--------------|
| 执行内容 | shell 命令 | LLM 驱动的 Agent |
| 通信 | 结果注入 | MessageBus |
| 生命周期 | 命令完成即结束 | lead 请求 shutdown 才退出 |
| 隔离 | SandboxExecutor | worktree（可选） |
