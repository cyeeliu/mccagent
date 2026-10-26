# Cron 调度器

按标准 5 字段 cron 表达式定时触发 prompt。调度器在 asyncio.Task 中
轮询 due jobs，AgentLoop 每轮迭代首 consume 并注入为 user 消息，
使模型在空闲时处理定时任务。durable job 持久化到 `.cron_jobs.json`，
进程重启后自动恢复。

## 架构概览

```
CronScheduler._poll_loop（asyncio.Task，每 poll_interval 秒）
 └── poll(now) → cron_matches 匹配 → due job 入队（asyncio.Queue）

AgentLoop 每轮迭代首
 ├── cron_scheduler.consume() → due jobs
 ├── format_cron_delivery → <cron_delivery> XML
 ├── messages.append(Message.user(notification))
 └── acknowledge(jobs) → recurring 重置 / one-shot 移除
```

## 使用方式

### schedule_cron 安排定时任务

```
> 每天早上九点跑一遍测试并汇报失败
[agent 调用 schedule_cron]
  schedule_cron(cron="0 9 * * *", prompt="运行测试并汇报失败用例")
  → "Scheduled cron_ab12cd34: 0 9 * * * -> 运行测试并汇报失败用例"

[到达 09:00，下一轮迭代注入]
  <cron_delivery>
    <job id="cron_ab12cd34" cron="0 9 * * *">
      运行测试并汇报失败用例
    </job>
  </cron_delivery>
```

### 查询与取消

| 工具 | 说明 |
|------|------|
| `schedule_cron` | 按 5 字段 cron 表达式调度 prompt |
| `list_crons` | 列出所有定时任务 |
| `cancel_cron` | 按 ID 取消定时任务 |

### cron 表达式

标准 5 字段：`minute hour day-of-month month day-of-week`。

| 表达式 | 含义 |
|--------|------|
| `0 9 * * *` | 每天 09:00 |
| `*/15 * * * *` | 每 15 分钟 |
| `0 9-17 * * 1-5` | 工作日 9 点到 17 点整点 |
| `0 9 * * 1` | 每周一 09:00 |

支持 `*`、步进（`*/15`）、范围（`9-17`）、列表（`1,15`）。
day-of-month 与 day-of-week 同时受限时为 OR 语义（与标准 cron 一致）。

### recurring 与 durable

- `recurring=true`（默认）：按计划重复触发；`false`：触发一次后移除
- `durable=true`（默认）：持久化到 `<workdir>/.cron_jobs.json`，重启恢复；
  `false`：仅本次会话有效

投递语义：due job 注入前标记 `pending_delivery`（持久化），成功注入后
acknowledge；若投递失败可 `restore` 重新入队，保证不丢任务。

## 核心组件

### CronScheduler

- `schedule(cron, prompt, recurring, durable) -> CronJob | str`：安排任务（返回错误字符串表示校验失败）
- `poll(moment) -> int`：检查 due jobs 入队（minute_marker 去重，同一分钟不重复触发）
- `consume() -> list[CronJob]`：批量取出 due jobs
- `acknowledge(jobs)`：recurring 重置 pending_delivery / one-shot 移除并持久化
- `restore(jobs)`：投递失败时重新入队
- `start() / stop()`：启动/停止 poll loop（幂等）

### CronJob

`id / cron / prompt / recurring / durable / pending_delivery / last_fired`
字段经 asdict 序列化持久化；加载时校验表达式、ID 前缀与 prompt，
非法条目跳过不中断。

## 配置

```json
{
  "cron": {
    "enabled": true,
    "poll_interval": 1.0,
    "durable": true
  }
}
```

| 字段 | 默认 | 说明 |
|------|------|------|
| `enabled` | `true` | 关闭时不创建调度器、不注册 cron 工具 |
| `poll_interval` | `1.0` | 轮询间隔（秒） |
| `durable` | `true` | 是否持久化到 `.cron_jobs.json` |

## 与后台任务的区别

| 特性 | 后台任务 | Cron 调度器 |
|------|----------|-------------|
| 触发方式 | 模型主动发起 | 时间表驱动 |
| 执行内容 | shell 命令 | prompt 注入（由 AgentLoop 处理） |
| 持久化 | 无 | `.cron_jobs.json` |
| 生命周期 | 命令完成即结束 | recurring 持续到取消 |
