# eval-platform — Agent 评测平台

批量运行 Agent、实时查看轨迹、对比不同版本、输出统计报表的一站式平台。
它是 `mcc` 的配套评测设施：`mcc` 本身（以及任何满足适配协议的 Agent 框架）都可以作为被测对象接入。

```
React (web/)  ──HTTP/SSE──►  API 服务 (server/, Python/FastAPI)  ──SQL──►  PostgreSQL
   轨迹回放                      │  REST + SSE 广播                     元数据/索引
   批次看板                      │  租约调度 + reaper 回收
   对比报告                      └──JSONL──►  轨迹对象存储（本地 / S3 兼容）
                                        ▲
Python Runner (runner/) ──lease/events/complete──┘
   领任务 → 起被测 Agent → 流式上报轨迹 → 汇报结果与评分
```

## 目录结构

| 目录 | 内容 |
|---|---|
| `schemas/` | TraceEvent JSON Schema —— server / web / runner 三端的单一事实来源 |
| `server/` | API 服务：REST + SSE、租约调度、轨迹存储门面、Prometheus 指标 |
| `runner/` | 执行端：领取任务、运行被测 Agent 适配器、批量上报事件 |
| `web/` | React 前端：批次看板、轨迹查看器、对比报告、仪表盘 |

本目录是独立的 uv workspace（`server` 与 `runner` 为其成员），与仓库根的 mcc workspace 互不干扰。

## 三条全局设计原则

1. **轨迹不进数据库** —— 单条轨迹可达数十 MB，正文存对象存储，库里只留索引。
2. **评分在执行端** —— 评分逻辑随 Runner 运行，与被测沙箱隔离，防止作弊。
3. **环境指纹不能少** —— Agent 配置精确记录沙箱镜像 digest；环境不同的批次，对比无意义，界面强提示。

## 领域模型

```
Agent 配置 (profile)：被测对象 = 框架 × 模型 × 版本 × 沙箱镜像
评测集：一批测试用例（任务 + 期望 + 预算）
批次 (batch)：一次评测 = 评测集 × Agent 配置（支持幂等提交）
运行 (run)：一个用例执行一次 → 状态 / 分数 / 成本
轨迹 (trace)：一次运行的完整事件流（JSONL，存对象存储）
```

run 生命周期：`queued → leased → running → passed | failed | error | timeout`；
租约到期未续由 reaper 回收重试，超过重试上限直接置 `error`。

## 快速开始（建设中）

平台按模块渐进交付，当前各部分状态：

| 部分 | 状态 |
|---|---|
| 事件契约（schemas + 三端对齐） | ✅ 可用 |
| server 领域类型 / 配置 / 错误 / API 契约 / 状态机 | ✅ 可用 |
| server 路由实现（REST + SSE + 调度 + 存储 + 指标） | 🚧 下一阶段 |
| runner 执行主循环 / 评分 / 适配器（mock / mcc） | 🚧 下一阶段 |
| web 功能页面（看板 / 轨迹 / 对比 / 仪表盘） | 🚧 建设中 |
| 部署（docker compose / K8s / 可观测栈） | 🚧 规划中 |

本地开发：

```bash
cd eval-platform
uv sync --all-packages          # server + runner 依赖
uv run --all-packages pytest    # 平台测试
ruff check .                    # lint（或使用仓库根的 ruff）
```

前端（`web/`）为标准 Vite 项目：`npm install && npm run dev` 启动开发服务器（代理 `/api` 到本地 API 服务）。

## 环境变量（server）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:dev@localhost:5432/evalplatform` | PostgreSQL 连接串 |
| `PORT` | `8080` | HTTP 监听端口 |
| `RUNNER_SECRET` | 未设置 | 内部接口 Bearer token；未设置为开发模式（放行并告警） |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | 允许的浏览器来源，逗号分隔 |
| `LEASE_TTL_S` | `120` | 任务租约时长（秒） |
