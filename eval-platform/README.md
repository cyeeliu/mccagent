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

## 关键实现机制

- **并发领取**：任务领取是一条 `FOR UPDATE SKIP LOCKED` 事务——批次并发上限的检查、
  优先级排序与租约写入原子完成，多个 Runner 并发领取互不阻塞、绝无双重分配。
- **SSE 慢消费者策略**：订阅队列写满时直接丢弃事件并累计 lagged 计数；
  订阅端读到 lagged 信号后由前端走 REST 全量补齐，服务端绝不反压事件摄取路径。
- **评分位置**：expectations 断言（工具调用 / 输出包含 / 无错误 / 调用次数，
  按权重叠加、阈值判定）由 Runner 在沙箱外计算，结果随 complete 上报，
  与被测执行环境隔离。
- **测试策略**：数据库访问抽象为 `DbLike` / `ConnLike` 协议——生产传 asyncpg 连接池，
  测试传内存替身；真 PostgreSQL 集成测试通过 `EVAL_PG_DSN` 环境变量开启，缺省跳过。

## 快速开始

平台按模块渐进交付，当前各部分状态：

| 部分 | 状态 |
|---|---|
| 事件契约（schemas + 三端对齐） | ✅ 可用 |
| server 领域类型 / 配置 / 错误 / API 契约 / 状态机 | ✅ 可用 |
| server 路由实现（REST + SSE + 调度 + 存储 + 指标） | ✅ 可用 |
| runner 执行主循环 / 评分 / 适配器（mock / anthropic / langgraph） | ✅ 可用 |
| web 功能页面（看板 / 轨迹 / 对比 / 仪表盘） | ✅ 可用 |
| 部署（docker compose / K8s / 可观测栈） | ✅ 可用 |

本地开发：

```bash
cd eval-platform
uv sync --all-packages          # server + runner 依赖
uv run --all-packages pytest    # 平台测试
ruff check .                    # lint（或使用仓库根的 ruff）
```

运行完整链路（需 PostgreSQL）：

```bash
# 1. 启动 API 服务（自动执行 migrations）
cd server && uv run python -m eval_server

# 2. 启动 Runner（默认 mock scaffold，无需 API key）
cd runner && uv run python -m runner.main

# 3. 提交批次：通过 POST /api/profiles + /api/batches，
#    Runner 会自动领取任务并流式上报轨迹
```

可选 extras：`uv sync --extra s3`（server，S3 轨迹后端）、
`uv sync --extra anthropic` / `--extra langgraph`（runner，真实 Agent 适配器）。

### web 前端

标准 Vite + React + TanStack Query 项目，所有网络数据先过 Zod 校验，未知事件类型降级为占位而不是崩溃。

```bash
cd web
npm install
npm run dev     # 开发服务器（/api 代理到本地 API 服务，SSE 不缓冲）
npm run build   # tsc 严格类型检查 + 生产构建
npm test        # vitest（Zod 契约 + 轨迹合并纯函数）
```

页面路由：

| 路由 | 页面 |
|---|---|
| `/` | 批次列表（运行中自动轮询，跑完自动停） |
| `/batches/:batchId` | 批次详情（run_stats 汇总 + runs 表） |
| `/runs/:runId` | 轨迹查看器（时间线 + 详情双栏，只看失败过滤） |
| `/dashboard` | 仪表盘（统计卡 + 通过率/Run 数双 Y 轴趋势 + 成本趋势） |
| `/compare?a=…&b=…` | 批次对比（URL 驱动可分享，verdict 过滤 + 可比性警告） |

**实时与历史无缝拼接**（`web/src/features/trace/useTrace.ts`）：run 无论已结束、正在跑、
还是看着看着跑完，组件拿到的都是同一份按 `seq` 排序去重后的事件流——
历史段分页拉全，实时段 SSE 续接（100ms 节流合帧），
SSE 掉帧读到 lagged 信号时触发全量重拉补偿，全部封装在 Hook 内。

## 部署与运维

### 本地一键全栈

```bash
cd eval-platform
docker compose up --build
```

| 服务 | 端口 | 说明 |
|---|---|---|
| web | http://localhost:5173 | 前端（nginx 托管 + /api 反代，SSE 不缓冲） |
| server | http://localhost:8081 | API 服务 |
| db | 5432 | PostgreSQL 17 |
| minio | http://localhost:9001 | S3 兼容对象存储（admin/devpassword） |
| prometheus | http://localhost:9090 | 指标采集 |
| grafana | http://localhost:3000 | 仪表盘（admin/admin，自动导入数据源与看板） |

健康检查链保证任意顺序重启幂等：`minio-init` 等 minio 健康后建 bucket，
server 等 db 健康 + bucket 就绪才起，runner 等 server 健康才连。

提交演示批次（mock agent，无需 API key）：

```bash
./scripts/demo-batch.sh
```

Runner 横向扩展（验证 `FOR UPDATE SKIP LOCKED` 并发领取）：

```bash
docker compose up --scale runner=4
```

### 本地与生产的三处刻意不同

| | 本地（图省事） | 生产（要安全） |
|---|---|---|
| 对象存储 | MinIO | 真 S3 + IAM Role（不用 Access Key） |
| 密钥 | `${RUNNER_SECRET:-dev-runner-secret}` | Kubernetes Secret + 外部 Vault 注入 |
| Grafana 密码 | `admin` | 强密码 + SSO |

### Kubernetes（Kustomize 管理环境差异）

```bash
kubectl apply -k deploy/k8s/overlays/staging/   # 1 副本、小资源
kubectl apply -k deploy/k8s/overlays/prod/      # 3 副本、NetworkPolicy、KEDA
```

生产 overlay 比 base 多三样关键东西：

- **NetworkPolicy**：runner 出方向只允许访问 server:8081 与集群 DNS——沙箱默认断网，攻破爆炸半径被框死
- **KEDA ScaledObject**：runner 副本数随 `queued` 队列深度自动伸缩（每 5 个任务 ≈ 1 个 runner），没任务缩到 0
- **PodDisruptionBudget**：节点维护时至少保留副本数，滚动更新 `maxUnavailable: 0`

### 可观测性

Prometheus 抓取 `server:8081/metrics`；Grafana provisioning 自动导入数据源与
「Eval Platform」看板（通过率趋势 / API 延迟 P99 / 队列积压 / 活跃 run / 成本累积）。
两条生产必装告警在 `deploy/alerts.yml`：

- `EvalQueueBacklog`：队列深度 > 50 持续 5 分钟 → 通知 oncall
- `EvalPassRateLow`：通过率连续 1 小时 < 60% → 通知算法团队

### 高频故障排查

| 故障现象 | 第一步排查 | 典型原因 |
|---|---|---|
| 任务全部卡在 queued | `eval_queue_depth` 飙高、`eval_active_runs` 为 0 | Runner 全挂 / 鉴权 token 不匹配 |
| 通过率突然下降 | 看最近批次的 error message | 模型 API 限速 / 沙箱 OOM |
| /healthz/ready 返回 500 | server 日志 | PostgreSQL 断连 / MinIO 不可达 |
| Prometheus 没数据 | `curl server:8081/metrics` 手动检查 | scrape 配错 / 路由未注册 |
| Runner 领不到任务 | server 日志看 401 | RUNNER_SECRET 两边不一致 |

## 环境变量（server）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:dev@localhost:5432/evalplatform` | PostgreSQL 连接串 |
| `PORT` | `8081` | HTTP 监听端口 |
| `RUNNER_SECRET` | 未设置 | 内部接口 Bearer token；未设置为开发模式（放行并告警） |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | 允许的浏览器来源，逗号分隔 |
| `LEASE_TTL_S` | `120` | 任务租约时长（秒） |
| `TRACE_BACKEND` | `local` | 轨迹后端：`local`（本地目录）或 `s3` / `minio` |
| `TRACE_DIR` | `./traces` | local 后端的轨迹目录 |
| `S3_ENDPOINT` / `S3_BUCKET` / `S3_ACCESS_KEY` / `S3_SECRET_KEY` | — | s3 后端连接信息（需安装 s3 extra） |
| `S3_REGION` / `S3_PREFIX` | `us-east-1` / 空 | s3 后端区域与 key 前缀 |

## 环境变量（runner）

| 变量 | 默认值 | 说明 |
|---|---|---|
| `EVAL_SERVER_URL` | `http://localhost:8081` | API 服务地址 |
| `RUNNER_ID` | `主机名-进程号` | Runner 标识（领取归属与事件校验） |
| `RUNNER_SECRET` | 未设置 | 内部接口 Bearer token；须与 server 一致 |
| `POLL_IDLE_S` | `2.0` | 无任务时的轮询间隔（秒） |
| `LG_MODEL_PROVIDER` | `anthropic` | langgraph 适配器的底层 LLM（anthropic / openai） |
