# 评测平台

`eval-platform/` 是仓库内的独立子项目：一个 Agent 评测平台，用于批量运行 Agent、实时查看轨迹、对比版本效果并输出统计报表。`mcc` 既是它的第一个被测对象，也是其能力上限的试金石。

- 平台门面文档（架构图、领域模型、进度表、环境变量）：见 [../eval-platform/README.md](../eval-platform/README.md)
- 本页说明平台与 mcc 的关系、目录形态与验证方式

## 平台与 mcc 的关系

平台通过**适配器协议**与被测 Agent 解耦：

```python
class AgentAdapter(Protocol):
    def run(self, req: RunRequest) -> AsyncIterator[TraceEvent]: ...
```

被测框架（mcc、其他 scaffold）各自实现一个适配器，把内部执行过程转译成标准轨迹事件流。平台不关心框架内部实现——只要事件契约满足 `schemas/trace-event.schema.json`，任何 Agent 都能被评测。

事件契约包含六种事件：`run_started` / `llm_request` / `llm_chunk` / `tool_call` / `tool_result` / `run_finished`，每个事件携带单调递增的 `seq` 序号，前端据此实现实时流与历史轨迹的无缝拼接。

## 目录形态

```
eval-platform/
├── README.md            平台门面文档
├── pyproject.toml       独立 uv workspace 根（server + runner）
├── schemas/             TraceEvent JSON Schema（三端单一事实来源）
├── server/              API 服务（Python / FastAPI / asyncpg）
│   ├── migrations/      PostgreSQL 表结构（profile / batch / run）
│   ├── src/eval_server/
│   └── tests/
├── runner/              执行端（Python / httpx）
│   ├── src/runner/
│   └── tests/
└── web/                 React 前端（Vite / TanStack Query / recharts）
```

要点：

- **独立 workspace**：`eval-platform/pyproject.toml` 自成 uv workspace，与仓库根（`packages/*`）互不干扰；根项目的测试与 lint 不受影响。
- **三端一致**：schema 是唯一真相；server（pydantic 模型）与 runner（事件定义）各有契约测试，钉住与 schema 的一致性，字段变更必须三端同步并提升 `schema_version`。
- **评分在执行端**：expectations 断言（工具序列 / 输出包含 / 无错误）由 Runner 在沙箱外计算，结果随 complete 上报。

## 与主项目协作验证

```bash
# 根项目（mcc 本体）
uv run ruff check .
uv run --all-packages pytest tests/ -q

# 平台（独立 workspace）
cd eval-platform
uv sync --all-packages
uv run --all-packages pytest

# 平台前端
cd web
npm run build   # tsc 严格类型检查 + vite 生产构建
npm test        # vitest
```

## 路线图

1. 事件契约与 server 骨架（本页所述，已完成）
2. server 完整实现：REST 路由、SSE 广播、租约调度、轨迹存储、指标暴露（已完成）
3. runner 完整实现：执行主循环、心跳续租、mock / anthropic / langgraph 适配器、评分（已完成）
4. web 功能页面：批次看板、轨迹查看器、对比报告、仪表盘（已完成）
5. 部署与运维：docker compose 全栈、Prometheus/Grafana、K8s 配置（已完成）

## server 与 runner 内部结构

```
server/src/eval_server/
├── api.py        三组路由（/api 公开、/internal Runner、系统端点）+ 端点契约清单
├── scheduler.py  租约领取 / 心跳续租 / reaper 自愈（FOR UPDATE SKIP LOCKED）
├── store.py      轨迹存储门面：本地目录与 S3/MinIO 两个后端
├── stream.py     SSE 广播中心：按 run / batch 主题分发，慢消费者丢帧计 lagged
├── metrics.py    Prometheus 指标（每实例独立 registry）
├── db.py         DbLike/ConnLike 协议 + 迁移执行（幂等）
└── app.py        组装：依赖注入、异常处理、CORS、HTTP 指标中间件

runner/src/runner/
├── main.py       主循环：lease → 执行 → 批量上报 JSONL → complete；心跳后台任务
├── scoring.py    expectations 评分：四类断言加权叠加，阈值判定
└── adapters/     适配器：mock（零依赖演示）、anthropic、langgraph（惰性导入）
```

## web 内部结构

```
web/src/
├── api/schemas.ts             Zod 契约：全部网络数据运行时校验，未知事件降级占位
├── api/client.ts              fetch + SSE 封装（EventSource 自动重连带 Last-Event-ID）
├── App.tsx                    路由（批次/详情/轨迹/仪表盘/对比）+ 顶部导航
├── features/runs/             批次列表 + 批次详情（run_stats 汇总，运行中自动轮询）
├── features/trace/            轨迹查看器 + useTrace Hook（实时/历史按 seq 拼接去重）
├── features/dashboard/        仪表盘：统计卡 + 双 Y 轴趋势图（通过率左轴 / Run 数右轴）
└── features/compare/          批次对比：URL 参数驱动，verdict 过滤 + 可比性警告
```

## 部署形态

```
eval-platform/
├── docker-compose.yaml          六服务本地全栈（db/server/runner/web/prometheus/grafana，轨迹存卷）
├── server/Dockerfile            uv workspace 多阶段构建（--extra s3）
├── runner/Dockerfile            uv workspace 多阶段构建（mock 适配器零依赖）
├── web/Dockerfile               node 构建 + nginx-unprivileged 托管（/api 反代 + SPA fallback）
├── scripts/demo-batch.sh        幂等提交演示批次（mock agent）
└── deploy/
    ├── prometheus.yml           抓取 server /metrics
    ├── alerts.yml               队列积压 / 通过率劣化两条告警
    ├── grafana/provisioning/    数据源 + 五面板看板自动导入
    └── k8s/
        ├── base/                namespace/db/server/runner/web/ingress（Kustomize）
        └── overlays/
            ├── staging/         1 副本
            └── prod/            3 副本 + NetworkPolicy（runner 断网）+ KEDA 队列扩缩
```

测试基线：server 走 FastAPI TestClient + 内存数据库替身全链路验证；
runner 用 httpx MockTransport 模拟内部接口契约；真 PG 集成测试按需开启。
