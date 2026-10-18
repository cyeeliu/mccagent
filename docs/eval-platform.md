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
```

## 路线图

1. 事件契约与 server 骨架（本页所述，已完成）
2. server 完整实现：REST 路由、SSE 广播、租约调度、轨迹存储、指标暴露
3. runner 完整实现：执行主循环、心跳续租、mock 与 mcc 适配器、评分
4. web 功能页面：批次看板、轨迹查看器、对比报告、仪表盘
5. 部署与运维：docker compose 全栈、Prometheus/Grafana、K8s 配置
