# 可观测性

结构化 JSON 日志、敏感字段脱敏与进程内指标聚合。四支柱中的
Replays（SessionRecorder JSONL）与成本归因（TurnCost/cache_hit_rate）
已分别由 mcc-session 与 mcc-harness 提供，本包负责 Logs 与 Metrics。
分布式 Traces（OTel/Prometheus）对单机 CLI 形态过重，明确不做。

## 架构概览

```
AgentLoop 埋点（builder.observability(sink)）
 ├── llm_call   → _call_with_retry 成功路径（model/latency/tokens/stop_reason/attempt）
 ├── tool_call  → 工具 execute 前后计时（tool/latency/is_error）
 ├── permission → 权限决策点（category/decision）
 └── iteration  → run_streaming 每轮迭代首

ObservabilitySink（mcc-observability）
 ├── logging.Logger("mcc") → JsonFormatter → .mcc/logs/mcc.log.jsonl
 └── Metrics（进程内聚合）→ TUI 退出时 → .mcc/last_metrics.json + stderr 一行汇总

所有日志字段先经 sanitize 脱敏（超长截断 + 敏感字段名置 ***）。
redact_secrets 提供凭据正则扫描（AKIA/sk-/sk-ant-/ghp_/PRIVATE KEY）。
```

## 日志

默认写 `<workdir>/.mcc/logs/mcc.log.jsonl`（`.mcc/` 已 gitignore），
每行一条 JSON：`ts/level/logger/message + 事件字段`。

`log_format=text` 时输出人类可读单行（开发调试用）。

### 关键字段约定

| 事件 | 字段 |
|------|------|
| llm_call | model, latency_ms, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, stop_reason, attempt |
| tool_call | tool, latency_ms, is_error |
| permission | category, decision |
| iteration | iteration |

### 脱敏规则

- 字段名（小写后）含 `password/secret/token/key/authorization/apikey`
  → 值置 `***`
- 字符串超过 500 字符 → 截断为前 200 字符 + `…[len=N]`
- 递归覆盖嵌套 dict/list/tuple

## 指标

进程内聚合（Counter/Histogram/Gauge），无外部依赖。TUI 退出时写
`.mcc/last_metrics.json` 并向 stderr 打印一行汇总
（llm_calls、p95 延迟、cache_hit_rate）。

| 指标 | 类型 | 标签 |
|------|------|------|
| `agent_llm_calls_total` | Counter | model, stop_reason |
| `agent_llm_tokens_total` | Counter | model, kind(input/output/cache_read/cache_write) |
| `agent_llm_latency_ms` | Histogram | model |
| `agent_cache_hit_rate` | Histogram | model |
| `agent_tool_calls_total` | Counter | tool, is_error |
| `agent_tool_duration_ms` | Histogram | tool |
| `agent_permission_decisions_total` | Counter | category, decision |
| `agent_loop_iterations` | Histogram | session_hash |
| `agent_loop_iterations_total` | Counter | — |

Histogram 快照含 count/sum/min/max/p50/p95。

## 配置

```json
{
  "observability": {
    "enabled": true,
    "log_format": "json",
    "log_level": "info",
    "sanitize": true,
    "metrics": true
  }
}
```

| 字段 | 默认 | 说明 |
|------|------|------|
| `enabled` | `true` | 关闭时不构造 sink、不写日志文件 |
| `log_format` | `json` | `json`（JSONL）或 `text`（人类可读） |
| `log_level` | `info` | logging 级别 |
| `sanitize` | `true` | 日志字段脱敏开关 |
| `metrics` | `true` | 进程内指标聚合与退出快照 |

## 设计取舍

- **不做 OTel/Prometheus**：单机 CLI 无分布式调用链，事件流 + JSONL
  日志 + Replays 已覆盖排障需求；未来若需对接 LLM 观测平台
  （LangSmith/Helicone），可在 sink 层加 exporter 而不改埋点。
- **埋点走 sink 注入而非 hooks**：hooks 是外部脚本扩展点
  （turn 级），观测埋点是内部固定路径（llm/tool/permission 粒度），
  两者语义不同，不混用。
