# 可靠性

LLM 调用的错误分类、退避重试、熔断、限流与 turn 超时。
把不稳定外部调用做成基础设施，故障可预期、可恢复。

## 错误分类

`classify_llm_error` 用 duck-typing 读取异常的 `response.status_code`
与 `headers`（兼容 httpx.HTTPStatusError），替代字符串匹配：

| 类别 | 可重试 | 等待策略 |
|---|---|---|
| 429 | 是 | 优先 `retry-after` 头（秒数 ×1000），否则走退避 |
| 5xx（500/502/503…） | 是 | 退避 |
| 超时/连接错误 | 是 | 退避 |
| 其他 4xx（400/401/404…） | 否 | 立即失败（bug 信号） |
| 未知异常 | 否 | 立即失败 |

## 退避策略

`RetryPolicy`：指数退避 `base * 2^attempt` 封顶 `max_delay`，
jitter 开启时取 `[cap/2, cap]` 均匀随机（避免惊群）。

```
attempt 1: ~0.5–1.0s
attempt 2: ~1.0–2.0s
attempt 3: ~2.0–4.0s（默认 max_attempts=4，含首次调用）
```

重试时发 Notice：`rate limited (429) (attempt 1/4), retrying in 1.0s...`

## 熔断器

三态：`closed` →（连续失败达 `circuit_failure_threshold`）→ `open`
（快速失败，抛 `CircuitOpenError`，不触达 LLM）→
（`circuit_reset_timeout` 秒后）→ `half-open`（放行一次探测：
成功归零回 closed，失败重开）。

默认 **禁用**（`circuit_failure_threshold: 0`）——CLI 单用户场景
熔断语义弱于服务端；多并发部署或作为子代理宿主时开启。

## 限流

`ThrottledLlm` 包装 provider：并发信号量 + RPM 最小发送间隔。
默认 **不限**（`rpm: 0`、`max_concurrency: 0`）。CLI 构造时
按需包装，对 AgentLoop 透明。

## 四层超时

| 层 | 时长 | 位置 |
|---|---|---|
| 单次 HTTP | 120s | httpx client |
| 单个 tool | 30s（可配） | run_bash timeout_secs |
| 权限等待 | 120s | permission_timeout_secs |
| 整个 turn | `turn_timeout_secs`（默认 0=不限） | 迭代首协作式检查 |

turn 超时是协作式检查（不取消进行中的调用，由 HTTP 层兜底），
触发时发 Notice 并抛 `AgentError`（recorder 标 errored 收尾）。

## 故障注入

`FlakyProvider(inner, failure_rate, seed)` 按概率抛 503（带
duck-typed response，分类判可重试），开发期验证重试/熔断行为，
seed 固定可复现。

## 配置

```json
{
  "reliability": {
    "max_attempts": 4,
    "base_delay": 0.5,
    "max_delay": 30,
    "jitter": true,
    "circuit_failure_threshold": 0,
    "circuit_reset_timeout": 30,
    "rpm": 0,
    "max_concurrency": 0,
    "turn_timeout_secs": 0
  }
}
```

| 字段 | 默认 | 说明 |
|---|---|---|
| `max_attempts` | `4` | 最大尝试次数（含首次） |
| `base_delay` | `0.5` | 退避基数（秒） |
| `max_delay` | `30` | 退避封顶（秒） |
| `jitter` | `true` | equal jitter 开关 |
| `circuit_failure_threshold` | `0` | 熔断阈值；0 禁用 |
| `circuit_reset_timeout` | `30` | open 后探测等待（秒） |
| `rpm` | `0` | 每分钟请求上限；0 不限 |
| `max_concurrency` | `0` | 并发上限；0 不限 |
| `turn_timeout_secs` | `0` | turn 超时；0 不限 |
