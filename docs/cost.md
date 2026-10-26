# 成本优化与预算护栏

mcc 提供三层成本控制：Prompt Caching 降本、预算护栏防失控、成本归因可观测。

## Prompt Caching

Anthropic Messages API 支持 prompt caching：把 system prompt 标记为 `ephemeral`，
后续请求命中缓存时按 1/10 价格计费（cache_read_input_tokens）。

mcc 默认开启 prompt caching：AgentLoop 把 system prompt 包成 TextBlock 列表，
在末尾打 `cache_control={"type": "ephemeral"}` 标记。system prompt 在会话中不变，
是最高频的缓存命中点。

### 工作原理

```
AgentLoop._build_system_blocks()
  → [TextBlock(text=system, cache_control=CacheControl(kind="ephemeral"))]

AnthropicClient._build_body()
  → body["system"] = [b.model_dump(by_alias=True) for b in blocks]
  → {"type": "text", "text": "...", "cache_control": {"type": "ephemeral"}}

OpenAiClient._build_body()
  → 剥离 cache_control，拼接成纯文本（OpenAI 标准不认）
```

### 关闭缓存

```python
agent = AgentLoopBuilder(llm, registry, ctx).enable_prompt_cache(False).build()
```

关闭后 system 以纯字符串传递，行为与缓存标记前一致。

## 预算护栏

三层护栏，从松到紧：

| 层级 | 配置 | 行为 |
|------|------|------|
| 告警阈值 | `warn_threshold` (默认 0.8) | 达会话预算的 80% 时发 Notice，不中断 |
| 单轮预算 | `max_usd_per_turn` (默认 0.0=不限) | 单轮成本超限时发 Notice，不中断 |
| 会话硬熔断 | `max_usd_per_session` (默认 2.0) | 达上限时 raise AgentError，终止运行 |

### 配置

`settings.json` 的 `budget` 段：

```json
{
  "budget": {
    "max_usd_per_session": 2.0,
    "max_iterations": 40,
    "warn_threshold": 0.8,
    "max_usd_per_turn": 0.0
  }
}
```

- `warn_threshold`: 0.0~1.0，达 `max_usd_per_session * warn_threshold` 时告警一次
- `max_usd_per_turn`: 0.0 表示不限；设为正数时每轮成本超限发 Notice

### 事件流

告警通过 `Notice` 事件下发，TUI 与评测轨迹均可见：

- `Budget warning: $1.60 / $2.00 (80%)` — 告警阈值
- `Turn 3 cost $0.05 exceeds per-turn limit $0.03` — 单轮预算
- `Budget limit $2.00 reached after 15 iteration(s)` — 硬熔断

## 成本归因

`AgentRun` 返回会话级成本归因：

```python
@dataclass
class AgentRun:
    final_text: str
    iterations: int
    total_usage: Usage
    cost_usd: float
    cache_hit_rate: float          # 缓存命中率
    cost_breakdown: list[TurnCost] # 逐轮成本
```

### cache_hit_rate

```
cache_hit_rate = cache_read_input_tokens
               / (input_tokens + cache_read_input_tokens + cache_creation_input_tokens)
```

Anthropic 的 `input_tokens` 不含缓存部分，分母需三者相加。命中率越高，实际成本越低。

### cost_breakdown

每轮 LLM 调用一个 `TurnCost` 条目：

```python
@dataclass
class TurnCost:
    iteration: int
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    cost_usd: float
```

可用于事后分析哪轮最贵、缓存命中分布是否合理。

## 模型定价

`model_price_per_1m(model)` 返回 `(input_usd_per_1m, output_usd_per_1m)`：

| 模型 | 输入 $/1M | 输出 $/1M |
|------|-----------|-----------|
| claude-opus-4 | 15.0 | 75.0 |
| claude-sonnet-4 | 3.0 | 15.0 |
| claude-haiku-4 | 0.80 | 4.0 |
| gpt-4o | 2.5 | 10.0 |
| gpt-4-turbo | 10.0 | 30.0 |
| gpt-3.5 | 0.5 | 1.5 |
| 其他 | 3.0 | 15.0 |

子代理的 token 预算通过 `token_budget_to_usd(tokens, model)` 换算成 USD 硬限。
