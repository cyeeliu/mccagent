# 配置

## 配置文件位置

| 位置 | 作用 |
|---|---|
| `~/.mcc/settings.json` | 用户级配置（首次运行自动创建默认配置） |
| `<project>/.mcc/settings.json` | 项目级配置，逐字段覆盖用户级 |

合并规则：**逐字段合并**，project 只覆盖显式写入的字段，不整体替换某一段（例如 project 只写 `model.subagent` 时，`model.main` 仍取用户级值）。

`mcc config` 打印合并后的有效配置。

## 完整配置示例

```json
{
  "model": {
    "main": "claude-opus-4-8",
    "subagent": null,
    "summarize": null
  },
  "permissions": {
    "mode": "default",
    "allow": ["Read(**)", "Bash(git status:*)"],
    "deny": ["Bash(rm:*)", "Read(.env)"]
  },
  "budget": {
    "max_usd_per_session": 2.0,
    "max_iterations": 40
  },
  "observability": {
    "log_format": "json",
    "otel_endpoint": null
  },
  "hooks": {
    "pre_tool_use": [{"matcher": "run_bash", "command": "scripts/guard.sh"}],
    "post_tool_use": []
  }
}
```

## 模型配置与解析

`model` 段支持三种用途类型：

| 类型 | 用途 |
|---|---|
| `main` | 主对话模型 |
| `subagent` | 子代理预设使用的模型 |
| `summarize` | 会话标题生成等摘要任务 |

**统一入口**：模型解析与 LLM provider 构建由 `mcc-config` 包提供，
CLI 与评测（mcc-eval）调用同一 API，不存在第二份实现：

```python
from mcc_config import resolve_model, build_llm_provider

model = resolve_model("main", explicit_arg, config=config)   # 解析模型名
llm = build_llm_provider(base_url=None)                       # 构建 provider
```

**解析优先级**（每类型独立生效）：

```
显式参数（CLI --model / 评测 --model）
  > 环境变量 MODEL
  > settings.json 中该类型的配置
  > settings.json 中的 main 配置（兜底）
  > DEFAULT_MODEL（与 ModelConfig.main 默认值同源）
```

即：`subagent` / `summarize` 未配置（`null`）时一律回落 `main`；设置 `MODEL` 环境变量会同时覆盖三者；`--model` 只覆盖主模型，不影响子代理与摘要。

**provider 构建**（`build_llm_provider`）：显式 `base_url`（CLI
`--llm-base-url`）或 `MCC_LLM_BASE_URL` 环境变量指向 OpenAI 兼容
自托管网关（vLLM / SGLang / Ollama 等）时走 OpenAiClient（API key
缺省占位值——本地网关通常不鉴权）；否则按 `ANTHROPIC_API_KEY` /
`OPENAI_API_KEY` 自动探测（Anthropic 优先），均缺失时抛
RuntimeError。

示例——主对话用 Opus、子代理用便宜的 Haiku：

```json
{
  "model": {
    "main": "claude-opus-4-8",
    "subagent": "claude-haiku-4-5"
  }
}
```

## 权限配置

见 [permissions.md](permissions.md)。

## 预算配置

| 字段 | 默认 | 说明 |
|---|---|---|
| `max_usd_per_session` | `2.0` | 单次会话成本上限（USD），超限熔断 |
| `max_iterations` | `40` | 单次运行最大迭代轮数 |

## Hooks 配置

见 [permissions.md](permissions.md#hooks)。

## .env

项目根目录的 `.env` 自动加载（参考 [.env.example](../.env.example)），已存在的环境变量不会被覆盖。
