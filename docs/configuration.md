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
  "llm": {
    "base_url": null,
    "api_key": null
  },
  "context": {
    "auto_compact": true,
    "char_limit": 120000,
    "keep_head": 3,
    "keep_tail": 46,
    "keep_recent_results": 3,
    "large_result_chars": 30000
  },
  "sandbox": {
    "mode": "auto",
    "network": "on",
    "env_policy": "inherit",
    "redirect_home": false,
    "max_memory_mb": 2048,
    "max_cpu_secs": 600,
    "max_processes": 128,
    "max_file_mb": 1024,
    "env_allow": ["PATH", "LANG", "LC_ALL", "TERM", "TMPDIR", "SHELL"]
  },
  "permissions": {
    "mode": "default",
    "allow": ["Read(**)", "Bash(git status:*)"],
    "deny": ["Bash(rm:*)", "Read(.env)"]
  },
  "budget": {
    "max_usd_per_session": 2.0,
    "max_iterations": 40,
    "warn_threshold": 0.8,
    "max_usd_per_turn": 0.0
  },
  "observability": {
    "log_format": "json",
    "otel_endpoint": null
  },
  "hooks": {
    "pre_tool_use": [{"matcher": "run_bash", "command": "scripts/guard.sh"}],
    "post_tool_use": []
  },
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    }
  },
  "todo": {
    "enabled": true,
    "reminder_interval": 3
  },
  "tasks": {
    "enabled": true
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
llm = build_llm_provider(base_url=None, config=config)        # 构建 provider
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

**provider 构建**（`build_llm_provider`）：`base_url` 与 `api_key`
各自独立生效，优先级均为：

```
显式参数（CLI --llm-base-url）
  > settings.json 的 llm 段（llm.base_url / llm.api_key）
  > 环境变量（MCC_LLM_BASE_URL / 各厂商 key）
```

分支语义：

- `base_url` 非空 → OpenAI 兼容网关（vLLM / SGLang / Ollama 等）
  走 OpenAiClient，api_key 仍缺省时用占位值——本地网关通常不鉴权；
- 无 `base_url` → 按 `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` 自动
  探测厂商（Anthropic 优先）；环境变量全空但 settings 配了
  `llm.api_key` 时按 Anthropic 构建（mcc 默认模型系列）；
- 两手皆空 → RuntimeError。

settings.json 配置自托管网关示例（`~/.mcc` 不入库，凭证留在本机）：

```json
{
  "model": {"main": "qwen2.5-coder-32b"},
  "llm": {
    "base_url": "http://vllm.local:8000/v1",
    "api_key": "sk-your-gateway-key"
  }
}
```

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
| `warn_threshold` | `0.8` | 告警阈值（0~1），达 `max_usd_per_session * warn_threshold` 时发 Notice |
| `max_usd_per_turn` | `0.0` | 单轮成本上限（USD），0.0 表示不限；超限发 Notice |

## Goal 配置

见 [goal-loop.md](goal-loop.md)。

| 字段 | 默认 | 说明 |
|---|---|---|
| `enabled` | `true` | 关闭时不注入 GoalController（行为与历史一致） |
| `evaluator_model` | `null` | evaluator 模型，null 回落 summarize_model → main |
| `block_cap` | `8` | 连续 block 上限，超了返回 limit 防死循环 |
| `max_goal_length` | `4000` | condition 最大长度 |

## Hooks 配置

见 [permissions.md](permissions.md#hooks)。

## MCP 配置

见 [mcp.md](mcp.md)。

`mcpServers` 段声明外部 MCP server，键名即 server 名：

```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {"GITHUB_TOKEN": "ghp_xxx"}
    }
  }
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `command` | `str` | 可执行命令 |
| `args` | `list[str]` | 命令参数 |
| `env` | `dict[str,str]` | 额外环境变量 |

工具注册名格式 `mcp__<server>__<tool>`，与 Claude Code 兼容。home 与 project 配置浅合并（project 覆盖同名 server）。

## Todo 配置

见 [todo.md](todo.md)。

```json
{"todo": {"enabled": true, "reminder_interval": 3}}
```

| 字段 | 默认 | 说明 |
|---|---|---|
| `enabled` | `true` | 关闭时不注册 todo_write、不注入 reminder |
| `reminder_interval` | `3` | 连续多少轮未更新触发提醒 |

## Task 配置

见 [tasks.md](tasks.md)。

```json
{"tasks": {"enabled": true}}
```

| 字段 | 默认 | 说明 |
|---|---|---|
| `enabled` | `true` | 关闭时不注册 6 个 task 工具 |

`.tasks/` 已 gitignore。

## .env

项目根目录的 `.env` 自动加载（参考 [.env.example](../.env.example)），已存在的环境变量不会被覆盖。
