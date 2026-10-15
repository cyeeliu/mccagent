# mcc — mini-claude-code (Python 版)

一个用 Python/asyncio 从零构建的类 Claude Code 编码助手。

## 架构

8 个 package，职责单一，自底向上：

```
          mcc-cli
             │
        ┌────┴────┐
        │         │
     mcc-tui   mcc-harness
                   │
      ┌────────────┼────────────┐
      │            │            │
   mcc-tools   mcc-session   mcc-llm
      │            │            │
      └────────────┴────┬───────┘
                        │
                    mcc-core
                        │
                    mcc-config
```

| package | 职责 |
|---|---|
| `mcc-core` | 核心类型：Message / ContentBlock / Tool / AgentEvent / AgentError |
| `mcc-config` | 配置加载 (settings.json，home + project 合并) |
| `mcc-llm` | LLM provider 抽象与实现 (Anthropic / OpenAI) |
| `mcc-tools` | 内建工具集合 (read/write/edit/bash/grep/glob/list) |
| `mcc-session` | session 持久化 + 记忆 (JSONL) |
| `mcc-harness` | agent loop、permissions、hooks、skills、subagent |
| `mcc-tui` | Rich/Textual 终端 UI |
| `mcc-cli` | 最终入口 `mcc` |

## 快速开始

```bash
uv sync
uv run pytest
```

## 环境变量

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# 或
export OPENAI_API_KEY=sk-...
```

## 许可证

MIT
