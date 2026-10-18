# CLI 用法

## 启动模式

### TUI 模式（默认）

```bash
uv run mcc            # 当前目录为工作区
uv run mcc --cwd /path/to/repo
```

| 按键 | 作用 |
|---|---|
| `Enter` | 发送输入 |
| `Ctrl-C` | 清空输入 / 退出（作为字符读取，不触发 SIGINT） |
| `y` | 权限确认：允许一次 |
| `a` | 权限确认：总是允许（写入运行时 allow 规则） |
| `n` | 权限确认：拒绝 |
| `e` | 权限确认：拒绝并附反馈（输入原因回传给 Agent） |

### headless 模式

```bash
uv run mcc -p "解释这段代码的作用"
uv run mcc -p "..." --quiet    # stderr 不输出工具进度与成本
```

- stdout：最终答案（可管道）
- stderr：工具进度（`⚙ run_bash…`）、错误、成本统计（`[$0.0012 USD]`）
- 退出码：`0` 成功，`1` 执行错误，`2` 配置/用法错误

权限确认在 headless 下无交互通道，`Ask` 决策默认拒绝。

## 会话管理

```bash
uv run mcc sessions                      # 列出全部会话
uv run mcc sessions list
uv run mcc sessions delete <session-id>  # 按 id 删除
uv run mcc sessions purge --older-than 30d
uv run mcc resume <session-id>           # 恢复会话继续对话
```

`list` 输出格式：

```
ID         STARTED           TURNS      COST  TITLE
7acdc77f   2026-10-18 04:33      2   $0.0011  修复超时问题
```

`resume` 从 JSONL 快照重放重建完整消息历史，标题、成本等元数据沿用原会话。详见 [sessions.md](sessions.md)。

## 其它命令

```bash
mcc config     # 打印合并后的有效配置（JSON）
mcc version
mcc update     # 检查 PyPI 新版本并提示升级方式
mcc completion bash   # 输出 shell 补全脚本（bash / zsh / fish）
```

## Mock 模式

无 API key 时体验完整链路（LLM 按预设脚本回放）：

```bash
uv run mcc --mock -p "hello"
uv run mcc --mock-script examples/write_then_read.json -p "write and read"
```

### mock 脚本格式

JSON 数组，每个元素是一轮事件列表，按顺序被 `MockLlm` 消费：

```json
[
  [
    {"kind": "tool_use_start", "id": "t1", "name": "write_file"},
    {"kind": "tool_use_input", "input": {"path": "a.txt", "content": "hello"}},
    {"kind": "message_stop", "stop_reason": "tool_use",
     "usage": {"input_tokens": 10, "output_tokens": 5}}
  ],
  [
    {"kind": "text", "text": "文件已写入"},
    {"kind": "message_stop", "stop_reason": "end_turn",
     "usage": {"input_tokens": 10, "output_tokens": 5}}
  ]
]
```

事件类型：

| kind | 字段 | 说明 |
|---|---|---|
| `text` | `text` | 流式文本增量 |
| `tool_use_start` | `id`, `name` | 工具调用开始 |
| `tool_use_input` | `input` | 工具参数（JSON 对象） |
| `message_stop` | `stop_reason`, `usage` | 轮结束（`end_turn` / `tool_use`） |

`examples/` 目录下有现成脚本：`simple_text.json`、`write_then_read.json`、`bash_echo.json`、`explore_codebase.json`、`parallel_writes.json`、`permission_deny.json`。

> **注意**：首轮快照录制后会触发会话标题生成，额外消耗一轮脚本（`MockLlm.complete` 与 `stream` 共享轮次计数）。mock 脚本请在主流程轮次之间预留一个"标题轮"。

## 环境变量

| 变量 | 说明 |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OPENAI_API_KEY` | OpenAI 兼容端点 key |
| `OPENAI_BASE_URL` | 兼容端点地址（默认 `https://api.openai.com/v1`） |
| `MCC_LLM_BASE_URL` | 自托管网关地址（同 `--llm-base-url`，强制走 OpenAI 兼容协议） |
| `MODEL` | 覆盖所有类型的模型选择 |
| `MCC_PROJECT` | 工作目录（同 `--cwd`） |

优先级详见 [configuration.md](configuration.md)。项目根目录的 `.env` 会被自动加载（已存在的环境变量不覆盖）。
