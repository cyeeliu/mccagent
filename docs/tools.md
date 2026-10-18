# 工具系统

主代理默认注册 10 个工具。工具实现 `Tool` 协议（`name` / `description` / `input_schema()` / `execute(input, ctx)`），在 `ToolContext`（cwd / session_id / depth）中执行，每次调用经过权限检查与 hooks（见 [permissions.md](permissions.md)）。

## 文件与进程工具（mcc-tools）

### read_file

读取文本文件，支持分页。

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `path` | string | 是 | 相对（基于 cwd）或绝对路径 |
| `offset` | integer | 否 | 起始行（0 基） |
| `limit` | integer | 否 | 读取行数上限 |

### list_dir

树状列出目录结构，跳过隐藏目录与 `__pycache__`。

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `path` | string | 是 | 目录路径 |
| `max_depth` | integer | 否 | 最大深度（默认 3，上限 10） |

### write_file

写入新文件。**已存在的文件会被拒绝**（防止意外覆盖，修改已有内容用 `edit_file`）。

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `path` | string | 是 | 文件路径 |
| `content` | string | 是 | 写入内容 |

### edit_file

精确字符串替换修改文件。

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `path` | string | 是 | 文件路径 |
| `old_string` | string | 是 | 被替换文本 |
| `new_string` | string | 是 | 替换后文本 |
| `replace_all` | boolean | 否 | 替换全部出现（默认 false，此时 old_string 必须唯一） |

### run_bash

`bash -c` 执行 shell 命令，返回 stdout/stderr 与退出码。

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `command` | string | 是 | 命令 |
| `timeout_secs` | integer | 否 | 超时（默认 30s） |

### grep

正则搜索文件内容，输出 `path:line: 内容`。

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `pattern` | string | 是 | 正则表达式 |
| `path` | string | 否 | 搜索根目录（默认 cwd） |
| `include` | string | 否 | 文件名过滤 glob（如 `*.py`） |

### glob

按 glob 模式列出文件路径，按修改时间排序。

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `pattern` | string | 是 | glob 模式（如 `src/**/*.ts`） |
| `path` | string | 否 | 根目录（默认 cwd） |

## 记忆工具

长期记忆的读写入口，数据落在 `~/.mcc/memory/`，详见 [sessions.md](sessions.md#长期记忆)。

### save_memory

保存一条记忆（name 需为文件名安全的形式，路径穿越会被拒绝）。

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `name` | string | 是 | 稳定 id，如 `user_role` |
| `type` | string | 是 | 记忆类型（见 sessions.md） |
| `description` | string | 是 | 一行描述（进入索引） |
| `body` | string | 是 | 正文 |

### read_memory

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `name` | string | 是 | 索引中的记忆 id |

## 子代理派生（mcc-harness）

### spawn_subagent

按预设派生子代理，独立上下文执行一次性任务，只返回精炼结论。详见 [subagents.md](subagents.md)。

| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `preset` | enum | 是 | `code_explorer` / `test_runner` / `summarizer` |
| `task` | string | 是 | 自包含的一次性指令 |

返回格式：

```
[subagent=code_explorer tokens=1523in/204out reason=end_turn]
<精炼结论>
```

## 工具注册表

`ToolRegistry` 提供 `register` / `get` / `names` / `as_api_schema`（LLM API schema，按名排序）；`subset(allowed)` 返回只含指定工具的新注册表——子代理即通过 subset 获得工具子集（天然不含 `spawn_subagent` 自身，防止递归派生）。
