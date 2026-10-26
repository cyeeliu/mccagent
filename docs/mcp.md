# MCP - Model Context Protocol

mcc 通过 stdio JSON-RPC 2.0 连接外部 MCP server，把 server 暴露的工具适配成 mcc 的 `Tool` 协议注册到 `ToolRegistry`，AgentLoop 透明使用——主循环无需任何改动。

## 协议

MCP server 是一个长驻子进程，与 mcc 通过 stdin/stdout 交换 newline-delimited JSON-RPC 2.0 消息：

```
mcc                          MCP server
 │                                │
 │── initialize ─────────────────→│
 │←── initialize result ──────────│
 │── notifications/initialized ──→│
 │── tools/list ─────────────────→│
 │←── tools list ─────────────────│
 │── tools/call {name, args} ────→│
 │←── call result ────────────────│
```

握手后 mcc 缓存工具列表，模型调用 `mcp__<server>__<tool>` 时转发给对应 server。

## 配置

在 `settings.json` 顶层 `mcpServers` 段声明 server（键名即 server 名，与 Claude Code 兼容）：

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    },
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
| `command` | `str` | 可执行命令（如 `npx`、`python`、`node`） |
| `args` | `list[str]` | 命令参数 |
| `env` | `dict[str,str]` | 额外环境变量（合并到 `os.environ` 之上） |

配置合并规则与其它段一致：home `settings.json` 与 project `.mcc/settings.json` 的 `mcpServers` 做 dict 浅合并（project 覆盖同名 server）。

## 工具命名

MCP 工具注册名格式：`mcp__<server>__<tool>`

- 前缀 `mcp__` 防与内建工具冲突
- server 名与工具名中的非法字符（非 `[a-zA-Z0-9_-]`）替换为下划线
- 总长度上限 64 字符

例：server `fs.server` 的工具 `read/file` → `mcp__fs_server__read_file`

## 生命周期

- **启动**：CLI `_build_source` 阶段 `asyncio.run(mcp_manager.connect_all(servers))`，逐个 spawn 子进程并握手；单个 server 失败不中断其他 server
- **注册**：`mcp_manager.register_all(registry)` 把发现的工具适配后注册到 `ToolRegistry`
- **调用**：模型发起 `mcp__<server>__<tool>` 调用时，`McpToolAdapter.execute` 通过 `McpClient.call_tool` 转发 JSON-RPC `tools/call`，把返回的 text content 拼成 `ToolOutput.content`，`isError` 映射到 `is_error`
- **退出**：主进程退出时子进程自动清理；`McpClient.close` 先 terminate，5 秒不退则 kill

## 容错

- server 命令不存在 → `McpError`，该 server 跳过，其他 server 正常
- server 返回 JSON-RPC error → `McpError` 抛出，工具调用失败但不崩溃
- server 提前关闭 stdout → `McpError`
- 工具名过长 / 缺 name → 该工具跳过，日志 warning

## 与 AgentLoop 的关系

AgentLoop **零改动**。MCP 工具通过 `McpToolAdapter` 适配 `Tool` 协议后注册到现有 `ToolRegistry`，主循环按普通工具调度。模型在 API schema 里看到 `mcp__*` 工具，调用流程与内建工具完全一致。

## 包结构

```
packages/mcc-mcp/
├── pyproject.toml          # 依赖 mcc-core
└── src/mcc_mcp/
    ├── __init__.py         # 导出 McpClient/McpClientManager/McpError/McpToolAdapter/normalize_mcp_name
    ├── adapter.py          # McpToolAdapter + normalize_mcp_name
    ├── client.py           # McpClient（stdio JSON-RPC）+ McpError
    └── manager.py          # McpClientManager（多 server 管理）
```

## 参考

- [MCP 协议规范](https://modelcontextprotocol.io/)
- [learn-claude-code s14](../../learn-claude-code/s14_mcp_plugin/) — in-process mock 蓝本
- [rust_agents ch14a](../../rust_agents/src/part3/ch14a-mcp.md) — stdio JSON-RPC 教材
