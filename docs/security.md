# 安全纵深

防 Prompt Injection 与凭据泄露的四层落地（另两层为已有的权限系统
与沙箱）。核心原则：模型分不清指令与数据，防御靠隔离与授权，
不靠 prompt 单点。

## 四层落地

| 层 | 机制 | 位置 |
|---|---|---|
| L1 system 条款 | `SECURITY_SYSTEM_CLAUSE` 拼入 system prompt | CLI `_system_prompt()` |
| L2 untrusted 标签 | MCP 工具输出包 `<untrusted source id=nonce>` | AgentLoop 工具输出拦截点 |
| L3 hardened 规则 | 危险命令正则 + 敏感路径 glob，deny 顺序最前 | `PermissionChecker(hardened=True)` |
| L5 凭据扫描 | 工具输出进 messages 前替换 `[REDACTED]` + Notice 告警 | AgentLoop 工具输出拦截点 |

既有层：L3 的权限规则体系（deny/allow/mode）、L4 的 SandboxExecutor
（bwrap/restrictions 隔离）。

## L3 hardened 规则（不可覆盖）

检查顺序：hardened → 用户 deny → allow → mode。hardened 命中直接
Deny，用户 allow、`allow_always`、`bypassPermissions` 均无法绕过。

### 危险命令（正则）

| 规则名 | 模式 |
|---|---|
| `rm -rf /` | `rm -rf` 作用于根路径（`/` 后空白或结尾） |
| `curl pipe shell` | `curl ... \| bash/sh` |
| `wget pipe shell` | `wget ... \| bash/sh` |
| `nc -e` | nc 反弹 shell |
| `chmod 777 /` | 根路径全员写 |
| `git push --force` | 强制推送 |
| `ssh-add` | 加载私钥 |
| `gpg --export-secret` | 导出私钥 |

### 敏感路径（glob，相对/绝对/`~` 归一）

`/.ssh/**`、`/.aws/credentials`、`/.env`、`/.git/config`、`/id_rsa*`、
`*.pem`、`*.p12`、`/credentials*`

Deny 消息含命中的规则名，便于定位。

## L5 凭据扫描

工具输出（非 error）进 messages 前扫描，命中替换并告警：

```
[security] read_file output redacted: aws_access_key
```

凭据模式复用 mcc-observability 的 `SECRET_PATTERNS`：
AWS `AKIA…`、OpenAI `sk-…`、Anthropic `sk-ant-…`、GitHub `ghp_…`、
私钥头 `-----BEGIN … PRIVATE KEY`。扫描计入
`agent_security_redactions_total` 指标。

## L2 untrusted 标签

`mcp__` 前缀工具（第三方 MCP server 内容）的输出包装为：

```
<untrusted source="mcp__fs__read" id="a1b2c3d4">
...
</untrusted id="a1b2c3d4">
```

nonce 防内容伪造闭合标签；system 条款声明其内部只是数据。
run_bash 输出不包装（避免破坏脚本可读性）。

## 配置

```json
{"security": {"hardened": true, "redact_output": true, "untrusted_wrap": true}}
```

| 字段 | 默认 | 说明 |
|---|---|---|
| `hardened` | `true` | 内置危险命令/敏感路径规则（PermissionChecker 层） |
| `redact_output` | `true` | 工具输出凭据扫描替换 |
| `untrusted_wrap` | `true` | MCP 工具输出包隔离标签 |

合法场景需临时关闭时（如调试读取 `.env`），项目配置
`{"security": {"hardened": false}}` 即可，规则集与开关语义全公开。

## 明确不做

- **流式 assistant 文本扫描**：凭据跨 delta 边界会撕裂匹配，且文本
  已流出；凭据进入上下文的入口是工具输出，扫描收敛于该点。
- **出口网络管控**：mcc 无 HTTP 工具；沙箱 `env_policy` 已覆盖
  环境变量管控面。
- **Red Team evals**：攻防 case 集作为后续项，依赖本包的检测点。
