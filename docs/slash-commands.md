# Slash 命令与首启配置

两种 slash 语义共存于同一输入框：

- **本地命令**（SlashRegistry）：/model /config /help /status /goal——
  直接执行不进 AgentLoop，结果以 Notice 回显
- **内容替换**（slash_resolver）：/skill-name——SKILL.md 注入
  prompt 后进 AgentLoop（Skills 机制，见 skills.md）

## 内置命令

| 命令 | 说明 |
|---|---|
| `/help` | 列出内置命令与已加载 skills |
| `/status` | 当前会话状态（session/model/历史消息数/网关） |
| `/model` | 无参显示当前模型；`/model <id>` 切换主模型（即刻生效并写配置）；`/model <main\|subagent\|summarize> <id>` 指定类型 |
| `/config` | 无参查看生效配置摘要（凭据脱敏）；`/config get <key>`；`/config set <key> <value>` |
| `/goal` | 查看/设置/清除 goal loop（`/goal clear` 清除；`/goal <文本>` 设置并把文本作为任务输入） |

### /config set 白名单与生效语义

| key | 生效时机 |
|---|---|
| `model.main` | 即刻（运行中热切换，history/session 保留） |
| `model.subagent` / `model.summarize` | 下个会话 |
| `llm.base_url` / `llm.api_key` | 下个会话 |
| `budget.max_usd_per_session` | 即刻 |
| `reliability.turn_timeout_secs` | 即刻 |
| `permissions.mode` / `sandbox.mode` | 下个会话 |

白名单外的 key 拒绝修改并列出可用项。写回的是 home 层
`~/.mcc/settings.json`（dict 级原子写，0600），不固化
"回落 main" 的派生值——project 层与环境变量的覆盖语义不变。

## 首次运行向导

TUI 启动且未配置 LLM 凭证（无 env key、无 `--llm-base-url`、
home settings 无 `llm.api_key`/`base_url`）时进入交互式向导：

```
1. Provider 类型：anthropic 官方 / gateway 自托管网关
2. Base URL（仅网关）
3. API key（密文输入，网关可留空）
4. 主模型 ID（默认 claude-opus-4-8）
```

写入 `~/.mcc/settings.json` 的 `llm` 与 `model.main` 段，
其余保持默认，配置完成直接进入 TUI 无需重启。

- headless（`-p`）不可交互：保留文本引导 + exit 2
- `--no-wizard` 显式跳过向导
- 已有 env key 或 settings 凭证时静默跳过
- 凭证安全：settings.json 0600；env 变量优先级最高；
  `/config` 展示一律脱敏（`sk-***…尾4`）

## 扩展命令

`mcc_cli.slash.build_default_registry()` 返回默认注册表；
`SlashRegistry.register(name, description, handler)` 可注册
自定义本地命令。handler 签名 `(source, args) -> SlashResult`
（可 async），`SlashResult.text` 回显、`SlashResult.prompt`
替换输入继续走后续链路。
