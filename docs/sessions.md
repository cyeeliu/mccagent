# 会话与记忆

## 会话录制

每次启动（TUI / headless / resume）自动开启会话录制，落在 `~/.mcc/sessions/<YYYY-MM-DD>/<id>/`：

| 文件 | 内容 |
|---|---|
| `<id>.jsonl` | 每轮一个 `TurnSnapshot`（JSON Lines 追加） |
| `<id>.meta.json` | 元数据：轮数、成本、状态、标题 |

`TurnSnapshot` 记录完整请求消息、assistant 内容块、工具输出（id / 内容 / 是否错误）、token 用量、模型与该轮成本——足以完整重放对话。

元数据在每轮录制时原子重写；`cost_usd` 只从 `TurnSnapshot.cost_usd` 累加（由 AgentLoop 按模型定价表计算），避免二次估算造成定价漂移。

## 会话标题

首轮快照录制后自动生成简短标题（摘要模型，见 [configuration.md](configuration.md#模型配置与解析)）：

- 输出配额 `TITLE_MAX_TOKENS = 512`：推理模型的思考 token 计入配额，须为其留足余量
- 后处理取首行并截断到 `TITLE_MAX_CHARS = 30` 字符（提示词长度约束不具强制力）
- 生成失败静默跳过，不影响主流程

## 会话管理

```bash
uv run mcc sessions                       # 列表（按开始时间降序）
uv run mcc sessions delete <id>           # 按 id 删除（meta + JSONL 全部移除）
uv run mcc sessions purge --older-than 30d
```

`delete` 与 `purge` 共用同一套文件删除逻辑（`.meta.json` / `.jsonl` / `.messages.json`）。

## 恢复会话

```bash
uv run mcc resume <session-id>
```

从 JSONL 逐行重放重建消息历史：每行快照携带完整 request_messages，assistant 块与工具结果按原顺序追加（工具结果以 `Role.tool` 消息回填，与运行时结构一致）。恢复后沿用原会话 id 继续录制，轮数与成本累计。

恢复的会话在 TUI 中**完整回显历史轨迹**：用户消息、助手文本、工具调用与结果按原顺序重放渲染，成本面板显示会话累计成本——历史不只是送进 LLM 的上下文前缀，界面上同样可见。

## 长期记忆

跨会话的 Markdown 记忆库，落在 `~/.mcc/memory/`：

```
~/.mcc/memory/
├── MEMORY.md          # 索引（name + type + description）
└── entries/
    └── user_role.md   # 单条记忆正文
```

Agent 通过两个工具读写记忆（详见 [tools.md](tools.md#记忆工具)）：

- `save_memory(name, type, description, body)` — name 做文件名安全校验（防路径穿越），type 限定固定枚举
- `read_memory(name)` — 按索引 id 读取

每次启动时，记忆索引（name + description）注入 system prompt 的 `# Memory Index` 段，Agent 据此判断何时读取完整条目。记忆读写属 `Memory` 权限类别，默认放行。
