# mcc — mini-claude-code

一个用 Python / asyncio 从零构建的企业级 AI 编码助手：终端 TUI 对话、流式输出、工具调用、权限门控、会话持久化与子代理并行执行。

## 特性

- **交互式 TUI** — 流式渲染、工具进度、成本面板、交互式权限确认（允许 / 总是允许 / 拒绝 / 附反馈拒绝）
- **10 个内建工具** — 读 / 写 / 编辑 / bash / grep / glob / 目录浏览 / 记忆读写 / 子代理派生
- **权限系统** — deny 优先的规则门控，bash 命令按词边界匹配前缀，支持交互确认与运行时放行
- **Hooks** — pre/post-tool 外部脚本钩子，可阻止工具执行或改写输入输出
- **会话持久化** — JSONL 逐轮快照落盘，`resume` 恢复上下文继续对话，自动生成会话标题
- **长期记忆** — Markdown 记忆库 + 记忆索引注入 system prompt
- **子代理并行执行** — 预设角色（代码探索 / 测试运行 / 摘要），信号量并发上限 + 超时 + token 预算三重护栏
- **多 provider** — Anthropic 原生 API + OpenAI 兼容端点（DeepSeek / Kimi / Qwen / Groq / vLLM / OpenRouter），`--llm-base-url` 一键接入企业自托管网关
- **成本追踪** — 按模型定价表计算 USD 成本，会话级预算熔断
- **开箱即用** — pipx/uv 一键安装、Docker 镜像、shell 补全、首次运行引导；无遥测

## 架构

8 个 package，职责单一，自底向上：

```
          mcc-cli
             │
        ┌────┴────┐
        │         │
     mcc-tui   mcc-harness
                   │
        ┌───────────┼───────────┐
        │           │           │
    mcc-tools   mcc-session   mcc-llm
        │           │           │
        └───────────┴─────┬─────┘
                          │
                      mcc-core
                          │
                      mcc-config
```

| package | 职责 |
|---|---|
| `mcc-core` | 核心类型：Message / ContentBlock / Tool / AgentEvent / AgentError |
| `mcc-config` | 配置加载（settings.json，home + project 合并） |
| `mcc-llm` | LLM provider 抽象与实现（Anthropic / OpenAI 兼容 / Mock） |
| `mcc-tools` | 内建工具集合（read / write / edit / bash / grep / glob / list） |
| `mcc-session` | 会话持久化 + 长期记忆（JSONL / Markdown） |
| `mcc-harness` | Agent 主循环、权限、hooks、子代理 |
| `mcc-tui` | Rich/Textual 终端 UI |
| `mcc-cli` | 最终入口 `mcc` |

详见 [docs/architecture.md](docs/architecture.md)。

仓库另含 **mcc-eval** 评测包（`packages/mcc-eval`）：轻量 Agent 评测 harness——单进程 CLI 批量跑基准（HumanEval/MBPP/SWE-bench 采样）、逐 case 落 JSONL 轨迹、离线重评与 Markdown 报告。用法见 [packages/mcc-eval/README.md](packages/mcc-eval/README.md)。

## 快速开始

安装（要求 Python ≥ 3.12；从源码运行：`uv sync` 后 `uv run mcc`）：

```bash
pipx install mcc-cli
# 或
uv tool install mcc-cli
```

配置 API key（也可写入项目根目录 `.env`，见 [.env.example](.env.example)）：

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # Anthropic 原生
# 或
export OPENAI_API_KEY=sk-...          # OpenAI 兼容端点
export OPENAI_BASE_URL=https://api.deepseek.com/v1
```

启动：

```bash
mcc            # TUI 模式
mcc -p "..."   # headless 单次执行
```

自托管网关（vLLM / SGLang / Ollama 等 OpenAI 兼容端点，代码不出域）：

```bash
mcc --llm-base-url https://llm-gateway.internal.corp/v1 -p "hi"
```

无 API key 时 headless 模式打印首次运行引导；体验完整链路可用 mock 模式：

```bash
mcc --mock -p "hello"
mcc --mock-script examples/write_then_read.json -p "write and read"
```

## CLI 速查

| 命令 | 说明 |
|---|---|
| `mcc` / `mcc tui` | 进入 TUI 交互模式 |
| `mcc -p <prompt>` | headless 单次执行，stdout 输出答案 |
| `mcc -p <prompt> --quiet` | 静默模式，只输出最终答案 |
| `mcc --model <name>` | 覆盖主模型 |
| `mcc --cwd <dir>` | 指定工作目录（或 `MCC_PROJECT` 环境变量） |
| `mcc config` | 打印合并后的有效配置 |
| `mcc sessions` | 列出会话（id / 时间 / 轮数 / 成本 / 标题） |
| `mcc sessions delete <id>` | 按 id 删除会话 |
| `mcc sessions purge --older-than 30d` | 清理 N 天前的会话 |
| `mcc resume <id>` | 恢复历史会话继续对话 |
| `mcc completion <bash\|zsh\|fish>` | 输出 shell 补全脚本 |
| `mcc update` | 检查 PyPI 新版本并提示升级 |
| `mcc version` | 版本号 |

## 文档

| 文档 | 内容 |
|---|---|
| [架构](docs/architecture.md) | 包依赖、事件流、Agent 主循环 |
| [CLI 用法](docs/usage.md) | 全命令详解、TUI 按键、mock 脚本格式 |
| [配置](docs/configuration.md) | settings.json、多类型模型解析、环境变量 |
| [工具系统](docs/tools.md) | 10 个内建工具的输入输出契约 |
| [权限与 Hooks](docs/permissions.md) | 规则语法、决策顺序、外部脚本钩子 |
| [会话与记忆](docs/sessions.md) | JSONL 快照、resume、Markdown 记忆库 |
| [子代理](docs/subagents.md) | 预设角色、并发护栏、fan-out 用法 |
| [评测](docs/evals.md) | 评测方法：三层断言、失败回流、多次采样 |
| [评测包](packages/mcc-eval/README.md) | mcc-eval：CLI、case 结构、产物布局 |
| [安装与自托管](docs/packaging.md) | pipx/Docker 安装、自托管 LLM 网关、发布流程 |
| [踩坑与排障](docs/troubleshooting.md) | 已修复的 bug 与已知陷阱记录 |

## 开发

```bash
uv run ruff check .                               # lint
uv run --all-packages pytest tests/ packages/ -q  # 全量测试
```

评测冒烟（mock，无需 API key，CI 的 eval-smoke job 同款）：

```bash
uv run --package mcc-eval mcc-eval run \
  --cases packages/mcc-eval/examples/mock-cases.json --mock --runs-per-case 2
uv run --package mcc-eval mcc-eval report eval-runs/<run-id>
```

## 隐私

mcc 不收集任何遥测数据，不会上报 prompt、代码或使用统计。

## 许可证

MIT
