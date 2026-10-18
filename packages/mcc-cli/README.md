# mcc — mini-claude-code

一个用 Python / asyncio 构建的企业级 AI 编码助手：终端 TUI 对话、流式输出、工具调用、权限门控、会话持久化与子代理并行执行。

## 安装

```bash
pipx install mcc-cli
# 或
uv tool install mcc-cli
```

## 快速开始

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # 或 OPENAI_API_KEY（兼容 DeepSeek/Kimi/Qwen/vLLM 等端点）

mcc                    # TUI 交互模式
mcc -p "解释这段代码"   # headless 单次执行
```

## 主要能力

- 10 个内建工具（读 / 写 / 编辑 / bash / grep / glob / 目录 / 记忆 / 子代理派生）
- 权限规则门控 + 交互式确认 + 外部脚本 hooks
- 会话录制、按 id 恢复、长期记忆
- 子代理并行执行（预设角色 + 超时 / 预算 / 并发三重护栏）
- 成本追踪与会话预算熔断

## 文档

完整文档见 [mccagent 仓库](https://github.com/cyeeliu/mccagent)：

- [CLI 用法](https://github.com/cyeeliu/mccagent/blob/main/docs/usage.md)
- [配置](https://github.com/cyeeliu/mccagent/blob/main/docs/configuration.md)
- [Docker 与自托管](https://github.com/cyeeliu/mccagent/blob/main/docs/packaging.md)
- [踩坑与排障](https://github.com/cyeeliu/mccagent/blob/main/docs/troubleshooting.md)

## 许可证

MIT
