# 安装、Docker 与自托管

## 安装

```bash
pipx install mcc-cli
# 或
uv tool install mcc-cli
```

从源码运行（开发模式）：

```bash
git clone https://github.com/cyeeliu/mccagent
cd mccagent
uv sync
uv run mcc
```

## Shell 补全

```bash
mcc completion bash > ~/.local/share/bash-completion/completions/mcc
mcc completion zsh  > ~/.zfunc/_mcc   # 并在 .zshrc 启用 compinit
mcc completion fish  > ~/.config/fish/completions/mcc.fish
```

## 版本更新检查

```bash
mcc update
```

比对 PyPI 上的最新发布版本并给出升级命令（`uv tool upgrade mcc-cli` / `pipx upgrade mcc-cli`）。仅按需检查——mcc 不会在后台静默请求网络。

## Docker

```bash
docker build -t mcc .

# 挂载当前目录为工作区
docker run --rm -it \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -v "$PWD":/workspace \
  mcc -p "list project files"
```

镜像设计：

- 多阶段构建：builder 阶段用 uv 锁定依赖（`uv sync --frozen`），runtime 基于 `python:3.12-slim`
- runtime 只装运行必需工具（bash / git / ripgrep / ca-certificates）
- 以非 root 用户 `agent` 运行，工作区固定挂载到 `/workspace`

## 企业自托管 LLM

内网部署（代码不出域）时，把 mcc 指向 OpenAI 兼容网关即可——vLLM、SGLang、Ollama、商用 LLM 网关均支持：

```bash
mcc --llm-base-url https://llm-gateway.internal.corp/v1 -p "hi"
# 或环境变量
export MCC_LLM_BASE_URL=https://llm-gateway.internal.corp/v1
```

- 显式 base-url 强制走 OpenAI 兼容协议（`OPENAI_API_KEY` 存在则带上，网关无鉴权时自动用占位值）
- 配合 `MODEL` 环境变量或 settings.json 指定网关后的模型名
- mock 优先级高于 base-url（`--mock` 仍可用于无网络测试）

## 发布流程

- **CI**（`.github/workflows/ci.yml`）：push / PR 触发 ruff + 全量 pytest
- **发布**（`.github/workflows/release.yml`）：推送 `v*` tag 触发——质量门通过后 `uv build --all-packages` 构建全部 8 个包，经 Trusted Publishing 发布到 PyPI

```bash
git tag v0.1.0
git push origin v0.1.0   # CI 自动构建并发布
```

首次发布前需在 PyPI 上为 8 个包名（mcc-core / mcc-config / mcc-llm / mcc-tools / mcc-session / mcc-harness / mcc-tui / mcc-cli）完成注册，并在 GitHub 仓库配置 `pypi` environment 的 Trusted Publisher。

## 隐私

mcc 不收集任何遥测数据，不会上报 prompt、代码或使用统计。
