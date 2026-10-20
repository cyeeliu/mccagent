# 多阶段构建：构建上下文为仓库根（需包含 eval-platform/ 与 packages/ 源码）
FROM ghcr.io/astral-sh/uv:0.9.5-python3.12-trixie-slim AS build
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
COPY eval-platform/pyproject.toml eval-platform/uv.lock ./
COPY eval-platform/server/pyproject.toml server/
COPY eval-platform/runner/pyproject.toml runner/
COPY eval-platform/runner/src runner/src
RUN uv sync --locked --no-dev --package eval-runner

# mcc 系列包不在 workspace 依赖中，从源码安装进同一 venv（mini-claude-code 适配器所需）
COPY packages/mcc-core packages/mcc-core
COPY packages/mcc-config packages/mcc-config
COPY packages/mcc-llm packages/mcc-llm
COPY packages/mcc-tools packages/mcc-tools
COPY packages/mcc-session packages/mcc-session
COPY packages/mcc-harness packages/mcc-harness
RUN uv pip install -e packages/mcc-core -e packages/mcc-config \
    -e packages/mcc-llm -e packages/mcc-tools \
    -e packages/mcc-session -e packages/mcc-harness

FROM ghcr.io/astral-sh/uv:0.9.5-python3.12-trixie-slim AS runtime
# git / ca-certificates：仓库类任务（clone https）所需；ripgrep：代码检索工具后端
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates ripgrep \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 10001 app
COPY --from=build --chown=app:app /app /app
USER app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python", "-m", "runner.main"]
