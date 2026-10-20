# 多阶段构建：uv workspace 上下文（构建上下文为 eval-platform/ 根）
FROM ghcr.io/astral-sh/uv:0.9.5-python3.12-trixie-slim AS build
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
COPY pyproject.toml uv.lock ./
COPY server/pyproject.toml server/
COPY runner/pyproject.toml runner/
COPY runner/src runner/src
# 缺省只装 mock 适配器依赖；真实 Agent 适配器按需加 --extra anthropic / --extra langgraph
RUN uv sync --locked --no-dev --package eval-runner

FROM ghcr.io/astral-sh/uv:0.9.5-python3.12-trixie-slim AS runtime
RUN useradd -m -u 10001 app
COPY --from=build --chown=app:app /app /app
USER app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python", "-m", "runner.main"]
