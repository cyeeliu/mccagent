# --- builder: 安装依赖到独立 venv ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY packages/ packages/
RUN uv sync --frozen --no-dev

# --- runtime: 最小镜像 + 工具依赖 ---
FROM python:3.12-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates bash git ripgrep \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -s /bin/bash agent
WORKDIR /workspace
COPY --from=builder /app /app
ENV PATH="/app/.venv/bin:$PATH"
USER agent
ENTRYPOINT ["mcc"]
