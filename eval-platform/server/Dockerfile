# 多阶段构建：uv workspace 上下文（构建上下文为 eval-platform/ 根）
FROM ghcr.io/astral-sh/uv:0.9.5-python3.12-trixie-slim AS build
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
# 锁文件校验需要 workspace 全部成员的元数据
COPY pyproject.toml uv.lock ./
COPY server/pyproject.toml server/
COPY runner/pyproject.toml runner/
COPY server/src server/src
COPY server/migrations server/migrations
# s3 extra：容器栈轨迹后端默认走 S3/MinIO
RUN uv sync --locked --no-dev --package eval-server --extra s3

FROM ghcr.io/astral-sh/uv:0.9.5-python3.12-trixie-slim AS runtime
RUN useradd -m -u 10001 app && \
    printf 'import os,sys,urllib.request\nport=os.environ.get("PORT","8081")\ntry:\n    ok=urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz/ready",timeout=3).status==200\nexcept Exception:\n    ok=False\nsys.exit(0 if ok else 1)\n' > /healthcheck.py
COPY --from=build --chown=app:app /app /app
USER app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
EXPOSE 8081
ENTRYPOINT ["python", "-m", "eval_server"]
