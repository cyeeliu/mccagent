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
# s3 extra：镜像内置 S3 轨迹后端（TRACE_BACKEND=s3 时启用）
RUN uv sync --locked --no-dev --package eval-server --extra s3

FROM ghcr.io/astral-sh/uv:0.9.5-python3.12-trixie-slim AS runtime
# 预创建轨迹目录并归属 app 用户：命名卷首次挂载会继承挂载点的属主，
# 不预创建则卷由 root 拥有，非 root 运行时 ready 探针写探测文件会失败
RUN useradd -m -u 10001 app && \
    mkdir -p /traces && chown app:app /traces && \
    printf 'import os,sys,urllib.request\nport=os.environ.get("PORT","8081")\ntry:\n    ok=urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz/ready",timeout=3).status==200\nexcept Exception:\n    ok=False\nsys.exit(0 if ok else 1)\n' > /healthcheck.py
COPY --from=build --chown=app:app /app /app
USER app
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
EXPOSE 8081
ENTRYPOINT ["python", "-m", "eval_server"]
