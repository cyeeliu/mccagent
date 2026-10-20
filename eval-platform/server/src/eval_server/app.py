"""应用组装：路由、中间件、异常处理与后台任务。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from eval_server import scheduler
from eval_server.api import (
    build_internal_router,
    build_public_router,
    build_system_router,
)
from eval_server.error import ApiError, BadRequestError
from eval_server.state import AppState


def create_app(
    state: AppState,
    *,
    cors_origins: tuple[str, ...] = (),
    start_reaper: bool = True,
) -> FastAPI:
    """按公开 / 内部 / 系统三组路由组装 FastAPI 应用。

    cors_origins 为空时不启用 CORS（进程内测试场景）；
    start_reaper 控制是否随应用生命周期启动租约回收循环。
    """

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        reaper_task = None
        if start_reaper:
            reaper_task = asyncio.create_task(
                scheduler.reaper_loop(state.db, on_tick=state.metrics.update_queue_gauges)
            )
        try:
            yield
        finally:
            if reaper_task is not None:
                reaper_task.cancel()

    app = FastAPI(title="eval-server", lifespan=lifespan)
    app.state.state = state

    public = build_public_router()
    internal = build_internal_router()
    system = build_system_router()
    # 指标标签使用带前缀的路由模板（低基数，路径参数不展开）；
    # 匹配后 scope 中的 route 对象是这里注册的原始对象，据此取回完整模板。
    for prefix, router in (("/api", public), ("/internal", internal), ("", system)):
        for r in router.routes:
            r.metrics_path = prefix + r.path  # type: ignore[attr-defined]
    app.include_router(public, prefix="/api")
    app.include_router(internal, prefix="/internal")
    app.include_router(system)

    @app.exception_handler(ApiError)
    async def on_api_error(_: Request, exc: ApiError):
        return exc.to_response()

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(_: Request, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        loc = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        message = first.get("msg", "invalid request")
        return BadRequestError(f"{loc}: {message}" if loc else message).to_response()

    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_metrics(request: Request, call_next):
        started = time.monotonic()
        response = await call_next(request)
        route = request.scope.get("route")
        path = getattr(route, "metrics_path", None) or request.url.path
        state.metrics.http_request(
            request.method, path, response.status_code, time.monotonic() - started
        )
        return response

    return app
