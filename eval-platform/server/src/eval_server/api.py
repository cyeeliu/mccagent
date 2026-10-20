"""HTTP 接口：公开端点、Runner 内部端点与端点契约清单。

/api 给前端与 CI，/internal 给 Runner（Bearer token 鉴权，生产建议
叠加网络级隔离）。端点清单常量（PUBLIC/INTERNAL/SYSTEM_ENDPOINTS）
是 API 的文档化契约，路由注册与清单必须保持一致（由测试钉住）。
"""

from __future__ import annotations

import asyncio
import json
import secrets
import uuid as uuid_module
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated, Any

import asyncpg
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from eval_server import scheduler
from eval_server.db import DbLike
from eval_server.domain import (
    TraceEvent,
    is_milestone,
    trace_event_adapter,
    validate_for_run,
)
from eval_server.error import (
    BadRequestError,
    ConflictError,
    NotFoundError,
    UnauthorizedError,
)
from eval_server.metrics import Metrics
from eval_server.state import AppState
from eval_server.store import TraceStoreProtocol
from eval_server.stream import Topic

VALID_SCAFFOLDS = ("mock", "anthropic", "langgraph", "openai-agents", "mini-claude-code")
VALID_COMPLETE_STATUSES = ("passed", "failed", "error", "timeout")
MAX_CASES_PER_BATCH = 10_000
MAX_PARALLELISM = 256
MAX_PAGE_LIMIT = 1_000
SSE_KEEPALIVE_S = 15.0

# ─── 端点契约清单 ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class EndpointSpec:
    method: str
    path: str
    summary: str


PUBLIC_ENDPOINTS: tuple[EndpointSpec, ...] = (
    EndpointSpec("POST", "/api/profiles", "创建 Agent 配置"),
    EndpointSpec("GET", "/api/profiles", "列出所有配置"),
    EndpointSpec("POST", "/api/batches", "提交批次（支持幂等键）"),
    EndpointSpec("GET", "/api/batches", "列出批次（含通过率/成本汇总）"),
    EndpointSpec("GET", "/api/batches/{id}", "批次详情"),
    EndpointSpec("POST", "/api/batches/{id}/cancel", "取消批次"),
    EndpointSpec("GET", "/api/runs", "查运行列表（过滤、分页）"),
    EndpointSpec("GET", "/api/runs/{id}", "单次运行详情"),
    EndpointSpec("GET", "/api/runs/{id}/trace", "轨迹分页读取"),
    EndpointSpec("GET", "/api/runs/{id}/stream", "SSE 实时事件流"),
    EndpointSpec("GET", "/api/reports/compare", "双批次对比报告"),
    EndpointSpec("GET", "/api/stats/dashboard", "仪表盘摘要统计"),
    EndpointSpec("GET", "/api/stats/trend", "通过率趋势"),
)

INTERNAL_ENDPOINTS: tuple[EndpointSpec, ...] = (
    EndpointSpec("POST", "/internal/lease", "Runner 领取任务"),
    EndpointSpec("POST", "/internal/runs/{id}/heartbeat", "心跳续租"),
    EndpointSpec("POST", "/internal/runs/{id}/events", "批量上报轨迹事件（JSONL）"),
    EndpointSpec("POST", "/internal/runs/{id}/complete", "标记运行完成"),
)

SYSTEM_ENDPOINTS: tuple[EndpointSpec, ...] = (
    EndpointSpec("GET", "/healthz/live", "存活探针"),
    EndpointSpec("GET", "/healthz/ready", "就绪探针（检查数据库与轨迹存储）"),
    EndpointSpec("GET", "/metrics", "Prometheus 文本格式指标"),
)


def all_endpoints() -> tuple[EndpointSpec, ...]:
    """三组端点的拼接视图。"""
    return PUBLIC_ENDPOINTS + INTERNAL_ENDPOINTS + SYSTEM_ENDPOINTS


# ─── 鉴权与应用状态 ──────────────────────────────────────────────────────


def get_state(request: Request) -> AppState:
    """从应用实例取共享状态（由 create_app 注入 app.state.state）。"""
    return request.app.state.state


StateDep = Annotated[AppState, Depends(get_state)]


def runner_auth(request: Request, state: StateDep) -> None:
    """内部接口鉴权：RUNNER_SECRET 未设置时放行（开发模式，启动时已告警）。"""
    if state.runner_secret is None:
        return
    provided = request.headers.get("authorization", "")
    expected = f"Bearer {state.runner_secret}"
    if not secrets.compare_digest(provided, expected):
        raise UnauthorizedError()


def _uuid_param(value: str, label: str = "id") -> str:
    """校验 UUID 形参；非法输入立即 400，而非让数据库驱动抛 500。"""
    try:
        uuid_module.UUID(value)
    except (ValueError, TypeError, AttributeError) as exc:
        raise BadRequestError(f"invalid {label}: {value!r} is not a UUID") from exc
    return value


def _is_unique_violation(exc: asyncpg.UniqueViolationError, name_part: str) -> bool:
    """判断唯一约束冲突是否来自指定约束（按约束名片段匹配）。"""
    constraint = getattr(exc, "constraint_name", None) or ""
    return name_part in constraint


# ─── 请求模型 ────────────────────────────────────────────────────────────


class CreateProfileReq(BaseModel):
    name: str
    scaffold: str
    model: str
    harness_version: str | None = None
    sandbox_image: str | None = None
    config: dict[str, Any] | None = None


class CreateBatchReq(BaseModel):
    name: str
    profile: str
    cases: list[dict[str, Any]]
    parallelism: int = 4
    priority: int = 0
    idempotency_key: str | None = None


class LeaseReq(BaseModel):
    runner_id: str


class HeartbeatReq(BaseModel):
    runner_id: str


class CompleteReq(BaseModel):
    runner_id: str
    status: str
    score: float | None = None
    cost_usd: float | None = None
    turns: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None
    duration_s: float | None = None


# ─── 公开路由 ────────────────────────────────────────────────────────────


def build_public_router() -> APIRouter:
    router = APIRouter()

    @router.get("/profiles")
    async def list_profiles(state: StateDep) -> dict[str, Any]:
        rows = await state.db.fetch(
            "SELECT id, name, scaffold, model, harness_version, sandbox_image"
            " FROM agent_profiles ORDER BY name"
        )
        items = [
            {
                "id": str(r["id"]),
                "name": r["name"],
                "scaffold": r["scaffold"],
                "model": r["model"],
                "harness_version": r["harness_version"],
                "sandbox_image": r["sandbox_image"],
            }
            for r in rows
        ]
        return {"items": items}

    @router.post("/profiles")
    async def create_profile(req: CreateProfileReq, state: StateDep) -> dict[str, Any]:
        name = req.name.strip()
        scaffold = req.scaffold.strip()
        model = req.model.strip()
        if not name:
            raise BadRequestError("name must not be empty")
        if not model:
            raise BadRequestError("model must not be empty")
        if scaffold not in VALID_SCAFFOLDS:
            raise BadRequestError(
                f"scaffold must be one of: {', '.join(VALID_SCAFFOLDS)}"
            )
        existing = await state.db.fetchrow(
            "SELECT id FROM agent_profiles WHERE name = $1", name
        )
        if existing is not None:
            raise ConflictError(f"profile {name!r} already exists")
        try:
            row = await state.db.fetchrow(
                "INSERT INTO agent_profiles"
                " (name, scaffold, model, harness_version, sandbox_image, config)"
                " VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
                name,
                scaffold,
                model,
                req.harness_version or "dev",
                req.sandbox_image or "local",
                json.dumps(req.config or {}),
            )
        except asyncpg.UniqueViolationError as exc:
            # 并发创建同名 profile 的竞态兜底（先查后插的窗口）
            raise ConflictError(f"profile {name!r} already exists") from exc
        return {"id": str(row["id"])}

    @router.post("/batches")
    async def create_batch(req: CreateBatchReq, state: StateDep) -> dict[str, Any]:
        name = req.name.strip()
        profile = req.profile.strip()
        if not name:
            raise BadRequestError("name must not be empty")
        if not profile:
            raise BadRequestError("profile must not be empty")
        if not req.cases:
            raise BadRequestError("cases must not be empty")
        if len(req.cases) > MAX_CASES_PER_BATCH:
            raise BadRequestError(f"cases must not exceed {MAX_CASES_PER_BATCH}")
        if not 1 <= req.parallelism <= MAX_PARALLELISM:
            raise BadRequestError(
                f"parallelism must be between 1 and {MAX_PARALLELISM}"
            )
        case_ids: set[str] = set()
        for case in req.cases:
            case_id = str(case.get("case_id") or "").strip()
            task = str(case.get("task") or "").strip()
            if not case_id or not task:
                raise BadRequestError("each case needs case_id and task")
            if case_id in case_ids:
                raise BadRequestError(f"duplicate case_id in batch: {case_id}")
            case_ids.add(case_id)

        if req.idempotency_key:
            existing = await state.db.fetchrow(
                "SELECT id FROM batches WHERE idempotency_key = $1",
                req.idempotency_key,
            )
            if existing is not None:
                return {"id": str(existing["id"]), "deduplicated": True}

        profile_row = await state.db.fetchrow(
            "SELECT id FROM agent_profiles WHERE name = $1", profile
        )
        if profile_row is None:
            raise NotFoundError("profile")

        try:
            async with state.db.acquire() as conn, conn.transaction():
                batch_row = await conn.fetchrow(
                    "INSERT INTO batches"
                    " (name, profile_id, parallelism, priority, cases, idempotency_key)"
                    " VALUES ($1, $2, $3, $4, $5, $6) RETURNING id",
                    name,
                    profile_row["id"],
                    req.parallelism,
                    req.priority,
                    json.dumps(req.cases),
                    req.idempotency_key,
                )
                batch_id = batch_row["id"]
                await conn.executemany(
                    "INSERT INTO runs (batch_id, case_id) VALUES ($1, $2)",
                    [(batch_id, case["case_id"]) for case in req.cases],
                )
        except asyncpg.UniqueViolationError as exc:
            # 并发提交同 idempotency_key 的竞态：重查返回已有批次（幂等语义）
            if req.idempotency_key and _is_unique_violation(exc, "idempotency_key"):
                existing = await state.db.fetchrow(
                    "SELECT id FROM batches WHERE idempotency_key = $1",
                    req.idempotency_key,
                )
                if existing is not None:
                    return {"id": str(existing["id"]), "deduplicated": True}
            raise ConflictError("batch conflict on unique constraint") from exc
        return {"id": str(batch_id)}

    @router.get("/batches")
    async def list_batches(state: StateDep) -> dict[str, Any]:
        rows = await state.db.fetch(
            "SELECT b.id, b.name, b.status,"
            " count(r.id) FILTER (WHERE r.status = 'passed') AS passed,"
            " count(r.id) AS total,"
            " COALESCE(sum(r.cost_usd), 0) AS cost, b.created_at"
            " FROM batches b LEFT JOIN runs r ON r.batch_id = b.id"
            " GROUP BY b.id ORDER BY b.created_at DESC LIMIT 100"
        )
        items = []
        for r in rows:
            total = int(r["total"])
            passed = int(r["passed"])
            items.append(
                {
                    "id": str(r["id"]),
                    "name": r["name"],
                    "status": r["status"],
                    "passed": passed,
                    "total": total,
                    "cost_usd": float(r["cost"] or 0),
                    "pass_rate": passed / total if total else 0.0,
                    "created_at": r["created_at"],
                }
            )
        return {"items": items}

    @router.get("/batches/{batch_id}")
    async def get_batch(batch_id: str, state: StateDep) -> dict[str, Any]:
        batch_id = _uuid_param(batch_id, "batch id")
        row = await state.db.fetchrow(
            "SELECT name, status, parallelism, created_at FROM batches WHERE id = $1",
            batch_id,
        )
        if row is None:
            raise NotFoundError("batch")
        stats = await state.db.fetch(
            "SELECT status, count(*) AS n FROM runs WHERE batch_id = $1 GROUP BY status",
            batch_id,
        )
        return {
            "id": batch_id,
            "name": row["name"],
            "status": row["status"],
            "parallelism": row["parallelism"],
            "created_at": row["created_at"],
            "run_stats": {r["status"]: int(r["n"]) for r in stats},
        }

    @router.post("/batches/{batch_id}/cancel")
    async def cancel_batch(batch_id: str, state: StateDep) -> dict[str, Any]:
        batch_id = _uuid_param(batch_id, "batch id")
        async with state.db.acquire() as conn, conn.transaction():
            await conn.execute(
                "UPDATE batches SET status = 'cancelled'"
                " WHERE id = $1 AND status IN ('pending','running')",
                batch_id,
            )
            await conn.execute(
                "UPDATE runs SET status = 'error', error = 'batch cancelled',"
                " finished_at = now() WHERE batch_id = $1 AND status = 'queued'",
                batch_id,
            )
        return {"ok": True}

    @router.get("/runs")
    async def list_runs(
        state: StateDep,
        batch: str | None = None,
        status: str | None = None,
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE_LIMIT)] = 100,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> dict[str, Any]:
        if status is not None and status not in scheduler.RUN_STATUSES:
            raise BadRequestError(f"invalid status: {status}")
        if batch is not None:
            batch = _uuid_param(batch, "batch filter")
        rows = await state.db.fetch(
            "SELECT id, batch_id, case_id, status, score, cost_usd, turns"
            " FROM runs"
            " WHERE ($1::text IS NULL OR batch_id = $1::uuid)"
            "   AND ($2::text IS NULL OR status = $2)"
            " ORDER BY created_at DESC LIMIT $3 OFFSET $4",
            batch,
            status,
            limit,
            offset,
        )
        items = [
            {
                "id": str(r["id"]),
                "batch_id": str(r["batch_id"]),
                "case_id": r["case_id"],
                "status": r["status"],
                "score": r["score"],
                "cost_usd": r["cost_usd"],
                "turns": r["turns"],
            }
            for r in rows
        ]
        return {"items": items}

    @router.get("/runs/{run_id}")
    async def get_run(run_id: str, state: StateDep) -> dict[str, Any]:
        run_id = _uuid_param(run_id, "run id")
        row = await state.db.fetchrow(
            "SELECT batch_id, case_id, status, score, cost_usd, turns, error,"
            " started_at, finished_at FROM runs WHERE id = $1",
            run_id,
        )
        if row is None:
            raise NotFoundError("run")
        return {
            "id": run_id,
            "batch_id": str(row["batch_id"]),
            "case_id": row["case_id"],
            "status": row["status"],
            "score": row["score"],
            "cost_usd": row["cost_usd"],
            "turns": row["turns"],
            "error": row["error"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
        }

    @router.get("/runs/{run_id}/trace")
    async def get_trace(
        run_id: str,
        state: StateDep,
        offset: int = 0,
        limit: int = 500,
    ) -> dict[str, Any]:
        run_id = _uuid_param(run_id, "run id")
        if not 1 <= limit <= MAX_PAGE_LIMIT:
            raise BadRequestError(f"limit must be between 1 and {MAX_PAGE_LIMIT}")
        page = await state.traces.read_page(run_id, offset, limit)
        return {
            "events": page.events,
            "total": page.total,
            "next_offset": page.next_offset,
        }

    @router.get("/runs/{run_id}/stream")
    async def stream_run(run_id: str, state: StateDep) -> StreamingResponse:
        _uuid_param(run_id, "run id")
        return StreamingResponse(
            _sse_events(state, run_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.get("/reports/compare")
    async def compare_batches(state: StateDep, a: str, b: str) -> dict[str, Any]:
        a = _uuid_param(a, "batch a")
        b = _uuid_param(b, "batch b")
        fingerprints = await state.db.fetch(
            "SELECT b.id, p.model, p.harness_version, p.sandbox_image"
            " FROM batches b JOIN agent_profiles p ON p.id = b.profile_id"
            " WHERE b.id IN ($1, $2)",
            a,
            b,
        )
        if len(fingerprints) != 2:
            raise NotFoundError("batch")
        fa, fb = fingerprints[0], fingerprints[1]
        warnings = []
        if fa["sandbox_image"] != fb["sandbox_image"]:
            warnings.append(
                f"sandbox_image 不同（{fa['sandbox_image']} vs {fb['sandbox_image']}），"
                "对比可能无意义"
            )
        if fa["harness_version"] != fb["harness_version"]:
            warnings.append(
                f"harness_version 不同（{fa['harness_version']} vs {fb['harness_version']}）"
            )
        if fa["model"] != fb["model"]:
            warnings.append(f"model 不同（{fa['model']} vs {fb['model']}）")

        rows = await state.db.fetch(
            "SELECT a.case_id, a.status AS status_a, b.status AS status_b,"
            " a.score AS score_a, b.score AS score_b,"
            " a.id AS run_a, b.id AS run_b"
            " FROM runs a JOIN runs b USING (case_id)"
            " WHERE a.batch_id = $1 AND b.batch_id = $2"
            " ORDER BY CASE WHEN a.status = 'passed' AND b.status != 'passed'"
            " THEN 0 ELSE 1 END, a.case_id",
            a,
            b,
        )
        regressions = improvements = 0
        cases = []
        for r in rows:
            verdict = "same"
            if r["status_a"] == "passed" and r["status_b"] != "passed":
                verdict = "regression"
                regressions += 1
            elif r["status_a"] != "passed" and r["status_b"] == "passed":
                verdict = "improvement"
                improvements += 1
            cases.append(
                {
                    "case_id": r["case_id"],
                    "status_a": r["status_a"],
                    "status_b": r["status_b"],
                    "score_a": r["score_a"],
                    "score_b": r["score_b"],
                    "run_a": str(r["run_a"]),
                    "run_b": str(r["run_b"]),
                    "verdict": verdict,
                }
            )
        total = len(cases)
        passed_a = sum(1 for c in cases if c["status_a"] == "passed")
        passed_b = sum(1 for c in cases if c["status_b"] == "passed")
        return {
            "comparability_warnings": warnings,
            "summary": {
                "total": total,
                "passed_a": passed_a,
                "passed_b": passed_b,
                "pass_rate_a": passed_a / total if total else 0.0,
                "pass_rate_b": passed_b / total if total else 0.0,
                "regressions": regressions,
                "improvements": improvements,
            },
            "cases": cases,
        }

    @router.get("/stats/dashboard")
    async def dashboard_stats(state: StateDep) -> dict[str, Any]:
        db: DbLike = state.db
        total_runs = int(await db.fetchval("SELECT count(*) FROM runs") or 0)
        total_batches = int(await db.fetchval("SELECT count(*) FROM batches") or 0)
        total_cost = float(
            await db.fetchval("SELECT COALESCE(sum(cost_usd), 0) FROM runs") or 0
        )
        status_counts = await db.fetch(
            "SELECT status, count(*) AS n FROM runs GROUP BY status"
        )
        breakdown = {r["status"]: int(r["n"]) for r in status_counts}
        passed = breakdown.get("passed", 0)
        total_finished = sum(
            n for s, n in breakdown.items() if s in scheduler.TERMINAL_STATUSES
        )
        runs_24h = int(
            await db.fetchval(
                "SELECT count(*) FROM runs WHERE created_at > now() - interval '24 hours'"
            )
            or 0
        )
        queued = int(
            await db.fetchval("SELECT count(*) FROM runs WHERE status = 'queued'") or 0
        )
        active = int(
            await db.fetchval(
                "SELECT count(*) FROM runs WHERE status IN ('leased','running')"
            )
            or 0
        )
        state.metrics.update_queue_gauges(queued, active)
        return {
            "total_runs": total_runs,
            "total_batches": total_batches,
            "total_cost_usd": total_cost,
            "pass_rate": passed / total_finished if total_finished else 0.0,
            "runs_last_24h": runs_24h,
            "queue_depth": queued,
            "active_runs": active,
            "status_breakdown": breakdown,
        }

    @router.get("/stats/trend")
    async def pass_rate_trend(
        state: StateDep,
        days: Annotated[int, Query(ge=1, le=365)] = 30,
        group_by: str = "day",
    ) -> dict[str, Any]:
        # 白名单硬限制，杜绝把用户输入拼进 SQL 模板
        trunc = {"hour": "hour", "week": "week"}.get(group_by, "day")
        rows = await state.db.fetch(
            f"SELECT date_trunc('{trunc}', finished_at) AS bucket,"
            " count(*) FILTER (WHERE status = 'passed') AS passed,"
            " count(*) AS total,"
            " COALESCE(sum(cost_usd), 0) AS cost"
            " FROM runs"
            " WHERE finished_at > now() - ($1::text || ' days')::interval"
            "   AND status IN ('passed', 'failed', 'error', 'timeout')"
            " GROUP BY bucket ORDER BY bucket",
            str(days),
        )
        points = []
        for r in rows:
            total = int(r["total"])
            passed = int(r["passed"])
            points.append(
                {
                    "time": r["bucket"],
                    "passed": passed,
                    "total": total,
                    "cost_usd": float(r["cost"] or 0),
                    "pass_rate": passed / total if total else 0.0,
                }
            )
        return {"points": points}

    return router


async def _sse_events(state: AppState, run_id: str) -> AsyncIterator[str]:
    """单 run 的 SSE 事件流；慢消费者掉队时发 lagged 信号由前端 REST 补齐。"""
    sub = state.hub.subscribe(Topic.run(run_id))
    try:
        while True:
            try:
                kind, payload = await asyncio.wait_for(
                    sub.next(), timeout=SSE_KEEPALIVE_S
                )
            except TimeoutError:
                yield ": keepalive\n\n"
                continue
            if kind == "lagged":
                yield f"event: lagged\ndata: {payload}\n\n"
            else:
                event: TraceEvent = payload  # type: ignore[assignment]
                yield f"id: {event.seq}\ndata: {event.model_dump_json()}\n\n"
    finally:
        sub.close()


# ─── 内部路由（Runner 专用）──────────────────────────────────────────────


def build_internal_router() -> APIRouter:
    router = APIRouter(dependencies=[Depends(runner_auth)])

    @router.post("/lease")
    async def lease(req: LeaseReq, state: StateDep) -> dict[str, Any]:
        leased = await scheduler.lease_next(state.db, req.runner_id, state.lease_ttl_s)
        if leased is None:
            return {"run": None}
        return {"run": leased.model_dump()}

    @router.post("/runs/{run_id}/heartbeat")
    async def heartbeat(run_id: str, req: HeartbeatReq, state: StateDep) -> dict[str, Any]:
        run_id = _uuid_param(run_id, "run id")
        alive = await scheduler.heartbeat(
            state.db, run_id, req.runner_id, state.lease_ttl_s
        )
        return {"alive": alive}

    @router.post("/runs/{run_id}/events")
    async def ingest_events(
        run_id: str, request: Request, state: StateDep
    ) -> dict[str, Any]:
        run_id = _uuid_param(run_id, "run id")
        body = await request.body()
        lines = [ln for ln in body.decode("utf-8").splitlines() if ln.strip()]
        if not lines:
            return {"accepted": 0, "rejected": 0}

        runner_id = request.headers.get("x-runner-id", "").strip()
        if not runner_id:
            raise BadRequestError("X-Runner-Id header is required")

        run_row = await state.db.fetchrow(
            "SELECT batch_id, runner_id, status FROM runs WHERE id = $1", run_id
        )
        if run_row is None:
            raise NotFoundError("run")
        if run_row["runner_id"] != runner_id:
            raise ConflictError("run not owned by this runner")
        if run_row["status"] not in scheduler.ACTIVE_STATUSES:
            raise ConflictError(
                f"run is not accepting events in status {run_row['status']}"
            )

        events: list[TraceEvent] = []
        for line in lines:
            try:
                ev = trace_event_adapter.validate_python(json.loads(line))
            except Exception as exc:
                raise BadRequestError(f"malformed trace event: {exc}") from exc
            try:
                validate_for_run(ev, run_id)
            except ValueError as exc:
                raise BadRequestError(str(exc)) from exc
            events.append(ev)

        traces: TraceStoreProtocol = state.traces
        await traces.append(run_id, lines)
        metrics: Metrics = state.metrics
        metrics.trace_append(len(lines))
        metrics.events_ingested_inc(len(events), 0)

        batch_id = str(run_row["batch_id"])
        for ev in events:
            state.hub.publish(Topic.run(run_id), ev)
            if is_milestone(ev):
                state.hub.publish(Topic.batch(batch_id), ev)

        return {"accepted": len(events), "rejected": 0}

    @router.post("/runs/{run_id}/complete")
    async def complete_run(
        run_id: str, req: CompleteReq, state: StateDep
    ) -> dict[str, Any]:
        run_id = _uuid_param(run_id, "run id")
        if req.status not in VALID_COMPLETE_STATUSES:
            raise BadRequestError(f"invalid status: {req.status}")
        trace_path = state.traces.finalize(run_id)
        status_tag = await state.db.execute(
            "UPDATE runs SET status = $2, score = $3, cost_usd = $4, turns = $5,"
            " input_tokens = $6, output_tokens = $7, error = $8,"
            " trace_path = $9, finished_at = now()"
            " WHERE id = $1 AND runner_id = $10 AND status IN ('leased','running')",
            run_id,
            req.status,
            req.score,
            req.cost_usd,
            req.turns,
            req.input_tokens,
            req.output_tokens,
            req.error,
            trace_path,
            req.runner_id,
        )
        if _rows_affected(status_tag) == 0:
            raise ConflictError("run not owned by this runner or already completed")

        metrics: Metrics = state.metrics
        metrics.run_completed(req.status, req.duration_s or 0.0, req.cost_usd)
        state.hub.retire(Topic.run(run_id))
        return {"ok": True}

    return router


def _rows_affected(status: str) -> int:
    parts = status.split()
    if len(parts) >= 2 and parts[-1].isdigit():
        return int(parts[-1])
    return 0


# ─── 系统端点 ────────────────────────────────────────────────────────────


def build_system_router() -> APIRouter:
    router = APIRouter()

    @router.get("/healthz/live")
    async def live() -> PlainTextResponse:
        return PlainTextResponse("ok")

    @router.get("/healthz/ready")
    async def ready(state: StateDep) -> PlainTextResponse:
        await state.db.fetchval("SELECT 1")
        await state.traces.health_check()
        return PlainTextResponse("ready")

    @router.get("/metrics")
    async def metrics_endpoint(state: StateDep) -> PlainTextResponse:
        return PlainTextResponse(state.metrics.render(), media_type="text/plain")

    return router
