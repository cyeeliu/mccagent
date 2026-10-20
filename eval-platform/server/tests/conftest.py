"""server 测试共享夹具：内存版数据库替身与本地轨迹存储。

FakeDb 按 SQL 语句特征分发到内存状态，语义对齐 PostgreSQL 实现
（调度用 ``FOR UPDATE SKIP LOCKED`` 的行为在此以顺序扫描模拟），
使 HTTP 层与调度器无需真实数据库即可全链路验证。
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import asyncpg
import pytest
from eval_server import scheduler
from eval_server.store import LocalTraceStore


def _now() -> datetime:
    return datetime.now(UTC)


class _NoopTx:
    async def __aenter__(self) -> _NoopTx:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


class FakeConn:
    """借出的连接：事务为空操作，查询委托给 FakeDb。"""

    def __init__(self, db: FakeDb) -> None:
        self._db = db

    def transaction(self) -> _NoopTx:
        return _NoopTx()

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        return await self._db.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        return await self._db.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        return await self._db.fetchval(query, *args)

    async def execute(self, query: str, *args: Any) -> str:
        return await self._db.execute(query, *args)

    async def executemany(
        self, query: str, args_seq: Iterable[Sequence[Any]]
    ) -> None:
        for args in args_seq:
            await self._db.execute(query, *args)


def _unique_violation(constraint: str) -> asyncpg.UniqueViolationError:
    """构造带约束名的唯一冲突（对齐真实 asyncpg 驱动的异常形态）。"""
    exc = asyncpg.UniqueViolationError("duplicate key value violates unique constraint")
    exc.constraint_name = constraint
    return exc


class FakeDb:
    """内存数据库替身：识别调度器与应用层使用的查询并操作内存状态。"""

    def __init__(self) -> None:
        self.profiles: list[dict[str, Any]] = []
        self.batches: list[dict[str, Any]] = []
        self.runs: list[dict[str, Any]] = []
        self.migrations: list[str] = []
        self.executed: list[tuple[str, str, tuple[Any, ...]]] = []

    # ── 池接口 ─────────────────────────────────────────────────────────

    @asynccontextmanager
    async def acquire(self):
        yield FakeConn(self)

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.executed.append(("fetch", query, args))
        return self._dispatch_fetch(query, args)

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        self.executed.append(("fetchrow", query, args))
        return self._dispatch_fetchrow(query, args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.executed.append(("fetchval", query, args))
        return self._dispatch_fetchval(query, args)

    async def execute(self, query: str, *args: Any) -> str:
        self.executed.append(("execute", query, args))
        return self._dispatch_execute(query, args)

    # ── fetch 分发 ─────────────────────────────────────────────────────

    def _dispatch_fetch(self, query: str, args: tuple[Any, ...]) -> list[dict[str, Any]]:
        if query == scheduler.LEASE_SELECT_SQL:
            row = self._lease_select()
            return [row] if row else []
        if "SELECT name FROM schema_migrations" in query:
            return [{"name": n} for n in self.migrations]
        if query.startswith("SELECT id, name, scaffold, model"):
            return [
                {
                    "id": p["id"],
                    "name": p["name"],
                    "scaffold": p["scaffold"],
                    "model": p["model"],
                    "harness_version": p["harness_version"],
                    "sandbox_image": p["sandbox_image"],
                }
                for p in sorted(self.profiles, key=lambda p: p["name"])
            ]
        if query.startswith("SELECT b.id, b.name, b.status,"):
            items = []
            for b in sorted(self.batches, key=lambda b: b["created_at"], reverse=True)[:100]:
                runs = [r for r in self.runs if r["batch_id"] == b["id"]]
                passed = sum(1 for r in runs if r["status"] == "passed")
                items.append(
                    {
                        "id": b["id"],
                        "name": b["name"],
                        "status": b["status"],
                        "passed": passed,
                        "total": len(runs),
                        "cost": sum(r["cost_usd"] or 0.0 for r in runs),
                        "created_at": b["created_at"],
                    }
                )
            return items
        if "FROM runs WHERE batch_id = $1 GROUP BY status" in query:
            batch_id = args[0]
            counts: dict[str, int] = {}
            for r in self.runs:
                if r["batch_id"] == batch_id:
                    counts[r["status"]] = counts.get(r["status"], 0) + 1
            return [{"status": s, "n": n} for s, n in counts.items()]
        if "FROM runs GROUP BY status" in query:
            counts: dict[str, int] = {}
            for r in self.runs:
                counts[r["status"]] = counts.get(r["status"], 0) + 1
            return [{"status": s, "n": n} for s, n in counts.items()]
        if query.startswith(
            "SELECT id, batch_id, case_id, status, score, cost_usd, turns, retries"
        ):
            batch, status, limit, offset = args
            rows = [
                {
                    "id": r["id"],
                    "batch_id": r["batch_id"],
                    "case_id": r["case_id"],
                    "status": r["status"],
                    "score": r["score"],
                    "cost_usd": r["cost_usd"],
                    "turns": r["turns"],
                    "retries": r["retries"],
                }
                for r in sorted(self.runs, key=lambda r: r["created_at"], reverse=True)
                if (batch is None or r["batch_id"] == batch)
                and (status is None or r["status"] == status)
            ]
            return rows[offset : offset + limit]
        if query.startswith("SELECT b.id, p.model"):
            rows = []
            for batch_id in args[:2]:
                batch = self._batch(batch_id)
                if batch is None:
                    continue
                profile = self._profile_by_id(batch["profile_id"])
                rows.append(
                    {
                        "id": batch["id"],
                        "model": profile["model"] if profile else "",
                        "harness_version": profile["harness_version"] if profile else "",
                        "sandbox_image": profile["sandbox_image"] if profile else "",
                    }
                )
            return rows
        if query.startswith("SELECT a.case_id"):
            batch_a, batch_b = args
            runs_a = {r["case_id"]: r for r in self.runs if r["batch_id"] == batch_a}
            runs_b = {r["case_id"]: r for r in self.runs if r["batch_id"] == batch_b}
            joined = []
            for case_id, ra in runs_a.items():
                rb = runs_b.get(case_id)
                if rb is not None:
                    joined.append(
                        {
                            "case_id": case_id,
                            "status_a": ra["status"],
                            "status_b": rb["status"],
                            "score_a": ra["score"],
                            "score_b": rb["score"],
                            "run_a": ra["id"],
                            "run_b": rb["id"],
                        }
                    )
            joined.sort(
                key=lambda x: (
                    0 if x["status_a"] == "passed" and x["status_b"] != "passed" else 1,
                    x["case_id"],
                )
            )
            return joined
        if query.startswith("SELECT date_trunc"):
            return []
        raise AssertionError(f"FakeDb.fetch: unhandled query: {query[:80]!r}")

    # ── fetchrow 分发 ──────────────────────────────────────────────────

    def _dispatch_fetchrow(self, query: str, args: tuple[Any, ...]) -> dict[str, Any] | None:
        if query == scheduler.LEASE_SELECT_SQL:
            return self._lease_select()
        if query == scheduler.LEASE_TASK_SQL:
            batch_id, case_id = args
            batch = self._batch(batch_id)
            if batch is None:
                return None
            case = next(
                (c for c in batch["cases"] if c.get("case_id") == case_id), None
            )
            if case is None:
                return None
            profile = self._profile_by_id(batch["profile_id"])
            return {
                "task": case.get("task", ""),
                "expectations": case.get("expectations", []),
                "scaffold": profile["scaffold"] if profile else "mock",
                "model": profile["model"] if profile else "",
            }
        if query == scheduler.QUEUE_DEPTH_SQL:
            return {
                "queued": sum(1 for r in self.runs if r["status"] == "queued"),
                "active": sum(
                    1 for r in self.runs if r["status"] in ("leased", "running")
                ),
            }
        if "INSERT INTO agent_profiles" in query:
            name, scaffold, model, harness, sandbox, config = args
            if any(p["name"] == name for p in self.profiles):
                raise _unique_violation("agent_profiles_name_key")
            profile = {
                "id": str(uuid.uuid4()),
                "name": name,
                "scaffold": scaffold,
                "model": model,
                "harness_version": harness,
                "sandbox_image": sandbox,
                "config": config,
                "created_at": _now(),
            }
            self.profiles.append(profile)
            return {"id": profile["id"]}
        if "SELECT id FROM agent_profiles WHERE name" in query:
            return next(
                ({"id": p["id"]} for p in self.profiles if p["name"] == args[0]), None
            )
        if "SELECT id FROM batches WHERE idempotency_key" in query:
            return next(
                ({"id": b["id"]} for b in self.batches if b["idempotency_key"] == args[0]),
                None,
            )
        if "INSERT INTO batches" in query:
            name, profile_id, parallelism, priority, cases, idem = args
            if idem is not None and any(
                b["idempotency_key"] == idem for b in self.batches
            ):
                raise _unique_violation("batches_idempotency_key")
            batch = {
                "id": str(uuid.uuid4()),
                "name": name,
                "profile_id": profile_id,
                "status": "running",
                "parallelism": parallelism,
                "priority": priority,
                "cases": json.loads(cases) if isinstance(cases, str) else cases,
                "idempotency_key": idem,
                "created_at": _now(),
            }
            self.batches.append(batch)
            return {"id": batch["id"]}
        if "SELECT name, status, parallelism, created_at FROM batches WHERE id" in query:
            batch = self._batch(args[0])
            if batch is None:
                return None
            return {
                "name": batch["name"],
                "status": batch["status"],
                "parallelism": batch["parallelism"],
                "created_at": batch["created_at"],
            }
        if query.startswith(
            "SELECT batch_id, case_id, status, score, cost_usd, turns, retries"
        ):
            run = self._run(args[0])
            if run is None:
                return None
            return {
                "batch_id": run["batch_id"],
                "case_id": run["case_id"],
                "status": run["status"],
                "score": run["score"],
                "cost_usd": run["cost_usd"],
                "turns": run["turns"],
                "retries": run["retries"],
                "lease_expires_at": run["lease_expires_at"],
                "error": run["error"],
                "started_at": run["started_at"],
                "finished_at": run["finished_at"],
            }
        if "SELECT batch_id, runner_id, status FROM runs WHERE id" in query:
            run = self._run(args[0])
            if run is None:
                return None
            return {
                "batch_id": run["batch_id"],
                "runner_id": run["runner_id"],
                "status": run["status"],
            }
        raise AssertionError(f"FakeDb.fetchrow: unhandled query: {query[:80]!r}")

    # ── fetchval 分发 ──────────────────────────────────────────────────

    def _dispatch_fetchval(self, query: str, args: tuple[Any, ...]) -> Any:
        if query.strip() == "SELECT 1":
            return 1
        if "WHERE created_at > now()" in query:
            cutoff = _now() - timedelta(hours=24)
            return sum(1 for r in self.runs if r["created_at"] > cutoff)
        if "WHERE status = 'queued'" in query:
            return sum(1 for r in self.runs if r["status"] == "queued")
        if "WHERE status IN ('leased','running')" in query:
            return sum(1 for r in self.runs if r["status"] in ("leased", "running"))
        if "count(*) FROM runs" in query:
            return len(self.runs)
        if "count(*) FROM batches" in query:
            return len(self.batches)
        if "COALESCE(sum(cost_usd)" in query:
            return sum(r["cost_usd"] or 0.0 for r in self.runs)
        raise AssertionError(f"FakeDb.fetchval: unhandled query: {query[:80]!r}")

    # ── execute 分发 ───────────────────────────────────────────────────

    def _dispatch_execute(self, query: str, args: tuple[Any, ...]) -> str:
        if query == scheduler.HEARTBEAT_SQL:
            run_id, runner_id, ttl = args
            run = self._run(run_id)
            if run and run["runner_id"] == runner_id and run["status"] in (
                "leased",
                "running",
            ):
                run["lease_expires_at"] = _now() + timedelta(seconds=ttl)
                run["status"] = "running"
                run["started_at"] = run["started_at"] or _now()
                return "UPDATE 1"
            return "UPDATE 0"
        if query == scheduler.LEASE_UPDATE_SQL:
            run_id, runner_id, ttl = args
            run = self._run(run_id)
            if run:
                run["status"] = "leased"
                run["runner_id"] = runner_id
                run["lease_expires_at"] = _now() + timedelta(seconds=ttl)
                return "UPDATE 1"
            return "UPDATE 0"
        if query == scheduler.REAPER_REQUEUE_SQL:
            n = 0
            for run in self._expired_leases():
                if run["retries"] < args[0]:
                    run["status"] = "queued"
                    run["runner_id"] = None
                    run["lease_expires_at"] = None
                    run["retries"] += 1
                    n += 1
            return f"UPDATE {n}"
        if query == scheduler.REAPER_DEAD_SQL:
            n = 0
            for run in self._expired_leases():
                run["status"] = "error"
                run["error"] = "lease expired without heartbeat, retries exhausted"
                run["finished_at"] = _now()
                n += 1
            return f"UPDATE {n}"
        if query == scheduler.BATCH_DONE_SQL:
            n = 0
            for batch in self.batches:
                if batch["status"] != "running":
                    continue
                active = [
                    r
                    for r in self.runs
                    if r["batch_id"] == batch["id"]
                    and r["status"] in ("queued", "leased", "running")
                ]
                if not active:
                    batch["status"] = "done"
                    n += 1
            return f"UPDATE {n}"
        if "INSERT INTO runs" in query:
            batch_id, case_id = args
            self.runs.append(
                {
                    "id": str(uuid.uuid4()),
                    "batch_id": batch_id,
                    "case_id": case_id,
                    "status": "queued",
                    "score": None,
                    "cost_usd": None,
                    "turns": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "trace_path": None,
                    "error": None,
                    "runner_id": None,
                    "retries": 0,
                    "lease_expires_at": None,
                    "started_at": None,
                    "finished_at": None,
                    "created_at": _now(),
                }
            )
            return "INSERT 0 1"
        if "UPDATE batches SET status = 'cancelled'" in query:
            batch = self._batch(args[0])
            if batch and batch["status"] in ("pending", "running"):
                batch["status"] = "cancelled"
                return "UPDATE 1"
            return "UPDATE 0"
        if "batch cancelled" in query:
            n = 0
            for run in self.runs:
                if run["batch_id"] == args[0] and run["status"] == "queued":
                    run["status"] = "error"
                    run["error"] = "batch cancelled"
                    run["finished_at"] = _now()
                    n += 1
            return f"UPDATE {n}"
        if "UPDATE runs SET status = $2" in query:
            (
                run_id,
                status,
                score,
                cost_usd,
                turns,
                in_tok,
                out_tok,
                error,
                trace_path,
                runner_id,
            ) = args
            run = self._run(run_id)
            if run and run["runner_id"] == runner_id and run["status"] in (
                "leased",
                "running",
            ):
                run.update(
                    status=status,
                    score=score,
                    cost_usd=cost_usd,
                    turns=turns,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    error=error,
                    trace_path=trace_path,
                    finished_at=_now(),
                )
                return "UPDATE 1"
            return "UPDATE 0"
        if "CREATE TABLE IF NOT EXISTS schema_migrations" in query:
            return "CREATE TABLE"
        if "INSERT INTO schema_migrations" in query:
            self.migrations.append(args[0])
            return "INSERT 0 1"
        return "OK"

    # ── 内部辅助 ───────────────────────────────────────────────────────

    def _run(self, run_id: str) -> dict[str, Any] | None:
        return next((r for r in self.runs if r["id"] == run_id), None)

    def _batch(self, batch_id: str) -> dict[str, Any] | None:
        return next((b for b in self.batches if b["id"] == batch_id), None)

    def _profile_by_id(self, profile_id: str) -> dict[str, Any] | None:
        return next((p for p in self.profiles if p["id"] == profile_id), None)

    def _lease_select(self) -> dict[str, Any] | None:
        for batch in sorted(
            self.batches, key=lambda b: (-b["priority"], b["created_at"])
        ):
            if batch["status"] != "running":
                continue
            active = sum(
                1
                for r in self.runs
                if r["batch_id"] == batch["id"] and r["status"] in ("leased", "running")
            )
            if active >= batch["parallelism"]:
                continue
            queued = sorted(
                (
                    r
                    for r in self.runs
                    if r["batch_id"] == batch["id"] and r["status"] == "queued"
                ),
                key=lambda r: r["created_at"],
            )
            if queued:
                return {
                    "id": queued[0]["id"],
                    "batch_id": batch["id"],
                    "case_id": queued[0]["case_id"],
                }
        return None

    def _expired_leases(self) -> list[dict[str, Any]]:
        now = _now()
        return [
            r
            for r in self.runs
            if r["status"] in ("leased", "running")
            and r["lease_expires_at"] is not None
            and r["lease_expires_at"] < now
        ]


# ── 夹具 ─────────────────────────────────────────────────────────────────


@pytest.fixture
def fake_db() -> FakeDb:
    return FakeDb()


@pytest.fixture
def trace_store(tmp_path: Path) -> LocalTraceStore:
    return LocalTraceStore(tmp_path / "traces")
