"""调度器测试：SQL 契约、租约领取、心跳续租与 reaper 自愈。"""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from conftest import FakeDb
from eval_server import scheduler
from eval_server.scheduler import LeasedRun, _rows_affected


def _seed_run(db: FakeDb, *, status: str = "queued", runner_id: str | None = None,
              retries: int = 0, expired: bool = False) -> dict:
    db.batches.append(
        {
            "id": "batch-1",
            "name": "b",
            "profile_id": db.profiles[0]["id"] if db.profiles else "p",
            "status": "running",
            "parallelism": 4,
            "priority": 0,
            "cases": [
                {"case_id": "case-1", "task": "do it", "expectations": [{"type": "no_tool_errors"}]}
            ],
            "idempotency_key": None,
            "created_at": datetime.now(UTC),
        }
    )
    run = {
        "id": "run-1",
        "batch_id": "batch-1",
        "case_id": "case-1",
        "status": status,
        "score": None,
        "cost_usd": None,
        "turns": None,
        "input_tokens": None,
        "output_tokens": None,
        "trace_path": None,
        "error": None,
        "runner_id": runner_id,
        "retries": retries,
        "lease_expires_at": datetime.now(UTC) - timedelta(seconds=1) if expired else None,
        "started_at": None,
        "finished_at": None,
        "created_at": datetime.now(UTC),
    }
    db.runs.append(run)
    return run


@pytest.fixture
def profile_db(fake_db: FakeDb) -> FakeDb:
    fake_db.profiles.append(
        {
            "id": "profile-1",
            "name": "p1",
            "scaffold": "mock",
            "model": "m1",
            "harness_version": "dev",
            "sandbox_image": "local",
            "config": "{}",
            "created_at": datetime.now(UTC),
        }
    )
    return fake_db


# ── SQL 契约 ────────────────────────────────────────────────────────────


def test_lease_select_uses_skip_locked_and_priority() -> None:
    sql = scheduler.LEASE_SELECT_SQL
    assert "FOR UPDATE OF r SKIP LOCKED" in sql
    assert "ORDER BY b.priority DESC, r.created_at" in sql
    assert "< b.parallelism" in sql  # 批次并发上限在同一条 SQL 内检查
    assert "r.status = 'queued'" in sql
    assert "b.status = 'running'" in sql


def test_heartbeat_updates_status_and_lease() -> None:
    sql = scheduler.HEARTBEAT_SQL
    assert "lease_expires_at = now() + make_interval" in sql
    assert "status = 'running'" in sql
    assert "runner_id = $2" in sql


def test_reaper_requeue_increments_retries_with_limit() -> None:
    sql = scheduler.REAPER_REQUEUE_SQL
    assert "retries = retries + 1" in sql
    assert "retries < $1" in sql
    assert "lease_expires_at < now()" in sql


def test_reaper_dead_marks_error() -> None:
    assert "status = 'error'" in scheduler.REAPER_DEAD_SQL


def test_batch_done_requires_no_active_runs() -> None:
    sql = scheduler.BATCH_DONE_SQL
    assert "NOT EXISTS" in sql
    assert "IN ('queued','leased','running')" in sql


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("UPDATE 1", 1),
        ("UPDATE 0", 0),
        ("INSERT 0 3", 3),
        ("DELETE 5", 5),
        ("UPDATE", 0),
        ("SELECT 42", 42),
        ("", 0),
    ],
)
def test_rows_affected(status: str, expected: int) -> None:
    assert _rows_affected(status) == expected


# ── lease_next ──────────────────────────────────────────────────────────


async def test_lease_next_returns_task_details(profile_db: FakeDb) -> None:
    _seed_run(profile_db)
    leased = await scheduler.lease_next(profile_db, "runner-a", 120)
    assert isinstance(leased, LeasedRun)
    assert leased.run_id == "run-1"
    assert leased.case_id == "case-1"
    assert leased.task == "do it"
    assert leased.scaffold == "mock"
    assert leased.model == "m1"
    assert leased.expectations == [{"type": "no_tool_errors"}]

    run = profile_db.runs[0]
    assert run["status"] == "leased"
    assert run["runner_id"] == "runner-a"
    assert run["lease_expires_at"] is not None


async def test_lease_next_returns_none_when_empty(profile_db: FakeDb) -> None:
    assert await scheduler.lease_next(profile_db, "runner-a", 120) is None


async def test_lease_next_respects_parallelism(profile_db: FakeDb) -> None:
    profile_db.batches.append(
        {
            "id": "batch-p",
            "name": "p",
            "profile_id": "profile-1",
            "status": "running",
            "parallelism": 1,
            "priority": 0,
            "cases": [
                {"case_id": "c1", "task": "t1"},
                {"case_id": "c2", "task": "t2"},
            ],
            "idempotency_key": None,
            "created_at": datetime.now(UTC),
        }
    )
    for case_id in ("c1", "c2"):
        profile_db.runs.append(
            {
                "id": f"run-{case_id}",
                "batch_id": "batch-p",
                "case_id": case_id,
                "status": "queued",
                "score": None, "cost_usd": None, "turns": None,
                "input_tokens": None, "output_tokens": None, "trace_path": None,
                "error": None, "runner_id": None, "retries": 0,
                "lease_expires_at": None, "started_at": None, "finished_at": None,
                "created_at": datetime.now(UTC),
            }
        )

    first = await scheduler.lease_next(profile_db, "r1", 120)
    assert first is not None
    second = await scheduler.lease_next(profile_db, "r2", 120)  # 并发上限 1
    assert second is None


async def test_lease_next_prefers_higher_priority_batch(profile_db: FakeDb) -> None:
    for i, prio in enumerate((0, 10)):
        batch_id = f"batch-{prio}"
        profile_db.batches.append(
            {
                "id": batch_id,
                "name": f"b{prio}",
                "profile_id": "profile-1",
                "status": "running",
                "parallelism": 4,
                "priority": prio,
                "cases": [{"case_id": f"case-{i}", "task": "t"}],
                "idempotency_key": None,
                "created_at": datetime.now(UTC) + timedelta(seconds=i),
            }
        )
        profile_db.runs.append(
            {
                "id": f"run-prio-{prio}",
                "batch_id": batch_id,
                "case_id": f"case-{i}",
                "status": "queued",
                "score": None, "cost_usd": None, "turns": None,
                "input_tokens": None, "output_tokens": None, "trace_path": None,
                "error": None, "runner_id": None, "retries": 0,
                "lease_expires_at": None, "started_at": None, "finished_at": None,
                "created_at": datetime.now(UTC) + timedelta(seconds=i),
            }
        )

    leased = await scheduler.lease_next(profile_db, "r1", 120)
    assert leased is not None
    assert leased.run_id == "run-prio-10"


# ── heartbeat ──────────────────────────────────────────────────────────


async def test_heartbeat_renews_lease(profile_db: FakeDb) -> None:
    run = _seed_run(profile_db, status="leased", runner_id="runner-a")
    run["lease_expires_at"] = datetime.now(UTC) + timedelta(seconds=5)

    assert await scheduler.heartbeat(profile_db, "run-1", "runner-a", 120) is True
    assert run["status"] == "running"
    assert run["started_at"] is not None
    assert run["lease_expires_at"] > datetime.now(UTC) + timedelta(seconds=100)


async def test_heartbeat_fails_for_wrong_runner(profile_db: FakeDb) -> None:
    _seed_run(profile_db, status="leased", runner_id="runner-a")
    assert await scheduler.heartbeat(profile_db, "run-1", "runner-b", 120) is False


async def test_heartbeat_fails_for_terminal_run(profile_db: FakeDb) -> None:
    _seed_run(profile_db, status="passed", runner_id="runner-a")
    assert await scheduler.heartbeat(profile_db, "run-1", "runner-a", 120) is False


# ── reaper ──────────────────────────────────────────────────────────────


async def test_reaper_requeues_expired_lease(profile_db: FakeDb) -> None:
    run = _seed_run(profile_db, status="leased", runner_id="r1", expired=True)
    await scheduler.run_reaper_once(profile_db)
    assert run["status"] == "queued"
    assert run["retries"] == 1
    assert run["runner_id"] is None


async def test_reaper_dead_letters_exhausted_retries(profile_db: FakeDb) -> None:
    run = _seed_run(
        profile_db, status="leased", runner_id="r1",
        retries=scheduler.MAX_RETRIES, expired=True,
    )
    await scheduler.run_reaper_once(profile_db)
    assert run["status"] == "error"
    assert "retries exhausted" in run["error"]


async def test_reaper_marks_batch_done(profile_db: FakeDb) -> None:
    run = _seed_run(profile_db, status="leased", runner_id="r1", expired=True)
    await scheduler.run_reaper_once(profile_db)  # 回队
    run["status"] = "passed"  # 模拟完成
    await scheduler.run_reaper_once(profile_db)
    assert profile_db.batches[0]["status"] == "done"


async def test_run_reaper_once_executes_in_order(fake_db: FakeDb) -> None:
    await scheduler.run_reaper_once(fake_db)
    executed = [q for kind, q, _ in fake_db.executed if kind == "execute"]
    assert executed[:3] == [
        scheduler.REAPER_REQUEUE_SQL,
        scheduler.REAPER_DEAD_SQL,
        scheduler.BATCH_DONE_SQL,
    ]


async def test_queue_depth(profile_db: FakeDb) -> None:
    _seed_run(profile_db, status="queued")
    queued, active = await scheduler.queue_depth(profile_db)
    assert (queued, active) == (1, 0)


async def test_reaper_loop_calls_on_tick(fake_db: FakeDb) -> None:
    ticks: list[tuple[int, int]] = []
    task = asyncio.create_task(
        scheduler.reaper_loop(fake_db, interval_s=0.01, on_tick=lambda q, a: ticks.append((q, a)))
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(ticks) >= 2
