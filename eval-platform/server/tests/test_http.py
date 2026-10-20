"""HTTP 接口全链路测试：FakeDb + LocalTraceStore 组装的真实路由栈。

覆盖公开端点、Runner 内部端点、错误分支与鉴权；
SSE 流因长连接特性以生成器单元测试的方式单独验证。
"""

import asyncio
import json
import uuid

import pytest
from conftest import FakeConn, FakeDb
from eval_server import api as api_module
from eval_server.api import _sse_events
from eval_server.app import create_app
from eval_server.domain import LlmChunk, RunFinished, RunStarted
from eval_server.state import AppState
from eval_server.store import LocalTraceStore
from eval_server.stream import SUBSCRIBER_BUFFER, Topic
from fastapi.testclient import TestClient


@pytest.fixture
def state(fake_db: FakeDb, trace_store: LocalTraceStore) -> AppState:
    return AppState(db=fake_db, traces=trace_store)


@pytest.fixture
def client(state: AppState) -> TestClient:
    app = create_app(state, start_reaper=False)
    with TestClient(app) as c:
        yield c


def _create_profile(client: TestClient, name: str = "p1", model: str = "m1") -> None:
    r = client.post(
        "/api/profiles", json={"name": name, "scaffold": "mock", "model": model}
    )
    assert r.status_code == 200
    assert r.json()["id"]


def _create_batch(
    client: TestClient,
    cases: list[dict],
    profile: str = "p1",
    **fields,
) -> str:
    payload = {"name": "batch", "profile": profile, "cases": cases}
    payload.update(fields)
    r = client.post("/api/batches", json=payload)
    assert r.status_code == 200
    return r.json()["id"]


def _case(case_id: str = "case-1", task: str = "do work") -> dict:
    return {"case_id": case_id, "task": task}


def _lease(client: TestClient, runner: str = "runner-a") -> dict | None:
    r = client.post("/internal/lease", json={"runner_id": runner})
    assert r.status_code == 200
    return r.json()["run"]


def _event_lines(run_id: str, case_id: str = "case-1") -> str:
    events = [
        RunStarted(seq=0, run_id=run_id, case_id=case_id, task="t", model="m"),
        LlmChunk(seq=1, turn=0, delta="working"),
        RunFinished(seq=2, status="passed", score=1.0, turns=1, cost_usd=0.01),
    ]
    return "\n".join(e.model_dump_json() for e in events)


# ─── profiles ────────────────────────────────────────────────────────────


def test_create_and_list_profiles(client: TestClient) -> None:
    _create_profile(client, "alpha", model="model-a")
    _create_profile(client, "beta", model="model-b")

    items = client.get("/api/profiles").json()["items"]
    names = [p["name"] for p in items]
    assert names == ["alpha", "beta"]  # 按名称排序
    alpha = items[0]
    assert alpha["scaffold"] == "mock"
    assert alpha["model"] == "model-a"
    assert alpha["harness_version"] == "dev"
    assert alpha["sandbox_image"] == "local"


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "", "scaffold": "mock", "model": "m"},
        {"name": "x", "scaffold": "not-a-scaffold", "model": "m"},
        {"name": "x", "scaffold": "mock", "model": " "},
        {"scaffold": "mock", "model": "m"},
    ],
)
def test_create_profile_validation(client: TestClient, payload: dict) -> None:
    r = client.post("/api/profiles", json=payload)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "bad_request"


def test_create_profile_duplicate_name_conflicts(client: TestClient) -> None:
    _create_profile(client, "dup", model="m1")
    r = client.post("/api/profiles", json={"name": "dup", "scaffold": "mock", "model": "m2"})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "conflict"


def test_create_profile_unique_violation_race_conflicts(
    client: TestClient, fake_db: FakeDb, monkeypatch: pytest.MonkeyPatch
) -> None:
    """先查后插窗口内的并发创建：INSERT 撞唯一约束也应 409 而非 500。"""
    _create_profile(client, "race", model="m1")
    original = fake_db.fetchrow

    async def racing_fetchrow(query: str, *args: object) -> dict | None:
        if "SELECT id FROM agent_profiles WHERE name" in query:
            return None  # 竞态窗口：检查时尚不可见
        return await original(query, *args)

    monkeypatch.setattr(fake_db, "fetchrow", racing_fetchrow)
    r = client.post("/api/profiles", json={"name": "race", "scaffold": "mock", "model": "m"})
    assert r.status_code == 409
    assert "already exists" in r.json()["error"]["message"]


# ─── batches ────────────────────────────────────────────────────────────


def test_create_batch_creates_queued_runs(client: TestClient) -> None:
    _create_profile(client)
    batch_id = _create_batch(client, [_case("c1"), _case("c2")])

    runs = client.get("/api/runs").json()["items"]
    assert len(runs) == 2
    assert all(r["status"] == "queued" for r in runs)
    assert {r["case_id"] for r in runs} == {"c1", "c2"}

    detail = client.get(f"/api/batches/{batch_id}").json()
    assert detail["name"] == "batch"
    assert detail["status"] == "running"
    assert detail["run_stats"] == {"queued": 2}


def test_create_batch_idempotency_key_deduplicates(client: TestClient) -> None:
    _create_profile(client)
    first = client.post(
        "/api/batches",
        json={"name": "b", "profile": "p1", "cases": [_case()], "idempotency_key": "k1"},
    ).json()
    second = client.post(
        "/api/batches",
        json={"name": "b", "profile": "p1", "cases": [_case()], "idempotency_key": "k1"},
    ).json()
    assert first["id"] == second["id"]
    assert second["deduplicated"] is True
    assert len(client.get("/api/runs").json()["items"]) == 1


def test_create_batch_inserts_runs_via_executemany(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """批次内 runs 批量插入：走 executemany 而非逐条往返。"""
    calls: list[tuple[str, list[tuple[object, ...]]]] = []
    original = FakeConn.executemany

    async def spy(self: FakeConn, query: str, args_seq: object) -> None:
        materialized = [tuple(a) for a in args_seq]  # type: ignore[arg-type]
        calls.append((query, materialized))
        await original(self, query, materialized)

    monkeypatch.setattr(FakeConn, "executemany", spy)

    _create_profile(client)
    batch_id = _create_batch(client, [_case("c1"), _case("c2")])

    assert len(calls) == 1
    query, arg_sets = calls[0]
    assert "INSERT INTO runs" in query
    assert {case_id for _, case_id in arg_sets} == {"c1", "c2"}
    assert all(bid == batch_id for bid, _ in arg_sets)


def test_create_batch_unknown_profile_404(client: TestClient) -> None:
    r = client.post("/api/batches", json={"name": "b", "profile": "ghost", "cases": [_case()]})
    assert r.status_code == 404


@pytest.mark.parametrize(
    "cases",
    [
        [],
        [{"case_id": "c1", "task": ""}],
        [{"case_id": " ", "task": "t"}],
        [{"case_id": "c1", "task": "t"}, {"case_id": "c1", "task": "t2"}],
    ],
)
def test_create_batch_validates_cases(client: TestClient, cases: list[dict]) -> None:
    _create_profile(client)
    r = client.post("/api/batches", json={"name": "b", "profile": "p1", "cases": cases})
    assert r.status_code == 400


@pytest.mark.parametrize("parallelism", [0, -1, 257])
def test_create_batch_validates_parallelism(client: TestClient, parallelism: int) -> None:
    _create_profile(client)
    r = client.post(
        "/api/batches",
        json={"name": "b", "profile": "p1", "cases": [_case()], "parallelism": parallelism},
    )
    assert r.status_code == 400


def test_batch_not_found(client: TestClient) -> None:
    assert client.get("/api/batches/00000000-0000-0000-0000-000000000000").status_code == 404


@pytest.mark.parametrize(
    ("method", "path", "params", "json_body"),
    [
        pytest.param("GET", "/api/batches/zzz", None, None, id="get_batch"),
        pytest.param("POST", "/api/batches/zzz/cancel", None, None, id="cancel_batch"),
        pytest.param("GET", "/api/runs/zzz", None, None, id="get_run"),
        pytest.param("GET", "/api/runs/zzz/trace", None, None, id="get_trace"),
        pytest.param("GET", "/api/runs/zzz/stream", None, None, id="stream_run"),
        pytest.param("GET", "/api/runs", {"batch": "zzz"}, None, id="list_runs_batch_filter"),
        pytest.param("GET", "/api/reports/compare", {"a": "zzz", "b": "zzz"}, None, id="compare"),
        pytest.param(
            "POST", "/internal/runs/zzz/heartbeat", None, {"runner_id": "r1"}, id="heartbeat"
        ),
        pytest.param("POST", "/internal/runs/zzz/events", None, None, id="ingest_events"),
        pytest.param(
            "POST", "/internal/runs/zzz/complete", None,
            {"runner_id": "r1", "status": "passed"}, id="complete",
        ),
    ],
)
def test_invalid_uuid_params_return_400(
    client: TestClient, method: str, path: str, params: dict | None, json_body: dict | None
) -> None:
    """非 UUID 形参应在入口 400，而非透传到数据库驱动抛 500。"""
    r = client.request(
        method, path, params=params, json=json_body, headers={"X-Runner-Id": "runner-a"}
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "bad_request"
    assert "not a UUID" in r.json()["error"]["message"]


def test_cancel_batch_marks_runs_error(client: TestClient) -> None:
    _create_profile(client)
    batch_id = _create_batch(client, [_case("c1"), _case("c2")])

    r = client.post(f"/api/batches/{batch_id}/cancel")
    assert r.json() == {"ok": True}
    assert client.get(f"/api/batches/{batch_id}").json()["status"] == "cancelled"

    runs = client.get("/api/runs", params={"batch": batch_id}).json()["items"]
    assert all(r["status"] == "error" for r in runs)
    assert _lease(client) is None  # 已取消批次不可再领取


def test_list_batches_aggregates(client: TestClient, fake_db: FakeDb) -> None:
    _create_profile(client)
    batch_id = _create_batch(client, [_case("c1"), _case("c2")])
    for run in fake_db.runs:
        run["status"] = "passed" if run["case_id"] == "c1" else "failed"
        run["cost_usd"] = 0.5

    item = next(
        b for b in client.get("/api/batches").json()["items"] if b["id"] == batch_id
    )
    assert item["passed"] == 1
    assert item["total"] == 2
    assert item["cost_usd"] == pytest.approx(1.0)
    assert item["pass_rate"] == pytest.approx(0.5)


# ─── runner 全链路 ───────────────────────────────────────────────────────


def test_full_runner_chain(client: TestClient, fake_db: FakeDb) -> None:
    _create_profile(client)
    _create_batch(
        client,
        [{"case_id": "case-1", "task": "do work",
          "expectations": [{"type": "no_tool_errors", "weight": 1.0}]}],
    )

    # 1. 领取
    leased = _lease(client, "runner-a")
    assert leased is not None
    run_id = leased["run_id"]
    assert leased["case_id"] == "case-1"
    assert leased["task"] == "do work"
    assert leased["scaffold"] == "mock"
    assert leased["expectations"] == [{"type": "no_tool_errors", "weight": 1.0}]

    # 2. 心跳
    hb = client.post(f"/internal/runs/{run_id}/heartbeat", json={"runner_id": "runner-a"})
    assert hb.status_code == 200
    assert hb.json() == {"alive": True}

    # 3. 事件上报
    r = client.post(
        f"/internal/runs/{run_id}/events",
        content=_event_lines(run_id),
        headers={"X-Runner-Id": "runner-a"},
    )
    assert r.status_code == 200
    assert r.json() == {"accepted": 3, "rejected": 0}

    # 4. 完成
    complete = client.post(
        f"/internal/runs/{run_id}/complete",
        json={
            "runner_id": "runner-a",
            "status": "passed",
            "score": 0.9,
            "cost_usd": 0.012,
            "turns": 3,
            "duration_s": 2.5,
        },
    )
    assert complete.json() == {"ok": True}

    # 5. 查询侧
    detail = client.get(f"/api/runs/{run_id}").json()
    assert detail["status"] == "passed"
    assert detail["score"] == 0.9
    assert detail["cost_usd"] == 0.012
    assert detail["turns"] == 3
    assert detail["finished_at"] is not None

    trace = client.get(f"/api/runs/{run_id}/trace").json()
    assert trace["total"] == 3
    assert [e["seq"] for e in trace["events"]] == [0, 1, 2]
    assert trace["next_offset"] is None

    batch_id = detail["batch_id"]
    assert client.get(f"/api/batches/{batch_id}").json()["run_stats"] == {"passed": 1}


def test_lease_returns_null_when_queue_empty(client: TestClient) -> None:
    _create_profile(client)
    assert _lease(client) is None


def test_lease_assigns_each_run_once(client: TestClient) -> None:
    _create_profile(client)
    _create_batch(client, [_case("c1"), _case("c2")])

    first = _lease(client, "runner-a")
    second = _lease(client, "runner-b")
    assert first is not None and second is not None
    assert first["run_id"] != second["run_id"]
    assert _lease(client, "runner-c") is None


def test_events_reject_missing_runner_header(client: TestClient) -> None:
    _create_profile(client)
    _create_batch(client, [_case()])
    run_id = _lease(client)["run_id"]

    r = client.post(f"/internal/runs/{run_id}/events", content=_event_lines(run_id))
    assert r.status_code == 400


def test_events_reject_wrong_runner(client: TestClient) -> None:
    _create_profile(client)
    _create_batch(client, [_case()])
    run_id = _lease(client, "runner-a")["run_id"]

    r = client.post(
        f"/internal/runs/{run_id}/events",
        content=_event_lines(run_id),
        headers={"X-Runner-Id": "runner-b"},
    )
    assert r.status_code == 409


def test_events_reject_malformed_json(client: TestClient) -> None:
    _create_profile(client)
    _create_batch(client, [_case()])
    run_id = _lease(client)["run_id"]

    r = client.post(
        f"/internal/runs/{run_id}/events",
        content="{not json",
        headers={"X-Runner-Id": "runner-a"},
    )
    assert r.status_code == 400
    assert "malformed" in r.json()["error"]["message"]


def test_events_reject_run_id_mismatch(client: TestClient) -> None:
    _create_profile(client)
    _create_batch(client, [_case()])
    run_id = _lease(client)["run_id"]

    r = client.post(
        f"/internal/runs/{run_id}/events",
        content=_event_lines("other-run"),
        headers={"X-Runner-Id": "runner-a"},
    )
    assert r.status_code == 400
    assert "does not match" in r.json()["error"]["message"]


def test_events_reject_after_complete(client: TestClient) -> None:
    _create_profile(client)
    _create_batch(client, [_case()])
    run_id = _lease(client)["run_id"]
    ok = client.post(
        f"/internal/runs/{run_id}/complete",
        json={"runner_id": "runner-a", "status": "passed"},
    )
    assert ok.status_code == 200

    r = client.post(
        f"/internal/runs/{run_id}/events",
        content=_event_lines(run_id),
        headers={"X-Runner-Id": "runner-a"},
    )
    assert r.status_code == 409


def test_events_empty_body_returns_zero(client: TestClient) -> None:
    _create_profile(client)
    _create_batch(client, [_case()])
    run_id = _lease(client)["run_id"]

    r = client.post(
        f"/internal/runs/{run_id}/events",
        content="",
        headers={"X-Runner-Id": "runner-a"},
    )
    assert r.json() == {"accepted": 0, "rejected": 0}


def test_events_unknown_run_404(client: TestClient) -> None:
    r = client.post(
        "/internal/runs/00000000-0000-0000-0000-000000000000/events",
        content="{}",
        headers={"X-Runner-Id": "runner-a"},
    )
    assert r.status_code == 404


def test_complete_conflict_for_wrong_runner(client: TestClient) -> None:
    _create_profile(client)
    _create_batch(client, [_case()])
    run_id = _lease(client, "runner-a")["run_id"]

    r = client.post(
        f"/internal/runs/{run_id}/complete",
        json={"runner_id": "runner-b", "status": "passed"},
    )
    assert r.status_code == 409


def test_complete_second_call_conflicts(client: TestClient) -> None:
    _create_profile(client)
    _create_batch(client, [_case()])
    run_id = _lease(client)["run_id"]

    first = client.post(
        f"/internal/runs/{run_id}/complete",
        json={"runner_id": "runner-a", "status": "passed"},
    )
    second = client.post(
        f"/internal/runs/{run_id}/complete",
        json={"runner_id": "runner-a", "status": "failed"},
    )
    assert first.status_code == 200
    assert second.status_code == 409


def test_complete_rejects_invalid_status(client: TestClient) -> None:
    _create_profile(client)
    _create_batch(client, [_case()])
    run_id = _lease(client)["run_id"]

    r = client.post(
        f"/internal/runs/{run_id}/complete",
        json={"runner_id": "runner-a", "status": "exploded"},
    )
    assert r.status_code == 400


# ─── runs 查询 ───────────────────────────────────────────────────────────


def test_runs_list_filters_and_pagination(client: TestClient, fake_db: FakeDb) -> None:
    _create_profile(client)
    _create_batch(client, [_case("c1"), _case("c2")])
    for run in fake_db.runs:
        run["status"] = "passed"

    passed = client.get("/api/runs", params={"status": "passed"}).json()["items"]
    assert len(passed) == 2

    limited = client.get("/api/runs", params={"limit": 1}).json()["items"]
    assert len(limited) == 1

    bad = client.get("/api/runs", params={"status": "bogus"})
    assert bad.status_code == 400


def test_run_not_found(client: TestClient) -> None:
    missing = str(uuid.uuid4())
    assert client.get(f"/api/runs/{missing}").status_code == 404


async def test_trace_pagination(state: AppState, client: TestClient) -> None:
    lines = [
        json.dumps({"schema_version": 1, "seq": i, "type": "llm_chunk", "turn": 0, "delta": "x"})
        for i in range(5)
    ]
    run_id = str(uuid.uuid4())
    await state.traces.append(run_id, lines)

    page1 = client.get(f"/api/runs/{run_id}/trace", params={"offset": 0, "limit": 2}).json()
    assert [e["seq"] for e in page1["events"]] == [0, 1]
    assert page1["total"] == 5
    assert page1["next_offset"] == 2

    page3 = client.get(f"/api/runs/{run_id}/trace", params={"offset": 4, "limit": 2}).json()
    assert page3["next_offset"] is None


@pytest.mark.parametrize("limit", [0, 1001])
def test_trace_validates_limit(client: TestClient, limit: int) -> None:
    r = client.get(f"/api/runs/{uuid.uuid4()}/trace", params={"limit": limit})
    assert r.status_code == 400


# ─── compare / stats ─────────────────────────────────────────────────────


def _seed_compare_batches(client: TestClient, fake_db: FakeDb, same_profile: bool) -> tuple:
    _create_profile(client, "p1", model="model-a")
    if not same_profile:
        _create_profile(client, "p2", model="model-b")
    batch_a = _create_batch(client, [_case("c1"), _case("c2")], profile="p1")
    batch_b = _create_batch(
        client, [_case("c1"), _case("c2")], profile="p1" if same_profile else "p2"
    )
    outcomes = {
        batch_a: {"c1": "passed", "c2": "failed"},
        batch_b: {"c1": "failed", "c2": "passed"},
    }
    for run in fake_db.runs:
        run["status"] = outcomes[run["batch_id"]][run["case_id"]]
    return batch_a, batch_b


def test_compare_reports_regressions_and_improvements(
    client: TestClient, fake_db: FakeDb
) -> None:
    batch_a, batch_b = _seed_compare_batches(client, fake_db, same_profile=True)
    report = client.get(
        "/api/reports/compare", params={"a": batch_a, "b": batch_b}
    ).json()

    assert report["comparability_warnings"] == []
    assert report["summary"] == {
        "total": 2,
        "passed_a": 1,
        "passed_b": 1,
        "pass_rate_a": 0.5,
        "pass_rate_b": 0.5,
        "regressions": 1,
        "improvements": 1,
    }
    by_case = {c["case_id"]: c for c in report["cases"]}
    assert by_case["c1"]["verdict"] == "regression"
    assert by_case["c2"]["verdict"] == "improvement"
    assert by_case["c1"]["status_a"] == "passed"
    assert by_case["c1"]["status_b"] == "failed"


def test_compare_warns_on_model_mismatch(client: TestClient, fake_db: FakeDb) -> None:
    batch_a, batch_b = _seed_compare_batches(client, fake_db, same_profile=False)
    report = client.get(
        "/api/reports/compare", params={"a": batch_a, "b": batch_b}
    ).json()
    assert len(report["comparability_warnings"]) == 1
    assert "model" in report["comparability_warnings"][0]


def test_compare_missing_batch_404(client: TestClient) -> None:
    _create_profile(client)
    batch_id = _create_batch(client, [_case()])
    r = client.get(
        "/api/reports/compare",
        params={"a": batch_id, "b": "00000000-0000-0000-0000-000000000000"},
    )
    assert r.status_code == 404


def test_dashboard_stats(client: TestClient, fake_db: FakeDb) -> None:
    _create_profile(client)
    _create_batch(client, [_case("c1"), _case("c2"), _case("c3")])
    statuses = {"c1": "passed", "c2": "failed", "c3": "queued"}
    for run in fake_db.runs:
        run["status"] = statuses[run["case_id"]]
        if run["status"] == "passed":
            run["cost_usd"] = 0.25

    stats = client.get("/api/stats/dashboard").json()
    assert stats["total_runs"] == 3
    assert stats["total_batches"] == 1
    assert stats["total_cost_usd"] == pytest.approx(0.25)
    assert stats["pass_rate"] == pytest.approx(0.5)  # 1 passed / 2 finished
    assert stats["queue_depth"] == 1
    assert stats["active_runs"] == 0
    assert stats["runs_last_24h"] == 3
    assert stats["status_breakdown"] == {"passed": 1, "failed": 1, "queued": 1}


def test_trend_returns_points(client: TestClient) -> None:
    r = client.get("/api/stats/trend", params={"days": 7, "group_by": "day"})
    assert r.status_code == 200
    assert r.json() == {"points": []}


def test_trend_validates_days(client: TestClient) -> None:
    assert client.get("/api/stats/trend", params={"days": 0}).status_code == 400
    assert client.get("/api/stats/trend", params={"days": 400}).status_code == 400


# ─── 系统端点与鉴权 ──────────────────────────────────────────────────────


def test_health_endpoints(client: TestClient) -> None:
    assert client.get("/healthz/live").text == "ok"
    assert client.get("/healthz/ready").text == "ready"


def test_metrics_endpoint_exposes_http_counter(client: TestClient) -> None:
    client.get("/api/profiles")
    body = client.get("/metrics").text
    assert "http_requests_total" in body
    assert 'path="/api/profiles"' in body


def test_internal_endpoints_require_bearer(
    fake_db: FakeDb, trace_store: LocalTraceStore
) -> None:
    secured = AppState(db=fake_db, traces=trace_store, runner_secret="s3cret")
    app = create_app(secured, start_reaper=False)
    with TestClient(app) as c:
        assert c.post("/internal/lease", json={"runner_id": "r"}).status_code == 401
        assert (
            c.post(
                "/internal/lease",
                json={"runner_id": "r"},
                headers={"Authorization": "Bearer wrong"},
            ).status_code
            == 401
        )
        ok = c.post(
            "/internal/lease",
            json={"runner_id": "r"},
            headers={"Authorization": "Bearer s3cret"},
        )
        assert ok.status_code == 200
        assert c.get("/api/profiles").status_code == 200  # 公开端点不受影响


# ─── SSE 流（生成器单元测试）────────────────────────────────────────────


async def test_sse_events_emits_event_frames(state: AppState) -> None:
    gen = _sse_events(state, "run-1")
    task = asyncio.create_task(gen.__anext__())
    await asyncio.sleep(0.02)  # 等生成器完成订阅

    state.hub.publish(Topic.run("run-1"), RunStarted(seq=7, run_id="run-1", case_id="c", task="t"))
    frame = await asyncio.wait_for(task, timeout=1.0)

    assert frame.startswith("id: 7\ndata: ")
    payload = json.loads(frame.split("data: ", 1)[1].strip())
    assert payload["type"] == "run_started"
    await gen.aclose()


async def test_sse_events_emits_lagged_signal(state: AppState) -> None:
    gen = _sse_events(state, "run-1")
    first_read = asyncio.create_task(gen.__anext__())
    await asyncio.sleep(0.02)  # 等生成器完成订阅

    for i in range(SUBSCRIBER_BUFFER + 3):
        ev = RunStarted(seq=i, run_id="run-1", case_id="c", task="t")
        state.hub.publish(Topic.run("run-1"), ev)

    # 先读到缓冲区内的事件帧
    first_frame = await asyncio.wait_for(first_read, timeout=1.0)
    assert first_frame.startswith("id: 0\n")

    # 随后读到掉队信号：丢弃 3 条
    lagged_frame = await asyncio.wait_for(gen.__anext__(), timeout=1.0)
    assert lagged_frame.startswith("event: lagged\ndata: 3\n")
    await gen.aclose()


async def test_sse_events_keepalive(
    state: AppState, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_module, "SSE_KEEPALIVE_S", 0.05)
    gen = _sse_events(state, "run-1")
    frame = await asyncio.wait_for(gen.__anext__(), timeout=1.0)
    assert frame == ": keepalive\n\n"
    await gen.aclose()
