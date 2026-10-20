"""Runner 主循环测试：用 MockTransport 模拟 eval-server 的内部接口。"""

import asyncio
import importlib.util
import json

import httpx
import pytest
from runner.adapters.base import RunRequest
from runner.adapters.mock import MockAdapter
from runner.events import RunFinished, RunStarted, Seq

from runner import main


class FakeServer:
    """记录请求并按内部接口契约返回固定响应。"""

    def __init__(self, heartbeat_alive: bool = True) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self.heartbeat_alive = heartbeat_alive

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = request.content.decode("utf-8")
        self.calls.append((request.method, request.url.path, body))
        if request.url.path.endswith("/events"):
            return httpx.Response(200, json={"accepted": 1, "rejected": 0})
        if request.url.path.endswith("/heartbeat"):
            return httpx.Response(200, json={"alive": self.heartbeat_alive})
        if request.url.path.endswith("/complete"):
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(200, json={})

    def posts_to(self, suffix: str) -> list[dict]:
        return [
            json.loads(body)
            for method, path, body in self.calls
            if method == "POST" and path.endswith(suffix)
        ]

    def event_batches(self, run_id: str) -> list[list[dict]]:
        batches = []
        for method, path, body in self.calls:
            if method == "POST" and path == f"/internal/runs/{run_id}/events":
                batches.append([json.loads(ln) for ln in body.splitlines() if ln.strip()])
        return batches


def _lease(**overrides) -> dict:
    base = {
        "run_id": "run-1",
        "batch_id": "batch-1",
        "case_id": "case-1",
        "task": "fix the divide function",
        "model": "mock-model-v1",
        "scaffold": "mock",
        "expectations": [],
    }
    base.update(overrides)
    return base


@pytest.fixture
def fast_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    class FastMockAdapter(MockAdapter):
        def __init__(self) -> None:
            super().__init__(pace_s=0.0)

    monkeypatch.setattr(main, "MockAdapter", FastMockAdapter)


async def test_execute_flushes_events_and_completes(
    fast_mock: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "FLUSH_INTERVAL_S", 999.0)  # 不触发中途 flush
    server = FakeServer()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(server.handler), base_url="http://test"
    ) as http:
        await main.execute(http, _lease())

    batches = server.event_batches("run-1")
    assert len(batches) == 1  # 结束时一次性 flush
    events = batches[0]
    assert len(events) == 3 * 5 + 2  # 三轮（request + 2 chunk + call + result）+ 首尾
    assert all(ev["schema_version"] == 1 for ev in events)
    assert events[0]["type"] == "run_started"
    assert events[-1]["type"] == "run_finished"
    seqs = [ev["seq"] for ev in events]
    assert seqs == sorted(seqs)

    completes = server.posts_to("/complete")
    assert len(completes) == 1
    assert completes[0]["runner_id"] == main.RUNNER_ID
    assert completes[0]["status"] == "passed"
    assert completes[0]["score"] == 1.0
    assert completes[0]["turns"] == 3
    assert completes[0]["cost_usd"] is not None
    assert completes[0]["duration_s"] >= 0


async def test_execute_expectations_override_status(fast_mock: None) -> None:
    server = FakeServer()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(server.handler), base_url="http://test"
    ) as http:
        await main.execute(
            http,
            _lease(
                task="make the tests fail",
                expectations=[{"type": "tool_called", "tool": "bash", "weight": 1.0}],
            ),
        )

    completes = server.posts_to("/complete")
    assert len(completes) == 1
    # mock 执行状态为 failed，但 expectations 判定通过 → 覆盖为 passed
    assert completes[0]["status"] == "passed"
    assert completes[0]["score"] == 1.0


async def test_execute_error_path_reports_error(fast_mock: None) -> None:
    server = FakeServer()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(server.handler), base_url="http://test"
    ) as http:
        await main.execute(http, _lease(scaffold="does-not-exist"))

    completes = server.posts_to("/complete")
    assert len(completes) == 1
    assert completes[0]["status"] == "error"
    assert "unknown scaffold" in completes[0]["error"]


async def test_execute_midway_flush_respects_interval(
    fast_mock: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "FLUSH_INTERVAL_S", 0.0)  # 每个事件都 flush
    server = FakeServer()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(server.handler), base_url="http://test"
    ) as http:
        await main.execute(http, _lease())

    batches = server.event_batches("run-1")
    assert len(batches) >= 2  # 中途至少 flush 一次
    total = sum(len(b) for b in batches)
    assert total == 3 * 5 + 2


class SlowAdapter:
    """耗时 adapter：制造心跳触发窗口。"""

    async def run(self, req: RunRequest):
        seq = Seq()
        yield RunStarted(
            seq=seq.next(),
            run_id=req.run_id,
            case_id=req.case_id,
            task=req.task,
            model=req.model,
        )
        await asyncio.sleep(0.05)
        yield RunFinished(seq=seq.next(), status="passed", score=1.0, turns=1)


async def test_execute_sends_heartbeat_while_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "HEARTBEAT_INTERVAL_S", 0.01)
    monkeypatch.setattr(main, "make_adapter", lambda scaffold: SlowAdapter())
    server = FakeServer(heartbeat_alive=True)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(server.handler), base_url="http://test"
    ) as http:
        await main.execute(http, _lease(scaffold="slow"))

    heartbeats = server.posts_to("/heartbeat")
    assert len(heartbeats) >= 1
    assert all(hb["runner_id"] == main.RUNNER_ID for hb in heartbeats)
    assert server.posts_to("/complete")[0]["status"] == "passed"


async def test_execute_aborts_when_lease_lost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "HEARTBEAT_INTERVAL_S", 0.01)
    monkeypatch.setattr(main, "make_adapter", lambda scaffold: SlowAdapter())
    server = FakeServer(heartbeat_alive=False)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(server.handler), base_url="http://test"
    ) as http:
        await main.execute(http, _lease(scaffold="slow"))

    completes = server.posts_to("/complete")
    assert len(completes) == 1
    assert completes[0]["status"] == "error"
    assert "lease lost" in completes[0]["error"]


def test_auth_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "RUNNER_SECRET", None)
    assert main._auth_headers() == {"X-Runner-Id": main.RUNNER_ID}

    monkeypatch.setattr(main, "RUNNER_SECRET", "s3cret")
    headers = main._auth_headers()
    assert headers["Authorization"] == "Bearer s3cret"
    assert headers["X-Runner-Id"] == main.RUNNER_ID


def test_make_adapter_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(main.make_adapter("mock"), MockAdapter)

    with pytest.raises(ValueError, match="unknown scaffold"):
        main.make_adapter("nope")


@pytest.mark.skipif(
    importlib.util.find_spec("anthropic") is not None, reason="anthropic extra installed"
)
def test_make_adapter_anthropic_requires_extra() -> None:
    with pytest.raises(ImportError):
        main.make_adapter("anthropic")


@pytest.mark.skipif(
    importlib.util.find_spec("langgraph") is not None, reason="langgraph extra installed"
)
def test_make_adapter_langgraph_requires_extra() -> None:
    with pytest.raises(ImportError):
        main.make_adapter("langgraph")


async def test_heartbeat_loop_uses_env_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "HEARTBEAT_INTERVAL_S", 0.01)
    server = FakeServer(heartbeat_alive=True)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(server.handler), base_url="http://test"
    ) as http:
        task = asyncio.create_task(main.heartbeat_loop(http, "run-x"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert len(server.posts_to("/heartbeat")) >= 2


class _ErrorWithSignalAdapter:
    """模拟 mcc 适配器的 error 路径：出错前已有工具调用与模型输出。"""

    async def run(self, req):
        from runner.events import LlmChunk, Seq, ToolCall, ToolResult

        seq = Seq()
        yield RunStarted(
            seq=seq.next(), run_id=req.run_id, case_id=req.case_id,
            task=req.task, model=req.model,
        )
        yield ToolCall(seq=seq.next(), turn=0, call_id="c1", tool_name="run_bash",
                       args={"command": "git clone https://example/repo.git"})
        yield ToolResult(seq=seq.next(), call_id="c1", output="cloned")
        yield LlmChunk(seq=seq.next(), turn=1, delta="cloned, starting fix")
        yield RunFinished(seq=seq.next(), status="error", score=None, turns=1)


class _ErrorNoSignalAdapter:
    """模拟零产出 error：LLM 首调即挂，无任何工具调用或模型输出。"""

    async def run(self, req):
        seq = Seq()
        yield RunStarted(
            seq=seq.next(), run_id=req.run_id, case_id=req.case_id,
            task=req.task, model=req.model,
        )
        yield RunFinished(seq=seq.next(), status="error", score=None, turns=0)


@pytest.mark.parametrize(
    ("adapter", "expected_status", "expected_score"),
    [
        pytest.param(
            _ErrorWithSignalAdapter, "passed", 1.0, id="error_with_signal_scored"
        ),
        pytest.param(_ErrorNoSignalAdapter, "error", None, id="error_no_signal_kept"),
    ],
)
async def test_execute_scores_error_runs_with_trace_signal(
    monkeypatch, adapter, expected_status, expected_score
) -> None:
    """执行中出错的 run：轨迹有实质内容时仍按 expectations 评分。

    error 只代表执行中有未恢复异常；git clone 超时等场景的轨迹
    依旧可评（no_tool_errors 断言自会扣分）。零产出 error 保留
    error 状态不评分，避免把基础设施故障误判成 Agent 低分。
    """
    monkeypatch.setattr(main, "make_adapter", lambda scaffold: adapter())
    server = FakeServer()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(server.handler), base_url="http://test"
    ) as http:
        await main.execute(
            http,
            _lease(
                expectations=[
                    {"type": "tool_called", "tool": "run_bash",
                     "args_regex": "git clone", "weight": 1.0},
                ]
            ),
        )

    completes = server.posts_to("/complete")
    assert len(completes) == 1
    assert completes[0]["status"] == expected_status
    assert completes[0]["score"] == expected_score

class FlakyHeartbeatServer(FakeServer):
    """前 N 次心跳返回 503（网络抖动替身），之后恢复 alive。"""

    def __init__(self, fail_times: int) -> None:
        super().__init__(heartbeat_alive=True)
        self._remaining = fail_times

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/heartbeat") and self._remaining > 0:
            self._remaining -= 1
            return httpx.Response(503, json={"detail": "transient"})
        return super().handler(request)


class FlakyEventsServer(FakeServer):
    """events 端点前 N 次返回 503，模拟上报期瞬时连接故障。"""

    def __init__(self, fail_times: int = 1) -> None:
        super().__init__()
        self._remaining = fail_times

    def handler(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/events") and self._remaining > 0:
            self._remaining -= 1
            return httpx.Response(503, json={"detail": "gateway reset"})
        return super().handler(request)


def _patch_fast_adapter(monkeypatch) -> None:
    monkeypatch.setattr(main, "HEARTBEAT_INTERVAL_S", 0.01)
    monkeypatch.setattr(main, "HEARTBEAT_RETRY_S", 0.01)
    monkeypatch.setattr(main, "make_adapter", lambda scaffold: MockAdapter(pace_s=0.0))


async def test_heartbeat_single_transient_failure_recovers(monkeypatch) -> None:
    """单次心跳 503 不判死：短间隔重试恢复，run 正常完成。"""
    server = FlakyHeartbeatServer(fail_times=1)
    _patch_fast_adapter(monkeypatch)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(server.handler), base_url="http://test"
    ) as http:
        await main.execute(http, _lease())

    completes = server.posts_to("/complete")
    assert len(completes) == 1
    assert completes[0]["status"] == "passed"


async def test_heartbeat_consecutive_failures_abort_unfinished_run(monkeypatch) -> None:
    """连续 3 次心跳失败判死：未完成的 run 上报 error。

    用慢 adapter 制造心跳触发窗口且 run_finished 尚未产出——
    已产出 run_finished 的 run 走评分保全路径，不在此测试范围。
    """
    server = FlakyHeartbeatServer(fail_times=99)
    monkeypatch.setattr(main, "HEARTBEAT_INTERVAL_S", 0.01)
    monkeypatch.setattr(main, "HEARTBEAT_RETRY_S", 0.01)
    monkeypatch.setattr(main, "make_adapter", lambda scaffold: SlowAdapter())
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(server.handler), base_url="http://test"
    ) as http:
        await main.execute(http, _lease(scaffold="slow"))

    completes = server.posts_to("/complete")
    assert len(completes) == 1
    assert completes[0]["status"] == "error"
    assert "heartbeat failed" in completes[0]["error"]


async def test_finish_stage_crash_keeps_scored_result(monkeypatch) -> None:
    """收尾上报瞬时 503 后重试成功：评分结果保留，error 字段附摘要。

    对应真实事故：轨迹里 run_finished=passed/score=1，run 状态却是
    error——收尾期上报故障把已完成的评分结果覆盖成了 error。
    """
    server = FlakyEventsServer(fail_times=1)
    _patch_fast_adapter(monkeypatch)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(server.handler), base_url="http://test"
    ) as http:
        await main.execute(
            http,
            _lease(
                expectations=[
                    {"type": "tool_called", "tool": "bash", "weight": 1.0},
                ]
            ),
        )

    completes = server.posts_to("/complete")
    assert len(completes) == 1
    # 评分保全：mock 轨迹含 bash 调用，判 passed score=1，
    # 而不是收尾异常覆盖成 error
    assert completes[0]["status"] == "passed"
    assert completes[0]["score"] == 1.0
    assert "finish error" in completes[0]["error"]
