"""适配器测试：Mock 行为、mcc 事件转译与全链路（MockLlm，无需 API key）。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from mcc_core.event import Error as ErrorEvent
from mcc_core.event import (
    Notice,
    PermissionRequest,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
    TurnEnd,
    UserEcho,
)
from mcc_eval.adapters import MccAdapter, MockAdapter, RunRequest, _Translator
from mcc_eval.events import LlmChunk, LlmRequest, RunFinished, RunStarted, ToolCall, ToolResult
from mcc_llm import MessageStop, ToolUseInputDelta, ToolUseStart
from mcc_llm import TextDelta as LlmTextDelta
from mcc_llm.mock import MockLlm, single_text_turn


def _req(**overrides) -> RunRequest:
    payload = {
        "run_id": "run-1",
        "case_id": "case-1",
        "task": "write a file",
        "model": "test-model",
        "work_dir": "/tmp/unused",
    }
    payload.update(overrides)
    return RunRequest(**payload)


def _drain(adapter, req: RunRequest) -> list:
    async def collect() -> list:
        return [ev async for ev in adapter.run(req)]

    return asyncio.run(collect())


# ─── 转译器单元测试 ───────────────────────────────────────────────────────


def test_translator_event_mapping() -> None:
    t = _Translator(_req())
    assert isinstance(t.run_started(), RunStarted)

    # 首个产出事件前补 llm_request
    out = t.translate(TextDelta(text="hello"))
    assert [type(e) for e in out] == [LlmRequest, LlmChunk]
    assert out[1].delta == "hello"

    # 工具调用：同轮不重复 llm_request
    call = t.translate(ToolCallStart(id="t1", name="write_file", args_preview=""))
    assert [type(e) for e in call] == [ToolCall]
    assert call[0].call_id == "t1"
    assert call[0].tool_name == "write_file"


def test_translator_tool_call_args_carry_preview() -> None:
    """ToolCall.args 取 args_preview：置 None 会让所有 args_regex 断言恒假。"""
    t = _Translator(_req())
    out = t.translate(ToolCallStart(id="t1", name="bash", args_preview="pytest -q"))
    call = out[-1]
    assert isinstance(call, ToolCall)
    assert call.tool_name == "bash"
    assert call.args == "pytest -q"

    result = t.translate(ToolCallEnd(id="t1", output="ok", is_error=False))
    assert isinstance(result[0], ToolResult)
    assert result[0].output == "ok"

    # 轮结束：不产事件；下一次 LLM 调用的 turn 序号递增
    assert t.translate(TurnEnd(cost_usd=0.01)) == []
    assert t.turns == 1
    assert t.cost_usd == 0.01
    nxt = t.translate(TextDelta(text="done"))
    assert [type(e) for e in nxt] == [LlmRequest, LlmChunk]
    assert nxt[0].turn == 1
    assert nxt[1].turn == 1


@pytest.mark.parametrize(
    "ev",
    [
        UserEcho(text="hi"),
        Notice(text="info"),
        PermissionRequest(id=1, message="allow?"),
    ],
    ids=lambda ev: type(ev).__name__,
)
def test_translator_skips_non_trace_events(ev) -> None:
    assert _Translator(_req()).translate(ev) == []


def test_translator_error_marks_run_error() -> None:
    t = _Translator(_req())
    t.translate(ErrorEvent(text="boom"))
    assert t.run_finished().status == "error"


def test_translator_seq_monotonic() -> None:
    t = _Translator(_req())
    seqs = [t.run_started().seq]
    for ev in (
        TextDelta(text="a"),
        ToolCallStart(id="t1", name="bash", args_preview=""),
        ToolCallEnd(id="t1", output="ok", is_error=True),
        TurnEnd(cost_usd=0.02),
    ):
        seqs.extend(e.seq for e in t.translate(ev))
    seqs.append(t.run_finished().seq)
    assert seqs == sorted(seqs)
    assert len(set(seqs)) == len(seqs)


# ─── MccAdapter 全链路（MockLlm，无需 API key）───────────────────────────


def test_adapter_text_only_turn() -> None:
    adapter = MccAdapter(llm=MockLlm([single_text_turn("Paris is the capital.")]))
    events = _drain(adapter, _req())

    assert [type(e).__name__ for e in events] == [
        "RunStarted",
        "LlmRequest",
        "LlmChunk",
        "RunFinished",
    ]
    chunks = [e for e in events if isinstance(e, LlmChunk)]
    assert chunks[0].delta == "Paris is the capital."
    finished = events[-1]
    assert isinstance(finished, RunFinished)
    assert finished.status == "passed"
    assert finished.turns == 1


def test_adapter_uses_request_work_dir(tmp_path: Path) -> None:
    """RunRequest.work_dir 是每 case 工作目录：Agent 产物落在这里。"""
    turn1 = [
        ToolUseStart(id="t1", name="write_file"),
        ToolUseInputDelta(partial_json='{"path": "out.txt", "content": "eval-me"}'),
        MessageStop(stop_reason="tool_use"),
    ]
    turn2 = [LlmTextDelta(text="file written"), MessageStop(stop_reason="end_turn")]
    adapter = MccAdapter(llm=MockLlm([turn1, turn2]))

    events = _drain(adapter, _req(task="write out.txt", work_dir=str(tmp_path)))

    # 工具真实执行：write_file 在请求携带的工作目录落盘
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "eval-me"

    calls = [e for e in events if isinstance(e, ToolCall)]
    results = [e for e in events if isinstance(e, ToolResult)]
    assert len(calls) == 1
    assert calls[0].tool_name == "write_file"
    assert len(results) == 1
    assert not results[0].is_error

    finished = events[-1]
    assert isinstance(finished, RunFinished)
    assert finished.turns == 1  # 单任务单回合；turns 计 TurnEnd 而非 LLM 轮次
    assert finished.status == "passed"


def test_adapter_error_event_maps_to_error_status() -> None:
    class BrokenLlm:
        def stream(self, request):
            return self._broken()

        async def _broken(self):
            raise RuntimeError("provider down")
            yield  # 使其成为 async generator（不可达）

    adapter = MccAdapter(llm=BrokenLlm())
    events = _drain(adapter, _req())

    assert isinstance(events[-1], RunFinished)
    assert events[-1].status == "error"


# ─── MockAdapter ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("task", "expect_status"),
    [("fix the bug", "passed"), ("this will fail hard", "failed")],
    ids=["pass", "fail"],
)
def test_mock_adapter_terminal_status(task: str, expect_status: str) -> None:
    events = _drain(MockAdapter(), _req(task=task))
    assert events[0].type == "run_started"
    assert events[-1].status == expect_status


def test_mock_adapter_emits_full_pipeline() -> None:
    """Mock 事件覆盖完整链路：请求→文本→工具调用→结果→汇报→终态。"""
    events = _drain(MockAdapter(), _req(task="anything"))
    kinds = [e.type for e in events]
    assert kinds.count("llm_request") == 3
    assert kinds.count("tool_call") == 3
    assert kinds.count("tool_result") == 3
    # 最终汇报文本（output_contains 断言的判定对象）
    assert any("PASSED" in e.delta for e in events if isinstance(e, LlmChunk))
    seqs = [e.seq for e in events]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
