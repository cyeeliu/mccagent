"""mini-claude-code 适配器测试：事件转译与 MockLlm 全链路。

mcc 系列包未安装时整体跳过（importorskip 门控）——
适配器是可选组件，安装方式见 adapters/mcc_agent.py 模块注释。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from runner.adapters.base import RunRequest
from runner.adapters.mcc_agent import MccAdapter, _Translator
from runner.events import LlmChunk, LlmRequest, RunFinished, RunStarted, ToolCall, ToolResult

pytest.importorskip("mcc_harness")

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
from mcc_llm import MessageStop, ToolUseInputDelta, ToolUseStart
from mcc_llm import TextDelta as LlmTextDelta
from mcc_llm.mock import MockLlm, single_text_turn


def _req(**overrides) -> RunRequest:
    payload = {
        "run_id": "run-1",
        "case_id": "case-1",
        "task": "write a file",
        "model": "test-model",
    }
    payload.update(overrides)
    return RunRequest(**payload)


def _drain(adapter: MccAdapter, req: RunRequest) -> list:
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


def test_adapter_tool_round_writes_file(tmp_path: Path) -> None:
    turn1 = [
        ToolUseStart(id="t1", name="write_file"),
        ToolUseInputDelta(
            partial_json=json.dumps({"path": "out.txt", "content": "eval-me"})
        ),
        MessageStop(stop_reason="tool_use"),
    ]
    turn2 = [LlmTextDelta(text="file written"), MessageStop(stop_reason="end_turn")]
    adapter = MccAdapter(llm=MockLlm([turn1, turn2]), work_dir=tmp_path)

    events = _drain(adapter, _req(task="write out.txt"))

    # 工具真实执行：write_file 在注入的工作目录落盘
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


def test_make_adapter_branch() -> None:
    from runner.adapters.mcc_agent import MccAdapter as Cls
    from runner.main import make_adapter

    assert isinstance(make_adapter("mini-claude-code"), Cls)


def test_make_adapter_unknown_scaffold() -> None:
    from runner.main import make_adapter

    with pytest.raises(ValueError, match="unknown scaffold"):
        make_adapter("nope")


# ─── runner 管线端到端：execute + MccAdapter + expectations 评分 ─────────


async def test_mcc_run_through_execute(monkeypatch) -> None:
    """MccAdapter 接入 runner 主循环：事件上报 + 评分覆盖执行状态。"""
    import httpx
    from test_main import FakeServer

    from runner import main as runner_main

    adapter = MccAdapter(llm=MockLlm([single_text_turn("done")]))
    monkeypatch.setattr(runner_main, "make_adapter", lambda scaffold: adapter)

    fake = FakeServer()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(fake.handler), base_url="http://test"
    ) as http:
        await runner_main.execute(
            http,
            {
                "run_id": "run-e2e",
                "case_id": "case-e2e",
                "task": "say done",
                "scaffold": "mini-claude-code",
                "model": "test-model",
                "expectations": [{"type": "no_tool_errors", "weight": 1.0}],
            },
        )

    complete = next(c for c in fake.calls if c[1].endswith("/complete"))
    payload = json.loads(complete[2])
    assert payload["status"] == "passed"
    assert payload["score"] == 1.0  # expectations 评分覆盖 adapter 的 score=None
    events = next(c for c in fake.calls if c[1].endswith("/events"))
    assert '"run_started"' in events[2]
    assert '"llm_chunk"' in events[2]
    assert '"run_finished"' in events[2]


class TestProviderFromEnv:
    """_provider_from_env 的分支选择：复用 mcc 的环境变量约定。"""

    def test_gateway_base_url_wins_over_api_keys(self, monkeypatch):
        monkeypatch.setenv("MCC_LLM_BASE_URL", "http://host.docker.internal:8000/v1")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-be-ignored")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-real")

        from mcc_llm.openai import OpenAiClient
        from runner.adapters.mcc_agent import _provider_from_env

        provider = _provider_from_env()
        assert isinstance(provider, OpenAiClient)
        assert provider._base_url == "http://host.docker.internal:8000/v1"
        assert provider._api_key == "sk-openai-real"

    def test_gateway_without_api_key_uses_placeholder(self, monkeypatch):
        monkeypatch.setenv("MCC_LLM_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        from mcc_llm.openai import OpenAiClient
        from runner.adapters.mcc_agent import _provider_from_env

        provider = _provider_from_env()
        assert isinstance(provider, OpenAiClient)
        assert provider._api_key == "dummy"

    def test_without_gateway_uses_anthropic_autodetect(self, monkeypatch):
        monkeypatch.delenv("MCC_LLM_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        from mcc_llm.anthropic import AnthropicClient
        from runner.adapters.mcc_agent import _provider_from_env

        assert isinstance(_provider_from_env(), AnthropicClient)
