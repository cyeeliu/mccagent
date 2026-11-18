"""Phase 2: AgentLoopExecutor 测试。

覆盖：成功执行、取消、异常、关闭状态、事件排空、属性。
使用 StubAgentLoop 避免真实 LLM 依赖。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcc_core.event import TextDelta, ToolCallEnd, ToolCallStart, TurnEnd, UserEcho
from mcc_core.runtime import RuntimeResult, RuntimeStatus, TaskStatus, TokenUsage
from mcc_core.runtime.task import RuntimeTask
from mcc_core.types import Usage
from mcc_harness.agent import AgentRun
from mcc_harness.execution.agent_loop_executor import AgentLoopExecutor
from mcc_harness.execution.cancellation import CancellationToken


class StubAgentLoop:
    """最小 AgentLoop 桩，不依赖 LLM/工具。"""

    def __init__(
        self,
        final_text: str = "done",
        iterations: int = 1,
        usage: Usage | None = None,
        cost_usd: float = 0.0,
        events: list[Any] | None = None,
        raise_exc: Exception | None = None,
        check_cancel: bool = False,
    ) -> None:
        self._final_text = final_text
        self._iterations = iterations
        self._usage = usage or Usage()
        self._cost_usd = cost_usd
        self._events = events or []
        self._raise_exc = raise_exc
        self._check_cancel = check_cancel
        self.call_count = 0
        self.last_user_input: str | None = None
        self.last_cancel_event: asyncio.Event | None = None

    async def run_streaming(
        self,
        user_input: str,
        tx: asyncio.Queue,
        instruction_queue: asyncio.Queue[str] | None = None,
        cancel_event: asyncio.Event | None = None,
        pause_event: asyncio.Event | None = None,
    ) -> AgentRun:
        self.call_count += 1
        self.last_user_input = user_input
        self.last_cancel_event = cancel_event

        if self._check_cancel and cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError()

        if self._raise_exc is not None:
            raise self._raise_exc

        for ev in self._events:
            await tx.put(ev)

        return AgentRun(
            final_text=self._final_text,
            iterations=self._iterations,
            total_usage=self._usage,
            cost_usd=self._cost_usd,
        )


def _make_task(objective: str = "test objective") -> RuntimeTask:
    return RuntimeTask(
        task_id="task-1",
        attempt_id="att-1",
        agent_id="agent-1",
        objective=objective,
    )


class TestAgentLoopExecutorProperties:
    def test_runtime_id_explicit(self):
        loop = StubAgentLoop()
        executor = AgentLoopExecutor(loop, runtime_id="rt-custom", agent_id="a1")
        assert executor.runtime_id == "rt-custom"

    def test_runtime_id_auto_generated(self):
        loop = StubAgentLoop()
        executor = AgentLoopExecutor(loop)
        assert executor.runtime_id.startswith("executor-")

    def test_agent_id(self):
        loop = StubAgentLoop()
        executor = AgentLoopExecutor(loop, agent_id="agent-x")
        assert executor.agent_id == "agent-x"

    def test_not_closed_on_creation(self):
        loop = StubAgentLoop()
        executor = AgentLoopExecutor(loop)
        assert executor.is_closed is False


class TestAgentLoopExecutorSuccess:
    @pytest.mark.asyncio
    async def test_success_returns_succeeded(self):
        loop = StubAgentLoop(final_text="hello", iterations=2)
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")
        result = await executor.execute(_make_task())
        assert result.status == TaskStatus.SUCCEEDED
        assert result.is_success is True
        assert result.is_terminal is True

    @pytest.mark.asyncio
    async def test_success_output_contains_final_text(self):
        loop = StubAgentLoop(final_text="the answer")
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")
        result = await executor.execute(_make_task())
        assert result.output["final_text"] == "the answer"

    @pytest.mark.asyncio
    async def test_success_output_contains_iterations(self):
        loop = StubAgentLoop(iterations=3)
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")
        result = await executor.execute(_make_task())
        assert result.output["iterations"] == 3

    @pytest.mark.asyncio
    async def test_success_summary_truncated(self):
        long_text = "x" * 600
        loop = StubAgentLoop(final_text=long_text)
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")
        result = await executor.execute(_make_task())
        assert len(result.summary) == 500

    @pytest.mark.asyncio
    async def test_success_summary_short_text(self):
        loop = StubAgentLoop(final_text="short")
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")
        result = await executor.execute(_make_task())
        assert result.summary == "short"

    @pytest.mark.asyncio
    async def test_success_usage_mapped(self):
        usage = Usage(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=10,
            cache_read_input_tokens=20,
        )
        loop = StubAgentLoop(usage=usage)
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")
        result = await executor.execute(_make_task())
        assert result.usage is not None
        assert result.usage.input_tokens == 100
        assert result.usage.output_tokens == 50
        assert result.usage.cache_creation_input_tokens == 10
        assert result.usage.cache_read_input_tokens == 20

    @pytest.mark.asyncio
    async def test_success_cost_mapped(self):
        loop = StubAgentLoop(cost_usd=0.015)
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")
        result = await executor.execute(_make_task())
        assert result.cost_usd is not None
        assert float(result.cost_usd) == 0.015

    @pytest.mark.asyncio
    async def test_success_cost_none_when_zero(self):
        loop = StubAgentLoop(cost_usd=0.0)
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")
        result = await executor.execute(_make_task())
        assert result.cost_usd is None

    @pytest.mark.asyncio
    async def test_success_ids_propagated(self):
        loop = StubAgentLoop()
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")
        result = await executor.execute(_make_task())
        assert result.task_id == "task-1"
        assert result.attempt_id == "att-1"
        assert result.runtime_id == "rt-1"
        assert result.agent_id == "a1"

    @pytest.mark.asyncio
    async def test_success_agent_id_falls_back_to_task(self):
        loop = StubAgentLoop()
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="")
        result = await executor.execute(_make_task())
        assert result.agent_id == "agent-1"

    @pytest.mark.asyncio
    async def test_success_timestamps_set(self):
        loop = StubAgentLoop()
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")
        result = await executor.execute(_make_task())
        assert result.started_at is not None
        assert result.finished_at is not None
        assert result.finished_at >= result.started_at

    @pytest.mark.asyncio
    async def test_delegates_objective_to_loop(self):
        loop = StubAgentLoop()
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")
        await executor.execute(_make_task(objective="do something"))
        assert loop.last_user_input == "do something"


class TestAgentLoopExecutorCancellation:
    @pytest.mark.asyncio
    async def test_cancel_via_cancelled_error(self):
        loop = StubAgentLoop(raise_exc=asyncio.CancelledError())
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")
        result = await executor.execute(_make_task())
        assert result.status == TaskStatus.CANCELLED
        assert result.is_terminal is True

    @pytest.mark.asyncio
    async def test_cancel_via_event_set_after_run(self):
        loop = StubAgentLoop()
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")
        token = CancellationToken()
        token.cancel(reason="user")

        result = await executor.execute(_make_task(), cancellation=token)
        assert result.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_via_event_checked_in_loop(self):
        loop = StubAgentLoop(check_cancel=True)
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")
        token = CancellationToken()
        token.cancel()

        result = await executor.execute(_make_task(), cancellation=token)
        assert result.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_result_has_summary(self):
        loop = StubAgentLoop(raise_exc=asyncio.CancelledError())
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")
        result = await executor.execute(_make_task())
        assert result.summary == "任务被取消"

    @pytest.mark.asyncio
    async def test_cancel_passes_event_to_loop(self):
        loop = StubAgentLoop()
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")
        token = CancellationToken()

        await executor.execute(_make_task(), cancellation=token)
        assert loop.last_cancel_event is not None
        assert loop.last_cancel_event is token.to_asyncio_event()


class TestAgentLoopExecutorError:
    @pytest.mark.asyncio
    async def test_generic_exception_returns_failed(self):
        loop = StubAgentLoop(raise_exc=RuntimeError("boom"))
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")
        result = await executor.execute(_make_task())
        assert result.status == TaskStatus.FAILED
        assert result.is_terminal is True

    @pytest.mark.asyncio
    async def test_error_contains_code_and_message(self):
        loop = StubAgentLoop(raise_exc=ValueError("bad value"))
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")
        result = await executor.execute(_make_task())
        assert result.error is not None
        assert result.error.code == "execution_error"
        assert "bad value" in result.error.message

    @pytest.mark.asyncio
    async def test_error_summary_is_message(self):
        loop = StubAgentLoop(raise_exc=RuntimeError("something failed"))
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")
        result = await executor.execute(_make_task())
        assert result.summary == "something failed"


class TestAgentLoopExecutorClosed:
    @pytest.mark.asyncio
    async def test_closed_returns_failed(self):
        loop = StubAgentLoop()
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")
        await executor.close()
        result = await executor.execute(_make_task())
        assert result.status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_closed_error_code(self):
        loop = StubAgentLoop()
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")
        await executor.close()
        result = await executor.execute(_make_task())
        assert result.error is not None
        assert result.error.code == "executor_closed"

    @pytest.mark.asyncio
    async def test_closed_does_not_call_loop(self):
        loop = StubAgentLoop()
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")
        await executor.close()
        await executor.execute(_make_task())
        assert loop.call_count == 0

    @pytest.mark.asyncio
    async def test_is_closed_true_after_close(self):
        loop = StubAgentLoop()
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")
        await executor.close()
        assert executor.is_closed is True


class TestAgentLoopExecutorEventDraining:
    @pytest.mark.asyncio
    async def test_events_drained_to_emitter(self):
        events = [
            UserEcho(text="hi"),
            TextDelta(text="chunk1"),
            ToolCallStart(id="t1", name="bash", args_preview="ls"),
            ToolCallEnd(id="t1", output="out", is_error=False),
            TurnEnd(cost_usd=0.01),
        ]
        loop = StubAgentLoop(events=events)
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")

        emitted: list = []
        emitter = MagicMock()
        emitter.emit = AsyncMock(side_effect=lambda ev: emitted.append(ev))

        await executor.execute(_make_task(), emit=emitter)
        assert len(emitted) == 5

    @pytest.mark.asyncio
    async def test_no_events_no_emit_calls(self):
        loop = StubAgentLoop(events=[])
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")

        emitter = MagicMock()
        emitter.emit = AsyncMock()

        await executor.execute(_make_task(), emit=emitter)
        assert emitter.emit.call_count == 0

    @pytest.mark.asyncio
    async def test_drained_events_have_correct_sequence(self):
        events = [TextDelta(text="a"), TextDelta(text="b"), TextDelta(text="c")]
        loop = StubAgentLoop(events=events)
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")

        emitted: list = []
        emitter = MagicMock()
        emitter.emit = AsyncMock(side_effect=lambda ev: emitted.append(ev))

        await executor.execute(_make_task(), emit=emitter)
        assert emitted[0].sequence == 1
        assert emitted[1].sequence == 2
        assert emitted[2].sequence == 3

    @pytest.mark.asyncio
    async def test_drained_events_have_task_ids(self):
        events = [TextDelta(text="a")]
        loop = StubAgentLoop(events=events)
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")

        emitted: list = []
        emitter = MagicMock()
        emitter.emit = AsyncMock(side_effect=lambda ev: emitted.append(ev))

        await executor.execute(_make_task(), emit=emitter)
        assert emitted[0].task_id == "task-1"
        assert emitted[0].attempt_id == "att-1"

    @pytest.mark.asyncio
    async def test_no_emit_no_drain(self):
        events = [TextDelta(text="a"), TextDelta(text="b")]
        loop = StubAgentLoop(events=events)
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")

        result = await executor.execute(_make_task())
        assert result.status == TaskStatus.SUCCEEDED


class TestAgentLoopExecutorResultSerialization:
    @pytest.mark.asyncio
    async def test_success_result_to_dict(self):
        loop = StubAgentLoop(final_text="ok", iterations=1)
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")
        result = await executor.execute(_make_task())
        d = result.to_dict()
        assert d["status"] == "succeeded"
        assert d["task_id"] == "task-1"
        assert d["runtime_id"] == "rt-1"
        assert d["output"]["final_text"] == "ok"

    @pytest.mark.asyncio
    async def test_error_result_to_dict(self):
        loop = StubAgentLoop(raise_exc=RuntimeError("fail"))
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")
        result = await executor.execute(_make_task())
        d = result.to_dict()
        assert d["status"] == "failed"
        assert d["error"]["code"] == "execution_error"

    @pytest.mark.asyncio
    async def test_cancelled_result_to_dict(self):
        loop = StubAgentLoop(raise_exc=asyncio.CancelledError())
        executor = AgentLoopExecutor(loop, runtime_id="rt-1", agent_id="a1")
        result = await executor.execute(_make_task())
        d = result.to_dict()
        assert d["status"] == "cancelled"
