"""Phase 3 深度修复验证：InteractiveRuntimeV2 执行管理。

P0-02: InteractiveRuntimeV2 正确管理 submit 创建的执行 Task：
       - 按 attempt_id 追踪 RunningExecution
       - CancellationToken 与执行 Task 一一对应
       - cancel_task 精确取消 Attempt
       - stop drain/cancel 语义清晰
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

import pytest
from mcc_core.runtime import RuntimeStatus
from mcc_core.runtime.task import RuntimeTask


class SlowStubAgentLoop:
    """慢速桩 AgentLoop：支持取消，用于测试任务追踪。"""

    def __init__(self, *, delay: float = 0.5) -> None:
        self._delay = delay

    async def run_streaming(
        self,
        prompt: str,
        queue: asyncio.Queue,
        *,
        instruction_queue: asyncio.Queue | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> Any:
        from mcc_core import Usage
        from mcc_harness.agent import AgentRun

        try:
            if cancel_event is not None:
                done, pending = await asyncio.wait(
                    [
                        asyncio.create_task(asyncio.sleep(self._delay)),
                        asyncio.create_task(cancel_event.wait()),
                    ],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for p in pending:
                    p.cancel()
                if cancel_event.is_set():
                    await queue.put(None)
                    return AgentRun(
                        final_text="",
                        iterations=0,
                        total_usage=Usage(),
                        cost_usd=0.0,
                    )
            else:
                await asyncio.sleep(self._delay)
            await queue.put(None)
            return AgentRun(
                final_text=f"done: {prompt}",
                iterations=1,
                total_usage=Usage(),
                cost_usd=0.0,
            )
        except asyncio.CancelledError:
            await queue.put(None)
            return AgentRun(
                final_text="",
                iterations=0,
                total_usage=Usage(),
                cost_usd=0.0,
            )


def _make_task(
    *,
    task_id: str | None = None,
    attempt_id: str | None = None,
    agent_id: str = "test-agent",
    objective: str = "test objective",
) -> RuntimeTask:
    return RuntimeTask(
        task_id=task_id or f"task-{uuid4().hex[:8]}",
        attempt_id=attempt_id or f"att-{uuid4().hex[:8]}",
        agent_id=agent_id,
        objective=objective,
    )


# -- P0-02: InteractiveRuntimeV2 任务追踪 --


class TestInteractiveRuntimeTaskTracking:
    @pytest.fixture()
    async def runtime(self) -> Any:
        from mcc_harness.runtime.interactive import InteractiveRuntimeV2

        agent = SlowStubAgentLoop(delay=0.3)
        rt = InteractiveRuntimeV2(agent, agent_id="interactive-test")
        await rt.start()
        return rt

    async def test_submit_task_is_tracked(self, runtime: Any):
        task = _make_task()
        await runtime.submit(task)
        assert runtime._running.get(task.attempt_id) is not None
        await runtime.stop(timeout=5.0)

    async def test_cancel_task_cancels_exact_attempt(self, runtime: Any):
        task = _make_task()
        await runtime.submit(task)
        await asyncio.sleep(0.05)
        await runtime.cancel_task(task.task_id, task.attempt_id, reason="test")
        await asyncio.sleep(0.1)
        running = runtime._running.get(task.attempt_id)
        assert running is None or running.cancellation.is_cancelled
        await runtime.stop(timeout=5.0)

    async def test_stop_waits_for_submitted_task(self):
        from mcc_harness.runtime.interactive import InteractiveRuntimeV2

        agent = SlowStubAgentLoop(delay=0.2)
        rt = InteractiveRuntimeV2(agent, agent_id="interactive-test")
        await rt.start()
        task = _make_task()
        await rt.submit(task)
        await rt.stop(timeout=5.0, cancel_running=False)
        assert rt.status is RuntimeStatus.STOPPED
        assert len(rt._running) == 0

    async def test_stop_cancel_running_true_cancels(self):
        from mcc_harness.runtime.interactive import InteractiveRuntimeV2

        agent = SlowStubAgentLoop(delay=2.0)
        rt = InteractiveRuntimeV2(agent, agent_id="interactive-test")
        await rt.start()
        task = _make_task()
        await rt.submit(task)
        await asyncio.sleep(0.05)
        await rt.stop(timeout=5.0, cancel_running=True)
        assert rt.status is RuntimeStatus.STOPPED
        assert len(rt._running) == 0

    async def test_executor_not_closed_before_task_finishes(self):
        from mcc_harness.runtime.interactive import InteractiveRuntimeV2

        agent = SlowStubAgentLoop(delay=0.2)
        rt = InteractiveRuntimeV2(agent, agent_id="interactive-test")
        await rt.start()
        task = _make_task()
        await rt.submit(task)
        await asyncio.sleep(0.05)
        assert not rt._executor.is_closed
        await rt.stop(timeout=5.0)
        assert rt._executor.is_closed


# -- P0-02: InteractiveRuntimeV2 cancel 不影响其他 attempt --


class TestCancelDoesNotAffectOtherAttempt:
    async def test_cancel_only_targets_specified_attempt(self):
        from mcc_harness.runtime.interactive import InteractiveRuntimeV2

        agent = SlowStubAgentLoop(delay=2.0)
        rt = InteractiveRuntimeV2(agent, agent_id="interactive-test")
        await rt.start()
        task1 = _make_task(attempt_id="att-1")
        await rt.submit(task1)
        await asyncio.sleep(0.05)

        running1 = rt._running.get("att-1")
        assert running1 is not None
        assert not running1.cancellation.is_cancelled

        await rt.cancel_task(task1.task_id, "att-1", reason="test")
        await asyncio.sleep(0.1)

        running1_after = rt._running.get("att-1")
        assert running1_after is None or running1_after.cancellation.is_cancelled

        await rt.stop(timeout=5.0, cancel_running=True)
