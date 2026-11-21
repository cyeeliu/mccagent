"""PR-15: Runtime Contract 扩展验证。

验证所有 Runtime 实现满足统一行为契约：
- Protocol 符合性（isinstance AgentRuntime）
- submit → TASK_ACCEPTED + TASK_STARTED 事件
- 事件顺序：TASK_ACCEPTED → TASK_STARTED → 终态事件
- cancel_task → TASK_CANCELLED 事件
- timeout_seconds → TASK_TIMED_OUT 事件
- stop 终止运行中任务
- 单任务 Runtime 拒绝并发 submit
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any
from uuid import uuid4

import pytest
from mcc_core.runtime import RuntimeEventType, TaskStatus
from mcc_core.runtime.protocols import AgentRuntime
from mcc_core.runtime.result import RuntimeResult
from mcc_core.runtime.task import RuntimeTask


class StubExecutor:
    """可配置桩执行器：支持成功/失败/延迟/取消。"""

    def __init__(
        self,
        *,
        delay: float = 0.0,
        status: TaskStatus = TaskStatus.SUCCEEDED,
        summary: str = "done",
    ) -> None:
        self.delay = delay
        self.status = status
        self.summary = summary
        self.started = asyncio.Event()
        self.finished = asyncio.Event()

    async def execute(
        self,
        task: RuntimeTask,
        *,
        emit: Any | None = None,
        cancellation: Any | None = None,
    ) -> RuntimeResult:
        self.started.set()
        try:
            if cancellation is not None and self.delay > 0:
                done, pending = await asyncio.wait(
                    [
                        asyncio.create_task(cancellation.wait()),
                        asyncio.create_task(asyncio.sleep(self.delay)),
                    ],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await t
                if cancellation.is_cancelled:
                    return RuntimeResult(
                        task_id=task.task_id,
                        attempt_id=task.attempt_id,
                        runtime_id="stub",
                        agent_id=task.agent_id,
                        status=TaskStatus.CANCELLED,
                        summary="cancelled",
                    )
            else:
                await asyncio.sleep(self.delay)
            return RuntimeResult(
                task_id=task.task_id,
                attempt_id=task.attempt_id,
                runtime_id="stub",
                agent_id=task.agent_id,
                status=self.status,
                summary=self.summary,
            )
        finally:
            self.finished.set()

    async def close(self) -> None:
        pass


class RecordingTaskStore:
    """记录 fail/complete 调用的测试桩。"""

    def __init__(self) -> None:
        self.fail_calls: list[tuple[str, str, Any]] = []
        self.complete_calls: list[str] = []

    async def fail_attempt(
        self,
        attempt_id: str,
        *,
        error: str,
        status: Any = None,
    ) -> None:
        from mcc_core.runtime.task_record import AttemptStatus

        self.fail_calls.append((attempt_id, error, status or AttemptStatus.FAILED))

    async def complete_attempt(
        self,
        attempt_id: str,
        **kwargs: Any,
    ) -> None:
        self.complete_calls.append(attempt_id)


def _make_task(*, timeout_seconds: float | None = None) -> RuntimeTask:
    return RuntimeTask(
        task_id=f"task-{uuid4().hex[:8]}",
        attempt_id=f"att-{uuid4().hex[:8]}",
        agent_id="test-agent",
        objective="contract test",
        timeout_seconds=timeout_seconds,
    )


def _make_task_runtime(
    *,
    executor: StubExecutor | None = None,
    task_store: Any | None = None,
) -> Any:
    from mcc_harness.runtime.task_runtime import TaskRuntime

    return TaskRuntime(
        executor or StubExecutor(),
        agent_id="contract-task",
        task_store=task_store,
    )


def _make_interactive_runtime(
    *,
    executor: StubExecutor | None = None,
    task_store: Any | None = None,
) -> Any:
    from mcc_harness.runtime.interactive import InteractiveRuntimeV2

    rt = InteractiveRuntimeV2(
        object(),
        runtime_id="contract-interactive",
        agent_id="contract-interactive",
        task_store=task_store or RecordingTaskStore(),
    )
    rt._executor = executor or StubExecutor()
    return rt


def _make_persistent_runtime(
    *,
    executor: StubExecutor | None = None,
) -> Any:
    from mcc_harness.runtime.persistent import PersistentRuntimeV2

    return PersistentRuntimeV2(
        executor or StubExecutor(),
        max_concurrency=2,
        agent_id="contract-persistent",
    )


RUNTIME_FACTORIES = {
    "TaskRuntime": _make_task_runtime,
    "InteractiveRuntimeV2": _make_interactive_runtime,
    "PersistentRuntimeV2": _make_persistent_runtime,
}


@pytest.fixture(params=list(RUNTIME_FACTORIES.keys()))
def runtime_factory(request: pytest.FixtureRequest) -> Any:
    return RUNTIME_FACTORIES[request.param]


async def _collect_events(
    runtime: Any,
    *targets: RuntimeEventType,
    timeout: float = 5.0,
) -> list[Any]:
    """收集事件直到看到所有 target 类型或超时。"""
    events: list[Any] = []
    seen: set[RuntimeEventType] = set()

    async def collector() -> None:
        async for ev in runtime.subscribe_events():
            events.append(ev)
            if ev.event_type in targets:
                seen.add(ev.event_type)
            if seen >= set(targets):
                break

    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(collector(), timeout=timeout)
    return events


class TestPR15ProtocolConformance:
    """所有 Runtime 实现满足 AgentRuntime Protocol。"""

    @pytest.mark.parametrize("rt_name", list(RUNTIME_FACTORIES.keys()))
    async def test_satisfies_agent_runtime_protocol(self, rt_name: str) -> None:
        rt = RUNTIME_FACTORIES[rt_name]()
        assert isinstance(rt, AgentRuntime), (
            f"{rt_name} 不满足 AgentRuntime Protocol"
        )


class TestPR15SubmitEventContract:
    """submit 发射 TASK_ACCEPTED + TASK_STARTED 事件。"""

    async def test_submit_emits_task_accepted(
        self, runtime_factory: Any,
    ) -> None:
        rt = runtime_factory()
        await rt.start()
        task = _make_task()

        collect_task = asyncio.create_task(
            _collect_events(rt, RuntimeEventType.TASK_ACCEPTED, timeout=3.0),
        )
        await asyncio.sleep(0.05)
        await rt.submit(task)
        events = await collect_task

        types = [ev.event_type for ev in events]
        assert RuntimeEventType.TASK_ACCEPTED in types, f"未收到 TASK_ACCEPTED，收到: {types}"
        await rt.stop(cancel_running=True)

    async def test_submit_emits_task_started(
        self, runtime_factory: Any,
    ) -> None:
        rt = runtime_factory()
        await rt.start()
        task = _make_task()

        collect_task = asyncio.create_task(
            _collect_events(rt, RuntimeEventType.TASK_STARTED, timeout=3.0),
        )
        await asyncio.sleep(0.05)
        await rt.submit(task)
        events = await collect_task

        types = [ev.event_type for ev in events]
        assert RuntimeEventType.TASK_STARTED in types, f"未收到 TASK_STARTED，收到: {types}"
        await rt.stop(cancel_running=True)


class TestPR15EventOrdering:
    """事件顺序：TASK_ACCEPTED → TASK_STARTED → 终态。"""

    async def test_event_order_success(self, runtime_factory: Any) -> None:
        rt = runtime_factory()
        await rt.start()
        task = _make_task()

        collect_task = asyncio.create_task(
            _collect_events(rt, RuntimeEventType.TASK_COMPLETED, timeout=3.0),
        )
        await asyncio.sleep(0.05)
        await rt.submit(task)
        events = await collect_task

        types = [ev.event_type for ev in events]
        assert RuntimeEventType.TASK_ACCEPTED in types
        assert RuntimeEventType.TASK_STARTED in types
        assert RuntimeEventType.TASK_COMPLETED in types
        accepted_idx = types.index(RuntimeEventType.TASK_ACCEPTED)
        started_idx = types.index(RuntimeEventType.TASK_STARTED)
        completed_idx = types.index(RuntimeEventType.TASK_COMPLETED)
        assert accepted_idx < started_idx < completed_idx, (
            f"事件顺序错误: {types}"
        )
        await rt.stop()


class TestPR15CancelContract:
    """cancel_task 发射 TASK_CANCELLED 事件。"""

    async def test_cancel_emits_task_cancelled(
        self, runtime_factory: Any,
    ) -> None:
        executor = StubExecutor(delay=10.0)
        rt = runtime_factory(executor=executor)
        await rt.start()
        task = _make_task()

        collect_task = asyncio.create_task(
            _collect_events(rt, RuntimeEventType.TASK_CANCELLED, timeout=5.0),
        )
        await asyncio.sleep(0.05)
        await rt.submit(task)
        await executor.started.wait()
        await rt.cancel_task(task.task_id, task.attempt_id, "test cancel")
        events = await collect_task

        types = [ev.event_type for ev in events]
        assert RuntimeEventType.TASK_CANCELLED in types, (
            f"未收到 TASK_CANCELLED，收到: {types}"
        )
        await rt.stop()


class TestPR15TimeoutContract:
    """timeout_seconds 发射 TASK_TIMED_OUT 事件。"""

    async def test_timeout_emits_timed_out(
        self, runtime_factory: Any,
    ) -> None:
        executor = StubExecutor(delay=5.0)
        rt = runtime_factory(executor=executor)
        await rt.start()
        task = _make_task(timeout_seconds=0.2)

        collect_task = asyncio.create_task(
            _collect_events(rt, RuntimeEventType.TASK_TIMED_OUT, timeout=3.0),
        )
        await asyncio.sleep(0.05)
        await rt.submit(task)
        t0 = time.monotonic()
        events = await collect_task
        elapsed = time.monotonic() - t0

        assert elapsed < 2.0, f"超时事件延迟 {elapsed:.2f}s"
        types = [ev.event_type for ev in events]
        assert RuntimeEventType.TASK_TIMED_OUT in types, (
            f"未收到 TASK_TIMED_OUT，收到: {types}"
        )
        await rt.stop()


class TestPR15StopTerminatesRunning:
    """stop 终止运行中任务。"""

    async def test_stop_cancels_running(self, runtime_factory: Any) -> None:
        executor = StubExecutor(delay=10.0)
        rt = runtime_factory(executor=executor)
        await rt.start()
        task = _make_task()
        await rt.submit(task)
        await executor.started.wait()

        t0 = time.monotonic()
        await rt.stop(cancel_running=True, timeout=3.0)
        elapsed = time.monotonic() - t0

        assert elapsed < 3.5, f"stop 耗时 {elapsed:.2f}s，未及时终止"
        assert executor.finished.is_set(), "executor 未在 stop 后终止"


class TestPR15DoubleSubmitRejected:
    """单任务 Runtime 拒绝并发 submit。"""

    @pytest.mark.parametrize("rt_name", ["TaskRuntime", "InteractiveRuntimeV2"])
    async def test_double_submit_rejected(self, rt_name: str) -> None:
        from mcc_core.runtime.errors import RuntimeNotReadyError

        executor = StubExecutor(delay=10.0)
        rt = RUNTIME_FACTORIES[rt_name](executor=executor)
        await rt.start()
        task1 = _make_task()
        task2 = _make_task()
        await rt.submit(task1)
        await executor.started.wait()

        with pytest.raises(RuntimeNotReadyError):
            await rt.submit(task2)
        await rt.stop(cancel_running=True)


class TestPR15PersistentMultiTask:
    """PersistentRuntimeV2 支持并发多任务。"""

    async def test_multiple_tasks_accepted(self) -> None:
        rt = _make_persistent_runtime(executor=StubExecutor(delay=0.05))
        await rt.start()
        tasks = [_make_task() for _ in range(3)]

        handles = []
        for task in tasks:
            handle = await rt.submit(task)
            handles.append(handle)

        assert len(handles) == 3
        for h, t in zip(handles, tasks, strict=True):
            assert h.task_id == t.task_id
        await rt.stop()


class TestPR15PersistentEventMapping:
    """PersistentRuntimeV2 事件映射覆盖所有终态。"""

    async def test_task_timed_out_event_mapped(self) -> None:
        rt = _make_persistent_runtime(executor=StubExecutor(delay=5.0))
        await rt.start()
        task = _make_task(timeout_seconds=0.2)

        collect_task = asyncio.create_task(
            _collect_events(rt, RuntimeEventType.TASK_TIMED_OUT, timeout=3.0),
        )
        await asyncio.sleep(0.05)
        await rt.submit(task)
        events = await collect_task

        types = [ev.event_type for ev in events]
        assert RuntimeEventType.TASK_TIMED_OUT in types, (
            f"PersistentRuntimeV2 未映射 TASK_TIMED_OUT，收到: {types}"
        )
        await rt.stop()
