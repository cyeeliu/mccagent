"""Phase 5 深度修复验证：Dispatcher 幂等状态机（P1-02）。

P1-02 修复内容：
(a) 幂等命中时返回的 TaskHandle 应使用 Attempt 原有 runtime_id，而非当前传入值。
(b) Attempt 缺少 PENDING → RUNNING 中间态转换：新增 start_attempt 方法。
(c) submit 异常不区分类型：CancelledError 应标记 CANCELLED 而非 FAILED。
(d) TaskStoreProtocol 缺少 start_attempt 方法。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from mcc_core.runtime import (
    AttemptNotFoundError,
    AttemptStatus,
    InMemoryTaskStore,
    InvalidTransitionError,
    RuntimeCapabilities,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeTask,
    TaskHandle,
    TaskStatus,
    TaskStoreProtocol,
)
from mcc_harness import DispatchError, RuntimeCoordinator, TaskDispatcher
from mcc_harness.runtime import TaskStoreAdapter
from mcc_tools.task import TaskStore

# -- 桩类 --


class _StubRuntime:
    """支持新 Protocol submit + subscribe_events 的 Runtime 桩。"""

    def __init__(self, runtime_id: str = "stub-rt") -> None:
        self._runtime_id = runtime_id
        self._event_listeners: list[asyncio.Queue] = []
        self._submitted_tasks: list[RuntimeTask] = []

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    @property
    def status(self) -> Any:
        return None

    @property
    def alive(self) -> bool:
        return True

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(supports_cancel=False, supports_health_check=False)

    async def submit(self, task: RuntimeTask) -> TaskHandle:
        self._submitted_tasks.append(task)
        return TaskHandle(
            task_id=task.task_id,
            attempt_id=task.attempt_id,
            runtime_id=self._runtime_id,
        )

    def subscribe_events(
        self,
        *,
        task_id: str | None = None,
        attempt_id: str | None = None,
    ) -> AsyncIterator[RuntimeEvent]:
        return self._event_iterator(task_id=task_id, attempt_id=attempt_id)

    async def _event_iterator(
        self,
        *,
        task_id: str | None = None,
        attempt_id: str | None = None,
    ) -> AsyncIterator[RuntimeEvent]:
        queue: asyncio.Queue[RuntimeEvent | None] = asyncio.Queue()
        self._event_listeners.append(queue)
        try:
            while True:
                ev = await queue.get()
                if ev is None:
                    break
                yield ev
        finally:
            self._event_listeners.remove(queue)

    async def stop(self, **kwargs: Any) -> None:
        pass


class _CancelSubmitRuntime:
    """Runtime 桩：submit 抛出 CancelledError，模拟任务被取消。"""

    def __init__(self, runtime_id: str = "cancel-rt") -> None:
        self._runtime_id = runtime_id

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    @property
    def status(self) -> Any:
        return None

    @property
    def alive(self) -> bool:
        return True

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(supports_cancel=False, supports_health_check=False)

    async def submit(self, task: RuntimeTask) -> TaskHandle:
        raise asyncio.CancelledError()

    def subscribe_events(
        self,
        *,
        task_id: str | None = None,
        attempt_id: str | None = None,
    ) -> AsyncIterator[RuntimeEvent]:
        return self._event_iterator()

    async def _event_iterator(self) -> AsyncIterator[RuntimeEvent]:
        if False:
            yield  # 使其成为 async generator

    async def stop(self, **kwargs: Any) -> None:
        pass


# -- (d) TaskStoreProtocol.start_attempt 方法 --


class TestStartAttemptProtocol:
    def test_in_memory_store_has_start_attempt(self):
        """TaskStoreProtocol 应包含 start_attempt 方法。"""
        store = InMemoryTaskStore()
        assert hasattr(store, "start_attempt")

    def test_adapter_has_start_attempt(self, tmp_path: Path):
        """TaskStoreAdapter 应实现 start_attempt 方法。"""
        store = TaskStore(tmp_path / "tasks")
        adapter = TaskStoreAdapter(store, persist_dir=tmp_path / "attempts")
        assert hasattr(adapter, "start_attempt")


# -- (b) start_attempt: PENDING → RUNNING 中间态转换 --


class TestStartAttemptInMemoryStore:
    async def test_start_attempt_transitions_pending_to_running(self):
        """start_attempt 应将 PENDING attempt 转为 RUNNING。"""
        store = InMemoryTaskStore()
        task = await store.create_task("s", "d")
        attempt = await store.create_attempt(task.task_id, "a-1")
        assert attempt.status is AttemptStatus.PENDING

        running = await store.start_attempt(attempt.attempt_id)
        assert running.status is AttemptStatus.RUNNING

        # 持久化确认
        fetched = await store.get_attempt(attempt.attempt_id)
        assert fetched is not None
        assert fetched.status is AttemptStatus.RUNNING

    async def test_start_attempt_rejects_non_pending(self):
        """start_attempt 对非 PENDING attempt 应抛 InvalidTransitionError。"""
        store = InMemoryTaskStore()
        task = await store.create_task("s", "d")
        attempt = await store.create_attempt(task.task_id, "a-1")
        await store.complete_attempt(attempt.attempt_id, summary="done")

        with pytest.raises(InvalidTransitionError):
            await store.start_attempt(attempt.attempt_id)

    async def test_start_attempt_nonexistent_raises(self):
        """start_attempt 对不存在的 attempt_id 应抛 AttemptNotFoundError。"""
        store = InMemoryTaskStore()
        with pytest.raises(AttemptNotFoundError):
            await store.start_attempt("nonexistent-att")


class TestStartAttemptAdapter:
    async def test_start_attempt_transitions_pending_to_running(self, tmp_path: Path):
        """TaskStoreAdapter.start_attempt 应将 PENDING 转为 RUNNING 并持久化。"""
        store = TaskStore(tmp_path / "tasks")
        adapter = TaskStoreAdapter(store, persist_dir=tmp_path / "attempts")
        task = await adapter.create_task("s", "d")
        attempt = await adapter.create_attempt(task.task_id, "a-1")
        assert attempt.status is AttemptStatus.PENDING

        running = await adapter.start_attempt(attempt.attempt_id)
        assert running.status is AttemptStatus.RUNNING

        # 新实例从磁盘加载，确认持久化
        adapter2 = TaskStoreAdapter(store, persist_dir=tmp_path / "attempts")
        fetched = await adapter2.get_attempt(attempt.attempt_id)
        assert fetched is not None
        assert fetched.status is AttemptStatus.RUNNING

    async def test_start_attempt_rejects_non_pending(self, tmp_path: Path):
        """TaskStoreAdapter.start_attempt 对非 PENDING 应抛 InvalidTransitionError。"""
        store = TaskStore(tmp_path / "tasks")
        adapter = TaskStoreAdapter(store, persist_dir=tmp_path / "attempts")
        task = await adapter.create_task("s", "d")
        attempt = await adapter.create_attempt(task.task_id, "a-1")
        await adapter.complete_attempt(attempt.attempt_id, summary="done")

        with pytest.raises(InvalidTransitionError):
            await adapter.start_attempt(attempt.attempt_id)


# -- (b) dispatch 调用 start_attempt，attempt 进入 RUNNING --


class TestDispatchStartsAttempt:
    async def test_dispatch_transitions_attempt_to_running(self):
        """dispatch 成功后 attempt 应处于 RUNNING 状态（非 PENDING）。"""
        store = InMemoryTaskStore()
        coord = RuntimeCoordinator()
        rt = _StubRuntime("rt-1")
        coord.register(rt)
        dispatcher = TaskDispatcher(store, coord)

        task = await store.create_task("s", "d")
        handle = await dispatcher.dispatch(
            task.task_id,
            "a-1",
            runtime_id="rt-1",
        )

        attempt = await store.get_attempt(handle.attempt_id)
        assert attempt is not None
        assert attempt.status is AttemptStatus.RUNNING


# -- (a) 幂等命中时返回原始 runtime_id --


class TestIdempotentDispatchRuntimeId:
    async def test_idempotent_dispatch_preserves_original_runtime_id(self):
        """幂等命中时 TaskHandle.runtime_id 应为首次 dispatch 的 runtime_id。"""
        store = InMemoryTaskStore()
        coord = RuntimeCoordinator()
        rt1 = _StubRuntime("rt-original")
        rt2 = _StubRuntime("rt-different")
        coord.register(rt1)
        coord.register(rt2)
        dispatcher = TaskDispatcher(store, coord)

        task = await store.create_task("s", "d")
        handle1 = await dispatcher.dispatch(
            task.task_id,
            "a-1",
            runtime_id="rt-original",
            idempotency_key="idem-1",
        )
        assert handle1.runtime_id == "rt-original"

        # 第二次 dispatch 使用不同 runtime_id 但相同 idempotency_key
        handle2 = await dispatcher.dispatch(
            task.task_id,
            "a-1",
            runtime_id="rt-different",
            idempotency_key="idem-1",
        )

        # 应返回原始 runtime_id，而非当前传入的
        assert handle2.runtime_id == "rt-original"
        assert handle2.attempt_id == handle1.attempt_id

    async def test_idempotent_dispatch_via_create_attempt_race(self):
        """create_attempt 内部竞态返回已有 attempt 时也保留原始 runtime_id。"""
        store = InMemoryTaskStore()
        coord = RuntimeCoordinator()
        rt = _StubRuntime("rt-original")
        coord.register(rt)
        dispatcher = TaskDispatcher(store, coord)

        task = await store.create_task("s", "d")

        # 预先在 store 中创建一个同 idempotency_key 的 RUNNING attempt
        pre_attempt = await store.create_attempt(
            task.task_id,
            "a-1",
            runtime_id="rt-original",
            idempotency_key="idem-race",
        )
        await store.start_attempt(pre_attempt.attempt_id)

        # dispatch 使用不同 runtime_id，但 create_attempt 返回已有 RUNNING attempt
        handle = await dispatcher.dispatch(
            task.task_id,
            "a-1",
            runtime_id="rt-different",
            idempotency_key="idem-race",
        )

        # 应返回 attempt 上的原始 runtime_id
        assert handle.runtime_id == "rt-original"
        assert handle.attempt_id == pre_attempt.attempt_id


# -- (c) CancelledError 标记 CANCELLED 而非 FAILED --


class TestDispatchCancelledError:
    async def test_cancelled_error_marks_attempt_cancelled(self):
        """runtime.submit 抛 CancelledError 时 attempt 应标记为 CANCELLED。"""
        store = InMemoryTaskStore()
        coord = RuntimeCoordinator()
        rt = _CancelSubmitRuntime("cancel-rt")
        coord.register(rt)
        dispatcher = TaskDispatcher(store, coord)

        task = await store.create_task("s", "d")
        with pytest.raises(asyncio.CancelledError):
            await dispatcher.dispatch(
                task.task_id,
                "a-1",
                runtime_id="cancel-rt",
            )

        # attempt 应标记为 CANCELLED，而非 FAILED
        attempts = await store.list_attempts(task.task_id)
        assert len(attempts) == 1
        assert attempts[0].status is AttemptStatus.CANCELLED
        assert attempts[0].error is not None

    async def test_generic_error_marks_attempt_failed(self):
        """runtime.submit 抛普通异常时 attempt 应标记为 FAILED（回归确认）。"""
        store = InMemoryTaskStore()
        coord = RuntimeCoordinator()

        class _ErrorRuntime(_StubRuntime):
            async def submit(self, task: RuntimeTask) -> TaskHandle:
                raise RuntimeError("boom")

        rt = _ErrorRuntime("err-rt")
        coord.register(rt)
        dispatcher = TaskDispatcher(store, coord)

        task = await store.create_task("s", "d")
        with pytest.raises(DispatchError):
            await dispatcher.dispatch(
                task.task_id,
                "a-1",
                runtime_id="err-rt",
            )

        attempts = await store.list_attempts(task.task_id)
        assert len(attempts) == 1
        assert attempts[0].status is AttemptStatus.FAILED
