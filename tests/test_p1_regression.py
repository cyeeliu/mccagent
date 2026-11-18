"""P1 回归测试：验证三个可靠性缺陷修复。

P1-01: TaskStoreAdapter 的 Attempt/Assignment 仅内存保存，非持久化。
P1-02: TaskDispatcher 幂等机制依赖内存 set，进程重启后失效。
P1-03: WorkerPool 关闭只放一个 sentinel，多 worker 时其余 worker 永久阻塞。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest
from mcc_core.runtime import (
    AttemptStatus,
    RuntimeCapabilities,
    TaskHandle,
)
from mcc_core.runtime.task_record import AssignmentStatus
from mcc_harness.runtime import (
    BoundedTaskQueue,
    RuntimeCoordinator,
    TaskDispatcher,
    TaskStoreAdapter,
    WorkerPool,
)
from mcc_tools.task import TaskStore

# -- 桩类 --


class _CountingRuntime:
    """Runtime 桩：仅记录 submit 次数，用于验证幂等不重复派发。"""

    def __init__(self, runtime_id: str = "rt-1") -> None:
        self._runtime_id = runtime_id
        self.submit_count = 0

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
        return RuntimeCapabilities(
            supports_cancel=False,
            supports_health_check=False,
        )

    async def submit(self, task: Any) -> TaskHandle:
        self.submit_count += 1
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
    ) -> AsyncIterator[Any]:
        return self._empty_events()

    async def _empty_events(self) -> AsyncIterator[Any]:
        """空事件流：立即结束。"""
        return
        yield  # 使函数成为 async generator，运行时立即 StopAsyncIteration

    async def stop(self, **kwargs: Any) -> None:
        return None


class _NoopExecutor:
    """Executor 桩：worker 取到 sentinel 即退出，不会真正执行。"""

    async def execute(self, task: Any) -> Any:
        return None


# -- P1-01: Attempt/Assignment 文件持久化 --


class TestP1_01:
    """P1-01: TaskStoreAdapter 应将 Attempt/Assignment 持久化到文件。"""

    async def test_attempt_persisted_across_restart(self, tmp_path):
        """重新构造 TaskStoreAdapter 后应能加载已持久化的 attempt。"""
        store = TaskStore(tmp_path / ".tasks")
        persist_dir = tmp_path / ".persist"

        adapter = TaskStoreAdapter(store, persist_dir=persist_dir)
        task = await adapter.create_task("subject-1", "desc-1")
        attempt = await adapter.create_attempt(
            task.task_id,
            "agent-1",
            runtime_id="rt-1",
            idempotency_key="idem-1",
        )
        await adapter.complete_attempt(attempt.attempt_id, summary="done-1")

        # 模拟进程重启：重新构造 adapter，复用同一 persist_dir
        adapter2 = TaskStoreAdapter(store, persist_dir=persist_dir)

        loaded = await adapter2.get_attempt(attempt.attempt_id)
        assert loaded is not None
        assert loaded.status is AttemptStatus.SUCCEEDED
        assert loaded.summary == "done-1"
        assert loaded.task_id == task.task_id
        assert loaded.agent_id == "agent-1"

        # idempotency 索引应从持久化数据派生恢复
        existing = await adapter2.find_by_idempotency_key("idem-1")
        assert existing is not None
        assert existing.attempt_id == attempt.attempt_id

        # list_attempts 应恢复
        attempts = await adapter2.list_attempts(task.task_id)
        assert len(attempts) == 1
        assert attempts[0].attempt_id == attempt.attempt_id

    async def test_assignment_persisted_across_restart(self, tmp_path):
        """重新构造 TaskStoreAdapter 后应能加载已持久化的 assignment。"""
        store = TaskStore(tmp_path / ".tasks")
        persist_dir = tmp_path / ".persist"

        adapter = TaskStoreAdapter(store, persist_dir=persist_dir)
        task = await adapter.create_task("subject-2", "desc-2")
        attempt = await adapter.create_attempt(task.task_id, "agent-2")
        asgn = await adapter.create_assignment(
            task.task_id,
            attempt.attempt_id,
            "agent-2",
            runtime_id="rt-2",
        )

        adapter2 = TaskStoreAdapter(store, persist_dir=persist_dir)
        loaded = await adapter2.get_assignment(asgn.assignment_id)
        assert loaded is not None
        assert loaded.status is AssignmentStatus.ACTIVE
        assert loaded.task_id == task.task_id
        assert loaded.agent_id == "agent-2"

        # list_assignments 应恢复
        assignments = await adapter2.list_assignments(task_id=task.task_id)
        assert len(assignments) == 1

    async def test_in_memory_mode_when_no_persist_dir(self, tmp_path):
        """persist_dir=None 时保持内存行为（向后兼容）。"""
        store = TaskStore(tmp_path / ".tasks")
        adapter = TaskStoreAdapter(store)

        task = await adapter.create_task("subject-3", "desc-3")
        attempt = await adapter.create_attempt(task.task_id, "agent-3")
        assert await adapter.get_attempt(attempt.attempt_id) is not None

        # 新实例无 persist_dir，不加载旧数据
        adapter2 = TaskStoreAdapter(store)
        assert await adapter2.get_attempt(attempt.attempt_id) is None


# -- P1-02: TaskDispatcher 幂等跨实例 --


class TestP1_02:
    """P1-02: 幂等应依赖 TaskStore 机制，而非内存 set。"""

    async def test_idempotency_across_dispatcher_instances(self, tmp_path):
        """新 dispatcher 实例（模拟重启）不应重复派发同一 idempotency_key。"""
        store = TaskStore(tmp_path / ".tasks")
        adapter = TaskStoreAdapter(store)
        coord = RuntimeCoordinator()
        rt = _CountingRuntime("rt-1")
        coord.register(rt)

        task = await adapter.create_task("s", "d")

        dispatcher1 = TaskDispatcher(adapter, coord)
        await dispatcher1.dispatch(
            task.task_id,
            "a-1",
            runtime_id="rt-1",
            idempotency_key="k-1",
        )
        assert rt.submit_count == 1

        # 模拟重启：新 dispatcher 实例，同一 store（持久化）
        dispatcher2 = TaskDispatcher(adapter, coord)
        await dispatcher2.dispatch(
            task.task_id,
            "a-1",
            runtime_id="rt-1",
            idempotency_key="k-1",
        )
        assert rt.submit_count == 1, "幂等 dispatch 不应重复 submit"

    async def test_no_dispatched_memory_set(self):
        """Dispatcher 不应维护内存 _dispatched set 做幂等。"""
        from mcc_core.runtime import InMemoryTaskStore

        store = InMemoryTaskStore()
        coord = RuntimeCoordinator()
        dispatcher = TaskDispatcher(store, coord)
        assert not hasattr(dispatcher, "_dispatched"), (
            "TaskDispatcher 不应依赖内存 _dispatched set 做幂等"
        )


# -- P1-03: WorkerPool 多 worker 优雅关闭 --


class TestP1_03:
    """P1-03: max_workers>1 时 stop 应优雅快速退出。"""

    async def test_stop_graceful_with_multiple_workers(self):
        """max_workers=2 时 stop 应快速优雅退出，不依赖超时+cancel。"""
        queue = BoundedTaskQueue()
        executor = _NoopExecutor()
        pool = WorkerPool(queue, executor, max_workers=2)
        pool.start()
        # 让 workers 启动并阻塞在 get()
        await asyncio.sleep(0.05)
        assert pool.active_workers == 2

        start = time.monotonic()
        await pool.stop(timeout=1.0)
        elapsed = time.monotonic() - start

        assert elapsed < 0.5, (
            f"stop 耗时 {elapsed:.3f}s，期望优雅快速退出（<0.5s），"
            "说明 sentinel 数量不足导致 worker 被超时 cancel"
        )
        assert not pool.is_started
        assert pool.active_workers == 0

    @pytest.mark.parametrize("max_workers", [1, 2, 4, 8])
    async def test_stop_graceful_various_worker_counts(self, max_workers):
        """不同 worker 数量下 stop 都应优雅退出。"""
        queue = BoundedTaskQueue()
        executor = _NoopExecutor()
        pool = WorkerPool(queue, executor, max_workers=max_workers)
        pool.start()
        await asyncio.sleep(0.05)

        start = time.monotonic()
        await pool.stop(timeout=1.0)
        elapsed = time.monotonic() - start

        assert elapsed < 0.5
        assert pool.active_workers == 0
