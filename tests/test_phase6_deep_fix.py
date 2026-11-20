"""Phase 6 深度修复测试：TeamOrchestrator 任务闭环重构（Phase F）。

验证两个关键修复：
- 工作项 11（Retry 隔离）：超时后重试前，旧 attempt 必须被标记为 TIMED_OUT。
- 工作项 14（级联取消）：cancel 时活跃 attempt 必须被标记为 CANCELLED。
"""

from __future__ import annotations

import asyncio
import contextlib
import unittest.mock
from pathlib import Path

import pytest
from mcc_core.runtime import (
    RuntimeCapabilities,
    RuntimeEvent,
    RuntimeEventType,
    TaskHandle,
)
from mcc_core.runtime.errors import TaskTimeoutError
from mcc_core.runtime.task_record import AttemptStatus
from mcc_harness import (
    AgentRegistry,
    MessageBus,
    PlanStore,
    RuntimeCoordinator,
    RuntimeResult,
    RuntimeStatus,
    TeamOrchestrator,
    TeamPolicy,
    TeamRunStatus,
    WaitTimeoutError,
)
from mcc_tools import TaskStore

# -- fixture --


@pytest.fixture
def tmp_base(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def registry() -> AgentRegistry:
    return AgentRegistry(heartbeat_timeout=30.0)


@pytest.fixture
def bus(tmp_base: Path) -> MessageBus:
    return MessageBus(tmp_base)


@pytest.fixture
def plan_store() -> PlanStore:
    return PlanStore()


@pytest.fixture
def task_store(tmp_base: Path) -> TaskStore:
    return TaskStore(tmp_base / ".tasks")


@pytest.fixture
def coordinator() -> RuntimeCoordinator:
    return RuntimeCoordinator()


def register_agents(orch: TeamOrchestrator, count: int = 1) -> list[str]:
    ids = [f"agent-{i}" for i in range(1, count + 1)]
    for i, aid in enumerate(ids):
        orch.register_agent(
            agent_id=aid,
            agent_name=f"worker-{i + 1}",
            capabilities=frozenset({"code_generation"}),
        )
    return ids


def make_team(orch: TeamOrchestrator) -> object:
    return orch.create_team("team-1", "alpha", "lead-1")


# -- StubRuntime：逐行复制自 test_orchestrator.py --


class StubRuntime:
    """测试用 Runtime 桩：实现 AgentRuntime + PausableRuntime 协议。

    通过预设 result 控制执行结果，通过 call_log 记录调用顺序。
    """

    def __init__(
        self,
        runtime_id: str,
        result: RuntimeResult | None = None,
        raise_timeout: bool = False,
        raise_error: Exception | None = None,
        delay: float = 0.0,
    ) -> None:
        self._runtime_id = runtime_id
        self._status = RuntimeStatus.CREATED
        self._result = result or RuntimeResult(
            runtime_id=runtime_id,
            status=RuntimeStatus.COMPLETED,
            summary="done",
        )
        self._raise_timeout = raise_timeout
        self._raise_error = raise_error
        self._delay = delay
        self.call_log: list[str] = []
        self.start_count = 0
        self.cancelled = False
        self.paused = False
        self.resumed = False
        self.stopped = False

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    @property
    def status(self) -> RuntimeStatus:
        return self._status

    @property
    def alive(self) -> bool:
        return self._status not in (
            RuntimeStatus.COMPLETED,
            RuntimeStatus.FAILED,
            RuntimeStatus.CANCELLED,
            RuntimeStatus.TIMEOUT,
            RuntimeStatus.SHUTDOWN,
        )

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(supports_cancel=True, supports_health_check=False)

    async def start(self, task: str = "", *, task_id: str | None = None) -> None:
        self.call_log.append(f"start:{task}")
        self.start_count += 1
        self._status = RuntimeStatus.RUNNING

    async def stop(self, *, timeout: float = 5.0) -> None:
        self.call_log.append("stop")
        self._status = RuntimeStatus.SHUTDOWN
        self.stopped = True

    def cancel(self) -> None:
        self.call_log.append("cancel")
        self.cancelled = True
        self._status = RuntimeStatus.CANCELLED

    def pause(self) -> None:
        self.call_log.append("pause")
        self.paused = True
        self._status = RuntimeStatus.PAUSED

    def resume(self) -> None:
        self.call_log.append("resume")
        self.resumed = True
        self._status = RuntimeStatus.RUNNING

    async def wait_done(self, *, timeout: float | None = None) -> RuntimeResult:
        self.call_log.append("wait_done")
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._raise_error is not None:
            raise self._raise_error
        if self._raise_timeout:
            raise WaitTimeoutError("rt-stub", 1.0)
        self._status = self._result.status
        return self._result

    async def submit_task(self, task: str, *, task_id: str | None = None) -> str:
        self.call_log.append(f"submit_task:{task}")
        self.start_count += 1
        self._status = RuntimeStatus.RUNNING
        return task_id or f"task::{task}"

    async def wait_task(self, task_id: str, *, timeout: float | None = None) -> RuntimeResult:
        self.call_log.append("wait_task")
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        if self._raise_error is not None:
            raise self._raise_error
        if self._raise_timeout:
            raise WaitTimeoutError("rt-stub", 1.0)
        self._status = self._result.status
        return self._result


# -- 测试用 Dispatcher 桩 --


class RetryIsolationDispatcher:
    """重试隔离测试桩：第一次 wait_for_result 超时，第二次成功。

    记录 fail_attempt 调用以验证旧 attempt 被标记为 TIMED_OUT。
    """

    def __init__(self) -> None:
        self.dispatch_count = 0
        self.wait_count = 0
        self.fail_attempt_calls: list[tuple[str, str, AttemptStatus]] = []
        self._first_wait = True

    async def dispatch(
        self,
        task_id: str,
        agent_id: str,
        *,
        runtime_id: str,
        objective: str | None = None,
        input_data: dict[str, object] | None = None,
        timeout_seconds: float | None = None,
        run_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> TaskHandle:
        self.dispatch_count += 1
        return TaskHandle(
            task_id=task_id,
            attempt_id=f"att-{self.dispatch_count}",
            runtime_id=runtime_id,
        )

    async def wait_for_result(
        self,
        handle: TaskHandle,
        *,
        timeout: float | None = None,
    ) -> RuntimeEvent:
        self.wait_count += 1
        if self._first_wait:
            self._first_wait = False
            raise TaskTimeoutError(handle.task_id, handle.attempt_id)
        return RuntimeEvent(
            event_id=f"evt-{handle.task_id}",
            event_type=RuntimeEventType.TASK_COMPLETED,
            runtime_id=handle.runtime_id,
            task_id=handle.task_id,
            attempt_id=handle.attempt_id,
            payload={"summary": "ok"},
        )

    async def fail_attempt(
        self,
        handle: TaskHandle,
        *,
        error: str,
        status: AttemptStatus = AttemptStatus.FAILED,
    ) -> None:
        self.fail_attempt_calls.append((handle.attempt_id, error, status))


class CancelTestDispatcher:
    """级联取消测试桩：wait_for_result 阻塞直到 fail_attempt 被调用。

    模拟真实行为：fail_attempt 标记 attempt 为 CANCELLED 后，
    wait_for_result 返回 TASK_CANCELLED 事件。
    """

    def __init__(self) -> None:
        self.dispatch_count = 0
        self.dispatched = asyncio.Event()
        self._wait_block = asyncio.Event()
        self.fail_attempt_calls: list[tuple[str, str, AttemptStatus]] = []

    async def dispatch(
        self,
        task_id: str,
        agent_id: str,
        *,
        runtime_id: str,
        objective: str | None = None,
        input_data: dict[str, object] | None = None,
        timeout_seconds: float | None = None,
        run_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> TaskHandle:
        self.dispatch_count += 1
        self.dispatched.set()
        return TaskHandle(
            task_id=task_id,
            attempt_id=f"att-{self.dispatch_count}",
            runtime_id=runtime_id,
        )

    async def wait_for_result(
        self,
        handle: TaskHandle,
        *,
        timeout: float | None = None,
    ) -> RuntimeEvent:
        await self._wait_block.wait()
        return RuntimeEvent(
            event_id=f"evt-cancel-{handle.task_id}",
            event_type=RuntimeEventType.TASK_CANCELLED,
            runtime_id=handle.runtime_id,
            task_id=handle.task_id,
            attempt_id=handle.attempt_id,
            payload={"error": "cancelled"},
        )

    async def fail_attempt(
        self,
        handle: TaskHandle,
        *,
        error: str,
        status: AttemptStatus = AttemptStatus.FAILED,
    ) -> None:
        self.fail_attempt_calls.append((handle.attempt_id, error, status))
        self._wait_block.set()


# -- 工作项 11：Retry 隔离 --


class TestRetryIsolation:
    """超时后重试前，旧 attempt 必须被标记为 TIMED_OUT。"""

    async def test_timeout_then_retry_fails_old_attempt(
        self,
        registry: AgentRegistry,
        bus: MessageBus,
        plan_store: PlanStore,
        task_store: TaskStore,
        coordinator: RuntimeCoordinator,
    ):
        """第一次执行超时 → 旧 attempt 标记 TIMED_OUT → 重试成功。"""
        policy = TeamPolicy(task_lease_secs=10.0, max_attempts=2)
        dispatcher = RetryIsolationDispatcher()
        orch = TeamOrchestrator(
            registry=registry,
            bus=bus,
            plan_store=plan_store,
            task_store=task_store,
            dispatcher=dispatcher,
            coordinator=coordinator,
        )
        agent_ids = register_agents(orch, 1)
        team = orch.create_team("team-1", "alpha", "lead-1", policy=policy)

        coordinator.register(StubRuntime("rt-1"))
        orch.bind_runtime(agent_ids[0], "rt-1")

        run = await orch.start(team, "build feature A")
        run = await orch.execute_run(run.run_id)

        # 重试成功
        assert run.status == TeamRunStatus.COMPLETED
        # dispatch 被调用 2 次（第一次超时，第二次成功）
        assert dispatcher.dispatch_count == 2
        assert dispatcher.wait_count == 2
        # 旧 attempt 被标记为 TIMED_OUT
        assert len(dispatcher.fail_attempt_calls) == 1
        attempt_id, error, status = dispatcher.fail_attempt_calls[0]
        assert status is AttemptStatus.TIMED_OUT
        assert attempt_id == "att-1"

    async def test_timeout_no_retry_still_fails_attempt(
        self,
        registry: AgentRegistry,
        bus: MessageBus,
        plan_store: PlanStore,
        task_store: TaskStore,
        coordinator: RuntimeCoordinator,
    ):
        """超时且不重试时，attempt 仍被标记为 TIMED_OUT。"""
        policy = TeamPolicy(task_lease_secs=10.0, max_attempts=1)
        dispatcher = RetryIsolationDispatcher()
        orch = TeamOrchestrator(
            registry=registry,
            bus=bus,
            plan_store=plan_store,
            task_store=task_store,
            dispatcher=dispatcher,
            coordinator=coordinator,
        )
        agent_ids = register_agents(orch, 1)
        team = orch.create_team("team-1", "alpha", "lead-1", policy=policy)

        coordinator.register(StubRuntime("rt-1"))
        orch.bind_runtime(agent_ids[0], "rt-1")

        run = await orch.start(team, "build feature A")
        run = await orch.execute_run(run.run_id)

        # 超时失败
        assert run.status == TeamRunStatus.FAILED
        # attempt 被标记为 TIMED_OUT
        assert len(dispatcher.fail_attempt_calls) == 1
        _, _, status = dispatcher.fail_attempt_calls[0]
        assert status is AttemptStatus.TIMED_OUT


# -- 工作项 14：级联取消 --


class TestCascadeCancel:
    """cancel 时活跃 attempt 必须被标记为 CANCELLED。"""

    async def test_cancel_fails_active_attempt(
        self,
        registry: AgentRegistry,
        bus: MessageBus,
        plan_store: PlanStore,
        task_store: TaskStore,
        coordinator: RuntimeCoordinator,
    ):
        """执行中 cancel → 活跃 attempt 标记 CANCELLED。"""
        policy = TeamPolicy(task_lease_secs=10.0)
        dispatcher = CancelTestDispatcher()
        orch = TeamOrchestrator(
            registry=registry,
            bus=bus,
            plan_store=plan_store,
            task_store=task_store,
            dispatcher=dispatcher,
            coordinator=coordinator,
        )
        agent_ids = register_agents(orch, 1)
        team = orch.create_team("team-1", "alpha", "lead-1", policy=policy)

        rt = StubRuntime("rt-1")
        coordinator.register(rt)
        orch.bind_runtime(agent_ids[0], "rt-1")

        run = await orch.start(team, "build feature A")

        # 在后台启动 execute_run
        execute_task = asyncio.create_task(orch.execute_run(run.run_id))

        # 等待 dispatch 发生（handle 已注册到 _active_handles）
        await asyncio.wait_for(dispatcher.dispatched.wait(), timeout=5.0)
        # 让事件循环运行一下，确保 handle 已注册
        await asyncio.sleep(0)

        # 设置 runtime 为活跃状态
        rt._status = RuntimeStatus.RUNNING

        # 取消 run
        await orch.cancel(run.run_id, reason="user abort")

        # 等待 execute_run 完成
        await asyncio.wait_for(execute_task, timeout=5.0)

        # 活跃 attempt 被标记为 CANCELLED
        assert len(dispatcher.fail_attempt_calls) == 1
        attempt_id, error, status = dispatcher.fail_attempt_calls[0]
        assert status is AttemptStatus.CANCELLED
        assert "abort" in error or "cancel" in error

    async def test_cancel_no_active_handles_is_noop(
        self,
        registry: AgentRegistry,
        bus: MessageBus,
        plan_store: PlanStore,
        task_store: TaskStore,
        coordinator: RuntimeCoordinator,
    ):
        """cancel 时无活跃 handle，不调用 fail_attempt。"""
        from mcc_core.runtime import RuntimeEvent, RuntimeEventType, TaskHandle

        class NoopDispatcher:
            def __init__(self) -> None:
                self.dispatch_count = 0
                self.fail_attempt_calls: list[tuple] = []

            async def dispatch(self, task_id, agent_id, *, runtime_id, **kwargs):
                self.dispatch_count += 1
                return TaskHandle(
                    task_id=task_id,
                    attempt_id=f"att-{self.dispatch_count}",
                    runtime_id=runtime_id,
                )

            async def wait_for_result(self, handle, *, timeout=None):
                return RuntimeEvent(
                    event_id=f"evt-{handle.task_id}",
                    event_type=RuntimeEventType.TASK_COMPLETED,
                    runtime_id=handle.runtime_id,
                    task_id=handle.task_id,
                    attempt_id=handle.attempt_id,
                    payload={"summary": "ok"},
                )

            async def fail_attempt(self, handle, *, error, status=AttemptStatus.FAILED):
                self.fail_attempt_calls.append((handle.attempt_id, error, status))

        dispatcher = NoopDispatcher()
        orch = TeamOrchestrator(
            registry=registry,
            bus=bus,
            plan_store=plan_store,
            task_store=task_store,
            dispatcher=dispatcher,
            coordinator=coordinator,
        )
        agent_ids = register_agents(orch, 1)
        team = make_team(orch)

        rt = StubRuntime("rt-1")
        coordinator.register(rt)
        orch.bind_runtime(agent_ids[0], "rt-1")

        run = await orch.start(team, "build feature A")
        rt._status = RuntimeStatus.RUNNING

        # cancel before execute_run — 无活跃 handle
        await orch.cancel(run.run_id, reason="user abort")

        assert len(dispatcher.fail_attempt_calls) == 0
        assert run.status == TeamRunStatus.CANCELLED
