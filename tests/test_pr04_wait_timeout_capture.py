"""PR-04 测试：TaskWaitTimeoutError 被 Orchestrator 显式捕获，不落入通用 retry。

P0-04 核心问题：
  Dispatcher.wait_for_result() 超时抛 TaskWaitTimeoutError。
  TaskWaitTimeoutError 继承 DomainError，不是 WaitTimeoutError / TaskTimeoutError。
  Orchestrator 的 except (WaitTimeoutError, TaskTimeoutError) 无法捕获它，
  导致 TaskWaitTimeoutError 落入通用 except Exception 分支，
  该分支在尚有重试次数时盲目 continue，不确认旧执行已终止，
  可能产生重叠 Retry。

修复后预期：
  - _dispatch_and_wait 显式捕获 TaskWaitTimeoutError，标记旧 attempt 为 TIMED_OUT。
  - _execute_assignment 显式捕获 TaskWaitTimeoutError，走超时重试路径（非通用 Exception 路径）。
  - 旧 attempt 被标记 TIMED_OUT 后才重试，不产生重叠执行。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from mcc_core.runtime import (
    RuntimeCapabilities,
    RuntimeEvent,
    RuntimeEventType,
    TaskHandle,
)
from mcc_core.runtime.errors import TaskWaitTimeoutError
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

# -- fixtures --


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


# -- StubRuntime：逐行复制自 test_phase6_deep_fix.py --


class StubRuntime:
    """测试用 Runtime 桩：实现 AgentRuntime + PausableRuntime 协议。"""

    def __init__(
        self,
        runtime_id: str,
        result: RuntimeResult | None = None,
    ) -> None:
        self._runtime_id = runtime_id
        self._status = RuntimeStatus.CREATED
        self._result = result or RuntimeResult(
            runtime_id=runtime_id,
            status=RuntimeStatus.COMPLETED,
            summary="done",
        )
        self.call_log: list[str] = []
        self.start_count = 0

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

    def cancel(self) -> None:
        self.call_log.append("cancel")
        self._status = RuntimeStatus.CANCELLED

    def pause(self) -> None:
        self.call_log.append("pause")
        self._status = RuntimeStatus.PAUSED

    def resume(self) -> None:
        self.call_log.append("resume")
        self._status = RuntimeStatus.RUNNING

    async def wait_done(self, *, timeout: float | None = None) -> RuntimeResult:
        self.call_log.append("wait_done")
        self._status = self._result.status
        return self._result

    async def submit_task(self, task: str, *, task_id: str | None = None) -> str:
        self.call_log.append(f"submit_task:{task}")
        self.start_count += 1
        self._status = RuntimeStatus.RUNNING
        return task_id or f"task::{task}"

    async def wait_task(self, task_id: str, *, timeout: float | None = None) -> RuntimeResult:
        self.call_log.append("wait_task")
        self._status = self._result.status
        return self._result


# -- 测试用 Dispatcher 桩 --


class WaitTimeoutDispatcher:
    """TaskWaitTimeoutError 测试桩：第一次 wait_for_result 抛 TaskWaitTimeoutError，第二次成功。

    记录 fail_attempt 调用以验证旧 attempt 被标记为 TIMED_OUT。
    与 RetryIsolationDispatcher 的区别：抛 TaskWaitTimeoutError 而非 TaskTimeoutError。
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
            raise TaskWaitTimeoutError(handle.task_id, timeout or 0.0)
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


class AlwaysWaitTimeoutDispatcher:
    """始终抛 TaskWaitTimeoutError 的桩，用于验证不重试场景。"""

    def __init__(self) -> None:
        self.dispatch_count = 0
        self.wait_count = 0
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
        raise TaskWaitTimeoutError(handle.task_id, timeout or 0.0)

    async def fail_attempt(
        self,
        handle: TaskHandle,
        *,
        error: str,
        status: AttemptStatus = AttemptStatus.FAILED,
    ) -> None:
        self.fail_attempt_calls.append((handle.attempt_id, error, status))


# -- 测试 --


class TestPR04WaitTimeoutCapture:
    """TaskWaitTimeoutError 被 Orchestrator 显式捕获，不落入通用 except Exception。"""

    async def test_wait_timeout_marks_attempt_timed_out_before_retry(
        self,
        registry: AgentRegistry,
        bus: MessageBus,
        plan_store: PlanStore,
        task_store: TaskStore,
        coordinator: RuntimeCoordinator,
    ):
        """第一次等待超时(TaskWaitTimeoutError) → 旧 attempt 标记 TIMED_OUT → 重试成功。"""
        policy = TeamPolicy(task_lease_secs=10.0, max_attempts=2)
        dispatcher = WaitTimeoutDispatcher()
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
        # 旧 attempt 被标记为 TIMED_OUT（而非落入通用 Exception 不标记）
        assert len(dispatcher.fail_attempt_calls) == 1
        attempt_id, error, status = dispatcher.fail_attempt_calls[0]
        assert status is AttemptStatus.TIMED_OUT
        assert attempt_id == "att-1"

    async def test_wait_timeout_no_retry_still_fails_attempt(
        self,
        registry: AgentRegistry,
        bus: MessageBus,
        plan_store: PlanStore,
        task_store: TaskStore,
        coordinator: RuntimeCoordinator,
    ):
        """等待超时且不重试时，attempt 仍被标记为 TIMED_OUT。"""
        policy = TeamPolicy(task_lease_secs=10.0, max_attempts=1)
        dispatcher = WaitTimeoutDispatcher()
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
        # attempt 被标记为 TIMED_OUT（而非通用 Exception 的 "execution error"）
        assert len(dispatcher.fail_attempt_calls) == 1
        _, _, status = dispatcher.fail_attempt_calls[0]
        assert status is AttemptStatus.TIMED_OUT

    async def test_wait_timeout_error_not_treated_as_generic_exception(
        self,
        registry: AgentRegistry,
        bus: MessageBus,
        plan_store: PlanStore,
        task_store: TaskStore,
        coordinator: RuntimeCoordinator,
    ):
        """TaskWaitTimeoutError 不落入通用 except Exception 分支。

        通用分支用 "execution error: {e}" 作为错误消息，
        超时分支用 "task timed out after ...s" 作为错误消息。
        若 TaskWaitTimeoutError 落入通用分支，错误消息会包含 "execution error"。
        """
        policy = TeamPolicy(task_lease_secs=5.0, max_attempts=1)
        dispatcher = AlwaysWaitTimeoutDispatcher()
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

        # 验证 TaskWaitTimeoutError 走超时分支而非通用 Exception 分支：
        # 超时分支会调用 fail_attempt(status=TIMED_OUT)，
        # 通用分支不会调用 fail_attempt（直接 complete_task with "execution error"）。
        # fail_attempt_calls 非空且 status=TIMED_OUT 即证明走了超时分支。
