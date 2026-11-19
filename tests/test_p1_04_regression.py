"""P1-04 回归测试：拆分 TeammateRuntime 为 Definition/MessageAdapter/ExecutionService。

验证：
- 静态架构守卫：PersistentRuntimeV2.from_executor / TeammateDefinition /
  TeammatePolicy / TeammateMessageAdapter / TeammateExecutionService 存在；
- TeammateDefinition 从 TeammateSpec 提取纯配置且可往返；
- TeammateMessageAdapter 封装 MessageBus 收发；
- TeammateExecutionService 通过注入的 AgentExecutor 执行 RuntimeTask，
  含 Plan/Worktree/Result 回写；
- PersistentRuntimeV2.from_executor 直接持有 AgentExecutor，端到端 submit/wait。

使用真实集成（MessageBus / PlanStore / TaskStore + Stub executor 桩避免 LLM 依赖），
不注入 mock。
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from mcc_core.runtime import TaskStatus
from mcc_core.runtime.result import RuntimeResult as CoreRuntimeResult
from mcc_core.runtime.task import RuntimeTask
from mcc_harness.mailbox import MessageBus
from mcc_harness.plan import PlanStore
from mcc_harness.runtime.base import RuntimeStatus
from mcc_harness.runtime.persistent import (
    PersistentRuntimeV2,
)
from mcc_harness.teammate import (
    LEAD_NAME,
    TeammateDefinition,
    TeammateExecutionService,
    TeammateMessageAdapter,
    TeammatePolicy,
    TeammateRuntime,
    TeammateSpec,
)
from mcc_llm.mock import MockLlm
from mcc_tools import ToolRegistry

# -- 共享 fixture：真实集成环境 --


@pytest.fixture
def work_dir(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture
def bus(work_dir: Path) -> MessageBus:
    return MessageBus(work_dir)


@pytest.fixture
def plan_store() -> PlanStore:
    return PlanStore()


def _make_spec(**overrides) -> TeammateSpec:
    """构造测试用 TeammateSpec，字段可定制。"""
    defaults = dict(
        name="worker",
        role="coder",
        system="you are a coder",
        model="claude-haiku-4-5",
        tools=["read_file", "list_dir"],
        max_iterations=8,
        max_tokens=1024,
        budget_tokens=50_000,
        require_plan=False,
        timeout_secs=60.0,
    )
    defaults.update(overrides)
    return TeammateSpec(**defaults)


def _runtime_task(
    *,
    task_id: str = "t1",
    attempt_id: str = "a1",
    agent_id: str = "worker",
    objective: str = "do something",
) -> RuntimeTask:
    return RuntimeTask(
        task_id=task_id,
        attempt_id=attempt_id,
        agent_id=agent_id,
        objective=objective,
    )


class _StubExecutor:
    """Stub AgentExecutor 桩：记录执行并返回确定性 CoreRuntimeResult。

    避免依赖 LLM，同时满足 AgentExecutor 协议（execute + close）。
    """

    def __init__(
        self,
        *,
        status: TaskStatus = TaskStatus.SUCCEEDED,
        summary: str = "stub done",
        runtime_id: str = "stub-exec",
    ) -> None:
        self._status = status
        self._summary = summary
        self._runtime_id = runtime_id
        self.closed = False
        self.executed: list[RuntimeTask] = []

    async def execute(self, task: RuntimeTask, **kwargs: object) -> CoreRuntimeResult:
        self.executed.append(task)
        now = datetime.now(UTC)
        return CoreRuntimeResult(
            task_id=task.task_id,
            attempt_id=task.attempt_id,
            runtime_id=self._runtime_id,
            agent_id=task.agent_id,
            status=self._status,
            summary=self._summary,
            started_at=now,
            finished_at=now,
        )

    async def close(self) -> None:
        self.closed = True


class _RecordingTaskStore:
    """Stub TaskStore 桩：记录回写调用，不强制 claim 前置。"""

    def __init__(self) -> None:
        self.completes: list[str] = []
        self.fails: list[tuple[str, str]] = []
        self.timeouts: list[str] = []
        self.cancels: list[str] = []

    def complete(self, task_id: str, **kwargs: object) -> None:
        self.completes.append(task_id)

    def fail(self, task_id: str, *, error: str = "", **kwargs: object) -> None:
        self.fails.append((task_id, error))

    def timeout(self, task_id: str, **kwargs: object) -> None:
        self.timeouts.append(task_id)

    def cancel(self, task_id: str, **kwargs: object) -> None:
        self.cancels.append(task_id)


# -- 静态架构守卫 --


class TestArchitectureGuards:
    """断言 P1-04 拆分后的新架构元素存在。"""

    @pytest.mark.parametrize(
        "cls_name",
        [
            "TeammateDefinition",
            "TeammatePolicy",
            "TeammateMessageAdapter",
            "TeammateExecutionService",
        ],
    )
    def test_teammate_split_classes_exist(self, cls_name: str) -> None:
        import mcc_harness.teammate as teammate_mod

        assert hasattr(teammate_mod, cls_name), f"{cls_name} 应存在于 teammate 模块"
        assert isinstance(getattr(teammate_mod, cls_name), type)

    def test_persistent_runtime_has_from_executor(self) -> None:
        assert hasattr(PersistentRuntimeV2, "from_executor")
        assert callable(PersistentRuntimeV2.from_executor)

    def test_from_executor_is_classmethod(self) -> None:
        import inspect

        assert isinstance(
            inspect.getattr_static(PersistentRuntimeV2, "from_executor"),
            classmethod,
        )


# -- TeammateDefinition --


class TestTeammateDefinition:
    """TeammateDefinition 从 TeammateSpec 提取纯配置且可往返。"""

    def test_from_spec_maps_identity_fields(self) -> None:
        spec = _make_spec(name="explorer", role="researcher", system="sys", model="m1")
        definition = TeammateDefinition.from_spec(spec)
        assert definition.name == "explorer"
        assert definition.role == "researcher"
        assert definition.system == "sys"
        assert definition.model == "m1"

    def test_from_spec_maps_tools_as_tuple(self) -> None:
        spec = _make_spec(tools=["read_file", "list_dir", "grep"])
        definition = TeammateDefinition.from_spec(spec)
        assert isinstance(definition.tools, tuple)
        assert definition.tools == ("read_file", "list_dir", "grep")

    @pytest.mark.parametrize(
        ("max_iterations", "budget_tokens", "timeout_secs", "require_plan"),
        [
            (5, 30_000, 30.0, False),
            (20, 150_000, 300.0, True),
        ],
    )
    def test_from_spec_maps_policy(
        self,
        max_iterations: int,
        budget_tokens: int,
        timeout_secs: float,
        require_plan: bool,
    ) -> None:
        spec = _make_spec(
            max_iterations=max_iterations,
            budget_tokens=budget_tokens,
            timeout_secs=timeout_secs,
            require_plan=require_plan,
        )
        definition = TeammateDefinition.from_spec(spec)
        assert isinstance(definition.policy, TeammatePolicy)
        assert definition.policy.max_iterations == max_iterations
        assert definition.policy.budget_tokens == budget_tokens
        assert definition.policy.timeout_secs == timeout_secs
        assert definition.policy.require_plan is require_plan

    def test_to_spec_roundtrip(self) -> None:
        spec = _make_spec()
        definition = TeammateDefinition.from_spec(spec)
        restored = definition.to_spec()
        assert restored.name == spec.name
        assert restored.role == spec.role
        assert restored.system == spec.system
        assert restored.model == spec.model
        assert restored.tools == spec.tools
        assert restored.max_iterations == spec.max_iterations
        assert restored.max_tokens == spec.max_tokens
        assert restored.budget_tokens == spec.budget_tokens
        assert restored.require_plan == spec.require_plan
        assert restored.timeout_secs == spec.timeout_secs

    def test_definition_is_frozen(self) -> None:
        definition = TeammateDefinition.from_spec(_make_spec())
        with pytest.raises((AttributeError, TypeError)):
            definition.name = "mutated"  # type: ignore[misc]

    def test_policy_is_frozen(self) -> None:
        policy = TeammatePolicy()
        with pytest.raises((AttributeError, TypeError)):
            policy.max_iterations = 999  # type: ignore[misc]

    def test_default_policy_values(self) -> None:
        policy = TeammatePolicy()
        assert policy.max_iterations == 20
        assert policy.max_tokens == 4096
        assert policy.budget_tokens == 150_000
        assert policy.require_plan is False
        assert policy.timeout_secs == 300.0


# -- TeammateMessageAdapter --


class TestTeammateMessageAdapter:
    """TeammateMessageAdapter 封装 MessageBus 收发与唤醒。"""

    def test_send_to_lead_delivers_message(self, bus: MessageBus, work_dir: Path) -> None:
        adapter = TeammateMessageAdapter(bus, name="worker", lead_name=LEAD_NAME)
        msg = asyncio.run(
            adapter.send_to_lead("hello lead", kind="work_done"),
        )
        assert msg.sender == "worker"
        assert msg.recipient == LEAD_NAME
        assert msg.content == "hello lead"
        assert msg.kind == "work_done"

    async def test_send_to_lead_then_recv_by_lead(
        self,
        bus: MessageBus,
    ) -> None:
        adapter = TeammateMessageAdapter(bus, name="worker", lead_name=LEAD_NAME)
        await adapter.send_to_lead("result", kind="work_done", task_id="t-1")
        inbox = await bus.consume(LEAD_NAME)
        assert len(inbox) == 1
        assert inbox[0].content == "result"
        assert inbox[0].task_id == "t-1"

    async def test_send_task_submitted_to_self(self, bus: MessageBus) -> None:
        adapter = TeammateMessageAdapter(bus, name="worker", lead_name=LEAD_NAME)
        await adapter.send_task_submitted(LEAD_NAME, "do task", task_id="t-2")
        inbox = await adapter.recv()
        assert len(inbox) == 1
        assert inbox[0].kind == "task_submitted"
        assert inbox[0].content == "do task"
        assert inbox[0].task_id == "t-2"

    async def test_has_messages_reflects_inbox(self, bus: MessageBus) -> None:
        adapter = TeammateMessageAdapter(bus, name="worker", lead_name=LEAD_NAME)
        assert await adapter.has_messages() is False
        await adapter.send_task_submitted(LEAD_NAME, "x", task_id="t-3")
        assert await adapter.has_messages() is True

    def test_notify_sets_wake_event(self, bus: MessageBus) -> None:
        adapter = TeammateMessageAdapter(bus, name="worker", lead_name=LEAD_NAME)
        wake = asyncio.Event()
        assert wake.is_set() is False
        adapter.notify(wake)
        assert wake.is_set() is True

    def test_properties_expose_bus_and_names(self, bus: MessageBus) -> None:
        adapter = TeammateMessageAdapter(bus, name="worker", lead_name=LEAD_NAME)
        assert adapter.name == "worker"
        assert adapter.lead_name == LEAD_NAME
        assert adapter.bus is bus


# -- TeammateExecutionService --


class TestTeammateExecutionService:
    """TeammateExecutionService 通过注入的 AgentExecutor 执行 RuntimeTask。"""

    def _make_service(
        self,
        executor: _StubExecutor,
        *,
        bus: MessageBus | None = None,
        task_store: _RecordingTaskStore | None = None,
        plan_store: PlanStore | None = None,
        require_plan: bool = False,
    ) -> TeammateExecutionService:
        spec = _make_spec(require_plan=require_plan)
        definition = TeammateDefinition.from_spec(spec)
        message_adapter = TeammateMessageAdapter(bus, name="worker") if bus else None
        return TeammateExecutionService(
            executor,
            definition=definition,
            message_adapter=message_adapter,
            plan_store=plan_store,
            task_store=task_store,
        )

    async def test_execute_returns_succeeded_result(self, bus: MessageBus) -> None:
        executor = _StubExecutor(status=TaskStatus.SUCCEEDED, summary="ok")
        service = self._make_service(executor, bus=bus)
        result = await service.execute(_runtime_task())
        assert result.status is TaskStatus.SUCCEEDED
        assert result.summary == "ok"
        assert len(executor.executed) == 1

    async def test_execute_notifies_lead_via_bus(self, bus: MessageBus) -> None:
        executor = _StubExecutor(summary="work summary")
        service = self._make_service(executor, bus=bus)
        await service.execute(_runtime_task(task_id="t-notify"))
        inbox = await bus.consume(LEAD_NAME)
        assert len(inbox) == 1
        assert inbox[0].kind == "work_done"
        assert inbox[0].content == "work summary"
        assert inbox[0].meta["task_id"] == "t-notify"

    async def test_execute_writes_back_to_task_store(self, bus: MessageBus) -> None:
        executor = _StubExecutor(status=TaskStatus.SUCCEEDED)
        store = _RecordingTaskStore()
        service = self._make_service(executor, bus=bus, task_store=store)
        await service.execute(_runtime_task(task_id="t-store"))
        assert store.completes == ["t-store"]

    @pytest.mark.parametrize(
        ("status", "expected_kind"),
        [
            (TaskStatus.SUCCEEDED, "work_done"),
            (TaskStatus.FAILED, "work_failed"),
            (TaskStatus.CANCELLED, "work_cancelled"),
            (TaskStatus.TIMED_OUT, "work_timeout"),
        ],
    )
    async def test_execute_status_maps_to_lead_kind(
        self,
        bus: MessageBus,
        status: TaskStatus,
        expected_kind: str,
    ) -> None:
        executor = _StubExecutor(status=status, summary="x")
        service = self._make_service(executor, bus=bus)
        await service.execute(_runtime_task(task_id="t-map"))
        inbox = await bus.consume(LEAD_NAME)
        assert inbox[0].kind == expected_kind

    async def test_execute_writes_failure_to_task_store(self, bus: MessageBus) -> None:
        executor = _StubExecutor(status=TaskStatus.FAILED, summary="boom")
        store = _RecordingTaskStore()
        service = self._make_service(executor, bus=bus, task_store=store)
        await service.execute(_runtime_task(task_id="t-fail"))
        assert store.fails == [("t-fail", "boom")]

    async def test_execute_without_message_adapter_skips_bus(self) -> None:
        executor = _StubExecutor()
        service = self._make_service(executor, bus=None)
        result = await service.execute(_runtime_task())
        assert result.status is TaskStatus.SUCCEEDED

    async def test_execute_executor_exception_returns_failed(
        self,
        bus: MessageBus,
    ) -> None:
        class _BoomExecutor:
            async def execute(self, task: RuntimeTask, **kw: object) -> CoreRuntimeResult:
                raise RuntimeError("kaboom")

            async def close(self) -> None:
                pass

        service = self._make_service(_BoomExecutor(), bus=bus)  # type: ignore[arg-type]
        result = await service.execute(_runtime_task(task_id="t-err"))
        assert result.status is TaskStatus.FAILED
        assert "kaboom" in result.summary

    async def test_execute_plan_rejected_returns_failed(
        self,
        bus: MessageBus,
        plan_store: PlanStore,
    ) -> None:
        executor = _StubExecutor()
        service = self._make_service(
            executor,
            bus=bus,
            plan_store=plan_store,
            require_plan=True,
        )

        async def _reject_pending_plan() -> None:
            """轮询 plan_store 找到 service.execute 创建的 pending 请求并拒绝。"""
            for _ in range(100):
                req = await plan_store.pending_for("worker")
                if req is not None:
                    await plan_store.review(
                        req.request_id,
                        approve=False,
                        feedback="no",
                    )
                    return
                await asyncio.sleep(0.01)

        result, _ = await asyncio.gather(
            service.execute(_runtime_task(task_id="t-plan")),
            _reject_pending_plan(),
        )
        assert result.status is TaskStatus.FAILED
        assert "plan rejected" in result.summary
        assert executor.executed == []

    async def test_execute_plan_approved_then_runs_executor(
        self,
        bus: MessageBus,
        plan_store: PlanStore,
    ) -> None:
        executor = _StubExecutor(status=TaskStatus.SUCCEEDED, summary="approved run")
        service = self._make_service(
            executor,
            bus=bus,
            plan_store=plan_store,
            require_plan=True,
        )

        async def _approve_pending_plan() -> None:
            for _ in range(100):
                req = await plan_store.pending_for("worker")
                if req is not None:
                    await plan_store.review(req.request_id, approve=True)
                    return
                await asyncio.sleep(0.01)

        result, _ = await asyncio.gather(
            service.execute(_runtime_task(task_id="t-approve")),
            _approve_pending_plan(),
        )
        assert result.status is TaskStatus.SUCCEEDED
        assert result.summary == "approved run"
        assert len(executor.executed) == 1

    async def test_close_delegates_to_executor(self, bus: MessageBus) -> None:
        executor = _StubExecutor()
        service = self._make_service(executor, bus=bus)
        await service.close()
        assert executor.closed is True


# -- PersistentRuntimeV2.from_executor --


class TestPersistentRuntimeFromExecutor:
    """PersistentRuntimeV2.from_executor 直接持有 AgentExecutor。"""

    async def test_from_executor_instance_has_no_teammate(self) -> None:
        executor = _StubExecutor()
        pr = PersistentRuntimeV2.from_executor(executor, agent_id="worker")
        assert pr._teammate is None
        assert pr.agent_id == "worker"

    async def test_from_executor_holds_message_adapter(
        self,
        bus: MessageBus,
    ) -> None:
        executor = _StubExecutor()
        adapter = TeammateMessageAdapter(bus, name="worker")
        pr = PersistentRuntimeV2.from_executor(
            executor,
            agent_id="worker",
            message_adapter=adapter,
        )
        assert pr._message_adapter is adapter

    async def test_from_executor_submit_wait_end_to_end(self) -> None:
        executor = _StubExecutor(status=TaskStatus.SUCCEEDED, summary="done")
        pr = PersistentRuntimeV2.from_executor(
            executor,
            agent_id="worker",
            runtime_id="pr-test",
        )
        await pr.start()
        try:
            tid = await pr.submit_task("do something", task_id="t-e2e")
            assert tid == "t-e2e"
            result = await pr.wait_task(tid, timeout=5.0)
            assert result.status is RuntimeStatus.COMPLETED
            assert result.summary == "done"
            assert len(executor.executed) == 1
        finally:
            await pr.stop(timeout=5.0)
        assert executor.closed is True

    async def test_from_executor_failure_status_mapped(self) -> None:
        executor = _StubExecutor(status=TaskStatus.FAILED, summary="boom")
        pr = PersistentRuntimeV2.from_executor(executor, agent_id="worker")
        await pr.start()
        try:
            tid = await pr.submit_task("fail task", task_id="t-fail")
            result = await pr.wait_task(tid, timeout=5.0)
            assert result.status is RuntimeStatus.FAILED
        finally:
            await pr.stop(timeout=5.0)

    async def test_from_executor_with_execution_service(self, bus: MessageBus) -> None:
        """完整新路径：from_executor 持有 TeammateExecutionService。"""
        inner = _StubExecutor(status=TaskStatus.SUCCEEDED, summary="via service")
        spec = _make_spec()
        definition = TeammateDefinition.from_spec(spec)
        message_adapter = TeammateMessageAdapter(bus, name="worker")
        service = TeammateExecutionService(
            inner,
            definition=definition,
            message_adapter=message_adapter,
        )
        pr = PersistentRuntimeV2.from_executor(
            service,
            agent_id="worker",
            message_adapter=message_adapter,
        )
        await pr.start()
        try:
            tid = await pr.submit_task("do via service", task_id="t-svc")
            result = await pr.wait_task(tid, timeout=5.0)
            assert result.status is RuntimeStatus.COMPLETED
            assert result.summary == "via service"
            assert len(inner.executed) == 1
        finally:
            await pr.stop(timeout=5.0)

    async def test_from_executor_wait_unknown_task(self) -> None:
        executor = _StubExecutor()
        pr = PersistentRuntimeV2.from_executor(executor, agent_id="worker")
        await pr.start()
        try:
            result = await pr.wait_task("nonexistent", timeout=1.0)
            assert result.status is RuntimeStatus.FAILED
        finally:
            await pr.stop(timeout=5.0)
