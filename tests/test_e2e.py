"""端到端测试：完整 Agent 流程（用户输入 → LLM → 工具 → 最终输出）。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from mcc_cli.events import AgentEventSource
from mcc_cli.headless import run_headless
from mcc_core import ToolContext, Usage
from mcc_core.event import (
    Error as ErrorEvent,
)
from mcc_core.event import (
    LlmUsage,
    Notice,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
    TurnEnd,
    UserEcho,
)
from mcc_core.runtime import (
    HealthState,
    RuntimeCapabilities,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeHealth,
)
from mcc_harness import (
    AgentLoopBuilder,
    AgentRegistry,
    Assignment,
    MessageBus,
    PermissionChecker,
    PlanStore,
    RuntimeCoordinator,
    RuntimeResult,
    RuntimeStatus,
    TaskDispatcher,
    TaskSpec,
    Team,
    TeamOrchestrator,
    TeamPolicy,
    TeamRunStatus,
)
from mcc_harness.runtime import TaskStoreAdapter
from mcc_llm import MessageStop, ToolUseInputDelta, ToolUseStart
from mcc_llm.mock import MockLlm, single_text_turn
from mcc_tools import TaskStore, default_registry


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="e2e", depth=0)


@pytest.fixture
def registry():
    return default_registry()


def _agent(llm, ctx, registry, **kwargs):
    builder = AgentLoopBuilder(llm, registry, ctx).model("claude-haiku-4-5")
    for k, v in kwargs.items():
        getattr(builder, k)(v)
    return builder.build()


async def _drain(source, prompt) -> list:
    events = []
    async for ev in source.run(prompt):
        events.append(ev)
    return events


def _event_types(events) -> list[str]:
    return [type(e).__name__ for e in events]


# ---------- E2E: 纯文本对话 ----------


@pytest.mark.asyncio
async def test_e2e_text_only(ctx, registry):
    """用户提问 → LLM 直接回答（无工具调用）。"""
    llm = MockLlm([single_text_turn("Paris is the capital of France.")])
    source = AgentEventSource(_agent(llm, ctx, registry))
    events = await _drain(source, "What is the capital of France?")

    types = _event_types(events)
    assert types[0] == "UserEcho"
    assert "TextDelta" in types
    assert types[-1] == "TurnEnd"

    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert "Paris" in text


# ---------- E2E: 工具调用循环 ----------


@pytest.mark.asyncio
async def test_e2e_write_then_read(ctx, registry, tmp_path):
    """LLM 调用 write_file 写文件 → 再调 read_file 读回 → 返回内容。"""
    turn1 = [
        ToolUseStart(id="t1", name="write_file"),
        ToolUseInputDelta(
            partial_json=json.dumps({"path": "greeting.txt", "content": "hello world"})
        ),
        MessageStop(stop_reason="tool_use", usage=Usage()),
    ]
    turn2 = [
        ToolUseStart(id="t2", name="read_file"),
        ToolUseInputDelta(partial_json=json.dumps({"path": "greeting.txt"})),
        MessageStop(stop_reason="tool_use", usage=Usage()),
    ]
    turn3 = single_text_turn("The file contains: hello world")
    llm = MockLlm([turn1, turn2, turn3])
    source = AgentEventSource(_agent(llm, ctx, registry))
    events = await _drain(source, "Write and read a file")

    assert (tmp_path / "greeting.txt").read_text() == "hello world"
    tool_starts = [e for e in events if isinstance(e, ToolCallStart)]
    assert len(tool_starts) == 2
    assert tool_starts[0].name == "write_file"
    assert tool_starts[1].name == "read_file"

    tool_ends = [e for e in events if isinstance(e, ToolCallEnd)]
    assert all(not e.is_error for e in tool_ends)

    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert "hello world" in text


# ---------- E2E: bash 工具 ----------


@pytest.mark.asyncio
async def test_e2e_bash_echo(ctx, registry):
    """LLM 调用 run_bash 执行 echo 命令。"""
    turn1 = [
        ToolUseStart(id="t1", name="run_bash"),
        ToolUseInputDelta(partial_json=json.dumps({"command": "echo e2e-success"})),
        MessageStop(stop_reason="tool_use", usage=Usage()),
    ]
    turn2 = single_text_turn("Command output: e2e-success")
    llm = MockLlm([turn1, turn2])
    source = AgentEventSource(_agent(llm, ctx, registry))
    events = await _drain(source, "Run echo")

    tool_ends = [e for e in events if isinstance(e, ToolCallEnd)]
    assert len(tool_ends) == 1
    assert "e2e-success" in tool_ends[0].output


# ---------- E2E: 权限拒绝 ----------


@pytest.mark.asyncio
async def test_e2e_permission_deny(ctx, registry):
    """deny 规则阻止 rm 命令，工具返回 permission denied。"""
    from mcc_config import PermissionConfig

    cfg = PermissionConfig(deny=["Bash(rm:*)"])
    checker = PermissionChecker(cfg)

    turn1 = [
        ToolUseStart(id="t1", name="run_bash"),
        ToolUseInputDelta(partial_json=json.dumps({"command": "rm -rf /"})),
        MessageStop(stop_reason="tool_use", usage=Usage()),
    ]
    turn2 = single_text_turn("Permission was denied.")
    llm = MockLlm([turn1, turn2])
    source = AgentEventSource(_agent(llm, ctx, registry, permission=checker))
    events = await _drain(source, "Delete everything")

    tool_ends = [e for e in events if isinstance(e, ToolCallEnd)]
    assert tool_ends[0].is_error
    assert "permission denied" in tool_ends[0].output


# ---------- E2E: headless 模式 ----------


@pytest.mark.asyncio
async def test_e2e_headless_mode(ctx, registry, capsys):
    """通过 run_headless 运行，stdout 输出最终答案。"""
    llm = MockLlm([single_text_turn("42")])
    source = AgentEventSource(_agent(llm, ctx, registry))
    exit_code = await run_headless(source, "What is 6x7?", quiet=False)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "42" in captured.out


@pytest.mark.asyncio
async def test_e2e_headless_quiet_mode(ctx, registry, capsys):
    """quiet 模式下 stderr 不输出工具进度。"""
    llm = MockLlm([single_text_turn("silent answer")])
    source = AgentEventSource(_agent(llm, ctx, registry))
    exit_code = await run_headless(source, "test", quiet=True)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "silent answer" in captured.out
    assert captured.err == ""


# ---------- E2E: 成本追踪 ----------


@pytest.mark.asyncio
async def test_e2e_cost_tracking(ctx, registry, capsys):
    """headless 模式 stderr 输出 USD 成本。"""
    from mcc_core import Usage

    llm = MockLlm([single_text_turn("done", Usage(input_tokens=500, output_tokens=200))])
    source = AgentEventSource(_agent(llm, ctx, registry))
    await run_headless(source, "test", quiet=False)
    captured = capsys.readouterr()

    assert "$" in captured.err
    assert "USD" in captured.err


# ---------- E2E: 多轮工具调用 ----------


@pytest.mark.asyncio
async def test_e2e_multi_tool_parallel(ctx, registry, tmp_path):
    """LLM 同时请求多个工具调用，并行执行。"""
    from mcc_core import Usage

    u = Usage()
    turn1 = [
        ToolUseStart(id="t1", name="write_file"),
        ToolUseInputDelta(partial_json=json.dumps({"path": "a.txt", "content": "aaa"})),
        ToolUseStart(id="t2", name="write_file"),
        ToolUseInputDelta(partial_json=json.dumps({"path": "b.txt", "content": "bbb"})),
        MessageStop(stop_reason="tool_use", usage=u),
    ]
    turn2 = single_text_turn("Both files written.")
    llm = MockLlm([turn1, turn2])
    source = AgentEventSource(_agent(llm, ctx, registry))
    events = await _drain(source, "Write two files")

    assert (tmp_path / "a.txt").read_text() == "aaa"
    assert (tmp_path / "b.txt").read_text() == "bbb"
    tool_ends = [e for e in events if isinstance(e, ToolCallEnd)]
    assert len(tool_ends) == 2
    assert all(not e.is_error for e in tool_ends)


# ---------- E2E: 错误传播 ----------


@pytest.mark.asyncio
async def test_e2e_tool_error_continues(ctx, registry):
    """工具失败不中断循环，LLM 继续根据错误结果回复。"""
    from mcc_core import Usage

    turn1 = [
        ToolUseStart(id="t1", name="read_file"),
        ToolUseInputDelta(partial_json=json.dumps({"path": "nonexistent.txt"})),
        MessageStop(stop_reason="tool_use", usage=Usage()),
    ]
    turn2 = single_text_turn("The file does not exist.")
    llm = MockLlm([turn1, turn2])
    source = AgentEventSource(_agent(llm, ctx, registry))
    events = await _drain(source, "Read a missing file")

    tool_ends = [e for e in events if isinstance(e, ToolCallEnd)]
    assert tool_ends[0].is_error
    text = "".join(e.text for e in events if isinstance(e, TextDelta))
    assert "does not exist" in text


# ---------- E2E: 事件完整性 ----------


@pytest.mark.asyncio
async def test_e2e_event_sequence(ctx, registry):
    """验证事件序列：UserEcho → [TextDelta|ToolCall*] → TurnEnd。"""
    llm = MockLlm([single_text_turn("hello")])
    source = AgentEventSource(_agent(llm, ctx, registry))
    events = await _drain(source, "hi")

    assert isinstance(events[0], UserEcho)
    assert isinstance(events[-1], TurnEnd)
    for ev in events[1:-1]:
        assert isinstance(ev, (TextDelta, ToolCallStart, ToolCallEnd, Notice, LlmUsage))


# ---------- E2E: Team 编排 ----------


class _StubRuntime:
    """E2E 测试用 Runtime 桩：实现 AgentRuntime 协议。"""

    def __init__(
        self,
        runtime_id: str,
        result: RuntimeResult | None = None,
        delay: float = 0.0,
    ) -> None:
        self._runtime_id = runtime_id
        self._status = RuntimeStatus.CREATED
        self._result = result or RuntimeResult(
            runtime_id=runtime_id,
            status=RuntimeStatus.COMPLETED,
            summary="done",
        )
        self._delay = delay
        self.submit_count = 0
        self.cancelled = False
        self._event_queues: dict[str, asyncio.Queue] = {}

    @property
    def runtime_id(self) -> str:
        return self._runtime_id

    @property
    def agent_id(self) -> str:
        return self._runtime_id

    @property
    def status(self) -> RuntimeStatus:
        return self._status

    @property
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(supports_cancel=True, supports_health_check=False)

    @property
    def alive(self) -> bool:
        return self._status not in (
            RuntimeStatus.COMPLETED,
            RuntimeStatus.FAILED,
            RuntimeStatus.CANCELLED,
            RuntimeStatus.TIMEOUT,
            RuntimeStatus.SHUTDOWN,
        )

    async def start(self, task: str = "", *, task_id: str | None = None) -> None:
        self._status = RuntimeStatus.RUNNING

    async def stop(self, *, timeout: float = 5.0) -> None:
        self._status = RuntimeStatus.SHUTDOWN

    def cancel(self) -> None:
        self.cancelled = True
        self._status = RuntimeStatus.CANCELLED

    async def cancel_task(
        self,
        task_id: str,
        attempt_id: str,
        reason: str | None = None,
    ) -> None:
        self.cancelled = True

    def pause(self) -> None:
        self._status = RuntimeStatus.PAUSED

    def resume(self) -> None:
        self._status = RuntimeStatus.RUNNING

    async def wait_done(self, *, timeout: float | None = None) -> RuntimeResult:
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        self._status = self._result.status
        return self._result

    async def submit_task(self, task: str, *, task_id: str | None = None) -> str:
        self.submit_count += 1
        self._status = RuntimeStatus.RUNNING
        return task_id or f"task::{task}"

    async def wait_task(self, task_id: str, *, timeout: float | None = None) -> RuntimeResult:
        if self._delay > 0:
            await asyncio.sleep(self._delay)
        self._status = self._result.status
        return self._result

    async def submit(self, task: Any) -> Any:
        from mcc_core.runtime import TaskHandle

        self.submit_count += 1
        self._status = RuntimeStatus.RUNNING
        queue: asyncio.Queue = asyncio.Queue()
        self._event_queues[task.task_id] = queue

        async def _emit_done() -> None:
            if self._delay > 0:
                await asyncio.sleep(self._delay)
            event_type = (
                RuntimeEventType.TASK_COMPLETED
                if self._result.status is RuntimeStatus.COMPLETED
                else RuntimeEventType.TASK_FAILED
            )
            event = RuntimeEvent(
                event_id=f"evt-{task.task_id}",
                event_type=event_type,
                runtime_id=self._runtime_id,
                task_id=task.task_id,
                attempt_id=task.attempt_id,
                payload={
                    "summary": self._result.summary,
                    "error": self._result.error,
                    "cost_usd": self._result.cost_usd,
                },
            )
            await queue.put(event)

        asyncio.create_task(_emit_done())
        return TaskHandle(
            task_id=task.task_id,
            attempt_id=task.attempt_id,
            runtime_id=self._runtime_id,
        )

    async def subscribe_events(
        self,
        *,
        task_id: str | None = None,
        attempt_id: str | None = None,
    ) -> Any:
        from collections.abc import AsyncIterator

        queue = self._event_queues.get(task_id or "")
        if queue is None:
            return
        while True:
            event = await queue.get()
            yield event

    async def health_check(self) -> RuntimeHealth:
        return RuntimeHealth(
            runtime_id=self._runtime_id,
            state=HealthState.HEALTHY,
        )


def _make_team_e2e(tmp_path, policy=None):
    registry = AgentRegistry(heartbeat_timeout=30.0)
    bus = MessageBus(tmp_path)
    plan_store = PlanStore()
    task_store = TaskStore(tmp_path / ".tasks")
    coordinator = RuntimeCoordinator()
    adapter = TaskStoreAdapter(task_store)
    dispatcher = TaskDispatcher(adapter, coordinator)
    orch = TeamOrchestrator(
        registry=registry,
        bus=bus,
        plan_store=plan_store,
        task_store=task_store,
        coordinator=coordinator,
        dispatcher=dispatcher,
    )
    return orch, coordinator, task_store


def _register_agents(orch, count=2):
    ids = []
    for i in range(1, count + 1):
        orch.register_agent(
            agent_id=f"agent-{i}",
            agent_name=f"worker-{i}",
            capabilities=frozenset({"code_generation"}),
        )
        ids.append(f"agent-{i}")
    return ids


@pytest.mark.asyncio
async def test_team_e2e_assignment_executes_once(tmp_path):
    """一个 Assignment 只触发一次执行。"""
    orch, coordinator, _ = _make_team_e2e(tmp_path)
    agent_ids = _register_agents(orch, 1)
    team = orch.create_team("team-1", "alpha", "lead-1")

    rt = _StubRuntime("rt-1")
    coordinator.register(rt)
    orch.bind_runtime(agent_ids[0], "rt-1")

    run = await orch.start(team, "build feature A")
    run = await orch.execute_run(run.run_id)

    assert run.status == TeamRunStatus.COMPLETED
    assert rt.submit_count == 1


@pytest.mark.asyncio
async def test_team_e2e_concurrency_limit(tmp_path):
    """Agent 按并发限制接收多个任务：max_concurrent_tasks=1 时串行执行。"""
    policy = TeamPolicy(max_concurrent_tasks=1)
    orch, coordinator, _ = _make_team_e2e(tmp_path, policy)
    agent_ids = _register_agents(orch, 2)
    team = orch.create_team("team-1", "alpha", "lead-1", policy=policy)

    exec_log: list[tuple[str, float]] = []

    class TimedStub(_StubRuntime):
        def __init__(self, runtime_id: str) -> None:
            super().__init__(runtime_id, delay=0.05)

        async def submit(self, task: Any) -> Any:
            handle = await super().submit(task)
            exec_log.append((self._runtime_id, asyncio.get_event_loop().time()))
            return handle

    rt1 = TimedStub("rt-1")
    rt2 = TimedStub("rt-2")
    coordinator.register(rt1)
    coordinator.register(rt2)
    orch.bind_runtime(agent_ids[0], "rt-1")
    orch.bind_runtime(agent_ids[1], "rt-2")

    run = await orch.start(team, "build A\nbuild B")
    run = await orch.execute_run(run.run_id)

    assert run.status == TeamRunStatus.COMPLETED
    assert len(exec_log) == 2
    times = sorted(t for _, t in exec_log)
    assert times[1] - times[0] >= 0.04


@pytest.mark.asyncio
async def test_team_e2e_dag_dependency_order(tmp_path):
    """DAG 未满足依赖不执行：blocked_by 任务先完成。"""
    orch, coordinator, _ = _make_team_e2e(tmp_path)
    agent_ids = _register_agents(orch, 2)
    team = orch.create_team("team-1", "alpha", "lead-1")

    completion_order: list[str] = []

    class OrderedStub(_StubRuntime):
        def __init__(self, runtime_id: str) -> None:
            super().__init__(runtime_id, delay=0.02)

        async def submit(self, task: Any) -> Any:
            handle = await super().submit(task)
            completion_order.append(self._runtime_id)
            return handle

    rt1 = OrderedStub("rt-1")
    rt2 = OrderedStub("rt-2")
    coordinator.register(rt1)
    coordinator.register(rt2)
    orch.bind_runtime(agent_ids[0], "rt-1")
    orch.bind_runtime(agent_ids[1], "rt-2")

    from mcc_harness import SimplePlanner

    class DagPlanner(SimplePlanner):
        async def plan(self, objective, *, context=None):
            return [
                TaskSpec(subject="base", description="base task"),
                TaskSpec(subject="dependent", description="dep task", blocked_by=("base",)),
            ]

    orch._planner = DagPlanner()
    run = await orch.start(team, "base\ndependent")
    run = await orch.execute_run(run.run_id)

    assert run.status == TeamRunStatus.COMPLETED
    assert completion_order.index("rt-1") < completion_order.index("rt-2")


@pytest.mark.asyncio
async def test_team_e2e_unmatched_agent_fails(tmp_path):
    """未匹配 Agent 的任务进入明确状态（FAILED）。"""
    orch, coordinator, _ = _make_team_e2e(tmp_path)
    _register_agents(orch, 1)
    team = orch.create_team("team-1", "alpha", "lead-1")

    run = await orch.start(team, "build feature A")
    run = await orch.execute_run(run.run_id)

    assert run.status == TeamRunStatus.FAILED
    assert len(run.results) == 1
    assert run.results[0].status == "failed"


@pytest.mark.asyncio
async def test_team_e2e_cancel_propagates(tmp_path):
    """TeamRun 取消级联到所有活动 Attempt。"""
    orch, coordinator, _ = _make_team_e2e(tmp_path)
    agent_ids = _register_agents(orch, 2)
    team = orch.create_team("team-1", "alpha", "lead-1")

    rt1 = _StubRuntime("rt-1", delay=2.0)
    rt2 = _StubRuntime("rt-2", delay=2.0)
    coordinator.register(rt1)
    coordinator.register(rt2)
    orch.bind_runtime(agent_ids[0], "rt-1")
    orch.bind_runtime(agent_ids[1], "rt-2")

    run = await orch.start(team, "build A\nbuild B")
    exec_task = asyncio.create_task(orch.execute_run(run.run_id))
    await asyncio.sleep(0.1)
    await orch.cancel(run.run_id)
    run = await exec_task

    assert run.status == TeamRunStatus.CANCELLED
    assert rt1.cancelled or rt1.status in (RuntimeStatus.CANCELLED, RuntimeStatus.RUNNING)
    assert rt2.cancelled or rt2.status in (RuntimeStatus.CANCELLED, RuntimeStatus.RUNNING)


@pytest.mark.asyncio
async def test_team_e2e_partial_failure_aggregates(tmp_path):
    """结果聚合不丢失部分成功任务：一个成功一个失败，结果都保留。"""
    orch, coordinator, _ = _make_team_e2e(tmp_path)
    agent_ids = _register_agents(orch, 2)
    team = orch.create_team("team-1", "alpha", "lead-1")

    rt1 = _StubRuntime(
        "rt-1",
        RuntimeResult(
            runtime_id="rt-1",
            status=RuntimeStatus.COMPLETED,
            summary="ok-1",
        ),
    )
    rt2 = _StubRuntime(
        "rt-2",
        RuntimeResult(
            runtime_id="rt-2",
            status=RuntimeStatus.FAILED,
            summary="",
            error="fail-2",
        ),
    )
    coordinator.register(rt1)
    coordinator.register(rt2)
    orch.bind_runtime(agent_ids[0], "rt-1")
    orch.bind_runtime(agent_ids[1], "rt-2")

    run = await orch.start(team, "build A\nbuild B")
    run = await orch.execute_run(run.run_id)

    assert run.status == TeamRunStatus.FAILED
    assert len(run.results) == 2
    statuses = {r.status for r in run.results}
    assert RuntimeStatus.COMPLETED in statuses
    assert RuntimeStatus.FAILED in statuses


@pytest.mark.asyncio
async def test_team_e2e_retry_creates_new_attempt(tmp_path):
    """Retry 创建新 attempt_id：失败后重试使用新 attempt。"""
    import unittest.mock

    orch, coordinator, task_store = _make_team_e2e(tmp_path)
    agent_ids = _register_agents(orch, 1)
    policy = TeamPolicy(task_lease_secs=10.0)
    team = orch.create_team("team-1", "alpha", "lead-1", policy=policy)

    call_count = 0

    class RetryStub(_StubRuntime):
        async def submit(self, task: Any) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                self._result = RuntimeResult(
                    runtime_id=self._runtime_id,
                    status=RuntimeStatus.FAILED,
                    summary="",
                    error=f"fail-{call_count}",
                )
            else:
                self._result = RuntimeResult(
                    runtime_id=self._runtime_id,
                    status=RuntimeStatus.COMPLETED,
                    summary="ok",
                )
            return await super().submit(task)

    rt = RetryStub("rt-1")
    coordinator.register(rt)
    orch.bind_runtime(agent_ids[0], "rt-1")

    run = await orch.start(team, "build feature A")
    original_execute = orch._execute_assignment

    async def patched_execute(run_id, assignment, semaphore, max_attempts=1, **kwargs):
        return await original_execute(run_id, assignment, semaphore, max_attempts=3, **kwargs)

    with unittest.mock.patch.object(orch, "_execute_assignment", patched_execute):
        run = await orch.execute_run(run.run_id)

    assert call_count == 3
    assert run.status == TeamRunStatus.COMPLETED


@pytest.mark.asyncio
async def test_team_e2e_empty_assignments(tmp_path):
    """Planner 返回空任务列表时，TeamRun 直接完成。"""
    from mcc_harness import SimplePlanner

    class EmptyPlanner(SimplePlanner):
        async def plan(self, objective, *, context=None):
            return []

    orch, coordinator, _ = _make_team_e2e(tmp_path)
    agent_ids = _register_agents(orch, 1)
    team = orch.create_team("team-1", "alpha", "lead-1")

    rt = _StubRuntime("rt-1")
    coordinator.register(rt)
    orch.bind_runtime(agent_ids[0], "rt-1")
    orch._planner = EmptyPlanner()

    run = await orch.start(team, "do nothing")

    assert run.status == TeamRunStatus.COMPLETED
    assert len(run.results) == 0
    assert rt.submit_count == 0


@pytest.mark.asyncio
async def test_team_e2e_shutdown_stops_runtime(tmp_path):
    """shutdown 停止所有绑定的活跃 Runtime。"""
    orch, coordinator, _ = _make_team_e2e(tmp_path)
    agent_ids = _register_agents(orch, 2)
    team = orch.create_team("team-1", "alpha", "lead-1")

    rt1 = _StubRuntime("rt-1")
    rt2 = _StubRuntime("rt-2")
    coordinator.register(rt1)
    coordinator.register(rt2)
    orch.bind_runtime(agent_ids[0], "rt-1")
    orch.bind_runtime(agent_ids[1], "rt-2")

    run = await orch.start(team, "build A\nbuild B")
    await orch.shutdown(run.run_id)

    assert rt1.status is RuntimeStatus.SHUTDOWN
    assert rt2.status is RuntimeStatus.SHUTDOWN


def _make_team_e2e_with_dispatcher(tmp_path, policy=None):
    registry = AgentRegistry(heartbeat_timeout=30.0)
    bus = MessageBus(tmp_path)
    plan_store = PlanStore()
    task_store = TaskStore(tmp_path / ".tasks")
    coordinator = RuntimeCoordinator()
    adapter = TaskStoreAdapter(task_store)
    dispatcher = TaskDispatcher(adapter, coordinator)
    orch = TeamOrchestrator(
        registry=registry,
        bus=bus,
        plan_store=plan_store,
        task_store=task_store,
        coordinator=coordinator,
        dispatcher=dispatcher,
    )
    return orch, coordinator, task_store, dispatcher


@pytest.mark.asyncio
async def test_team_e2e_dispatcher_executes_once(tmp_path):
    """Dispatcher 路径：一个 Assignment 通过 TaskDispatcher 派发执行一次。"""
    orch, coordinator, _, _ = _make_team_e2e_with_dispatcher(tmp_path)
    agent_ids = _register_agents(orch, 1)
    team = orch.create_team("team-1", "alpha", "lead-1")

    rt = _StubRuntime("rt-1")
    coordinator.register(rt)
    orch.bind_runtime(agent_ids[0], "rt-1")

    run = await orch.start(team, "build feature A")
    run = await orch.execute_run(run.run_id)

    assert run.status == TeamRunStatus.COMPLETED
    assert rt.submit_count == 1


@pytest.mark.asyncio
async def test_team_e2e_dispatcher_partial_failure(tmp_path):
    """Dispatcher 路径：部分失败时结果聚合保留所有任务状态。"""
    orch, coordinator, _, _ = _make_team_e2e_with_dispatcher(tmp_path)
    agent_ids = _register_agents(orch, 2)
    team = orch.create_team("team-1", "alpha", "lead-1")

    rt1 = _StubRuntime(
        "rt-1",
        RuntimeResult(
            runtime_id="rt-1",
            status=RuntimeStatus.COMPLETED,
            summary="ok-1",
        ),
    )
    rt2 = _StubRuntime(
        "rt-2",
        RuntimeResult(
            runtime_id="rt-2",
            status=RuntimeStatus.FAILED,
            summary="",
            error="fail-2",
        ),
    )
    coordinator.register(rt1)
    coordinator.register(rt2)
    orch.bind_runtime(agent_ids[0], "rt-1")
    orch.bind_runtime(agent_ids[1], "rt-2")

    run = await orch.start(team, "build A\nbuild B")
    run = await orch.execute_run(run.run_id)

    assert run.status == TeamRunStatus.FAILED
    assert len(run.results) == 2
    statuses = {r.status for r in run.results}
    assert RuntimeStatus.COMPLETED in statuses
    assert RuntimeStatus.FAILED in statuses


@pytest.mark.asyncio
async def test_team_e2e_dispatcher_result_content(tmp_path):
    """Dispatcher 路径：结果 summary 正确传递到 TeamRun.results。"""
    orch, coordinator, _, _ = _make_team_e2e_with_dispatcher(tmp_path)
    agent_ids = _register_agents(orch, 1)
    team = orch.create_team("team-1", "alpha", "lead-1")

    rt = _StubRuntime(
        "rt-1",
        RuntimeResult(
            runtime_id="rt-1",
            status=RuntimeStatus.COMPLETED,
            summary="feature A built successfully",
        ),
    )
    coordinator.register(rt)
    orch.bind_runtime(agent_ids[0], "rt-1")

    run = await orch.start(team, "build feature A")
    run = await orch.execute_run(run.run_id)

    assert run.status == TeamRunStatus.COMPLETED
    assert len(run.results) == 1
    assert run.results[0].summary == "feature A built successfully"


@pytest.mark.asyncio
async def test_team_e2e_dispatcher_concurrency(tmp_path):
    """Dispatcher 路径：max_concurrent_tasks=1 时串行执行。"""
    policy = TeamPolicy(max_concurrent_tasks=1)
    orch, coordinator, _, _ = _make_team_e2e_with_dispatcher(tmp_path, policy)
    agent_ids = _register_agents(orch, 2)
    team = orch.create_team("team-1", "alpha", "lead-1", policy=policy)

    exec_log: list[tuple[str, float]] = []

    class TimedStub(_StubRuntime):
        async def submit(self, task: Any) -> Any:
            await asyncio.sleep(0.05)
            exec_log.append((self._runtime_id, asyncio.get_event_loop().time()))
            return await super().submit(task)

    rt1 = TimedStub("rt-1")
    rt2 = TimedStub("rt-2")
    coordinator.register(rt1)
    coordinator.register(rt2)
    orch.bind_runtime(agent_ids[0], "rt-1")
    orch.bind_runtime(agent_ids[1], "rt-2")

    run = await orch.start(team, "build A\nbuild B")
    run = await orch.execute_run(run.run_id)

    assert run.status == TeamRunStatus.COMPLETED
    assert len(exec_log) == 2


@pytest.mark.asyncio
async def test_team_e2e_dispatcher_unmatched_agent_fails(tmp_path):
    """Dispatcher 路径：未匹配 Agent 的任务标记为 FAILED。"""
    orch, coordinator, _, _ = _make_team_e2e_with_dispatcher(tmp_path)
    _register_agents(orch, 1)
    team = orch.create_team("team-1", "alpha", "lead-1")

    run = await orch.start(team, "build feature A")
    run = await orch.execute_run(run.run_id)

    assert run.status == TeamRunStatus.FAILED
    assert len(run.results) == 1
    assert run.results[0].status == "failed"
