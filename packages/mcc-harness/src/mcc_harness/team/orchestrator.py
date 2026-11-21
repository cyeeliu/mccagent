"""TeamOrchestrator 编排层：Team 能力的统一入口。

将任务规划、调度、Agent 管理和结果汇总集中到清晰的编排边界，
不直接实现工具执行细节，通过 Protocol 接口与 Planner、TaskScheduler、
ResultCollector 解耦，通过 AgentRuntime 控制底层 Agent 生命周期。

编排组件复用已有抽象：
- AgentRegistry：Agent 注册与发现
- MessageBus：Agent 间消息总线
- PlanStore：plan 审批协议
- TaskStore：任务持久化与状态机
- AgentBudget：统一预算管理
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from mcc_core.runtime import RuntimeEventType, TaskHandle, TaskStatus, TaskStoreProtocol
from mcc_core.runtime.errors import TaskTimeoutError, TaskWaitTimeoutError
from mcc_core.runtime.task_record import AttemptStatus

from mcc_harness.budget import AgentBudget, BudgetPolicy
from mcc_harness.context import AgentContext
from mcc_harness.mailbox import MessageBus
from mcc_harness.plan import PlanStore
from mcc_harness.registry import AgentDescriptor, AgentRegistry, RegistrationStatus
from mcc_harness.runtime import (
    RuntimeCoordinator,
    RuntimeMode,
    WaitTimeoutError,
)
from mcc_harness.runtime.dispatcher import TaskDispatcher

logger = logging.getLogger(__name__)


class TeamRunStatus(StrEnum):
    """TeamRun 生命周期状态。"""

    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SHUTDOWN = "shutdown"


class DependencyPolicy(StrEnum):
    """DAG 依赖策略：控制依赖任务在何种条件下才执行。

    - ALL_SUCCESS：所有依赖必须成功才执行（默认，Coding Agent DAG）。
    - ALL_TERMINAL：所有依赖到达任意终态即执行（容错编排）。
    - ANY_SUCCESS：至少一个依赖成功即执行（可选依赖）。
    """

    ALL_SUCCESS = "all_success"
    ALL_TERMINAL = "all_terminal"
    ANY_SUCCESS = "any_success"


_TERMINAL_RUN_STATUSES = frozenset(
    {
        TeamRunStatus.COMPLETED,
        TeamRunStatus.FAILED,
        TeamRunStatus.CANCELLED,
        TeamRunStatus.SHUTDOWN,
    }
)

_TASK_STATUS_TO_RESULT: dict[TaskStatus, str] = {
    TaskStatus.SUCCEEDED: "completed",
    TaskStatus.FAILED: "failed",
    TaskStatus.CANCELLED: "cancelled",
    TaskStatus.TIMED_OUT: "timeout",
}


class OrchestratorError(RuntimeError):
    """编排层基础错误。"""


class TeamNotFoundError(OrchestratorError):
    """团队不存在。"""


class RunNotFoundError(OrchestratorError):
    """TeamRun 不存在。"""


class RunStateError(OrchestratorError):
    """TeamRun 状态冲突：操作在当前状态下不合法。"""


@dataclass(frozen=True)
class TeamPolicy:
    """团队策略：约束一次 TeamRun 的行为。

    所有字段有默认值，新增配置必须有默认值和校验逻辑。
    """

    max_agents: int = 4
    max_concurrent_tasks: int = 4
    heartbeat_interval_secs: float = 10.0
    task_lease_secs: float = 300.0
    auto_cancel_on_failure: bool = False
    require_plan: bool = False
    max_attempts: int = 1
    budget_policy: BudgetPolicy | None = None
    dependency_policy: DependencyPolicy = DependencyPolicy.ALL_SUCCESS

    def __post_init__(self) -> None:
        if self.max_agents < 1:
            raise OrchestratorError("max_agents must be >= 1")
        if self.max_concurrent_tasks < 1:
            raise OrchestratorError("max_concurrent_tasks must be >= 1")
        if self.heartbeat_interval_secs <= 0:
            raise OrchestratorError("heartbeat_interval_secs must be > 0")
        if self.task_lease_secs <= 0:
            raise OrchestratorError("task_lease_secs must be > 0")
        if self.max_attempts < 1:
            raise OrchestratorError("max_attempts must be >= 1")


@dataclass(frozen=True)
class Team:
    """团队定义：Lead + 策略 + 标签。

    team_id 全局唯一，lead_agent_id 指向 Registry 中注册的 Lead Agent。
    """

    team_id: str
    name: str
    lead_agent_id: str
    policy: TeamPolicy = field(default_factory=TeamPolicy)
    tags: frozenset[str] = field(default_factory=frozenset)
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class TaskSpec:
    """任务规格：Planner 输出，Scheduler 输入。

    subject 为任务标题，description 为详细描述，
    blocked_by 为依赖的任务 subject 列表（规划阶段用 subject 引用），
    required_capabilities 为执行该任务所需的 Agent 能力集合，
    priority 越大越优先。
    """

    subject: str
    description: str = ""
    blocked_by: tuple[str, ...] = ()
    required_capabilities: frozenset[str] = frozenset()
    priority: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Assignment:
    """任务分配：Scheduler 输出。

    将 task_id 分配给 agent_id 执行。
    """

    task_id: str
    agent_id: str
    task_spec: TaskSpec


@dataclass(frozen=True)
class SchedulableTask:
    """可调度任务：将 TaskStore 中的 task_id 与 TaskSpec 绑定。

    Scheduler 接收 SchedulableTask 列表而非裸 TaskSpec 列表，
    使 Assignment.task_id 直接来源于 SchedulableTask.task_id，
    消除数组索引关联带来的错位风险。
    """

    task_id: str
    spec: TaskSpec


@dataclass(frozen=True)
class AgentResult:
    """单个 Agent 执行结果。"""

    agent_id: str
    task_id: str
    status: str
    summary: str = ""
    error: str | None = None
    cost_usd: float = 0.0


@dataclass(frozen=True)
class TeamRunResult:
    """TeamRun 汇总结果：ResultCollector 输出。"""

    run_id: str
    status: TeamRunStatus
    results: tuple[AgentResult, ...] = ()
    completed_count: int = 0
    failed_count: int = 0
    total_count: int = 0
    error: str | None = None

    @property
    def success_rate(self) -> float:
        if self.total_count == 0:
            return 0.0
        return self.completed_count / self.total_count


@dataclass(frozen=True)
class TeamProgress:
    """TeamRun 进度快照。"""

    run_id: str
    status: TeamRunStatus
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    active_agents: int
    elapsed_secs: float

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_RUN_STATUSES

    @property
    def pending_tasks(self) -> int:
        return self.total_tasks - self.completed_tasks - self.failed_tasks


@dataclass
class TeamRun:
    """一次编排执行的状态。

    可变 dataclass：Orchestrator 在编排过程中更新状态字段。
    通过 get_run 返回快照时调用 snapshot() 生成不可变副本。
    """

    run_id: str
    team: Team
    objective: str
    status: TeamRunStatus = TeamRunStatus.CREATED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    agent_ids: set[str] = field(default_factory=set)
    task_ids: set[str] = field(default_factory=set)
    assignments: list[Assignment] = field(default_factory=list)
    results: list[AgentResult] = field(default_factory=list)
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> TeamRunSnapshot:
        """生成不可变快照。"""
        return TeamRunSnapshot(
            run_id=self.run_id,
            team_id=self.team.team_id,
            status=self.status,
            created_at=self.created_at,
            started_at=self.started_at,
            finished_at=self.finished_at,
            agent_ids=frozenset(self.agent_ids),
            task_ids=frozenset(self.task_ids),
            error=self.error,
        )

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_RUN_STATUSES


@dataclass(frozen=True)
class TeamRunSnapshot:
    """TeamRun 不可变快照：用于查询和序列化。"""

    run_id: str
    team_id: str
    status: TeamRunStatus
    created_at: float
    started_at: float | None
    finished_at: float | None
    agent_ids: frozenset[str]
    task_ids: frozenset[str]
    error: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in _TERMINAL_RUN_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "team_id": self.team_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "agent_ids": sorted(self.agent_ids),
            "task_ids": sorted(self.task_ids),
            "error": self.error,
        }


# -- Protocol 接口：给 P2-07/P2-08/P2-09 预留 --


@runtime_checkable
class Planner(Protocol):
    """任务规划器：将目标拆分为可执行任务规格。"""

    async def plan(
        self,
        objective: str,
        *,
        context: AgentContext | None = None,
    ) -> list[TaskSpec]: ...


@runtime_checkable
class TaskScheduler(Protocol):
    """任务调度器：将任务规格分配给可用 Agent。

    依赖 AgentRegistry 进行能力发现，返回 Assignment 列表。
    接收 SchedulableTask 列表，每个元素已绑定 task_id，
    Scheduler 须将 SchedulableTask.task_id 原样填入 Assignment.task_id。
    """

    async def schedule(
        self,
        tasks: list[SchedulableTask],
        registry: AgentRegistry,
        *,
        exclude_agents: frozenset[str] = frozenset(),
    ) -> list[Assignment]: ...


@runtime_checkable
class ResultCollector(Protocol):
    """结果收集器：汇总各 Agent 执行结果。"""

    async def collect(self, run_id: str) -> TeamRunResult: ...


# -- 默认实现 --


class SimplePlanner:
    """默认 Planner：将目标按换行拆分为任务规格。

    无 LLM 依赖，用于测试和简单场景。
    每行一个任务，空行跳过。
    """

    async def plan(
        self,
        objective: str,
        *,
        context: AgentContext | None = None,
    ) -> list[TaskSpec]:
        lines = [line.strip() for line in objective.strip().splitlines() if line.strip()]
        return [
            TaskSpec(
                subject=line,
                description=f"子任务：{line}",
                priority=len(lines) - i,
            )
            for i, line in enumerate(lines)
        ]


class CapabilityScheduler:
    """默认 TaskScheduler：按能力匹配分配任务。

    策略：对每个 SchedulableTask，在 Registry 中按 required_capabilities
    查找可用 Agent，取第一个分配。无能力要求时分配给任意可用 Agent。
    Assignment.task_id 直接取自 SchedulableTask.task_id。
    """

    async def schedule(
        self,
        tasks: list[SchedulableTask],
        registry: AgentRegistry,
        *,
        exclude_agents: frozenset[str] = frozenset(),
    ) -> list[Assignment]:
        assignments: list[Assignment] = []
        used: set[str] = set(exclude_agents)
        for schedulable in tasks:
            agent_id = self._find_agent(schedulable.spec, registry, frozenset(used))
            if agent_id is not None:
                used.add(agent_id)
                assignments.append(
                    Assignment(
                        task_id=schedulable.task_id,
                        agent_id=agent_id,
                        task_spec=schedulable.spec,
                    )
                )
        return assignments

    def _find_agent(
        self,
        spec: TaskSpec,
        registry: AgentRegistry,
        exclude_agents: frozenset[str],
    ) -> str | None:
        if spec.required_capabilities:
            candidates = registry.find_by_capabilities(spec.required_capabilities)
        else:
            candidates = registry.list(status=RegistrationStatus.AVAILABLE)
        for reg in candidates:
            if reg.agent_id not in exclude_agents:
                return reg.agent_id
        return None


class SimpleResultCollector:
    """默认 ResultCollector：从 TeamRun 中汇总结果。"""

    def __init__(self, orchestrator: TeamOrchestrator) -> None:
        self._orch = orchestrator

    async def collect(self, run_id: str) -> TeamRunResult:
        run = self._orch._runs.get(run_id)
        if run is None:
            raise RunNotFoundError(f"unknown run: {run_id}")
        results = await self._orch._aggregate_results(run)
        completed = sum(1 for r in results if r.status == "completed")
        failed = sum(
            1 for r in results if r.status in ("failed", "timeout", "cancelled", "budget_exceeded")
        )
        return TeamRunResult(
            run_id=run_id,
            status=run.status,
            results=results,
            completed_count=completed,
            failed_count=failed,
            total_count=len(run.task_ids),
            error=run.error,
        )


# -- TeamOrchestrator --


class TeamOrchestrator:
    """Team 编排层：统一入口，编排多组件完成一次 TeamRun。

    持有 Registry、MessageBus、PlanStore、TaskStore、Budget，
    通过 Planner/TaskScheduler/ResultCollector Protocol 解耦业务决策，
    不直接实现工具执行细节。

    生命周期：create_team → start → (pause/resume)* → cancel/shutdown → 终态。
    """

    def __init__(
        self,
        registry: AgentRegistry,
        bus: MessageBus,
        plan_store: PlanStore,
        dispatcher: TaskDispatcher,
        task_store: Any | None = None,
        budget: AgentBudget | None = None,
        planner: Planner | None = None,
        scheduler: TaskScheduler | None = None,
        coordinator: RuntimeCoordinator | None = None,
        async_task_store: TaskStoreProtocol | None = None,
    ) -> None:
        self._registry = registry
        self._bus = bus
        self._plan_store = plan_store
        self._task_store = task_store
        self._async_task_store = async_task_store
        self._budget = budget
        self._planner = planner or SimplePlanner()
        self._scheduler = scheduler or CapabilityScheduler()
        self._coordinator = coordinator
        self._dispatcher = dispatcher
        self._agent_runtime_map: dict[str, str] = {}
        self._teams: dict[str, Team] = {}
        self._runs: dict[str, TeamRun] = {}
        self._run_counter = 0
        self._active_handles: dict[str, dict[str, TaskHandle]] = {}

    @property
    def registry(self) -> AgentRegistry:
        return self._registry

    @property
    def bus(self) -> MessageBus:
        return self._bus

    @property
    def plan_store(self) -> PlanStore:
        return self._plan_store

    @property
    def task_store(self) -> Any | None:
        return self._task_store

    @property
    def async_task_store(self) -> TaskStoreProtocol | None:
        return self._async_task_store

    @property
    def dispatcher(self) -> TaskDispatcher:
        return self._dispatcher

    @property
    def budget(self) -> AgentBudget | None:
        return self._budget

    @property
    def coordinator(self) -> RuntimeCoordinator | None:
        return self._coordinator

    @property
    def planner(self) -> Planner:
        return self._planner

    def bind_runtime(self, agent_id: str, runtime_id: str) -> None:
        """绑定 agent_id 到 runtime_id，用于执行闭环查找 Runtime。"""
        self._agent_runtime_map[agent_id] = runtime_id

    def unbind_runtime(self, agent_id: str) -> None:
        """解除 agent_id 的 runtime 绑定。"""
        self._agent_runtime_map.pop(agent_id, None)

    def create_team(
        self,
        team_id: str,
        name: str,
        lead_agent_id: str,
        *,
        policy: TeamPolicy | None = None,
        tags: frozenset[str] | None = None,
    ) -> Team:
        """创建团队定义。幂等：重复创建相同 team_id 返回已有团队。"""
        if team_id in self._teams:
            return self._teams[team_id]
        team = Team(
            team_id=team_id,
            name=name,
            lead_agent_id=lead_agent_id,
            policy=policy or TeamPolicy(),
            tags=tags or frozenset(),
        )
        self._teams[team_id] = team
        return team

    def get_team(self, team_id: str) -> Team:
        team = self._teams.get(team_id)
        if team is None:
            raise TeamNotFoundError(f"unknown team: {team_id}")
        return team

    async def start(
        self,
        team: Team,
        objective: str,
        *,
        run_id: str | None = None,
    ) -> TeamRun:
        """启动一次 TeamRun：规划任务、注册 Agent、调度分配。

        流程：Planner.plan → start_from_plan。
        TeamRun 进入 RUNNING 状态。若 Planner 返回空任务列表，直接进入 COMPLETED。
        """
        if not objective.strip():
            raise OrchestratorError("objective cannot be empty")

        self._run_counter += 1
        rid = run_id or f"run_{self._run_counter:04d}"
        if rid in self._runs:
            raise RunStateError(f"run already exists: {rid}")

        run = TeamRun(run_id=rid, team=team, objective=objective)
        run.started_at = time.time()
        self._runs[rid] = run

        try:
            specs = await self._planner.plan(objective)
        except Exception as e:
            run.status = TeamRunStatus.FAILED
            run.finished_at = time.time()
            run.error = f"planner failed: {e}"
            raise OrchestratorError(f"planner failed: {e}") from e

        return await self._start_from_plan(run, specs)

    async def start_from_plan(
        self,
        team: Team,
        objective: str,
        specs: list[TaskSpec],
        *,
        run_id: str | None = None,
    ) -> TeamRun:
        """用预规划的任务列表启动 TeamRun，不调用 Planner。

        允许调用方先执行 Planner.plan 并基于结果做准备（如创建 worker），
        再将 specs 传入此方法，确保规划只执行一次。
        """
        if not objective.strip():
            raise OrchestratorError("objective cannot be empty")

        self._run_counter += 1
        rid = run_id or f"run_{self._run_counter:04d}"
        if rid in self._runs:
            raise RunStateError(f"run already exists: {rid}")

        run = TeamRun(run_id=rid, team=team, objective=objective)
        run.started_at = time.time()
        self._runs[rid] = run

        return await self._start_from_plan(run, specs)

    async def _start_from_plan(
        self,
        run: TeamRun,
        specs: list[TaskSpec],
    ) -> TeamRun:
        """从已创建的 run 和 specs 执行任务创建与调度。"""
        if not specs:
            run.status = TeamRunStatus.COMPLETED
            run.finished_at = time.time()
            return run

        run.status = TeamRunStatus.RUNNING

        task_ids = await self._create_tasks(specs)
        run.task_ids = set(task_ids)

        schedulable_tasks = [
            SchedulableTask(task_id=tid, spec=spec)
            for tid, spec in zip(task_ids, specs, strict=True)
        ]
        assignments = await self._scheduler.schedule(schedulable_tasks, self._registry)
        for assignment in assignments:
            run.assignments.append(assignment)
            run.agent_ids.add(assignment.agent_id)
            if assignment.task_id and self._async_task_store is not None:
                with contextlib.suppress(Exception):
                    await self._async_task_store.claim_task(
                        assignment.task_id,
                        assignment.agent_id,
                        lease_seconds=run.team.policy.task_lease_secs,
                    )
            elif assignment.task_id and self._task_store is not None:
                with contextlib.suppress(Exception):
                    self._task_store.claim(assignment.task_id, assignment.agent_id)

        return run

    async def _create_tasks(self, specs: list[TaskSpec]) -> list[str]:
        """在 TaskStore 中创建任务，返回 task_id 列表。

        无 TaskStore 时生成唯一占位 ID。
        """
        if self._async_task_store is not None:
            task_ids: list[str] = []
            for spec in specs:
                task = await self._async_task_store.create_task(
                    subject=spec.subject,
                    description=spec.description,
                    priority=spec.priority,
                )
                task_ids.append(task.task_id)
            return task_ids
        if self._task_store is None:
            return [f"task_{i:04d}" for i in range(len(specs))]
        task_ids = []
        for spec in specs:
            task = self._task_store.create(
                subject=spec.subject,
                description=spec.description,
            )
            task_ids.append(task.id)
        return task_ids

    async def pause(self, run_id: str) -> TeamRun:
        """暂停 TeamRun：向所有活跃 Agent 发送 pause 信号。"""
        run = self._get_run(run_id)
        if run.status != TeamRunStatus.RUNNING:
            raise RunStateError(f"cannot pause run in {run.status.value} state")
        run.status = TeamRunStatus.PAUSED
        if self._coordinator is not None:
            for agent_id in run.agent_ids:
                runtime = self._find_runtime(agent_id)
                if runtime is not None and runtime.alive:
                    with contextlib.suppress(Exception):
                        runtime.pause()
        for agent_id in run.agent_ids:
            await self._bus.send(
                run.team.lead_agent_id,
                agent_id,
                "pause",
                kind="pause",
            )
        return run

    async def resume(self, run_id: str) -> TeamRun:
        """恢复已暂停的 TeamRun。"""
        run = self._get_run(run_id)
        if run.status != TeamRunStatus.PAUSED:
            raise RunStateError(f"cannot resume run in {run.status.value} state")
        run.status = TeamRunStatus.RUNNING
        if self._coordinator is not None:
            for agent_id in run.agent_ids:
                runtime = self._find_runtime(agent_id)
                if runtime is not None and runtime.alive:
                    with contextlib.suppress(Exception):
                        runtime.resume()
        for agent_id in run.agent_ids:
            await self._bus.send(
                run.team.lead_agent_id,
                agent_id,
                "resume",
                kind="resume",
            )
        return run

    async def cancel(self, run_id: str, reason: str = "") -> TeamRun:
        """取消 TeamRun：向所有活跃 Agent 发送 cancel 信号，标记终态。"""
        run = self._get_run(run_id)
        if run.is_terminal:
            raise RunStateError(f"cannot cancel run in terminal {run.status.value} state")
        run.status = TeamRunStatus.CANCELLING
        if self._coordinator is not None:
            for agent_id in run.agent_ids:
                runtime = self._find_runtime(agent_id)
                if runtime is not None and runtime.alive:
                    with contextlib.suppress(Exception):
                        runtime.cancel()
        for agent_id in run.agent_ids:
            await self._bus.send(
                run.team.lead_agent_id,
                agent_id,
                "cancel",
                kind="cancel",
            )
        if self._async_task_store is not None:
            for task_id in run.task_ids:
                if task_id:
                    with contextlib.suppress(Exception):
                        await self._async_transition_to_terminal(
                            task_id, TaskStatus.CANCELLED,
                        )
        elif self._task_store is not None:
            for task_id in run.task_ids:
                if task_id:
                    with contextlib.suppress(Exception):
                        self._task_store.cancel(task_id, reason=reason)
        run.status = TeamRunStatus.CANCELLED
        run.finished_at = time.time()
        run.error = reason or None

        # 工作项 14：级联取消 — 标记所有活跃 attempt 为 CANCELLED
        # 在 run.status 设为 CANCELLED 之后执行，确保 _execute_assignment
        # 中的 complete_task 被 RunStateError 抑制而不干扰取消流程
        active = self._active_handles.get(run_id, {})
        for handle in list(active.values()):
            await self._fail_attempt_quietly(
                handle,
                error=reason or "cancelled",
                status=AttemptStatus.CANCELLED,
            )

        return run

    async def shutdown(self, run_id: str) -> TeamRun:
        """关闭 TeamRun：向所有活跃 Agent 发送 shutdown 信号。"""
        run = self._get_run(run_id)
        if run.status == TeamRunStatus.SHUTDOWN:
            return run
        if self._coordinator is not None:
            for agent_id in run.agent_ids:
                runtime = self._find_runtime(agent_id)
                if runtime is not None and runtime.alive:
                    with contextlib.suppress(Exception):
                        await runtime.stop()
        for agent_id in run.agent_ids:
            await self._bus.send(
                run.team.lead_agent_id,
                agent_id,
                "shutdown",
                kind="shutdown",
            )
        run.status = TeamRunStatus.SHUTDOWN
        run.finished_at = time.time()
        return run

    async def complete_task(
        self,
        run_id: str,
        agent_id: str,
        task_id: str,
        *,
        summary: str = "",
        error: str | None = None,
        cost_usd: float = 0.0,
    ) -> TeamRun:
        """记录一个 Agent 完成任务的结果。

        当所有任务都有结果时，TeamRun 自动进入 COMPLETED 或 FAILED 终态。
        """
        run = self._get_run(run_id)
        if run.is_terminal:
            raise RunStateError(f"run {run_id} is terminal: {run.status.value}")

        status = "completed" if error is None else "failed"
        result = AgentResult(
            agent_id=agent_id,
            task_id=task_id,
            status=status,
            summary=summary,
            error=error,
            cost_usd=cost_usd,
        )
        run.results.append(result)

        if self._async_task_store is not None and task_id:
            try:
                target = TaskStatus.SUCCEEDED if error is None else TaskStatus.FAILED
                await self._async_transition_to_terminal(
                    task_id, target, agent_id=agent_id,
                )
            except Exception:
                logger.warning(
                    "async task_store update failed for task %s (agent %s)",
                    task_id,
                    agent_id,
                    exc_info=True,
                )
        elif self._task_store is not None and task_id:
            try:
                if error is None:
                    self._task_store.complete(task_id, agent_id)
                else:
                    self._task_store.fail(task_id, agent_id, error=error)
            except Exception:
                logger.warning(
                    "task_store update failed for task %s (agent %s)",
                    task_id,
                    agent_id,
                    exc_info=True,
                )

        await self._check_run_completion(run)
        return run

    async def execute_run(self, run_id: str) -> TeamRun:
        """执行 TeamRun 的所有 Assignment：触发 Runtime 执行，收集结果。

        对每个 Assignment，通过 agent-runtime 映射找到对应 Runtime，
        调用 start + wait_done 执行任务，将 RuntimeResult 映射为 AgentResult，
        通过 complete_task 更新 TeamRun 和 TaskStore 状态。

        并发度受 TeamPolicy.max_concurrent_tasks 约束。
        无 coordinator 或无绑定 Runtime 的 Assignment 记录为失败。
        """
        run = self._get_run(run_id)
        if run.status not in (TeamRunStatus.RUNNING, TeamRunStatus.PAUSED):
            raise RunStateError(f"cannot execute run in {run.status.value} state")

        # 为未被 Scheduler 分配的任务记录失败，使其参与终态判定
        assigned_task_ids = {a.task_id for a in run.assignments if a.task_id}
        unassigned_task_ids = run.task_ids - assigned_task_ids
        for tid in unassigned_task_ids:
            with contextlib.suppress(RunStateError):
                await self.complete_task(
                    run_id,
                    "unassigned",
                    tid,
                    error="no agent available for task",
                )

        if not run.assignments:
            return run

        run.status = TeamRunStatus.RUNNING
        semaphore = asyncio.Semaphore(run.team.policy.max_concurrent_tasks)
        max_attempts = run.team.policy.max_attempts
        dependency_policy = run.team.policy.dependency_policy

        # P1-10: DAG 依赖预解析为 task_id，消除 subject 中转。
        # subject_to_task_ids 仅在此处一次性构建，后续以 dep_task_ids 传递。
        subject_to_task_ids: dict[str, list[str]] = {}
        for a in run.assignments:
            subject_to_task_ids.setdefault(a.task_spec.subject, []).append(a.task_id)
        # 为每个 assignment 预解析 blocked_by 中的 subject 为 dep_task_ids
        assignment_dep_task_ids: dict[str, frozenset[str]] = {}
        for a in run.assignments:
            dep_ids: set[str] = set()
            for dep_subject in a.task_spec.blocked_by:
                dep_ids.update(subject_to_task_ids.get(dep_subject, []))
            assignment_dep_task_ids[a.task_id] = frozenset(dep_ids)
        completion_events: dict[str, asyncio.Event] = {
            a.task_id: asyncio.Event() for a in run.assignments
        }
        task_outcomes: dict[str, str] = {}

        tasks = [
            self._execute_assignment(
                run_id,
                assignment,
                semaphore,
                max_attempts,
                completion_events=completion_events,
                dep_task_ids=assignment_dep_task_ids.get(assignment.task_id, frozenset()),
                task_outcomes=task_outcomes,
                dependency_policy=dependency_policy,
            )
            for assignment in run.assignments
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
        return run

    async def _execute_assignment(
        self,
        run_id: str,
        assignment: Assignment,
        semaphore: asyncio.Semaphore,
        max_attempts: int = 1,
        *,
        completion_events: dict[str, asyncio.Event] | None = None,
        dep_task_ids: frozenset[str] = frozenset(),
        task_outcomes: dict[str, str] | None = None,
        dependency_policy: DependencyPolicy = DependencyPolicy.ALL_SUCCESS,
    ) -> None:
        """执行单个 Assignment：Runtime 执行 → 结果回流。

        支持失败重试：当 Runtime 执行失败且 attempt < max_attempts 时，
        通过 TaskStore.retry 重置任务状态后重试。
        支持 DAG：等待 dep_task_ids 依赖的任务完成后再执行。
        依赖策略：按 dependency_policy 检查依赖终态，不满足时跳过本任务。

        P1-10 修复后 dep_task_ids 为预解析的依赖 task_id 集合，
        不再通过 subject_to_task_ids 中转。
        """
        outcome = "failed"
        try:
            if completion_events and dep_task_ids:
                for dep_tid in dep_task_ids:
                    dep_event = completion_events.get(dep_tid)
                    if dep_event is not None:
                        await dep_event.wait()

            if (
                assignment.task_spec.blocked_by
                and task_outcomes is not None
                and not self._dependencies_satisfied(
                    self._collect_dep_outcomes(
                        dep_task_ids,
                        task_outcomes,
                        assignment.task_spec.blocked_by,
                    ),
                    dependency_policy,
                )
            ):
                outcome = "skipped"
                run = self._get_run(run_id)
                if not run.is_terminal:
                    with contextlib.suppress(RunStateError):
                        await self.complete_task(
                            run_id,
                            assignment.agent_id,
                            assignment.task_id,
                            error="skipped: dependency not satisfied",
                        )
                return

            async with semaphore:
                for attempt in range(1, max_attempts + 1):
                    run = self._get_run(run_id)
                    if run.is_terminal:
                        return

                    runtime = self._find_runtime(assignment.agent_id)
                    if runtime is None:
                        with contextlib.suppress(RunStateError):
                            await self.complete_task(
                                run_id,
                                assignment.agent_id,
                                assignment.task_id,
                                error=f"no runtime bound for agent {assignment.agent_id}",
                            )
                        return

                    try:
                        result_summary, result_error, result_cost = await self._dispatch_and_wait(
                            assignment,
                            runtime,
                            run_id,
                            run.team.policy.task_lease_secs,
                        )
                        if result_error is None:
                            with contextlib.suppress(RunStateError):
                                await self.complete_task(
                                    run_id,
                                    assignment.agent_id,
                                    assignment.task_id,
                                    summary=result_summary,
                                    cost_usd=result_cost,
                                )
                            outcome = "succeeded"
                            return
                        if attempt < max_attempts:
                            if self._async_task_store is not None and assignment.task_id:
                                with contextlib.suppress(Exception):
                                    await self._async_retry_task(assignment.task_id)
                            elif self._task_store and assignment.task_id:
                                with contextlib.suppress(Exception):
                                    self._task_store.retry(assignment.task_id)
                            continue
                        with contextlib.suppress(RunStateError):
                            await self.complete_task(
                                run_id,
                                assignment.agent_id,
                                assignment.task_id,
                                summary=result_summary,
                                error=result_error,
                                cost_usd=result_cost,
                            )
                    except (WaitTimeoutError, TaskTimeoutError, TaskWaitTimeoutError):
                        if attempt < max_attempts:
                            if self._async_task_store is not None and assignment.task_id:
                                with contextlib.suppress(Exception):
                                    await self._async_retry_task(assignment.task_id)
                            elif self._task_store and assignment.task_id:
                                with contextlib.suppress(Exception):
                                    self._task_store.retry(assignment.task_id)
                            continue
                        with contextlib.suppress(RunStateError):
                            await self.complete_task(
                                run_id,
                                assignment.agent_id,
                                assignment.task_id,
                                error=f"task timed out after {run.team.policy.task_lease_secs}s",
                            )
                    except Exception as e:
                        if attempt < max_attempts:
                            continue
                        with contextlib.suppress(RunStateError):
                            await self.complete_task(
                                run_id,
                                assignment.agent_id,
                                assignment.task_id,
                                error=f"execution error: {e}",
                            )
        finally:
            if task_outcomes is not None:
                task_outcomes[assignment.task_id] = outcome
            if completion_events:
                event = completion_events.get(assignment.task_id)
                if event is not None:
                    event.set()

    def _collect_dep_outcomes(
        self,
        dep_task_ids: frozenset[str],
        task_outcomes: dict[str, str],
        blocked_by: tuple[str, ...] = (),
    ) -> list[str]:
        """收集依赖 task_id 的 outcome 列表。

        P1-10 修复后直接接收预解析的 dep_task_ids，
        不再通过 (blocked_by, subject_to_task_ids) 中转。
        对于不存在于 task_outcomes 的 task_id（未调度或不存在），
        计为 "unassigned" 以保持与旧实现一致的跳过语义。

        当 blocked_by 非空但 dep_task_ids 为空时，说明所有依赖
        subject 被 Scheduler 过滤掉（无 Agent 匹配），此时每个
        被过滤的 subject 计为 "unassigned"，确保 dependent 被跳过。
        """
        if not dep_task_ids and blocked_by:
            return ["unassigned"] * len(blocked_by)
        return [task_outcomes.get(tid, "unassigned") for tid in dep_task_ids]

    def _dependencies_satisfied(
        self,
        dep_outcomes: list[str],
        policy: DependencyPolicy,
    ) -> bool:
        """按 DependencyPolicy 检查依赖终态是否满足执行条件。

        依赖 outcome 取值：succeeded / failed / skipped / unassigned。
        - ALL_SUCCESS：所有依赖必须 succeeded。
        - ALL_TERMINAL：所有依赖到达任意终态（非 unassigned）。
        - ANY_SUCCESS：至少一个依赖 succeeded。
        """
        if policy is DependencyPolicy.ALL_SUCCESS:
            return all(o == "succeeded" for o in dep_outcomes)
        if policy is DependencyPolicy.ALL_TERMINAL:
            return all(o != "unassigned" for o in dep_outcomes)
        if policy is DependencyPolicy.ANY_SUCCESS:
            return any(o == "succeeded" for o in dep_outcomes)
        return True

    async def _dispatch_and_wait(
        self,
        assignment: Assignment,
        runtime: Any,
        run_id: str,
        lease_secs: float,
    ) -> tuple[str, str | None, float]:
        """通过 TaskDispatcher 派发任务并等待结果。

        返回 (summary, error, cost_usd)。error 为 None 表示成功。

        Phase 6 工作项 11：超时时标记旧 attempt 为 TIMED_OUT，避免重试后
        旧 attempt 永远停留在 RUNNING 状态。
        Phase 6 工作项 14：注册 handle 到 _active_handles，供 cancel 级联取消。
        """
        runtime_id = self._agent_runtime_map.get(assignment.agent_id, "")
        handle = await self._dispatcher.dispatch(
            assignment.task_id,
            assignment.agent_id,
            runtime_id=runtime_id,
            objective=assignment.task_spec.subject,
            timeout_seconds=lease_secs,
            run_id=run_id,
        )

        # 注册活跃 handle，供 cancel 级联取消
        self._active_handles.setdefault(run_id, {})[assignment.task_id] = handle

        try:
            event = await self._dispatcher.wait_for_result(
                handle,
                timeout=lease_secs,
            )
        except (WaitTimeoutError, TaskTimeoutError, TaskWaitTimeoutError):
            # 工作项 11：超时隔离 — 标记旧 attempt 为 TIMED_OUT
            await self._fail_attempt_quietly(
                handle,
                error="timed out",
                status=AttemptStatus.TIMED_OUT,
            )
            raise
        finally:
            # 清理活跃 handle
            self._active_handles.get(run_id, {}).pop(assignment.task_id, None)

        summary = str(event.payload.get("summary", ""))
        cost = float(event.payload.get("cost_usd", 0.0))
        if event.event_type is RuntimeEventType.TASK_COMPLETED:
            return summary, None, cost
        error_msg = str(event.payload.get("error", event.event_type.value))
        return summary, error_msg, cost

    async def _fail_attempt_quietly(
        self,
        handle: TaskHandle,
        *,
        error: str,
        status: AttemptStatus = AttemptStatus.FAILED,
    ) -> None:
        """通过 dispatcher 标记 attempt 为失败/超时/取消，吞异常。

        供超时隔离和级联取消使用。attempt 可能已终态或不存在，
        这些情况不应影响主流程。
        """
        fail = getattr(self._dispatcher, "fail_attempt", None)
        if fail is not None:
            with contextlib.suppress(Exception):
                await fail(handle, error=error, status=status)

    def _find_runtime(self, agent_id: str) -> Any | None:
        """通过 agent-runtime 映射查找 Runtime。"""
        if self._coordinator is None:
            return None
        runtime_id = self._agent_runtime_map.get(agent_id)
        if runtime_id is None:
            return None
        return self._coordinator.get(runtime_id)

    async def _async_transition_to_terminal(
        self,
        task_id: str,
        target: TaskStatus,
        *,
        agent_id: str | None = None,
    ) -> None:
        """通过 async store 将任务转移到终态，自动走中间状态。

        SUCCEEDED 只能从 RUNNING 转入，因此从 DISPATCHED/ACCEPTED 等
        中间状态需逐步迁移。FAILED 和 CANCELLED 可从多数非终态直接转入。
        """
        store = self._async_task_store
        if store is None:
            return
        task = await store.get_task(task_id)
        if task is None or task.is_terminal:
            return

        if target is TaskStatus.SUCCEEDED:
            intermediate: dict[TaskStatus, list[TaskStatus]] = {
                TaskStatus.PENDING: [
                    TaskStatus.DISPATCHED,
                    TaskStatus.ACCEPTED,
                    TaskStatus.RUNNING,
                ],
                TaskStatus.QUEUED: [
                    TaskStatus.DISPATCHED,
                    TaskStatus.ACCEPTED,
                    TaskStatus.RUNNING,
                ],
                TaskStatus.DISPATCHED: [TaskStatus.ACCEPTED, TaskStatus.RUNNING],
                TaskStatus.ACCEPTED: [TaskStatus.RUNNING],
                TaskStatus.RUNNING: [],
            }
            steps = intermediate.get(task.status, [])
            for next_status in steps:
                await store.transition_task(
                    task_id, task.status, next_status, agent_id=agent_id,
                )
                task = await store.get_task(task_id)
                if task is None:
                    return
            await store.transition_task(
                task_id, task.status, TaskStatus.SUCCEEDED, agent_id=agent_id,
            )
        elif target is TaskStatus.FAILED:
            await store.transition_task(
                task_id, task.status, TaskStatus.FAILED, agent_id=agent_id,
            )
        elif target is TaskStatus.CANCELLED:
            await store.transition_task(
                task_id, task.status, TaskStatus.CANCELLED, agent_id=agent_id,
            )

    async def _async_retry_task(self, task_id: str) -> None:
        """通过 async store 重置失败/超时任务以供重试。"""
        store = self._async_task_store
        if store is None:
            return
        task = await store.get_task(task_id)
        if task is None:
            return
        if task.status in (TaskStatus.FAILED, TaskStatus.TIMED_OUT):
            await store.transition_task(
                task_id, task.status, TaskStatus.DISPATCHED,
            )

    async def _check_run_completion(self, run: TeamRun) -> None:
        """检查是否所有任务已完成，更新 TeamRun 终态。

        通过 _aggregate_results 从 TaskStore 聚合结果（唯一事实来源），
        只检查终态结果（completed/failed/timeout/cancelled/budget_exceeded），
        非终态 task 不参与判定，避免提前误判。
        """
        total = len(run.task_ids)
        if total == 0:
            return
        results = await self._aggregate_results(run)
        terminal_statuses = frozenset(
            {"completed", "failed", "timeout", "cancelled", "budget_exceeded"}
        )
        terminal_results = [r for r in results if r.status in terminal_statuses]
        result_task_ids = {r.task_id for r in terminal_results}
        expected_task_ids = run.task_ids
        if not expected_task_ids.issubset(result_task_ids):
            return
        failed = any(r.status != "completed" for r in terminal_results)
        if failed:
            run.status = TeamRunStatus.FAILED
            run.finished_at = time.time()
            run.error = "one or more tasks failed"
        else:
            run.status = TeamRunStatus.COMPLETED
            run.finished_at = time.time()

    def _get_run(self, run_id: str) -> TeamRun:
        run = self._runs.get(run_id)
        if run is None:
            raise RunNotFoundError(f"unknown run: {run_id}")
        return run

    def get_run(self, run_id: str) -> TeamRunSnapshot:
        """查询 TeamRun 快照。"""
        return self._get_run(run_id).snapshot()

    def list_runs(self, *, team_id: str | None = None) -> list[TeamRunSnapshot]:
        """列出所有 TeamRun 快照，可按 team_id 过滤。"""
        snapshots = [run.snapshot() for run in self._runs.values()]
        if team_id is not None:
            snapshots = [s for s in snapshots if s.team_id == team_id]
        return snapshots

    async def _aggregate_results(self, run: TeamRun) -> tuple[AgentResult, ...]:
        """从 TaskStore 终态聚合 AgentResult，TaskStore 为唯一事实来源。

        无 TaskStore 时 fallback 到 run.results 内存副本。
        summary / cost_usd 不在 TaskStore 中，从 run.results 缓存补充。
        当 TaskStore 状态非终态（如 pending）但缓存有终态结果时，
        使用缓存结果（处理 TaskStore 写入失败的边界情况）。
        """
        if self._async_task_store is not None:
            return await self._async_aggregate_results(run)
        if self._task_store is None:
            return tuple(run.results)

        terminal_statuses = frozenset(
            {"completed", "failed", "timeout", "cancelled", "budget_exceeded"}
        )
        cache: dict[str, AgentResult] = {r.task_id: r for r in run.results}
        results: list[AgentResult] = []
        for tid in run.task_ids:
            try:
                task = self._task_store.load(tid)
            except Exception:
                cached = cache.get(tid)
                if cached is not None:
                    results.append(cached)
                continue
            cached = cache.get(tid)
            # TaskStore 终态 → 以 TaskStore 为准
            if task.status in terminal_statuses:
                results.append(
                    AgentResult(
                        agent_id=task.owner_agent_id or task.owner or "",
                        task_id=task.id,
                        status=task.status,
                        summary=cached.summary if cached else "",
                        error=task.error,
                        cost_usd=cached.cost_usd if cached else 0.0,
                    )
                )
            elif cached is not None and cached.status in terminal_statuses:
                # TaskStore 非终态但缓存有终态 → 使用缓存（TaskStore 写入失败）
                results.append(cached)
            else:
                results.append(
                    AgentResult(
                        agent_id=task.owner_agent_id or task.owner or "",
                        task_id=task.id,
                        status=task.status,
                        summary=cached.summary if cached else "",
                        error=task.error,
                        cost_usd=cached.cost_usd if cached else 0.0,
                    )
                )
        return tuple(results)

    async def _async_aggregate_results(self, run: TeamRun) -> tuple[AgentResult, ...]:
        """从 async TaskStore 聚合 AgentResult。

        TaskStatus 映射为结果字符串：SUCCEEDED→completed, FAILED→failed,
        CANCELLED→cancelled, TIMED_OUT→timeout, 其他原值。
        """
        cache: dict[str, AgentResult] = {r.task_id: r for r in run.results}
        results: list[AgentResult] = []
        for tid in run.task_ids:
            try:
                task = await self._async_task_store.get_task(tid)
            except Exception:
                cached = cache.get(tid)
                if cached is not None:
                    results.append(cached)
                continue
            if task is None:
                cached = cache.get(tid)
                if cached is not None:
                    results.append(cached)
                continue
            cached = cache.get(tid)
            status_str = _TASK_STATUS_TO_RESULT.get(task.status, task.status.value)
            results.append(
                AgentResult(
                    agent_id=task.agent_id or "",
                    task_id=task.task_id,
                    status=status_str,
                    summary=cached.summary if cached else "",
                    error=task.error,
                    cost_usd=cached.cost_usd if cached else 0.0,
                )
            )
        return tuple(results)

    async def progress(self, run_id: str) -> TeamProgress:
        """查询 TeamRun 整体进度。"""
        run = self._get_run(run_id)
        total = len(run.task_ids)
        results = await self._aggregate_results(run)
        completed = sum(1 for r in results if r.status == "completed")
        failed = sum(
            1 for r in results if r.status in ("failed", "timeout", "cancelled", "budget_exceeded")
        )
        active = len(run.agent_ids)
        elapsed = 0.0
        if run.started_at is not None:
            end = run.finished_at or time.time()
            elapsed = end - run.started_at
        return TeamProgress(
            run_id=run_id,
            status=run.status,
            total_tasks=total,
            completed_tasks=completed,
            failed_tasks=failed,
            active_agents=active,
            elapsed_secs=elapsed,
        )

    async def collect_results(self, run_id: str) -> TeamRunResult:
        """收集 TeamRun 汇总结果。

        结果从 TaskStore 终态聚合，TaskStore 为唯一事实来源。
        """
        run = self._get_run(run_id)
        results = await self._aggregate_results(run)
        completed = sum(1 for r in results if r.status == "completed")
        failed = sum(
            1 for r in results if r.status in ("failed", "timeout", "cancelled", "budget_exceeded")
        )
        return TeamRunResult(
            run_id=run_id,
            status=run.status,
            results=results,
            completed_count=completed,
            failed_count=failed,
            total_count=len(run.task_ids),
            error=run.error,
        )

    def register_agent(
        self,
        agent_id: str,
        agent_name: str,
        *,
        capabilities: frozenset[str] = frozenset(),
        mode: RuntimeMode = RuntimeMode.PERSISTENT,
        tenant: str = "default",
        tags: frozenset[str] = frozenset(),
    ) -> Any:
        """便捷方法：通过 Registry 注册 Agent。

        返回 AgentRegistration。
        """
        desc = AgentDescriptor(
            agent_id=agent_id,
            agent_name=agent_name,
            capabilities=capabilities,
            mode=mode,
            tenant=tenant,
            tags=tags,
        )
        return self._registry.register(desc)


__all__ = [
    "AgentResult",
    "Assignment",
    "CapabilityScheduler",
    "OrchestratorError",
    "Planner",
    "ResultCollector",
    "RunNotFoundError",
    "RunStateError",
    "SimplePlanner",
    "SimpleResultCollector",
    "TaskScheduler",
    "TaskSpec",
    "Team",
    "TeamNotFoundError",
    "TeamOrchestrator",
    "TeamPolicy",
    "TeamProgress",
    "TeamRun",
    "TeamRunResult",
    "TeamRunSnapshot",
    "TeamRunStatus",
]
