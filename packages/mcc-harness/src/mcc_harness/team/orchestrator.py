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

import contextlib
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

from mcc_harness.budget import AgentBudget, BudgetPolicy
from mcc_harness.context import AgentContext
from mcc_harness.mailbox import MessageBus
from mcc_harness.plan import PlanStore
from mcc_harness.registry import AgentDescriptor, AgentRegistry, RegistrationStatus
from mcc_harness.runtime import RuntimeMode


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


_TERMINAL_RUN_STATUSES = frozenset({
    TeamRunStatus.COMPLETED,
    TeamRunStatus.FAILED,
    TeamRunStatus.CANCELLED,
    TeamRunStatus.SHUTDOWN,
})


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
    budget_policy: BudgetPolicy | None = None

    def __post_init__(self) -> None:
        if self.max_agents < 1:
            raise OrchestratorError("max_agents must be >= 1")
        if self.max_concurrent_tasks < 1:
            raise OrchestratorError("max_concurrent_tasks must be >= 1")
        if self.heartbeat_interval_secs <= 0:
            raise OrchestratorError("heartbeat_interval_secs must be > 0")
        if self.task_lease_secs <= 0:
            raise OrchestratorError("task_lease_secs must be > 0")


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
    ) -> list[TaskSpec]:
        ...


@runtime_checkable
class TaskScheduler(Protocol):
    """任务调度器：将任务规格分配给可用 Agent。

    依赖 AgentRegistry 进行能力发现，返回 Assignment 列表。
    """

    async def schedule(
        self,
        tasks: list[TaskSpec],
        registry: AgentRegistry,
        *,
        exclude_agents: frozenset[str] = frozenset(),
    ) -> list[Assignment]:
        ...


@runtime_checkable
class ResultCollector(Protocol):
    """结果收集器：汇总各 Agent 执行结果。"""

    async def collect(self, run_id: str) -> TeamRunResult:
        ...


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

    策略：对每个 TaskSpec，在 Registry 中按 required_capabilities
    查找可用 Agent，取第一个分配。无能力要求时分配给任意可用 Agent。
    """

    async def schedule(
        self,
        tasks: list[TaskSpec],
        registry: AgentRegistry,
        *,
        exclude_agents: frozenset[str] = frozenset(),
    ) -> list[Assignment]:
        assignments: list[Assignment] = []
        used: set[str] = set(exclude_agents)
        for spec in tasks:
            agent_id = self._find_agent(spec, registry, frozenset(used))
            if agent_id is not None:
                used.add(agent_id)
                assignments.append(
                    Assignment(
                        task_id="",
                        agent_id=agent_id,
                        task_spec=spec,
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
        results = tuple(run.results)
        completed = sum(1 for r in results if r.status == "completed")
        failed = sum(
            1 for r in results
            if r.status in ("failed", "timeout", "cancelled", "budget_exceeded")
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
        task_store: Any | None = None,
        budget: AgentBudget | None = None,
        planner: Planner | None = None,
        scheduler: TaskScheduler | None = None,
    ) -> None:
        self._registry = registry
        self._bus = bus
        self._plan_store = plan_store
        self._task_store = task_store
        self._budget = budget
        self._planner = planner or SimplePlanner()
        self._scheduler = scheduler or CapabilityScheduler()
        self._teams: dict[str, Team] = {}
        self._runs: dict[str, TeamRun] = {}
        self._run_counter = 0

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
    def budget(self) -> AgentBudget | None:
        return self._budget

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

        流程：Planner.plan → TaskStore.create → Scheduler.schedule → MessageBus 通知。
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

        if not specs:
            run.status = TeamRunStatus.COMPLETED
            run.finished_at = time.time()
            return run

        run.status = TeamRunStatus.RUNNING

        task_ids = await self._create_tasks(specs)
        run.task_ids = set(task_ids)

        assignments = await self._scheduler.schedule(specs, self._registry)
        for i, assignment in enumerate(assignments):
            task_id = task_ids[i] if i < len(task_ids) else ""
            assignment = Assignment(
                task_id=task_id,
                agent_id=assignment.agent_id,
                task_spec=assignment.task_spec,
            )
            run.assignments.append(assignment)
            run.agent_ids.add(assignment.agent_id)
            if task_id and self._task_store is not None:
                with contextlib.suppress(Exception):
                    self._task_store.claim(task_id, assignment.agent_id)
            await self._bus.send(
                team.lead_agent_id,
                assignment.agent_id,
                assignment.task_spec.subject,
                kind="assignment",
                task_id=task_id,
            )

        return run

    async def _create_tasks(self, specs: list[TaskSpec]) -> list[str]:
        """在 TaskStore 中创建任务，返回 task_id 列表。

        无 TaskStore 时生成唯一占位 ID。
        """
        if self._task_store is None:
            return [f"task_{i:04d}" for i in range(len(specs))]
        task_ids: list[str] = []
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
        for agent_id in run.agent_ids:
            await self._bus.send(
                run.team.lead_agent_id,
                agent_id,
                "cancel",
                kind="cancel",
            )
        if self._task_store is not None:
            for task_id in run.task_ids:
                if task_id:
                    with contextlib.suppress(Exception):
                        self._task_store.cancel(task_id, reason=reason)
        run.status = TeamRunStatus.CANCELLED
        run.finished_at = time.time()
        run.error = reason or None
        return run

    async def shutdown(self, run_id: str) -> TeamRun:
        """关闭 TeamRun：向所有活跃 Agent 发送 shutdown 信号。"""
        run = self._get_run(run_id)
        if run.status == TeamRunStatus.SHUTDOWN:
            return run
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

        if self._task_store is not None and task_id:
            try:
                if error is None:
                    self._task_store.complete(task_id, agent_id)
                else:
                    self._task_store.fail(task_id, agent_id, error=error)
            except Exception:
                pass

        self._check_run_completion(run)
        return run

    def _check_run_completion(self, run: TeamRun) -> None:
        """检查是否所有任务已完成，更新 TeamRun 终态。"""
        total = len(run.task_ids)
        if total == 0:
            return
        result_task_ids = {r.task_id for r in run.results}
        expected_task_ids = run.task_ids
        if not expected_task_ids.issubset(result_task_ids):
            return
        failed = any(r.status != "completed" for r in run.results)
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

    def progress(self, run_id: str) -> TeamProgress:
        """查询 TeamRun 整体进度。"""
        run = self._get_run(run_id)
        total = len(run.task_ids)
        completed = sum(1 for r in run.results if r.status == "completed")
        failed = sum(
            1 for r in run.results
            if r.status in ("failed", "timeout", "cancelled", "budget_exceeded")
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
        """收集 TeamRun 汇总结果。"""
        run = self._get_run(run_id)
        results = tuple(run.results)
        completed = sum(1 for r in results if r.status == "completed")
        failed = sum(
            1 for r in results
            if r.status in ("failed", "timeout", "cancelled", "budget_exceeded")
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
