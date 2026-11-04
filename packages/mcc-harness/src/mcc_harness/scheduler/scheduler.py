"""任务调度器实现：依赖检测、优先级排序、能力匹配与负载约束。

基于 TaskStore 的依赖状态和 AgentRegistry 的能力发现，
按 SchedulingPolicy 做分配决策。调度与 claim 原子衔接：
分配成功后立即在 TaskStore 中 claim 任务，防止重复分配。

实现 TaskScheduler Protocol（team.orchestrator），可直接注入
TeamOrchestrator 替换默认 CapabilityScheduler。
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable

from mcc_harness.registry import AgentRegistration, AgentRegistry, RegistrationStatus
from mcc_harness.team.orchestrator import Assignment, TaskSpec

from .base import (
    AgentCandidate,
    PriorityOrder,
    RetryPolicy,
    SchedulingPolicy,
    SchedulingResult,
    TaskCandidate,
)


class TaskSchedulerImpl:
    """任务调度器：依赖检测 → 优先级排序 → 能力匹配 → 负载约束 → claim。

    调度流程：
    1. 将 TaskSpec 包装为 TaskCandidate，检查依赖状态
    2. 按 priority 排序（DESC 优先级高在前，ASC 低在前）
    3. 对每个就绪任务，从 Registry 中查找能力匹配的 Agent
    4. 过滤掉已满载的 Agent，选择负载最低的
    5. 分配后通过 claim_callback 原子占位，防止重复分配

    与 TaskStore 集成时，claim_callback 调用 task_store.claim()。
    无 TaskStore 时 claim_callback 为空操作。
    """

    def __init__(
        self,
        policy: SchedulingPolicy | None = None,
        *,
        claim_callback: Callable[[str, str], None] | None = None,
        dependency_check: Callable[[str], bool] | None = None,
    ) -> None:
        self._policy = policy or SchedulingPolicy()
        self._claim_callback = claim_callback
        self._dependency_check = dependency_check
        self._agent_loads: dict[str, int] = {}

    @property
    def policy(self) -> SchedulingPolicy:
        return self._policy

    async def schedule(
        self,
        tasks: list[TaskSpec],
        registry: AgentRegistry,
        *,
        exclude_agents: frozenset[str] = frozenset(),
        task_ids: list[str] | None = None,
    ) -> list[Assignment]:
        """TaskScheduler Protocol 兼容接口。

        返回成功分配的 Assignment 列表。
        依赖未满足或无可用 Agent 的任务被跳过。
        """
        result = await self.schedule_round(
            tasks,
            registry,
            exclude_agents=exclude_agents,
            task_ids=task_ids,
        )
        return list(result.assignments)

    async def schedule_round(
        self,
        tasks: list[TaskSpec],
        registry: AgentRegistry,
        *,
        exclude_agents: frozenset[str] = frozenset(),
        task_ids: list[str] | None = None,
    ) -> SchedulingResult:
        """完整调度轮次，返回结构化结果。

        Args:
            tasks: 待调度的任务规格列表
            registry: Agent 注册表
            exclude_agents: 排除的 Agent ID 集合
            task_ids: 对应的 TaskStore 任务 ID（None 时用占位 ID）

        Returns:
            SchedulingResult: 包含分配、推迟、拒绝和重试信息
        """
        ids = task_ids or [f"task_{i:04d}" for i in range(len(tasks))]
        candidates = self._build_candidates(tasks, ids)
        candidates = self._sort_by_priority(candidates)

        assignments: list[Assignment] = []
        deferred: list[str] = []
        rejected: list[str] = []
        used_agents: set[str] = set(exclude_agents)
        assigned_count = 0

        for candidate in candidates:
            if not candidate.is_ready:
                deferred.append(candidate.task_id)
                continue

            max_total = self._policy.max_total_assignments
            if max_total is not None and assigned_count >= max_total:
                deferred.append(candidate.task_id)
                continue

            agent = self._select_agent(
                candidate.spec,
                registry,
                used_agents,
            )
            if agent is None:
                rejected.append(candidate.task_id)
                continue

            assignment = Assignment(
                task_id=candidate.task_id,
                agent_id=agent.agent_id,
                task_spec=candidate.spec,
            )
            assignments.append(assignment)
            self._agent_loads[agent.agent_id] = (
                self._agent_loads.get(agent.agent_id, 0) + 1
            )
            assigned_count += 1

            if self._claim_callback is not None:
                with contextlib.suppress(Exception):
                    self._claim_callback(candidate.task_id, agent.agent_id)

        return SchedulingResult(
            assignments=tuple(assignments),
            deferred=tuple(deferred),
            rejected=tuple(rejected),
        )

    def schedule_retry(
        self,
        task_id: str,
        attempt: int,
    ) -> float:
        """计算失败任务的重试延迟。

        Returns:
            重试延迟（秒）。若重试次数已耗尽，抛出 RetryExhaustedError。
        """
        retry = self._policy.retry_policy
        if not retry.should_retry(attempt):
            from .base import RetryExhaustedError
            raise RetryExhaustedError(
                f"task {task_id} exhausted {retry.max_retries} retries",
            )
        return retry.delay_for(attempt)

    def reset_loads(self) -> None:
        """重置所有 Agent 的负载计数。

        在新一轮调度开始前调用。
        """
        self._agent_loads.clear()

    def get_load(self, agent_id: str) -> int:
        """获取 Agent 当前负载。"""
        return self._agent_loads.get(agent_id, 0)

    def _build_candidates(
        self,
        tasks: list[TaskSpec],
        task_ids: list[str],
    ) -> list[TaskCandidate]:
        """将 TaskSpec 列表包装为 TaskCandidate 列表。"""
        candidates: list[TaskCandidate] = []
        for i, spec in enumerate(tasks):
            tid = task_ids[i] if i < len(task_ids) else f"task_{i:04d}"
            deps_satisfied = True
            if self._dependency_check is not None:
                with contextlib.suppress(Exception):
                    deps_satisfied = self._dependency_check(tid)
            candidates.append(
                TaskCandidate(
                    task_id=tid,
                    spec=spec,
                    dependencies_satisfied=deps_satisfied,
                )
            )
        return candidates

    def _sort_by_priority(
        self,
        candidates: list[TaskCandidate],
    ) -> list[TaskCandidate]:
        """按优先级排序。"""
        reverse = self._policy.priority_order == PriorityOrder.DESC
        return sorted(
            candidates,
            key=lambda c: c.spec.priority,
            reverse=reverse,
        )

    def _select_agent(
        self,
        spec: TaskSpec,
        registry: AgentRegistry,
        exclude_agents: set[str],
    ) -> AgentCandidate | None:
        """为任务选择最佳 Agent：能力匹配 → 排除 → 负载最低。"""
        capable = self._find_capable_agents(spec, registry)
        available = [
            agent for agent in capable
            if agent.agent_id not in exclude_agents and agent.has_capacity
        ]
        if not available:
            return None
        return min(available, key=lambda a: (a.current_load, a.agent_id))

    def _find_capable_agents(
        self,
        spec: TaskSpec,
        registry: AgentRegistry,
    ) -> list[AgentCandidate]:
        """从 Registry 中查找能力匹配的 Agent，转为 AgentCandidate。"""
        if spec.required_capabilities:
            registrations = registry.find_by_capabilities(
                spec.required_capabilities,
            )
        else:
            registrations = registry.list(
                status=RegistrationStatus.AVAILABLE,
            )
        return [
            self._to_candidate(reg) for reg in registrations
        ]

    def _to_candidate(self, reg: AgentRegistration) -> AgentCandidate:
        """将 AgentRegistration 转为 AgentCandidate，附加负载信息。"""
        load = self._agent_loads.get(reg.agent_id, 0)
        return AgentCandidate(
            agent_id=reg.agent_id,
            capabilities=reg.capabilities,
            current_load=load,
            max_concurrent=self._policy.max_concurrent_per_agent,
        )


class RetryHandler:
    """失败任务重试处理器。

    根据 RetryPolicy 决定是否重试、计算退避延迟，
    并通过回调执行重试动作（如 TaskStore 状态转换）。
    """

    def __init__(
        self,
        policy: RetryPolicy | None = None,
        *,
        retry_callback: Callable[[str, float], None] | None = None,
    ) -> None:
        self._policy = policy or RetryPolicy()
        self._retry_callback = retry_callback
        self._retry_counts: dict[str, int] = {}

    @property
    def policy(self) -> RetryPolicy:
        return self._policy

    def should_retry(self, task_id: str) -> bool:
        """判断任务是否还应重试。"""
        attempt = self._retry_counts.get(task_id, 0) + 1
        return self._policy.should_retry(attempt)

    def schedule_retry(self, task_id: str) -> float:
        """安排重试，返回延迟秒数。

        若重试次数已耗尽，抛出 RetryExhaustedError。
        """
        attempt = self._retry_counts.get(task_id, 0) + 1
        delay = self._policy.delay_for(attempt)
        if not self._policy.should_retry(attempt):
            from .base import RetryExhaustedError
            raise RetryExhaustedError(
                f"task {task_id} exhausted {self._policy.max_retries} retries",
            )
        self._retry_counts[task_id] = attempt
        if self._retry_callback is not None:
            with contextlib.suppress(Exception):
                self._retry_callback(task_id, delay)
        return delay

    def get_retry_count(self, task_id: str) -> int:
        """获取任务已重试次数。"""
        return self._retry_counts.get(task_id, 0)

    def reset(self, task_id: str) -> None:
        """重置任务的重试计数。"""
        self._retry_counts.pop(task_id, None)

    def reset_all(self) -> None:
        """重置所有重试计数。"""
        self._retry_counts.clear()


class LeaseManager:
    """任务租约管理器。

    跟踪已分配任务的租约到期时间，
    支持超时回收（将过期任务标记为 TIMEOUT 并释放 Agent 负载）。
    """

    def __init__(
        self,
        *,
        lease_secs: float = 300.0,
        timeout_callback: Callable[[str], None] | None = None,
    ) -> None:
        self._lease_secs = lease_secs
        self._timeout_callback = timeout_callback
        self._leases: dict[str, tuple[str, float]] = {}

    @property
    def lease_secs(self) -> float:
        return self._lease_secs

    def grant(self, task_id: str, agent_id: str) -> float:
        """授予租约，返回到期时间戳。"""
        expires_at = time.time() + self._lease_secs
        self._leases[task_id] = (agent_id, expires_at)
        return expires_at

    def renew(self, task_id: str) -> float | None:
        """续租，返回新的到期时间戳。任务不存在则返回 None。"""
        if task_id not in self._leases:
            return None
        agent_id = self._leases[task_id][0]
        expires_at = time.time() + self._lease_secs
        self._leases[task_id] = (agent_id, expires_at)
        return expires_at

    def revoke(self, task_id: str) -> str | None:
        """撤销租约，返回关联的 agent_id。任务不存在则返回 None。"""
        entry = self._leases.pop(task_id, None)
        if entry is None:
            return None
        return entry[0]

    def is_expired(self, task_id: str) -> bool:
        """检查租约是否已过期。"""
        entry = self._leases.get(task_id)
        if entry is None:
            return False
        return time.time() > entry[1]

    def collect_expired(self) -> list[tuple[str, str]]:
        """收集所有过期租约，返回 (task_id, agent_id) 列表。

        对每个过期任务调用 timeout_callback（如有），
        并从内部跟踪中移除。
        """
        now = time.time()
        expired: list[tuple[str, str]] = []
        to_remove: list[str] = []
        for task_id, (agent_id, expires_at) in self._leases.items():
            if now > expires_at:
                expired.append((task_id, agent_id))
                to_remove.append(task_id)
        for tid in to_remove:
            self._leases.pop(tid, None)
            if self._timeout_callback is not None:
                with contextlib.suppress(Exception):
                    self._timeout_callback(tid)
        return expired

    def active_leases(self) -> dict[str, str]:
        """返回当前活跃租约的 {task_id: agent_id} 映射。"""
        return {
            tid: agent_id
            for tid, (agent_id, _) in self._leases.items()
        }
