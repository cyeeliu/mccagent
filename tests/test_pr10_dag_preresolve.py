"""PR-10: DAG 依赖预解析为 task_id，消除 subject 中转。

P1-10 要求 blocked_by 中的 subject 在 execute_run 入口一次性解析为
dep_task_ids: frozenset[str]，后续 _execute_assignment 和
_collect_dep_outcomes 直接以 task_id 为键操作，不再通过
subject_to_task_ids 中转。

修复后：
- _collect_dep_outcomes 接收 dep_task_ids 而非 (blocked_by, subject_to_task_ids)
- _execute_assignment 接收 dep_task_ids 而非 subject_to_task_ids
- DAG 节点完全以 task_id 为键
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from mcc_harness import DependencyPolicy
from mcc_harness.team.orchestrator import TeamOrchestrator


@pytest.fixture
def orch(tmp_path: Path) -> TeamOrchestrator:
    """创建一个最小 orchestrator 仅用于测试内部方法。"""
    from mcc_harness import AgentRegistry, MessageBus, PlanStore

    from tests.test_orchestrator import FakeDispatcher

    return TeamOrchestrator(
        registry=AgentRegistry(),
        bus=MessageBus(tmp_path),
        plan_store=PlanStore(),
        dispatcher=FakeDispatcher(),
    )


class TestPr10CollectDepOutcomesByTaskId:
    """_collect_dep_outcomes 直接接收 dep_task_ids。"""

    def test_collect_outcomes_by_task_id(self, orch: TeamOrchestrator):
        """_collect_dep_outcomes 接收 frozenset[str] 的 dep_task_ids。"""
        dep_task_ids = frozenset({"task-001", "task-002"})
        task_outcomes = {"task-001": "succeeded", "task-002": "failed"}

        result = orch._collect_dep_outcomes(dep_task_ids, task_outcomes)

        assert sorted(result) == ["failed", "succeeded"]

    def test_collect_outcomes_missing_task_is_unassigned(
        self,
        orch: TeamOrchestrator,
    ):
        """dep_task_ids 中不存在于 task_outcomes 的记为 unassigned。"""
        dep_task_ids = frozenset({"task-001", "task-999"})
        task_outcomes = {"task-001": "succeeded"}

        result = orch._collect_dep_outcomes(dep_task_ids, task_outcomes)

        assert sorted(result) == ["succeeded", "unassigned"]

    def test_collect_outcomes_empty_dep(self, orch: TeamOrchestrator):
        """空 dep_task_ids 返回空列表。"""
        result = orch._collect_dep_outcomes(frozenset(), {"task-001": "succeeded"})
        assert result == []


class TestPr10DependencyPolicyByTaskId:
    """依赖策略检查基于 task_id 解析的 outcomes。"""

    def test_all_success_with_one_failure(self, orch: TeamOrchestrator):
        """ALL_SUCCESS：一个依赖失败 → 不满足。"""
        dep_task_ids = frozenset({"t1", "t2"})
        task_outcomes = {"t1": "succeeded", "t2": "failed"}
        outcomes = orch._collect_dep_outcomes(dep_task_ids, task_outcomes)
        assert not orch._dependencies_satisfied(outcomes, DependencyPolicy.ALL_SUCCESS)

    def test_all_success_all_succeed(self, orch: TeamOrchestrator):
        """ALL_SUCCESS：全部成功 → 满足。"""
        dep_task_ids = frozenset({"t1", "t2"})
        task_outcomes = {"t1": "succeeded", "t2": "succeeded"}
        outcomes = orch._collect_dep_outcomes(dep_task_ids, task_outcomes)
        assert orch._dependencies_satisfied(outcomes, DependencyPolicy.ALL_SUCCESS)

    def test_all_terminal_with_mixed(self, orch: TeamOrchestrator):
        """ALL_TERMINAL：成功+失败都到达终态 → 满足。"""
        dep_task_ids = frozenset({"t1", "t2"})
        task_outcomes = {"t1": "succeeded", "t2": "failed"}
        outcomes = orch._collect_dep_outcomes(dep_task_ids, task_outcomes)
        assert orch._dependencies_satisfied(outcomes, DependencyPolicy.ALL_TERMINAL)

    def test_all_terminal_with_unassigned(self, orch: TeamOrchestrator):
        """ALL_TERMINAL：有 unassigned → 不满足。"""
        dep_task_ids = frozenset({"t1", "t999"})
        task_outcomes = {"t1": "succeeded"}
        outcomes = orch._collect_dep_outcomes(dep_task_ids, task_outcomes)
        assert not orch._dependencies_satisfied(outcomes, DependencyPolicy.ALL_TERMINAL)


class TestPr10ExecuteAssignmentDepTaskIds:
    """_execute_assignment 接收 dep_task_ids 参数。"""

    async def test_execute_assignment_accepts_dep_task_ids(
        self,
        orch: TeamOrchestrator,
    ):
        """_execute_assignment 签名包含 dep_task_ids 参数。

        旧签名接收 subject_to_task_ids: dict[str, list[str]] | None，
        新签名接收 dep_task_ids: frozenset[str]。
        通过 inspect 验证参数名。
        """
        import inspect

        sig = inspect.signature(orch._execute_assignment)
        assert "dep_task_ids" in sig.parameters, (
            "_execute_assignment should accept dep_task_ids parameter"
        )
        assert "subject_to_task_ids" not in sig.parameters, (
            "_execute_assignment should not accept subject_to_task_ids parameter"
        )
