"""Team 编排层：TeamOrchestrator 统一入口。

将任务规划、调度、Agent 管理和结果汇总集中到清晰的编排边界，
通过 Protocol 接口与 Planner、TaskScheduler、ResultCollector 解耦。
"""

from __future__ import annotations

from mcc_harness.team.orchestrator import (
    AgentResult,
    Assignment,
    CapabilityScheduler,
    OrchestratorError,
    Planner,
    ResultCollector,
    RunNotFoundError,
    RunStateError,
    SimplePlanner,
    SimpleResultCollector,
    TaskScheduler,
    TaskSpec,
    Team,
    TeamNotFoundError,
    TeamOrchestrator,
    TeamPolicy,
    TeamProgress,
    TeamRun,
    TeamRunResult,
    TeamRunSnapshot,
    TeamRunStatus,
)

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
