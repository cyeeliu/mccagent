"""AgentExecutor 适配层。

将 AgentLoop 执行能力从 Runtime 和旧 Runner 中解耦。
Phase 2 交付物：AgentLoopExecutor 内部委托现有 AgentLoop。
Phase 3 交付物：DefaultAgentFactory 收拢构造入口。
"""

from __future__ import annotations

from mcc_harness.execution.agent_loop_executor import AgentLoopExecutor
from mcc_harness.execution.cancellation import CancellationToken
from mcc_harness.execution.context import ExecutionContext
from mcc_harness.execution.event_adapter import EventAdapter
from mcc_harness.execution.factory import DefaultAgentFactory, SessionContext

__all__ = [
    "AgentLoopExecutor",
    "CancellationToken",
    "DefaultAgentFactory",
    "ExecutionContext",
    "EventAdapter",
    "SessionContext",
]
