"""Execution 层协议重导出与扩展。

从 mcc_core.runtime 重导出 AgentExecutor / EventEmitter / CancellationToken，
并提供 mcc-harness 层的便捷类型。
"""

from __future__ import annotations

from mcc_core.runtime import AgentExecutor, CancellationToken, EventEmitter

from mcc_harness.execution.cancellation import CancellationToken as CancellationTokenImpl
from mcc_harness.execution.context import ExecutionContext
from mcc_harness.execution.event_adapter import EventAdapter

__all__ = [
    "AgentExecutor",
    "CancellationToken",
    "CancellationTokenImpl",
    "EventAdapter",
    "EventEmitter",
    "ExecutionContext",
]
