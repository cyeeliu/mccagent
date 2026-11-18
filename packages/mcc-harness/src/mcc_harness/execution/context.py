"""执行上下文：每次 AgentExecutor.execute 建立独立上下文。

ExecutionContext 隔离单次执行的临时状态，
不共享 Runtime 生命周期状态。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from mcc_core.runtime import RuntimeTask


@dataclass
class ExecutionContext:
    task: RuntimeTask
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    instruction_queue: asyncio.Queue[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        return self.task.task_id

    @property
    def attempt_id(self) -> str:
        return self.task.attempt_id

    @property
    def agent_id(self) -> str:
        return self.task.agent_id


__all__ = ["ExecutionContext"]
