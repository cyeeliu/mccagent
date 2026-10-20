"""统一适配协议：被测 Agent 以 TraceEvent 流为契约接入平台。

平台不关心 scaffold 的内部实现——只要适配器把执行过程
转译成标准轨迹事件流，任何框架都可以成为被测对象。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from pydantic import BaseModel

from runner.events import TraceEvent


class RunRequest(BaseModel):
    """一次评测任务的执行请求（由 lease 响应转换而来）。"""

    run_id: str
    case_id: str
    task: str
    model: str
    max_turns: int = 20


class AgentAdapter(Protocol):
    """所有 scaffold 适配器实现同一个方法：执行任务并流式产出标准轨迹事件。"""

    def run(self, req: RunRequest) -> AsyncIterator[TraceEvent]: ...
