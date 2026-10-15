"""事件源抽象与模拟实现。

EventSource 协议定义 Agent 事件流的统一接口，
前端（TUI/headless）只消费事件，不直接调用 Agent。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from mcc_core import AgentEvent, Notice, TextDelta, ToolCallEnd, ToolCallStart, TurnEnd, UserEcho


@runtime_checkable
class EventSource(Protocol):
    """事件源协议：接收用户输入，产出 AgentEvent 事件流。"""

    async def run(self, prompt: str) -> AsyncIterator[AgentEvent]: ...


class EchoEventSource:
    """模拟事件源：回显用户输入并模拟流式文本输出。

    用于在前端接入真实 Agent 前演示交互与流式渲染效果。
    """

    def __init__(self, chunk_size: int = 12, delay: float = 0.02) -> None:
        self._chunk_size = chunk_size
        self._delay = delay

    async def run(self, prompt: str) -> AsyncIterator[AgentEvent]:
        yield UserEcho(text=prompt)
        yield ToolCallStart(id="c0", name="echo", args_preview=prompt[:60])
        await asyncio.sleep(self._delay)
        yield ToolCallEnd(id="c0", output=f"received: {prompt}", is_error=False)
        yield Notice(text="generating response…")

        response = (
            f"已收到你的请求：「{prompt}」\n"
            "这是一个模拟响应，真实 Agent 集成后将由 LLM 生成。"
        )
        for i in range(0, len(response), self._chunk_size):
            yield TextDelta(text=response[i : i + self._chunk_size])
            await asyncio.sleep(self._delay)

        yield TurnEnd(cost_usd=0.0)
