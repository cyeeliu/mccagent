"""终端 UI。

基于 Rich 的三栏布局终端界面，消费 AgentEvent 事件流做流式渲染。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from mcc_core import AgentEvent


async def run_tui(events: AsyncIterator[AgentEvent]) -> None:
    """运行 TUI，消费 AgentEvent 事件流做流式渲染。"""
    raise NotImplementedError("TUI 尚未实现")


__all__ = ["run_tui"]
__version__ = "0.1.0"
