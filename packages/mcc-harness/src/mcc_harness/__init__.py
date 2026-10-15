"""Agent 主循环、权限、hooks、skills、subagent。

驱动 LLM 调用与工具执行的迭代循环，集成权限检查与事件分发。
"""

from __future__ import annotations

from dataclasses import dataclass

from mcc_core import Message, ToolContext
from mcc_llm import LlmProvider
from mcc_tools import ToolRegistry


@dataclass
class AgentLoopConfig:
    """AgentLoop 运行参数。"""

    system: str = ""
    model: str = ""
    max_tokens: int = 4096
    max_iterations: int = 40
    temperature: float | None = None
    max_cost_usd: float | None = None


class AgentLoop:
    """Agent 主循环，驱动 LLM 调用与工具执行的迭代。"""

    def __init__(
        self,
        llm: LlmProvider,
        registry: ToolRegistry,
        ctx: ToolContext,
        config: AgentLoopConfig | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.ctx = ctx
        self.config = config or AgentLoopConfig()
        self._messages: list[Message] = []

    @property
    def messages(self) -> list[Message]:
        return self._messages


__all__ = ["AgentLoop", "AgentLoopConfig"]
__version__ = "0.1.0"
