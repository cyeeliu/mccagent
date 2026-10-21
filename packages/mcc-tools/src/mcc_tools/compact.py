"""compact 工具：模型主动请求压缩当前会话上下文。

自动阈值只知道上下文有多大；模型自己知道「一个阶段结束了，后续
工作只需要保留摘要」。本工具无副作用——真正的压缩由 AgentLoop 在
本批工具结果全部闭合后执行（避免文件已写却丢执行记录，导致模型
重复同一副作用）。
"""

from __future__ import annotations

from typing import Any

from mcc_core import ToolContext, ToolOutput

CONFIRM_TEXT = (
    "Compaction requested. The harness will summarize the conversation "
    "after this tool batch completes; earlier details will be replaced by "
    "a summary with recoverable tool-output paths."
)


class CompactTool:
    """请求压缩上下文（无参数、无副作用，由主循环兑现）。"""

    @property
    def name(self) -> str:
        return "compact"

    @property
    def description(self) -> str:
        return (
            "Request compaction of the conversation history to free context "
            "space. Call this when a phase of work is complete and only a "
            "summary is needed going forward. Takes no arguments; the "
            "harness summarizes after the current tool batch finishes."
        )

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        return ToolOutput.ok(CONFIRM_TEXT)
