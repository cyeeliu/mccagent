"""测试用 Mock LLM provider。

按预设脚本依次返回流式事件，支持多轮对话（工具调用循环）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from mcc_core import Usage

from mcc_llm import (
    CompleteRequest,
    MessageResponse,
    MessageStop,
    StreamEvent,
    TextDelta,
)


class MockLlm:
    """按预设脚本返回流式事件的 Mock LLM。

    每次 stream 调用消费脚本中的一轮事件列表。
    若脚本耗尽，返回空文本 + end_turn。
    """

    def __init__(self, turns: list[list[StreamEvent]] | None = None) -> None:
        self._turns = turns or []
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    async def complete(self, req: CompleteRequest) -> MessageResponse:
        events = self._next_turn()
        text_parts: list[str] = []
        stop_reason = "end_turn"
        usage = Usage(input_tokens=10, output_tokens=5)
        for ev in events:
            if isinstance(ev, TextDelta):
                text_parts.append(ev.text)
            elif isinstance(ev, MessageStop):
                stop_reason = ev.stop_reason
                usage = ev.usage
        from mcc_core import TextBlock

        return MessageResponse(
            content=[TextBlock(text="".join(text_parts))],
            stop_reason=stop_reason,
            usage=usage,
        )

    async def stream(self, req: CompleteRequest) -> AsyncIterator[StreamEvent]:
        for ev in self._next_turn():
            yield ev

    def _next_turn(self) -> list[StreamEvent]:
        if self._call_count < len(self._turns):
            turn = self._turns[self._call_count]
        else:
            turn = [
                TextDelta(text=""),
                MessageStop(stop_reason="end_turn", usage=Usage()),
            ]
        self._call_count += 1
        return turn


def single_text_turn(text: str, usage: Usage | None = None) -> list[StreamEvent]:
    """构造单轮纯文本响应。"""
    return [
        TextDelta(text=text),
        MessageStop(
            stop_reason="end_turn",
            usage=usage or Usage(input_tokens=10, output_tokens=5),
        ),
    ]
