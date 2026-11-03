"""模型思考 thinking/reasoning 全链路测试。

覆盖 6 层：
- mcc-llm: Anthropic thinking_delta 解析 + OpenAI reasoning_content 解析
- mcc-core: ThinkingBlock 序列化 + ContentBlock 联合类型
- mcc-harness: AgentLoop 转发 thinking 事件 + 组装 ThinkingBlock
- mcc-tui: apply_event 累积 streaming_thinking + flush 顺序
- mcc-cli: history_events 回显 ThinkingBlock → ThinkingDelta
- mcc-config: ThinkingConfig 默认值 + Config 集成
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from mcc_core import (
    Message,
    Role,
    TextBlock,
    ThinkingBlock,
    ToolContext,
    ToolUseBlock,
    Usage,
)
from mcc_core.event import (
    TextDelta,
    ThinkingDelta,
    TurnEnd,
    UserEcho,
)
from mcc_harness import AgentLoopBuilder
from mcc_llm import (
    CompleteRequest,
    MessageStop,
)
from mcc_llm import (
    TextDelta as LlmTextDelta,
)
from mcc_llm import (
    ThinkingDelta as LlmThinkingDelta,
)
from mcc_llm.anthropic import AnthropicClient, _parse_content_blocks, _parse_sse
from mcc_llm.mock import MockLlm
from mcc_llm.openai import OpenAiClient, _parse_oai_content
from mcc_tools import ToolRegistry, default_registry
from mcc_tui.app import App

# ========== mcc-core: ThinkingBlock ==========


class TestThinkingBlock:
    def test_serialization(self) -> None:
        block = ThinkingBlock(text="reasoning here")
        dumped = block.model_dump()
        assert dumped == {"type": "thinking", "text": "reasoning here"}

    def test_roundtrip(self) -> None:
        block = ThinkingBlock(text="step by step")
        restored = ThinkingBlock.model_validate(block.model_dump())
        assert restored == block

    def test_in_content_block_union(self) -> None:
        from mcc_core import ContentBlock

        block: ContentBlock = ThinkingBlock(text="ok")
        assert isinstance(block, ThinkingBlock)

    def test_message_with_thinking(self) -> None:
        msg = Message(
            role=Role.assistant,
            content=[
                ThinkingBlock(text="let me think..."),
                TextBlock(text="answer"),
            ],
        )
        dumped = msg.model_dump()
        assert dumped["content"][0]["type"] == "thinking"
        assert dumped["content"][1]["type"] == "text"


# ========== mcc-llm Anthropic: thinking_delta / thinking block ==========


class TestAnthropicThinking:
    def test_parse_sse_thinking_delta(self) -> None:
        result = _parse_sse(
            "content_block_delta",
            {"delta": {"type": "thinking_delta", "thinking": "reasoning"}},
            Usage(),
        )
        assert isinstance(result, LlmThinkingDelta)
        assert result.text == "reasoning"

    def test_parse_sse_thinking_delta_empty(self) -> None:
        result = _parse_sse(
            "content_block_delta",
            {"delta": {"type": "thinking_delta", "thinking": ""}},
            Usage(),
        )
        assert result is None

    def test_parse_content_blocks_thinking(self) -> None:
        blocks = _parse_content_blocks([
            {"type": "thinking", "thinking": "thought process"},
            {"type": "text", "text": "answer"},
        ])
        assert len(blocks) == 2
        assert isinstance(blocks[0], ThinkingBlock)
        assert blocks[0].text == "thought process"
        assert isinstance(blocks[1], TextBlock)

    def test_build_body_thinking_passthrough(self) -> None:
        client = AnthropicClient(api_key="sk-test")
        req = CompleteRequest(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[Message(role=Role.user, content=[TextBlock(text="hi")])],
            thinking={"type": "enabled", "budget_tokens": 10000},
        )
        body = client._build_body(req, stream=False)
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 10000}

    def test_build_body_no_thinking(self) -> None:
        client = AnthropicClient(api_key="sk-test")
        req = CompleteRequest(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[Message(role=Role.user, content=[TextBlock(text="hi")])],
        )
        body = client._build_body(req, stream=False)
        assert "thinking" not in body

    @pytest.mark.asyncio
    async def test_stream_emits_thinking_delta(self) -> None:
        sse = (
            "event: content_block_delta\n"
            'data: {"delta": {"type": "thinking_delta", "thinking": "let me think"}}\n\n'
            "event: content_block_delta\n"
            'data: {"delta": {"type": "text_delta", "text": "answer"}}\n\n'
            "event: message_stop\n"
            'data: {}\n\n'
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=sse,
            )

        client = AnthropicClient(api_key="sk-test")
        client._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        req = CompleteRequest(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[Message(role=Role.user, content=[TextBlock(text="hi")])],
        )
        events = [ev async for ev in client.stream(req)]
        thinking_events = [e for e in events if isinstance(e, LlmThinkingDelta)]
        text_events = [e for e in events if isinstance(e, LlmTextDelta)]
        assert len(thinking_events) == 1
        assert thinking_events[0].text == "let me think"
        assert len(text_events) == 1
        assert text_events[0].text == "answer"


# ========== mcc-llm OpenAI: reasoning_content ==========


def _sse(lines: list[dict]) -> bytes:
    return "".join(f"data: {json.dumps(ln)}\n\n" for ln in lines) + "data: [DONE]\n\n"


def _oai_client_with(transport: httpx.MockTransport) -> OpenAiClient:
    client = OpenAiClient(api_key="sk-test", base_url="http://gw:8000/v1")
    client._http = httpx.AsyncClient(transport=transport)
    return client


def _oai_req() -> CompleteRequest:
    return CompleteRequest(
        model="deepseek-r1",
        max_tokens=100,
        messages=[Message(role=Role.user, content=[TextBlock(text="hi")])],
    )


class TestOpenAiReasoning:
    def test_parse_oai_content_reasoning(self) -> None:
        blocks = _parse_oai_content({
            "reasoning_content": "step by step",
            "content": "final answer",
        })
        assert len(blocks) == 2
        assert isinstance(blocks[0], ThinkingBlock)
        assert blocks[0].text == "step by step"
        assert isinstance(blocks[1], TextBlock)

    def test_parse_oai_content_no_reasoning(self) -> None:
        blocks = _parse_oai_content({"content": "just text"})
        assert len(blocks) == 1
        assert isinstance(blocks[0], TextBlock)

    @pytest.mark.asyncio
    async def test_stream_emits_thinking_delta(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_sse([
                    {"choices": [{"delta": {
                        "reasoning_content": "thinking...",
                    }, "finish_reason": None}]},
                    {"choices": [{"delta": {
                        "content": "answer",
                    }, "finish_reason": "stop"}]},
                ]),
            )

        client = _oai_client_with(httpx.MockTransport(handler))
        events = [ev async for ev in client.stream(_oai_req())]
        thinking_events = [e for e in events if isinstance(e, LlmThinkingDelta)]
        text_events = [e for e in events if isinstance(e, LlmTextDelta)]
        assert len(thinking_events) == 1
        assert thinking_events[0].text == "thinking..."
        assert len(text_events) == 1
        assert text_events[0].text == "answer"

    @pytest.mark.asyncio
    async def test_stream_reasoning_and_tool_calls(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_sse([
                    {"choices": [{"delta": {
                        "reasoning_content": "need to call tool",
                    }, "finish_reason": None}]},
                    {"choices": [{"delta": {
                        "tool_calls": [{"index": 0, "id": "c1",
                                        "function": {"name": "run_bash",
                                                     "arguments": '{"command":"ls"}'}}],
                    }, "finish_reason": "tool_calls"}]},
                ]),
            )

        client = _oai_client_with(httpx.MockTransport(handler))
        events = [ev async for ev in client.stream(_oai_req())]
        assert any(isinstance(e, LlmThinkingDelta) for e in events)
        from mcc_llm import ToolUseStart

        assert any(isinstance(e, ToolUseStart) for e in events)


# ========== mcc-harness: AgentLoop thinking 转发 ==========


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test-thinking", depth=0)


@pytest.fixture
def registry() -> ToolRegistry:
    return default_registry()


def _thinking_turn(thinking: str, text: str) -> list:
    return [
        LlmThinkingDelta(text=thinking),
        LlmTextDelta(text=text),
        MessageStop(stop_reason="end_turn", usage=Usage(input_tokens=10, output_tokens=5)),
    ]


class TestAgentLoopThinking:
    @pytest.mark.asyncio
    async def test_thinking_event_emitted(self, ctx, registry) -> None:
        llm = MockLlm([_thinking_turn("let me think...", "answer")])
        agent = AgentLoopBuilder(llm, registry, ctx).model("test-model").build()
        queue: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("hi", queue)

        events = []
        while not queue.empty():
            events.append(queue.get_nowait())
        thinking_events = [e for e in events if isinstance(e, ThinkingDelta)]
        assert len(thinking_events) == 1
        assert thinking_events[0].text == "let me think..."

    @pytest.mark.asyncio
    async def test_thinking_block_in_assistant_message(self, ctx, registry) -> None:
        llm = MockLlm([_thinking_turn("reasoning", "response")])
        agent = AgentLoopBuilder(llm, registry, ctx).model("test-model").build()
        queue: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("hi", queue)

        assert len(agent.history) >= 2
        assistant_msg = agent.history[1]
        assert assistant_msg.role == Role.assistant
        thinking_blocks = [
            b for b in assistant_msg.content if isinstance(b, ThinkingBlock)
        ]
        text_blocks = [
            b for b in assistant_msg.content if isinstance(b, TextBlock)
        ]
        assert len(thinking_blocks) == 1
        assert thinking_blocks[0].text == "reasoning"
        assert len(text_blocks) == 1
        assert text_blocks[0].text == "response"

    @pytest.mark.asyncio
    async def test_thinking_block_before_text_block(self, ctx, registry) -> None:
        llm = MockLlm([_thinking_turn("think", "act")])
        agent = AgentLoopBuilder(llm, registry, ctx).model("test-model").build()
        queue: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("hi", queue)

        assistant_msg = agent.history[1]
        types_in_order = [type(b).__name__ for b in assistant_msg.content]
        thinking_idx = types_in_order.index("ThinkingBlock")
        text_idx = types_in_order.index("TextBlock")
        assert thinking_idx < text_idx

    @pytest.mark.asyncio
    async def test_no_thinking_no_block(self, ctx, registry) -> None:
        from mcc_llm.mock import single_text_turn

        llm = MockLlm([single_text_turn("just text")])
        agent = AgentLoopBuilder(llm, registry, ctx).model("test-model").build()
        queue: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("hi", queue)

        assistant_msg = agent.history[1]
        assert not any(
            isinstance(b, ThinkingBlock) for b in assistant_msg.content
        )

    @pytest.mark.asyncio
    async def test_thinking_config_passthrough(self, ctx, registry) -> None:
        captured: list[CompleteRequest] = []

        class CapturingLlm(MockLlm):
            async def stream(self, req: CompleteRequest):
                captured.append(req)
                async for ev in super().stream(req):
                    yield ev

        llm = CapturingLlm([_thinking_turn("t", "a")])
        agent = (
            AgentLoopBuilder(llm, registry, ctx)
            .model("test-model")
            .thinking({"type": "enabled", "budget_tokens": 5000})
            .build()
        )
        queue: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("hi", queue)
        assert captured[0].thinking == {"type": "enabled", "budget_tokens": 5000}


# ========== mcc-tui: streaming_thinking ==========


class TestTuiThinking:
    def test_accumulates_thinking_deltas(self) -> None:
        app = App(session_short_id="test")
        app.apply_event(ThinkingDelta(text="let me "))
        app.apply_event(ThinkingDelta(text="think..."))
        assert app.streaming_thinking == "let me think..."

    def test_thinking_and_text_coexist(self) -> None:
        app = App(session_short_id="test")
        app.apply_event(ThinkingDelta(text="reasoning"))
        app.apply_event(TextDelta(text="answer"))
        assert app.streaming_thinking == "reasoning"
        assert app.streaming_text == "answer"

    def test_flush_thinking_before_text(self) -> None:
        app = App(session_short_id="test")
        app.apply_event(ThinkingDelta(text="thought"))
        app.apply_event(TextDelta(text="response"))
        app.flush_streaming()
        items = list(app.items)
        assert items[-2].kind == "thinking"
        assert items[-2].text == "thought"
        assert items[-1].kind == "assistant"
        assert items[-1].text == "response"

    def test_flush_only_thinking(self) -> None:
        app = App(session_short_id="test")
        app.apply_event(ThinkingDelta(text="just thinking"))
        app.flush_streaming()
        assert app.streaming_thinking == ""
        assert app.items[-1].kind == "thinking"

    def test_flush_only_text(self) -> None:
        app = App(session_short_id="test")
        app.apply_event(TextDelta(text="just text"))
        app.flush_streaming()
        assert app.streaming_text == ""
        assert app.items[-1].kind == "assistant"

    def test_est_stream_tokens_includes_thinking(self) -> None:
        app = App(session_short_id="test")
        app.apply_event(ThinkingDelta(text="x" * 40))
        app.apply_event(TextDelta(text="y" * 40))
        assert app.est_stream_tokens == 20

    def test_turn_end_flushes_thinking(self) -> None:
        app = App(session_short_id="test")
        app.agent_busy = True
        app.apply_event(ThinkingDelta(text="reasoning"))
        app.apply_event(TextDelta(text="answer"))
        app.apply_event(TurnEnd(cost_usd=0.01))
        kinds = [it.kind for it in app.items]
        assert "thinking" in kinds
        assert "assistant" in kinds
        assert kinds.index("thinking") < kinds.index("assistant")


# ========== mcc-cli: history_events 回显 ==========


class TestHistoryEventsThinking:
    def test_thinking_block_replayed_as_delta(self) -> None:
        from mcc_cli.events import history_events

        history = [
            Message.user("hi"),
            Message(
                role=Role.assistant,
                content=[
                    ThinkingBlock(text="I should consider..."),
                    TextBlock(text="Here is my answer"),
                ],
            ),
        ]
        events = history_events(history)
        thinking_events = [e for e in events if isinstance(e, ThinkingDelta)]
        text_events = [e for e in events if isinstance(e, TextDelta)]
        assert len(thinking_events) == 1
        assert thinking_events[0].text == "I should consider..."
        assert len(text_events) == 1

    def test_thinking_before_text_in_replay(self) -> None:
        from mcc_cli.events import history_events

        history = [
            Message(
                role=Role.assistant,
                content=[
                    ThinkingBlock(text="thought"),
                    TextBlock(text="answer"),
                ],
            ),
        ]
        events = history_events(history)
        types_in_order = [
            type(e).__name__ for e in events
            if isinstance(e, (ThinkingDelta, TextDelta))
        ]
        assert types_in_order == ["ThinkingDelta", "TextDelta"]

    def test_no_thinking_no_replay(self) -> None:
        from mcc_cli.events import history_events

        history = [
            Message(
                role=Role.assistant,
                content=[TextBlock(text="just text")],
            ),
        ]
        events = history_events(history)
        assert not any(isinstance(e, ThinkingDelta) for e in events)


# ========== mcc-config: ThinkingConfig ==========


class TestThinkingConfig:
    def test_default_disabled(self) -> None:
        from mcc_config.models import ThinkingConfig

        cfg = ThinkingConfig()
        assert cfg.enabled is False
        assert cfg.budget_tokens == 10000

    def test_config_has_thinking(self) -> None:
        from mcc_config import Config

        cfg = Config()
        assert hasattr(cfg, "thinking")
        assert cfg.thinking.enabled is False

    def test_thinking_enabled(self) -> None:
        from mcc_config.models import ThinkingConfig

        cfg = ThinkingConfig(enabled=True, budget_tokens=20000)
        assert cfg.enabled is True
        assert cfg.budget_tokens == 20000
