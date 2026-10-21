"""OpenAI 兼容客户端测试：流式 usage 解析（include_usage 尾 chunk）。

流式模式下用量只出现在最后一个 choices 为空的 chunk 里；不开
stream_options.include_usage 时网关不发这个 chunk，usage 恒为 0，
成本与 token 统计全部失真。
"""

from __future__ import annotations

import json

import httpx
import pytest
from mcc_core import Message, Role, TextBlock
from mcc_llm import CompleteRequest
from mcc_llm.openai import OpenAiClient


def _sse(lines: list[dict]) -> bytes:
    return "".join(f"data: {json.dumps(ln)}\n\n" for ln in lines) + "data: [DONE]\n\n"


def _client_with(transport: httpx.MockTransport) -> OpenAiClient:
    """注入 MockTransport 的客户端（替换内部 httpx.AsyncClient）。"""
    client = OpenAiClient(api_key="sk-test", base_url="http://gw:8000/v1")
    client._http = httpx.AsyncClient(transport=transport)
    return client


def _req() -> CompleteRequest:
    return CompleteRequest(
        model="qwen-test",
        max_tokens=100,
        messages=[Message(role=Role.user, content=[TextBlock(text="hi")])],
    )


@pytest.mark.asyncio
async def test_stream_parses_usage_tail_chunk() -> None:
    """usage-only 尾 chunk（choices 为空）被吸收进 MessageStop.usage。"""
    body: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body.update(json.loads(request.content.decode()))
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse([
                {"choices": [{"delta": {"content": "he"}, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {
                    "prompt_tokens": 1520, "completion_tokens": 310,
                }},
            ]),
        )

    client = _client_with(httpx.MockTransport(handler))
    events = [ev async for ev in client.stream(_req())]

    assert body["stream_options"] == {"include_usage": True}
    stop = events[-1]
    assert stop.usage.input_tokens == 1520
    assert stop.usage.output_tokens == 310


@pytest.mark.asyncio
async def test_stream_without_usage_chunk_keeps_zero() -> None:
    """网关不发 usage chunk（兼容旧网关）：usage 维持 0 不崩溃。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse([
                {"choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}]},
            ]),
        )

    client = _client_with(httpx.MockTransport(handler))
    events = [ev async for ev in client.stream(_req())]

    assert events[-1].usage.input_tokens == 0


@pytest.mark.asyncio
async def test_stream_tool_calls_with_usage() -> None:
    """工具调用流的 usage 同样在尾 chunk——文本与工具混合不丢用量。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse([
                {"choices": [{"delta": {
                    "tool_calls": [{"index": 0, "id": "call_1",
                                    "function": {"name": "run_bash",
                                                 "arguments": "{\"command\":"}}],
                }, "finish_reason": None}]},
                {"choices": [{"delta": {
                    "tool_calls": [{"index": 0,
                                    "function": {"arguments": "\"ls\"}"}}],
                }, "finish_reason": "tool_calls"}]},
                {"choices": [], "usage": {
                    "prompt_tokens": 900, "completion_tokens": 80,
                }},
            ]),
        )

    client = _client_with(httpx.MockTransport(handler))
    events = [ev async for ev in client.stream(_req())]

    assert events[-1].usage.input_tokens == 900
    assert events[-1].usage.output_tokens == 80
