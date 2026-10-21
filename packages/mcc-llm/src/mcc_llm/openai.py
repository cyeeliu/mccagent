"""OpenAI Chat Completions 兼容客户端。

支持 DeepSeek / Kimi / Qwen / Groq / vLLM / OpenRouter 等端点，
通过 OPENAI_BASE_URL 环境变量指定。
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

import httpx
from mcc_core import ContentBlock, Message, Role, TextBlock, ToolResultBlock, ToolUseBlock, Usage

from mcc_llm import (
    CompleteRequest,
    MessageResponse,
    MessageStop,
    StreamEvent,
    TextDelta,
    ToolUseInputDelta,
    ToolUseStart,
)


def _map_stop_reason(sr: str | None) -> str | None:
    """将 OpenAI finish_reason 映射为 Anthropic 兼容的 stop_reason。"""
    return {
        "stop": "end_turn",
        "tool_calls": "tool_use",
        "length": "max_tokens",
    }.get(sr, sr)


class OpenAiClient:
    """OpenAI Chat Completions 兼容客户端。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=timeout)

    @classmethod
    def from_env(cls) -> OpenAiClient:
        api_key = os.environ["OPENAI_API_KEY"]
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return cls(api_key, base_url)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "content-type": "application/json",
        }

    def _build_body(self, req: CompleteRequest, stream: bool) -> dict:
        messages: list[dict] = []
        if req.system:
            messages.append({"role": "system", "content": req.system})
        for m in req.messages:
            converted = _message_to_oai(m)
            if isinstance(converted, list):
                messages.extend(converted)
            else:
                messages.append(converted)

        body: dict = {
            "model": req.model,
            "max_tokens": req.max_tokens,
            "messages": messages,
        }
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.tools:
            body["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {}),
                    },
                }
                for t in req.tools
            ]
        if stream:
            body["stream"] = True
            # 请求流尾 usage chunk（OpenAI 兼容网关约定：最后一个
            # chunk choices 为空、携带 usage）。不开的话流式用量
            # 恒为 0，成本与 token 统计全部失真
            body["stream_options"] = {"include_usage": True}
        return body

    async def complete(self, req: CompleteRequest) -> MessageResponse:
        body = self._build_body(req, stream=False)
        resp = await self._http.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers(),
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        choice = choices[0] if choices else {}
        msg = choice.get("message", {})
        blocks = _parse_oai_content(msg)
        usage_raw = data.get("usage", {})
        usage = Usage(
            input_tokens=usage_raw.get("prompt_tokens", 0),
            output_tokens=usage_raw.get("completion_tokens", 0),
        )
        return MessageResponse(
            content=blocks,
            stop_reason=_map_stop_reason(choice.get("finish_reason")),
            usage=usage,
        )

    async def stream(self, req: CompleteRequest) -> AsyncIterator[StreamEvent]:
        body = self._build_body(req, stream=True)
        stop_reason = "end_turn"
        usage = Usage()
        tool_call_started: set[int] = set()

        async with self._http.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            headers=self._headers(),
            json=body,
        ) as resp:
            resp.raise_for_status()

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                # usage-only 尾 chunk（include_usage）：choices 为空、
                # 携带最终用量——先吸收再跳过 choice 解析
                usage_raw = chunk.get("usage")
                if usage_raw:
                    usage = Usage(
                        input_tokens=usage_raw.get("prompt_tokens", 0),
                        output_tokens=usage_raw.get("completion_tokens", 0),
                    )
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta", {})

                if "content" in delta and delta["content"]:
                    yield TextDelta(text=delta["content"])
                if "tool_calls" in delta:
                    for tc in delta["tool_calls"]:
                        idx = tc.get("index", 0)
                        fn = tc.get("function", {})
                        name = fn.get("name", "")
                        if name and idx not in tool_call_started:
                            tool_call_started.add(idx)
                            yield ToolUseStart(
                                id=tc.get("id", ""),
                                name=name,
                            )
                        args = fn.get("arguments", "")
                        if args:
                            yield ToolUseInputDelta(partial_json=args)
                sr = _map_stop_reason(choice.get("finish_reason"))
                if sr:
                    stop_reason = sr

        yield MessageStop(stop_reason=stop_reason, usage=usage)

    async def close(self) -> None:
        await self._http.aclose()


def _message_to_oai(m: Message) -> dict | list[dict]:
    """把 Message 转为 OpenAI Chat Completions 消息格式。

    - assistant 中的 ToolUseBlock 转为 tool_calls 字段
    - tool 角色的 ToolResultBlock 转为 role=tool 消息
    - 纯文本保持 content 字段
    """
    if m.role == Role.assistant:
        text_parts = [b.text for b in m.content if isinstance(b, TextBlock)]
        tool_calls = [
            {
                "id": b.id,
                "type": "function",
                "function": {
                    "name": b.name,
                    "arguments": json.dumps(b.input),
                },
            }
            for b in m.content
            if isinstance(b, ToolUseBlock)
        ]
        msg: dict = {"role": "assistant"}
        if text_parts:
            msg["content"] = "\n".join(text_parts)
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg

    if m.role == Role.tool:
        return [
            {
                "role": "tool",
                "tool_call_id": b.tool_use_id,
                "content": b.content,
            }
            for b in m.content
            if isinstance(b, ToolResultBlock)
        ]

    return {"role": m.role.value, "content": _msg_to_text(m)}


def _msg_to_text(m: Message) -> str:
    parts = []
    for b in m.content:
        if isinstance(b, TextBlock):
            parts.append(b.text)
        elif isinstance(b, ToolUseBlock):
            parts.append(json.dumps(b.input))
    return "\n".join(parts)


def _parse_oai_content(msg: dict) -> list[ContentBlock]:
    blocks: list[ContentBlock] = []
    if msg.get("content"):
        blocks.append(TextBlock(text=msg["content"]))
    for tc in msg.get("tool_calls", []):
        fn = tc.get("function", {})
        try:
            inp = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            inp = {}
        blocks.append(
            ToolUseBlock(
                id=tc.get("id", ""),
                name=fn.get("name", ""),
                input=inp,
            )
        )
    return blocks
