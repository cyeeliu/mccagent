"""OpenAI Chat Completions 兼容客户端。

支持 DeepSeek / Kimi / Qwen / Groq / vLLM / OpenRouter 等端点，
通过 OPENAI_BASE_URL 环境变量指定。
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

import httpx
from mcc_core import ContentBlock, Message, TextBlock, ToolUseBlock, Usage

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
            messages.append({"role": m.role.value, "content": _msg_to_text(m)})

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
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta", {})

                if "content" in delta and delta["content"]:
                    yield TextDelta(text=delta["content"])
                if "tool_calls" in delta:
                    for tc in delta["tool_calls"]:
                        if "function" in tc:
                            fn = tc["function"]
                            if "name" in fn:
                                yield ToolUseStart(
                                    id=tc.get("id", ""),
                                    name=fn.get("name", ""),
                                )
                            if "arguments" in fn and fn["arguments"]:
                                yield ToolUseInputDelta(partial_json=fn["arguments"])
                sr = _map_stop_reason(choice.get("finish_reason"))
                if sr:
                    stop_reason = sr

        yield MessageStop(stop_reason=stop_reason, usage=usage)

    async def close(self) -> None:
        await self._http.aclose()


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
