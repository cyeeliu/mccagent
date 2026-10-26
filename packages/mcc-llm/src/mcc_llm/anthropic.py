"""Anthropic Messages API 客户端，支持非流式与 SSE 流式。"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator

import httpx
from mcc_core import ContentBlock, TextBlock, ToolUseBlock, Usage

from mcc_llm import (
    CompleteRequest,
    MessageResponse,
    MessageStop,
    StreamEvent,
    TextDelta,
    ToolUseInputDelta,
    ToolUseStart,
)


class AnthropicClient:
    """Anthropic Messages API 客户端。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        timeout: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(timeout=timeout)

    @classmethod
    def from_env(cls) -> AnthropicClient:
        api_key = os.environ["ANTHROPIC_API_KEY"]
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        return cls(api_key, base_url)

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _build_body(self, req: CompleteRequest, stream: bool) -> dict:
        body: dict = {
            "model": req.model,
            "max_tokens": req.max_tokens,
            "messages": [m.model_dump() for m in req.messages],
        }
        if req.system:
            if isinstance(req.system, str):
                body["system"] = req.system
            else:
                body["system"] = [
                    b.model_dump(by_alias=True, exclude_none=True)
                    for b in req.system
                ]
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.tools:
            body["tools"] = req.tools
        if stream:
            body["stream"] = True
        return body

    async def complete(self, req: CompleteRequest) -> MessageResponse:
        body = self._build_body(req, stream=False)
        resp = await self._http.post(
            f"{self._base_url}/v1/messages",
            headers=self._headers(),
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()
        return MessageResponse(
            content=_parse_content_blocks(data.get("content", [])),
            stop_reason=data.get("stop_reason"),
            usage=Usage(**data.get("usage", {})),
        )

    async def stream(self, req: CompleteRequest) -> AsyncIterator[StreamEvent]:
        body = self._build_body(req, stream=True)
        stop_reason = "end_turn"
        usage = Usage()

        async with self._http.stream(
            "POST",
            f"{self._base_url}/v1/messages",
            headers=self._headers(),
            json=body,
        ) as resp:
            resp.raise_for_status()

            event_type = ""
            data_parts: list[str] = []

            async for line in resp.aiter_lines():
                if line == "":
                    if event_type and data_parts:
                        raw = json.loads("".join(data_parts))
                        ev = _parse_sse(event_type, raw, usage)
                        if isinstance(ev, tuple):
                            stop_reason, usage = ev
                        elif ev is not None:
                            yield ev
                    event_type = ""
                    data_parts = []
                elif line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: "):
                    data_parts.append(line[6:])
                elif line.startswith("data:"):
                    data_parts.append(line[5:])

            if event_type and data_parts:
                raw = json.loads("".join(data_parts))
                ev = _parse_sse(event_type, raw, usage)
                if isinstance(ev, tuple):
                    stop_reason, usage = ev
                elif ev is not None:
                    yield ev

        yield MessageStop(stop_reason=stop_reason, usage=usage)

    async def close(self) -> None:
        await self._http.aclose()


def _parse_content_blocks(raw: list[dict]) -> list[ContentBlock]:
    blocks: list[ContentBlock] = []
    for item in raw:
        t = item.get("type")
        if t == "text":
            blocks.append(TextBlock(text=item.get("text", "")))
        elif t == "tool_use":
            blocks.append(
                ToolUseBlock(
                    id=item.get("id", ""),
                    name=item.get("name", ""),
                    input=item.get("input", {}),
                )
            )
    return blocks


def _parse_sse(
    event_type: str, data: dict, current_usage: Usage
) -> StreamEvent | tuple[str, Usage] | None:
    if event_type == "content_block_start":
        cb = data.get("content_block", {})
        if cb.get("type") == "tool_use":
            return ToolUseStart(id=cb.get("id", ""), name=cb.get("name", ""))
    elif event_type == "content_block_delta":
        delta = data.get("delta", {})
        dt = delta.get("type")
        if dt == "text_delta":
            text = delta.get("text", "")
            if text:
                return TextDelta(text=text)
        elif dt == "input_json_delta":
            return ToolUseInputDelta(partial_json=delta.get("partial_json", ""))
    elif event_type == "message_delta":
        sr = data.get("delta", {}).get("stop_reason")
        u_raw = data.get("usage", {})
        new_usage = Usage(**u_raw) if u_raw else current_usage
        if sr:
            return sr, new_usage
        return None
    elif event_type == "message_start":
        u_raw = data.get("message", {}).get("usage", {})
        if u_raw:
            return "end_turn", Usage(**u_raw)
    return None
