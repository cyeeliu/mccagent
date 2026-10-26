"""Prompt caching：system prompt 以 ContentBlock 数组透传 cache_control。

Anthropic 端保留 cache_control 标记（by_alias 序列化为 {"type": "ephemeral"}）；
OpenAI 兼容端不认 cache_control，剥离成纯文本拼接。
"""

from __future__ import annotations

from mcc_core import CacheControl, Message, Role, TextBlock
from mcc_llm import CompleteRequest
from mcc_llm.anthropic import AnthropicClient
from mcc_llm.openai import OpenAiClient


def _req(system) -> CompleteRequest:
    return CompleteRequest(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[Message(role=Role.user, content=[TextBlock(text="hi")])],
        system=system,
    )


# ---------- Anthropic：透传 cache_control ----------


def test_anthropic_system_blocks_preserve_cache_control() -> None:
    client = AnthropicClient(api_key="sk-test")
    req = _req([
        TextBlock(
            text="You are helpful.",
            cache_control=CacheControl(kind="ephemeral"),
        ),
    ])
    body = client._build_body(req, stream=False)
    assert body["system"] == [
        {
            "type": "text",
            "text": "You are helpful.",
            "cache_control": {"type": "ephemeral"},
        },
    ]


def test_anthropic_system_blocks_multiple() -> None:
    client = AnthropicClient(api_key="sk-test")
    req = _req([
        TextBlock(text="base prompt"),
        TextBlock(
            text="cached tail",
            cache_control=CacheControl(kind="ephemeral"),
        ),
    ])
    body = client._build_body(req, stream=False)
    assert len(body["system"]) == 2
    assert body["system"][0] == {"type": "text", "text": "base prompt"}
    assert body["system"][1]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_system_str_still_works() -> None:
    client = AnthropicClient(api_key="sk-test")
    body = client._build_body(_req("plain string"), stream=False)
    assert body["system"] == "plain string"


# ---------- OpenAI：剥离 cache_control ----------


def test_openai_system_blocks_strip_cache_control() -> None:
    client = OpenAiClient(api_key="sk-test")
    req = _req([
        TextBlock(
            text="Part 1",
            cache_control=CacheControl(kind="ephemeral"),
        ),
        TextBlock(text="Part 2"),
    ])
    body = client._build_body(req, stream=False)
    assert body["messages"][0] == {
        "role": "system",
        "content": "Part 1\n\nPart 2",
    }


def test_openai_system_str_still_works() -> None:
    client = OpenAiClient(api_key="sk-test")
    body = client._build_body(_req("plain string"), stream=False)
    assert body["messages"][0] == {"role": "system", "content": "plain string"}
