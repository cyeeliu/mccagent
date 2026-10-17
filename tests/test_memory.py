"""长期记忆测试：Markdown 存储、Agent 工具与权限放行。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from mcc_config import PermissionConfig
from mcc_core import ToolContext, ToolUseBlock
from mcc_core.event import PermissionRequest as PermissionRequestEvent
from mcc_harness import AgentLoopBuilder, PermissionChecker
from mcc_llm import MessageStop, ToolUseInputDelta, ToolUseStart
from mcc_llm.mock import MockLlm, single_text_turn
from mcc_session import (
    MarkdownMemoryStore,
    ReadMemoryTool,
    SaveMemoryTool,
    memory_tools,
)
from mcc_tools import default_registry


@pytest.fixture
def store(tmp_path: Path) -> MarkdownMemoryStore:
    return MarkdownMemoryStore(base_dir=tmp_path / "memory")


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="mem-test")


class TestMarkdownMemoryStore:
    def test_save_creates_entry_and_index(self, store):
        path = store.save(
            "user_role",
            "user",
            "用户是后端工程师",
            "用户主要写 Python，熟悉 asyncio。",
            tags=["python"],
        )
        assert path.exists()
        assert "用户主要写 Python" in path.read_text(encoding="utf-8")
        assert "user_role" in store.index_text()
        assert "entries/user_role.md" in store.index_text()

    def test_save_same_name_replaces_index_line(self, store):
        store.save("a", "user", "第一版描述", "body v1")
        store.save("a", "user", "第二版描述", "body v2")
        index = store.index_text()
        assert index.count("- [a] ") == 1
        assert "第二版描述" in index

    def test_read_roundtrip(self, store):
        store.save("pref", "feedback", "描述", "正文内容")
        content = store.read("pref")
        assert "正文内容" in content
        assert "description: 描述" in content

    def test_read_missing_raises(self, store):
        with pytest.raises(KeyError):
            store.read("ghost")

    @pytest.mark.parametrize(
        "bad_name",
        ["../escape", "UPPER", "with space", "", "a" * 100, "中文"],
    )
    def test_invalid_name_rejected(self, store, bad_name):
        with pytest.raises(ValueError):
            store.save(bad_name, "user", "d", "b")
        with pytest.raises(ValueError):
            store.read(bad_name)

    def test_invalid_type_rejected(self, store):
        with pytest.raises(ValueError, match="invalid memory type"):
            store.save("ok_name", "bogus", "d", "b")

    def test_index_text_empty_when_missing(self, store):
        assert store.index_text() == ""

    def test_list_names(self, store):
        store.save("one", "user", "d1", "b1")
        store.save("two", "project", "d2", "b2")
        assert store.list_names() == ["one", "two"]


class TestMemoryTools:
    async def test_save_tool_writes_entry(self, store, ctx):
        tool = SaveMemoryTool(store)
        out = await tool.execute(
            {
                "name": "user_pref",
                "type": "user",
                "description": "偏好",
                "body": "喜欢简洁回复",
            },
            ctx,
        )
        assert out.is_error is False
        assert "user_pref" in store.list_names()

    async def test_save_tool_invalid_input_errors(self, store, ctx):
        tool = SaveMemoryTool(store)
        out = await tool.execute(
            {"name": "BAD NAME", "type": "user", "description": "d", "body": "b"},
            ctx,
        )
        assert out.is_error is True
        assert "save_memory failed" in out.content

    async def test_read_tool_returns_body(self, store, ctx):
        store.save("fact", "project", "d", "重要事实")
        tool = ReadMemoryTool(store)
        out = await tool.execute({"name": "fact"}, ctx)
        assert out.is_error is False
        assert "重要事实" in out.content

    async def test_read_tool_missing_errors(self, store, ctx):
        tool = ReadMemoryTool(store)
        out = await tool.execute({"name": "ghost"}, ctx)
        assert out.is_error is True

    def test_memory_tools_pair(self, store):
        tools = memory_tools(store)
        assert [t.name for t in tools] == ["save_memory", "read_memory"]
        schema = tools[0].input_schema()
        assert set(schema["required"]) == {"name", "type", "description", "body"}
        assert "enum" in str(schema["properties"]["type"])


class TestMemoryPermission:
    def test_memory_tools_allowed_in_default_mode(self):
        """default 模式下记忆工具免确认（无 Ask、无 PermissionRequest 事件）。"""
        from mcc_harness.agent import permission_request_for
        from mcc_harness.permission import Allow

        checker = PermissionChecker(PermissionConfig(mode="default"))
        for tool_name in ("save_memory", "read_memory"):
            req = permission_request_for(tool_name, {"name": "x"})
            assert req.category == "Memory"
            assert isinstance(checker.check(req), Allow)

    async def test_save_memory_via_agent_no_ask(self, store, ctx, tmp_path: Path):
        registry = default_registry()
        for t in memory_tools(store):
            registry.register(t)
        checker = PermissionChecker(PermissionConfig(mode="default"))

        evq: asyncio.Queue = asyncio.Queue()
        llm = MockLlm([
            [
                ToolUseStart(id="m1", name="save_memory"),
                ToolUseInputDelta(partial_json=(
                    '{"name": "agent_saved", "type": "user", '
                    '"description": "agent 记录", "body": "自动保存"}'
                )),
                MessageStop(stop_reason="tool_use"),
            ],
            single_text_turn("saved"),
            single_text_turn("标题"),
        ])
        agent = (
            AgentLoopBuilder(llm, registry, ctx)
            .permission(checker)
            .build()
        )
        await agent.run_streaming("记住我的偏好", evq)

        events = []
        while not evq.empty():
            events.append(evq.get_nowait())
        assert not any(isinstance(e, PermissionRequestEvent) for e in events)
        assert "agent_saved" in store.list_names()


class TestSystemPromptInjection:
    def test_build_source_injects_index_and_tools(self, tmp_path, monkeypatch):
        """索引存在时注入 system prompt 并注册记忆工具。"""
        from mcc_cli import _build_source

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        home_store = MarkdownMemoryStore()
        home_store.save("tip", "user", "一条提示", "记得用 uv")

        source = _build_source(tmp_path, None, True, None)
        assert "# Memory Index" in source.agent.system
        assert "tip" in source.agent.system
        names = source.agent.registry.names()
        assert "save_memory" in names and "read_memory" in names

    def test_build_source_without_index_keeps_system(self, tmp_path, monkeypatch):
        from mcc_cli import _build_source

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        source = _build_source(tmp_path, None, True, None)
        assert "# Memory Index" not in source.agent.system
        assert "save_memory" in source.agent.registry.names()
