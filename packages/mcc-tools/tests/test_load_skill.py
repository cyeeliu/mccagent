"""load_skill 工具测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from mcc_core import ToolContext
from mcc_skills import Skill, SkillManifest
from mcc_tools import LoadSkillTool


@pytest.fixture
def skills() -> dict[str, Skill]:
    return {
        "x": Skill(SkillManifest(name="x", description="d"), "full content", Path()),
    }


@pytest.fixture
def ctx(tmp_path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="s")


class TestLoadSkillTool:
    async def test_load_hit(self, skills, ctx):
        out = await LoadSkillTool(skills).execute({"name": "x"}, ctx)
        assert not out.is_error
        assert out.content == "full content"

    async def test_load_miss_lists_available(self, skills, ctx):
        out = await LoadSkillTool(skills).execute({"name": "nope"}, ctx)
        assert out.is_error
        assert "Unknown skill" in out.content
        assert "x" in out.content

    async def test_empty_name_with_empty_registry(self, ctx):
        out = await LoadSkillTool({}).execute({"name": ""}, ctx)
        assert out.is_error
        assert "none" in out.content

    def test_name_and_schema(self, skills):
        tool = LoadSkillTool(skills)
        assert tool.name == "load_skill"
        assert "skill" in tool.description.lower()
        schema = tool.input_schema()
        assert schema["properties"]["name"]["type"] == "string"
        assert "name" in schema["required"]
