"""load_skill 工具：按名称加载完整 SKILL.md 内容。

懒加载的核心——system prompt 只有轻量 catalog（name + description），
模型按需调用本工具读取完整技能指令。返回的 SKILL.md 作为 tool_result
进入上下文，未激活的技能不占 token。
"""

from __future__ import annotations

from typing import Any

from mcc_core import ToolContext, ToolOutput
from mcc_skills import Skill


class LoadSkillTool:
    """按名称加载技能的完整 SKILL.md。"""

    def __init__(self, skills: dict[str, Skill]) -> None:
        self._skills = skills

    @property
    def name(self) -> str:
        return "load_skill"

    @property
    def description(self) -> str:
        return (
            "Load the full SKILL.md content for a named skill. "
            "The system prompt lists available skills by name and "
            "description; call this to read the complete instructions "
            "when a skill applies to the current task."
        )

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name (from the available skills catalog)",
                }
            },
            "required": ["name"],
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        name = str(input.get("name", "")).strip()
        skill = self._skills.get(name)
        if skill is None:
            available = ", ".join(self._skills) or "none"
            return ToolOutput.err(f"Unknown skill '{name}'. Available: {available}")
        return ToolOutput.ok(skill.content)
