"""Skill 路由：catalog 注入 system prompt + slash 触发匹配。"""

from __future__ import annotations

from mcc_skills.types import Skill


class SkillRouter:
    """根据已加载的 skills 生成 catalog 与触发匹配。"""

    def __init__(self, skills: dict[str, Skill]) -> None:
        self.skills = skills

    def catalog_for_prompt(self) -> str:
        """生成轻量 catalog：仅 name + description，供 system prompt 注入。"""
        if not self.skills:
            return ""
        lines = ["# Available Skills"]
        for sk in self.skills.values():
            lines.append(f"- **{sk.manifest.name}** — {sk.manifest.description}")
        lines.append(
            "Use the load_skill tool to read the full instructions when a skill applies."
        )
        return "\n".join(lines)

    def match_slash(self, user_input: str) -> Skill | None:
        """字面匹配 triggers（如 ``/review``）。"""
        parts = user_input.strip().split(None, 1)
        if not parts:
            return None
        cmd = parts[0]
        for sk in self.skills.values():
            if cmd in sk.manifest.triggers:
                return sk
        return None
