"""Skill 与 Slash Command 加载。

懒加载设计：system prompt 只注入轻量 catalog（name + description），
模型自主调用 ``load_skill`` 工具按需读取完整 ``SKILL.md``。
"""

from __future__ import annotations

from mcc_skills.loader import (
    SkillLoader,
    SlashRegistry,
    load_commands_merged,
    load_skills_merged,
    parse_frontmatter,
    resolve_slash,
)
from mcc_skills.router import SkillRouter
from mcc_skills.types import Skill, SkillManifest, SlashCommand

__all__ = [
    "SlashCommand",
    "SlashRegistry",
    "Skill",
    "SkillLoader",
    "SkillManifest",
    "SkillRouter",
    "load_commands_merged",
    "load_skills_merged",
    "parse_frontmatter",
    "resolve_slash",
]
__version__ = "0.1.0"
