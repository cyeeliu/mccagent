"""Skill 与 Slash Command 的数据类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SkillManifest:
    """SKILL.md frontmatter 元数据。

    - ``triggers``：字面触发词（如 ``/review``），用于 slash 匹配
    - ``model``：激活时覆盖当前模型（可选）
    - ``requires_tools``：声明所需工具（写入 instructions 供模型自律）
    """

    name: str
    description: str
    triggers: list[str] = field(default_factory=list)
    model: str | None = None
    requires_tools: list[str] = field(default_factory=list)


@dataclass
class Skill:
    """一个已加载的技能：元数据 + 完整 SKILL.md 内容 + 来源目录。"""

    manifest: SkillManifest
    content: str
    dir: Path


@dataclass
class SlashCommand:
    """一条 slash 命令：触发词 + 替换正文 + 描述。"""

    name: str
    body: str
    description: str
