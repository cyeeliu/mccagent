"""Skill 与 Slash Command 加载器。

扫描 ``skills/*/SKILL.md`` 与 ``commands/*.md``，解析 YAML frontmatter。
支持 home + project 两层合并（project 同名覆盖 home）。
"""

from __future__ import annotations

from pathlib import Path

import yaml

from mcc_skills.types import Skill, SkillManifest, SlashCommand

SKILL_FILE = "SKILL.md"


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """拆分 YAML frontmatter 与正文。

    无 frontmatter 或解析失败时返回 ``({}, text)``。
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        return {}, text
    closing = next(
        (i for i, line in enumerate(lines[1:], start=1)
         if line.rstrip("\r\n") == "---"),
        None,
    )
    if closing is None:
        return {}, text
    fm = "".join(lines[1:closing])
    body = "".join(lines[closing + 1:]).strip()
    try:
        meta = yaml.safe_load(fm) or {}
    except yaml.YAMLError:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, body


def _clean_description(raw: object, fallback: str) -> str:
    desc = raw.strip() if isinstance(raw, str) else ""
    desc = desc or fallback
    return " ".join(str(desc).lstrip("# ").split())


def _as_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _safe_relative_to(path: Path, root: Path) -> bool:
    try:
        return path.is_relative_to(root)
    except ValueError:
        return False


class SkillLoader:
    """扫描目录下的 ``skills/*/SKILL.md``。"""

    def __init__(self, skills_dir: Path) -> None:
        self.skills_dir = skills_dir
        self.skills: dict[str, Skill] = {}
        self.scan()

    def scan(self) -> None:
        self.skills.clear()
        if not self.skills_dir.exists():
            return
        root = self.skills_dir.resolve()
        for manifest_path in sorted(self.skills_dir.glob(f"*/{SKILL_FILE}")):
            if not manifest_path.is_file():
                continue
            if not _safe_relative_to(manifest_path.resolve(), root):
                continue
            content = manifest_path.read_text(encoding="utf-8")
            meta, body = parse_frontmatter(content)
            raw_name = meta.get("name")
            name = raw_name.strip() if isinstance(raw_name, str) else ""
            name = name or manifest_path.parent.name
            description = _clean_description(
                meta.get("description"), body.split("\n", 1)[0]
            )
            model = meta.get("model")
            model = str(model) if isinstance(model, str) else None
            self.skills[name] = Skill(
                manifest=SkillManifest(
                    name=name,
                    description=description,
                    triggers=_as_str_list(meta.get("triggers")),
                    model=model,
                    requires_tools=_as_str_list(meta.get("requires_tools")),
                ),
                content=content,
                dir=manifest_path.parent,
            )

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)

    def names(self) -> list[str]:
        return list(self.skills.keys())


def load_skills_merged(home: Path | None, project: Path | None) -> dict[str, Skill]:
    """加载 home + project 两层 skills，project 同名覆盖 home。"""
    merged: dict[str, Skill] = {}
    if home is not None:
        merged.update(SkillLoader(home).skills)
    if project is not None:
        merged.update(SkillLoader(project).skills)
    return merged


def resolve_slash(
    commands: dict[str, SlashCommand], user_input: str
) -> str | None:
    """命中 slash 命令时返回替换后的文本，否则 None。"""
    parts = user_input.strip().split(None, 1)
    if not parts:
        return None
    sc = commands.get(parts[0])
    if sc is None:
        return None
    rest = parts[1] if len(parts) > 1 else ""
    if rest:
        return f"{sc.body}\n\n用户补充参数：{rest}"
    return sc.body


class SlashRegistry:
    """扫描目录下的 ``commands/*.md``。"""

    def __init__(self, commands_dir: Path) -> None:
        self.commands_dir = commands_dir
        self.commands: dict[str, SlashCommand] = {}
        self.scan()

    def scan(self) -> None:
        self.commands.clear()
        if not self.commands_dir.exists():
            return
        root = self.commands_dir.resolve()
        for path in sorted(self.commands_dir.glob("*.md")):
            if not path.is_file():
                continue
            if not _safe_relative_to(path.resolve(), root):
                continue
            raw = path.read_text(encoding="utf-8")
            meta, body = parse_frontmatter(raw)
            cmd = meta.get("command")
            if not isinstance(cmd, str) or not cmd.strip():
                cmd = "/" + path.stem
            desc = _clean_description(meta.get("description"), body.split("\n", 1)[0])
            self.commands[cmd] = SlashCommand(name=cmd, body=body, description=desc)

    def resolve(self, user_input: str) -> str | None:
        """命中 slash 命令时返回替换后的文本，否则 None。"""
        return resolve_slash(self.commands, user_input)


def load_commands_merged(
    home: Path | None, project: Path | None
) -> dict[str, SlashCommand]:
    """加载 home + project 两层 commands，project 同名覆盖 home。"""
    merged: dict[str, SlashCommand] = {}
    if home is not None:
        merged.update(SlashRegistry(home).commands)
    if project is not None:
        merged.update(SlashRegistry(project).commands)
    return merged
