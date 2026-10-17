"""长期记忆：Markdown 存储与 Agent 工具封装。

索引（MEMORY.md）在启动时注入 system prompt，条目正文
由 Agent 通过 save_memory / read_memory 工具自主维护。
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mcc_core import Tool, ToolContext, ToolOutput

NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
VALID_TYPES = ("user", "project", "feedback", "reference")


def default_memory_base() -> Path:
    """默认记忆目录（调用时求值，跟随当前 HOME）。"""
    return Path.home() / ".mcc" / "memory"


class MarkdownMemoryStore:
    """Markdown 记忆存储。

    布局：基目录下 MEMORY.md 为索引，entries/{name}.md 为条目正文。
    索引行格式：`- [{name}] entries/{name}.md — {description} ({type})`。
    """

    def __init__(self, base_dir: Path | None = None) -> None:
        self.base = base_dir or default_memory_base()
        self.index_path = self.base / "MEMORY.md"
        self.entries_dir = self.base / "entries"

    @staticmethod
    def is_valid_name(name: str) -> bool:
        return bool(NAME_PATTERN.match(name))

    def entry_path(self, name: str) -> Path:
        if not self.is_valid_name(name):
            msg = f"invalid memory name: {name!r}"
            raise ValueError(msg)
        return self.entries_dir / f"{name}.md"

    def save(
        self,
        name: str,
        type: str,
        description: str,
        body: str,
        tags: list[str] | None = None,
    ) -> Path:
        """写入条目并更新索引（同名条目替换索引行）。"""
        if type not in VALID_TYPES:
            msg = f"invalid memory type: {type!r} (choose from {VALID_TYPES})"
            raise ValueError(msg)

        self.base.mkdir(parents=True, exist_ok=True)
        self.entries_dir.mkdir(parents=True, exist_ok=True)

        path = self.entry_path(name)
        header = (
            f"---\nname: {name}\ntype: {type}\n"
            f"description: {description}\n"
            f"tags: {', '.join(tags or [])}\nupdated: "
            f"{datetime.now(UTC).isoformat()}\n---\n\n"
        )
        path.write_text(header + body + "\n", encoding="utf-8")

        line = f"- [{name}] entries/{name}.md — {description} ({type})"
        lines = self._read_index_lines()
        kept = [line for line in lines if not line.startswith(f"- [{name}] ")]
        kept.append(line)
        self.index_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        return path

    def read(self, name: str) -> str:
        """读取条目正文（含 frontmatter）。"""
        path = self.entry_path(name)
        if not path.exists():
            msg = f"memory not found: {name}"
            raise KeyError(msg)
        return path.read_text(encoding="utf-8")

    def list_names(self) -> list[str]:
        """列出全部条目名（按索引出现顺序）。"""
        names = []
        for line in self._read_index_lines():
            m = re.match(r"^- \[([a-z0-9_-]+)\] ", line)
            if m:
                names.append(m.group(1))
        return names

    def index_text(self) -> str:
        """返回索引全文，供 system prompt 注入。"""
        if not self.index_path.exists():
            return ""
        return self.index_path.read_text(encoding="utf-8").strip()

    def _read_index_lines(self) -> list[str]:
        if not self.index_path.exists():
            return []
        return [
            line
            for line in self.index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


class SaveMemoryTool:
    """save_memory 工具：持久化一条记忆。"""

    @property
    def name(self) -> str:
        return "save_memory"

    @property
    def description(self) -> str:
        return (
            "Save a durable memory entry. Use when the user shares a "
            "preference, project decision, or reusable fact worth "
            "remembering across conversations."
        )

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["name", "type", "description", "body"],
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Stable filename-safe id, e.g. 'user_role'",
                },
                "type": {
                    "enum": list(VALID_TYPES),
                    "description": "Memory category",
                },
                "description": {
                    "type": "string",
                    "description": "One line describing when to use",
                },
                "body": {
                    "type": "string",
                    "description": "The actual memory content (markdown)",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional search tags",
                },
            },
        }

    def __init__(self, store: MarkdownMemoryStore) -> None:
        self._store = store

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        try:
            path = self._store.save(
                name=str(input["name"]),
                type=str(input["type"]),
                description=str(input["description"]),
                body=str(input["body"]),
                tags=[str(t) for t in input.get("tags", [])],
            )
        except (ValueError, KeyError) as e:
            return ToolOutput.err(f"save_memory failed: {e}")
        return ToolOutput.ok(f"saved: {path.name}")


class ReadMemoryTool:
    """read_memory 工具：读取指定记忆条目。"""

    @property
    def name(self) -> str:
        return "read_memory"

    @property
    def description(self) -> str:
        return (
            "Read a memory entry by name. Use when the memory index "
            "suggests a relevant entry and details are needed."
        )

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["name"],
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Memory entry id from the index",
                },
            },
        }

    def __init__(self, store: MarkdownMemoryStore) -> None:
        self._store = store

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        try:
            content = self._store.read(str(input["name"]))
        except (ValueError, KeyError) as e:
            return ToolOutput.err(f"read_memory failed: {e}")
        return ToolOutput.ok(content)


def memory_tools(store: MarkdownMemoryStore) -> list[Tool]:
    """返回记忆工具对，便于注册到 ToolRegistry。"""
    return [SaveMemoryTool(store), ReadMemoryTool(store)]


__all__ = [
    "default_memory_base",
    "MarkdownMemoryStore",
    "ReadMemoryTool",
    "SaveMemoryTool",
    "memory_tools",
]
