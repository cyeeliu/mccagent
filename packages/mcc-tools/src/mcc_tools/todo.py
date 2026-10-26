"""todo_write 工具：会话内轻量任务清单。

模型一次性提交完整列表（替换式），harness 在连续多轮未更新时
注入提醒。状态机：pending → in_progress → completed，同时只允许
一个 in_progress。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from mcc_core import ToolContext, ToolOutput

TodoStatus = Literal["pending", "in_progress", "completed"]
_MAX_TODOS = 20
_MARKERS = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}


@dataclass
class TodoItem:
    content: str
    status: TodoStatus


class TodoError(ValueError):
    """todo 校验错误。"""


class TodoManager:
    """会话内 todo 状态，替换式更新。"""

    def __init__(self) -> None:
        self.items: list[TodoItem] = []

    def update(self, todos: list[dict[str, Any]]) -> str:
        """校验并替换整个列表，返回渲染文本。"""
        if not isinstance(todos, list):
            raise TodoError("todos must be a list")
        if len(todos) > _MAX_TODOS:
            raise TodoError(f"max {_MAX_TODOS} todos allowed")

        validated: list[TodoItem] = []
        in_progress = 0
        for i, todo in enumerate(todos):
            if not isinstance(todo, dict):
                raise TodoError(f"todos[{i}] must be an object")
            content = str(todo.get("content", "")).strip()
            if not content:
                raise TodoError(f"todos[{i}] requires non-empty content")
            status = str(todo.get("status", "pending")).lower()
            if status not in _MARKERS:
                raise TodoError(f"todos[{i}] invalid status {status!r}")
            if status == "in_progress":
                in_progress += 1
            validated.append(TodoItem(content=content, status=status))

        if in_progress > 1:
            raise TodoError("only one todo can be in_progress at a time")

        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."
        lines = [
            f"{_MARKERS[t.status]} {t.content}" for t in self.items
        ]
        done = sum(t.status == "completed" for t in self.items)
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)

    @property
    def has_items(self) -> bool:
        return bool(self.items)


class TodoWriteTool:
    """todo_write：创建/替换当前会话的任务清单。"""

    def __init__(self, manager: TodoManager) -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return "todo_write"

    @property
    def description(self) -> str:
        return (
            "Create and manage a task list for the current coding session. "
            "Submit the full list each call (replacement, not append). "
            "At most one item may be in_progress at a time."
        )

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["todos"],
            "properties": {
                "todos": {
                    "type": "array",
                    "maxItems": _MAX_TODOS,
                    "items": {
                        "type": "object",
                        "required": ["content", "status"],
                        "properties": {
                            "content": {"type": "string", "minLength": 1},
                            "status": {
                                "type": "string",
                                "enum": list(_MARKERS),
                            },
                        },
                    },
                }
            },
        }

    async def execute(
        self, input: dict[str, Any], ctx: ToolContext
    ) -> ToolOutput:
        todos = input.get("todos")
        try:
            output = self._manager.update(todos)
        except TodoError as e:
            return ToolOutput.err(f"invalid todos: {e}")
        return ToolOutput.ok(output)
