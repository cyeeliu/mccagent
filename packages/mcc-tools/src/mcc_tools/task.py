"""task 工具集：持久化任务记录与依赖图。

落盘 .tasks/task_{8hex}.json，支持依赖（blockedBy）与生命周期
pending → claim → in_progress → complete → completed。环检测防
循环依赖，路径校验防 traversal。
"""

from __future__ import annotations

import json
import secrets
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mcc_core import ToolContext, ToolOutput

_TASK_ID_PATTERN = "task_[0-9a-f]{8}"
_MAX_ID_RETRIES = 100


@dataclass
class Task:
    id: str
    subject: str
    description: str
    status: str
    owner: str | None
    blockedBy: list[str] = field(default_factory=list)


class TaskError(ValueError):
    """task 操作错误。"""


class TaskStore:
    """.tasks/ 目录下的任务持久化。"""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _root(self, create: bool = False) -> Path:
        if create:
            self.directory.mkdir(parents=True, exist_ok=True)
        root = self.directory.resolve()
        return root

    def _path(self, task_id: str, create_root: bool = False) -> Path:
        if not isinstance(task_id, str) or len(task_id) != 13:
            raise TaskError(f"invalid task ID: {task_id!r}")
        if not task_id.startswith("task_"):
            raise TaskError(f"invalid task ID: {task_id!r}")
        try:
            int(task_id[5:], 16)
        except ValueError as e:
            raise TaskError(f"invalid task ID: {task_id!r}") from e
        root = self._root(create=create_root)
        return root / f"{task_id}.json"

    def exists(self, task_id: str) -> bool:
        try:
            return self._path(task_id).is_file()
        except TaskError:
            return False

    def create(self, subject: str, description: str = "") -> Task:
        subject = subject.strip()
        if not subject:
            raise TaskError("subject cannot be empty")
        self._root(create=True)
        for _ in range(_MAX_ID_RETRIES):
            task = Task(
                id=f"task_{secrets.token_hex(4)}",
                subject=subject,
                description=description,
                status="pending",
                owner=None,
                blockedBy=[],
            )
            path = self._path(task.id, create_root=True)
            if not path.exists():
                path.write_text(
                    json.dumps(asdict(task), indent=2), encoding="utf-8"
                )
                return task
        raise TaskError("could not allocate a unique task ID")

    def save(self, task: Task) -> None:
        path = self._path(task.id, create_root=True)
        path.write_text(json.dumps(asdict(task), indent=2), encoding="utf-8")

    def load(self, task_id: str) -> Task:
        path = self._path(task_id)
        if not path.is_file():
            raise TaskError(f"task not found: {task_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        task = Task(**data)
        if task.id != task_id:
            raise TaskError(f"task file ID mismatch: {task_id}")
        if task.status not in ("pending", "in_progress", "completed"):
            raise TaskError(f"invalid task status: {task.status}")
        return task

    def list(self) -> list[Task]:
        if not self.directory.exists():
            return []
        root = self._root()
        tasks: list[Task] = []
        for path in sorted(root.glob("task_*.json")):
            try:
                tasks.append(self.load(path.stem))
            except TaskError:
                continue
        return tasks

    def _depends_on(self, task_id: str, target_id: str) -> bool:
        """task_id 是否传递依赖 target_id。"""
        pending = [task_id]
        visited: set[str] = set()
        while pending:
            current = pending.pop()
            if current == target_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            try:
                pending.extend(self.load(current).blockedBy)
            except TaskError:
                continue
        return False

    def update_dependencies(
        self, task_id: str, add_blocked_by: list[str]
    ) -> Task:
        if not isinstance(add_blocked_by, list):
            raise TaskError("addBlockedBy must be a list")
        task = self.load(task_id)
        if task.status != "pending" or task.owner is not None:
            raise TaskError(
                f"task {task_id} dependencies only editable while pending"
            )
        deps = list(dict.fromkeys(add_blocked_by))
        for dep in deps:
            if dep == task_id:
                raise TaskError("task cannot depend on itself")
            if not self.exists(dep):
                raise TaskError(f"dependency not found: {dep}")
            if dep not in task.blockedBy and self._depends_on(dep, task_id):
                raise TaskError(f"dependency cycle: {task_id} -> {dep}")
        task.blockedBy.extend(
            d for d in deps if d not in task.blockedBy
        )
        self.save(task)
        return task

    def can_start(self, task_id: str) -> bool:
        task = self.load(task_id)
        for dep in task.blockedBy:
            try:
                if self.load(dep).status != "completed":
                    return False
            except TaskError:
                return False
        return True

    def claim(self, task_id: str, owner: str = "agent") -> Task:
        task = self.load(task_id)
        if task.status != "pending":
            raise TaskError(f"task {task_id} is {task.status}, cannot claim")
        incomplete = [
            d for d in task.blockedBy
            if not self.exists(d) or self.load(d).status != "completed"
        ]
        if incomplete:
            raise TaskError(f"blocked by: {incomplete}")
        task.owner = owner
        task.status = "in_progress"
        self.save(task)
        return task

    def complete(self, task_id: str, owner: str = "agent") -> Task:
        task = self.load(task_id)
        if task.status != "in_progress":
            raise TaskError(f"task {task_id} is {task.status}, cannot complete")
        if task.owner != owner:
            raise TaskError(f"task {task_id} owned by {task.owner}, not {owner}")
        task.status = "completed"
        self.save(task)
        return task


def _format_task(task: Task) -> str:
    deps = f" (blockedBy: {', '.join(task.blockedBy)})" if task.blockedBy else ""
    owner = f" [{task.owner}]" if task.owner else ""
    return f"{task.id}: {task.subject} [{task.status}]{owner}{deps}"


class _TaskToolBase:
    """task 工具基类，共享 store 引用。"""

    def __init__(self, store: TaskStore) -> None:
        self._store = store


class CreateTaskTool(_TaskToolBase):
    @property
    def name(self) -> str:
        return "create_task"

    @property
    def description(self) -> str:
        return "Create a task and return its runtime-generated ID."

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["subject"],
            "properties": {
                "subject": {"type": "string", "minLength": 1},
                "description": {"type": "string"},
            },
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        subject = input.get("subject", "")
        desc = input.get("description", "")
        try:
            task = self._store.create(subject, desc)
        except TaskError as e:
            return ToolOutput.err(str(e))
        return ToolOutput.ok(f"Created {task.id}: {task.subject}")


class UpdateTaskTool(_TaskToolBase):
    @property
    def name(self) -> str:
        return "update_task"

    @property
    def description(self) -> str:
        return "Add dependencies to a pending task using task IDs."

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["task_id", "addBlockedBy"],
            "properties": {
                "task_id": {"type": "string", "pattern": _TASK_ID_PATTERN},
                "addBlockedBy": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "pattern": _TASK_ID_PATTERN},
                },
            },
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        task_id = input.get("task_id", "")
        deps = input.get("addBlockedBy", [])
        try:
            task = self._store.update_dependencies(task_id, deps)
        except TaskError as e:
            return ToolOutput.err(str(e))
        deps_str = ", ".join(task.blockedBy) or "(none)"
        return ToolOutput.ok(f"Updated {task.id} blockedBy: {deps_str}")


class ListTasksTool(_TaskToolBase):
    @property
    def name(self) -> str:
        return "list_tasks"

    @property
    def description(self) -> str:
        return "List all tasks with status, owner, and dependencies."

    def input_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        tasks = self._store.list()
        if not tasks:
            return ToolOutput.ok("No tasks.")
        markers = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}
        lines = [
            f"{markers.get(t.status, '[?]')} {_format_task(t)}"
            for t in tasks
        ]
        return ToolOutput.ok("\n".join(lines))


class GetTaskTool(_TaskToolBase):
    @property
    def name(self) -> str:
        return "get_task"

    @property
    def description(self) -> str:
        return "Get a single task by ID as JSON."

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["task_id"],
            "properties": {
                "task_id": {"type": "string", "pattern": _TASK_ID_PATTERN},
            },
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        task_id = input.get("task_id", "")
        try:
            task = self._store.load(task_id)
        except TaskError as e:
            return ToolOutput.err(str(e))
        return ToolOutput.ok(json.dumps(asdict(task), indent=2))


class ClaimTaskTool(_TaskToolBase):
    @property
    def name(self) -> str:
        return "claim_task"

    @property
    def description(self) -> str:
        return "Claim a pending task whose dependencies are all completed."

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["task_id"],
            "properties": {
                "task_id": {"type": "string", "pattern": _TASK_ID_PATTERN},
            },
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        task_id = input.get("task_id", "")
        try:
            task = self._store.claim(task_id, owner="agent")
        except TaskError as e:
            return ToolOutput.err(str(e))
        return ToolOutput.ok(f"Claimed {task.id} ({task.subject})")


class CompleteTaskTool(_TaskToolBase):
    @property
    def name(self) -> str:
        return "complete_task"

    @property
    def description(self) -> str:
        return "Complete a task claimed by this agent."

    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "required": ["task_id"],
            "properties": {
                "task_id": {"type": "string", "pattern": _TASK_ID_PATTERN},
            },
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        task_id = input.get("task_id", "")
        try:
            task = self._store.complete(task_id, owner="agent")
        except TaskError as e:
            return ToolOutput.err(str(e))
        unblocked = [
            t.subject for t in self._store.list()
            if t.status == "pending" and t.blockedBy and self._store.can_start(t.id)
        ]
        msg = f"Completed {task.id} ({task.subject})"
        if unblocked:
            msg += f"\nUnblocked: {', '.join(unblocked)}"
        return ToolOutput.ok(msg)
