"""task 工具集：持久化任务记录与依赖图。

落盘 .tasks/task_{8hex}.json，支持依赖（blockedBy）与生命周期
状态机管理。环检测防循环依赖，路径校验防 traversal。
"""

from __future__ import annotations

import fcntl
import json
import secrets
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from mcc_core import ToolContext, ToolOutput

_TASK_ID_PATTERN = "task_[0-9a-f]{8}"
_MAX_ID_RETRIES = 100


class TaskStatus(StrEnum):
    PENDING = "pending"
    READY = "ready"
    CLAIMED = "claimed"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.PENDING: frozenset({TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED}),
    TaskStatus.READY: frozenset({TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED}),
    TaskStatus.CLAIMED: frozenset({TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED}),
    TaskStatus.IN_PROGRESS: frozenset({
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.TIMEOUT,
        TaskStatus.CANCELLED,
    }),
    TaskStatus.BLOCKED: frozenset({TaskStatus.PENDING, TaskStatus.CANCELLED}),
    TaskStatus.FAILED: frozenset({TaskStatus.RETRYING}),
    TaskStatus.TIMEOUT: frozenset({TaskStatus.RETRYING}),
    TaskStatus.RETRYING: frozenset({TaskStatus.PENDING}),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
}

_VALID_STATUSES = frozenset(s.value for s in TaskStatus)

_TERMINAL_STATUSES = frozenset({
    TaskStatus.COMPLETED.value,
    TaskStatus.FAILED.value,
    TaskStatus.TIMEOUT.value,
    TaskStatus.CANCELLED.value,
})


@dataclass
class Task:
    id: str
    subject: str
    description: str
    status: str
    owner: str | None
    blockedBy: list[str] = field(default_factory=list)
    attempt: int = 0
    max_attempts: int | None = None
    owner_agent_id: str | None = None
    deadline_at: str | None = None
    error: str | None = None
    result_ref: str | None = None
    version: int = 0
    lease_expires_at: float | None = None


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
        data.setdefault("attempt", 0)
        data.setdefault("max_attempts", None)
        data.setdefault("owner_agent_id", None)
        data.setdefault("deadline_at", None)
        data.setdefault("error", None)
        data.setdefault("result_ref", None)
        data.setdefault("version", 0)
        data.setdefault("lease_expires_at", None)
        task = Task(**data)
        if task.id != task_id:
            raise TaskError(f"task file ID mismatch: {task_id}")
        if task.status not in _VALID_STATUSES:
            raise TaskError(f"invalid task status: {task.status}")
        return task

    def _transition(
        self,
        task: Task,
        target: TaskStatus,
        *,
        error: str | None = None,
    ) -> Task:
        current = TaskStatus(task.status)
        allowed = _TRANSITIONS.get(current, frozenset())
        if target not in allowed:
            raise TaskError(
                f"task {task.id} cannot transition from "
                f"{current.value} to {target.value}"
            )
        task.status = target.value
        if error is not None:
            task.error = error
        task.version += 1
        self.save(task)
        return task

    def _check_owner(self, task: Task, owner: str) -> None:
        actual = task.owner_agent_id or task.owner
        if actual != owner:
            raise TaskError(
                f"task {task.id} owned by {actual}, not {owner}"
            )

    def _is_lease_expired(self, task: Task) -> bool:
        if task.lease_expires_at is None:
            return False
        return time.time() > task.lease_expires_at

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
        if task.status != TaskStatus.PENDING.value or task.owner is not None:
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
                if self.load(dep).status != TaskStatus.COMPLETED.value:
                    return False
            except TaskError:
                return False
        return True

    def claim(
        self,
        task_id: str,
        owner: str,
        *,
        lease_seconds: float = 300.0,
    ) -> Task:
        path = self._path(task_id)
        if not path.is_file():
            raise TaskError(f"task not found: {task_id}")
        with path.open("r+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            task = self.load(task_id)
            if task.status == TaskStatus.IN_PROGRESS.value:
                if not self._is_lease_expired(task):
                    raise TaskError(
                        f"task {task_id} is in_progress, cannot claim"
                    )
            elif task.status != TaskStatus.PENDING.value:
                raise TaskError(
                    f"task {task_id} is {task.status}, cannot claim"
                )
            incomplete = [
                d for d in task.blockedBy
                if not self.exists(d) or self.load(d).status != TaskStatus.COMPLETED.value
            ]
            if incomplete:
                raise TaskError(f"blocked by: {incomplete}")
            task.owner = owner
            task.owner_agent_id = owner
            task.lease_expires_at = time.time() + lease_seconds
            if task.status == TaskStatus.IN_PROGRESS.value:
                task.version += 1
                self.save(task)
            else:
                task = self._transition(task, TaskStatus.IN_PROGRESS)
            return task

    def renew_lease(
        self,
        task_id: str,
        owner: str,
        *,
        lease_seconds: float = 300.0,
    ) -> Task:
        task = self.load(task_id)
        if task.status != TaskStatus.IN_PROGRESS.value:
            raise TaskError(
                f"task {task_id} is {task.status}, cannot renew lease"
            )
        self._check_owner(task, owner)
        task.lease_expires_at = time.time() + lease_seconds
        self.save(task)
        return task

    def complete(self, task_id: str, owner: str) -> Task:
        task = self.load(task_id)
        if task.status != TaskStatus.IN_PROGRESS.value:
            raise TaskError(f"task {task_id} is {task.status}, cannot complete")
        self._check_owner(task, owner)
        return self._transition(task, TaskStatus.COMPLETED)

    def fail(
        self, task_id: str, owner: str, error: str = ""
    ) -> Task:
        task = self.load(task_id)
        if task.status != TaskStatus.IN_PROGRESS.value:
            raise TaskError(f"task {task_id} is {task.status}, cannot fail")
        self._check_owner(task, owner)
        return self._transition(
            task, TaskStatus.FAILED, error=error or "task failed"
        )

    def timeout(self, task_id: str, owner: str) -> Task:
        task = self.load(task_id)
        if task.status != TaskStatus.IN_PROGRESS.value:
            raise TaskError(f"task {task_id} is {task.status}, cannot timeout")
        self._check_owner(task, owner)
        return self._transition(
            task, TaskStatus.TIMEOUT, error="task timed out"
        )

    def cancel(
        self,
        task_id: str,
        owner: str | None = None,
        reason: str = "",
    ) -> Task:
        task = self.load(task_id)
        if owner is not None and task.owner_agent_id is not None:
            self._check_owner(task, owner)
        return self._transition(
            task, TaskStatus.CANCELLED, error=reason or "task cancelled"
        )

    def retry(self, task_id: str) -> Task:
        task = self.load(task_id)
        if task.status not in (
            TaskStatus.FAILED.value,
            TaskStatus.TIMEOUT.value,
        ):
            raise TaskError(
                f"task {task_id} is {task.status}, can only retry "
                f"failed or timed-out tasks"
            )
        task = self._transition(task, TaskStatus.RETRYING)
        task.attempt += 1
        task.owner = None
        task.owner_agent_id = None
        task.error = None
        task.lease_expires_at = None
        return self._transition(task, TaskStatus.PENDING)


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
        markers = {
            "pending": "[ ]",
            "ready": "[R]",
            "claimed": "[C]",
            "in_progress": "[>]",
            "blocked": "[B]",
            "completed": "[x]",
            "failed": "[!]",
            "timeout": "[T]",
            "cancelled": "[-]",
            "retrying": "[~]",
        }
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
                "agent_id": {"type": "string"},
            },
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        task_id = input.get("task_id", "")
        agent_id = input.get("agent_id") or ctx.session_id
        try:
            task = self._store.claim(task_id, owner=agent_id)
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
                "agent_id": {"type": "string"},
            },
        }

    async def execute(self, input: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        task_id = input.get("task_id", "")
        agent_id = input.get("agent_id") or ctx.session_id
        try:
            task = self._store.complete(task_id, owner=agent_id)
        except TaskError as e:
            return ToolOutput.err(str(e))
        unblocked = [
            t.subject for t in self._store.list()
            if t.status == TaskStatus.PENDING.value and t.blockedBy and self._store.can_start(t.id)
        ]
        msg = f"Completed {task.id} ({task.subject})"
        if unblocked:
            msg += f"\nUnblocked: {', '.join(unblocked)}"
        return ToolOutput.ok(msg)
