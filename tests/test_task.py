"""task 工具集测试：TaskStore 持久化、依赖图、生命周期、6 工具转发。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from mcc_core import ToolContext, ToolOutput
from mcc_tools import (
    ClaimTaskTool,
    CompleteTaskTool,
    CreateTaskTool,
    GetTaskTool,
    ListTasksTool,
    TaskStore,
    UpdateTaskTool,
)

TASK_ID = "task_11223344"


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=tmp_path, session_id="test", depth=0)


@pytest.fixture
def store(tmp_path: Path) -> TaskStore:
    return TaskStore(tmp_path / ".tasks")


class TestTaskStore:
    def test_create_and_load(self, store: TaskStore) -> None:
        task = store.create("write tests", "cover all branches")
        assert task.id.startswith("task_")
        assert task.status == "pending"
        assert task.owner is None
        loaded = store.load(task.id)
        assert loaded.subject == "write tests"
        assert loaded.description == "cover all branches"

    def test_create_empty_subject_rejected(self, store: TaskStore) -> None:
        with pytest.raises(ValueError, match="subject"):
            store.create("  ")

    def test_list_returns_all(self, store: TaskStore) -> None:
        t1 = store.create("a")
        t2 = store.create("b")
        ids = [t.id for t in store.list()]
        assert {t1.id, t2.id} == set(ids)

    def test_list_empty_when_no_dir(self, store: TaskStore) -> None:
        assert store.list() == []

    def test_invalid_id_rejected(self, store: TaskStore) -> None:
        with pytest.raises(ValueError, match="invalid task ID"):
            store.load("not-a-task-id")
        with pytest.raises(ValueError, match="invalid task ID"):
            store.load("task_xyz")

    def test_load_nonexistent(self, store: TaskStore) -> None:
        with pytest.raises(ValueError, match="not found"):
            store.load(TASK_ID)

    def test_persisted_to_disk(self, store: TaskStore, tmp_path: Path) -> None:
        task = store.create("persisted")
        path = tmp_path / ".tasks" / f"{task.id}.json"
        assert path.is_file()
        data = json.loads(path.read_text())
        assert data["subject"] == "persisted"


class TestDependencies:
    def test_add_dependency(self, store: TaskStore) -> None:
        t1 = store.create("schema")
        t2 = store.create("api")
        store.update_dependencies(t2.id, [t1.id])
        assert store.load(t2.id).blockedBy == [t1.id]

    def test_self_dependency_rejected(self, store: TaskStore) -> None:
        t = store.create("self")
        with pytest.raises(ValueError, match="itself"):
            store.update_dependencies(t.id, [t.id])

    def test_cycle_rejected(self, store: TaskStore) -> None:
        t1 = store.create("a")
        t2 = store.create("b")
        store.update_dependencies(t2.id, [t1.id])
        with pytest.raises(ValueError, match="cycle"):
            store.update_dependencies(t1.id, [t2.id])

    def test_dependency_only_while_pending(self, store: TaskStore) -> None:
        t1 = store.create("a")
        t2 = store.create("b")
        store.claim(t2.id)
        with pytest.raises(ValueError, match="pending"):
            store.update_dependencies(t2.id, [t1.id])

    def test_nonexistent_dependency_rejected(self, store: TaskStore) -> None:
        t = store.create("a")
        with pytest.raises(ValueError, match="not found"):
            store.update_dependencies(t.id, [TASK_ID])

    def test_dedup_dependencies(self, store: TaskStore) -> None:
        t1 = store.create("a")
        t2 = store.create("b")
        store.update_dependencies(t2.id, [t1.id, t1.id])
        assert store.load(t2.id).blockedBy == [t1.id]


class TestLifecycle:
    def test_can_start_no_deps(self, store: TaskStore) -> None:
        t = store.create("free")
        assert store.can_start(t.id)

    def test_can_start_blocked_by_pending(self, store: TaskStore) -> None:
        t1 = store.create("schema")
        t2 = store.create("api")
        store.update_dependencies(t2.id, [t1.id])
        assert not store.can_start(t2.id)

    def test_can_start_after_dep_completed(self, store: TaskStore) -> None:
        t1 = store.create("schema")
        t2 = store.create("api")
        store.update_dependencies(t2.id, [t1.id])
        store.claim(t1.id)
        store.complete(t1.id)
        assert store.can_start(t2.id)

    def test_claim_blocked_by_incomplete_dep(self, store: TaskStore) -> None:
        t1 = store.create("schema")
        t2 = store.create("api")
        store.update_dependencies(t2.id, [t1.id])
        with pytest.raises(ValueError, match="blocked"):
            store.claim(t2.id)

    def test_claim_sets_in_progress_and_owner(self, store: TaskStore) -> None:
        t = store.create("work")
        claimed = store.claim(t.id, owner="agent-1")
        assert claimed.status == "in_progress"
        assert claimed.owner == "agent-1"

    def test_claim_non_pending_rejected(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id)
        with pytest.raises(ValueError, match="cannot claim"):
            store.claim(t.id)

    def test_complete_owner_mismatch(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        with pytest.raises(ValueError, match="owned by"):
            store.complete(t.id, owner="agent-2")

    def test_complete_non_in_progress_rejected(self, store: TaskStore) -> None:
        t = store.create("work")
        with pytest.raises(ValueError, match="cannot complete"):
            store.complete(t.id)

    def test_complete_sets_completed(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id)
        done = store.complete(t.id)
        assert done.status == "completed"


class TestTools:
    @pytest.mark.asyncio
    async def test_create_task_tool(self, store: TaskStore, ctx: ToolContext) -> None:
        tool = CreateTaskTool(store)
        out = await tool.execute({"subject": "new task", "description": "d"}, ctx)
        assert out.is_error is False
        assert "Created task_" in out.content
        tasks = store.list()
        assert len(tasks) == 1
        assert tasks[0].subject == "new task"

    @pytest.mark.asyncio
    async def test_create_task_empty_subject(self, store: TaskStore, ctx: ToolContext) -> None:
        tool = CreateTaskTool(store)
        out = await tool.execute({"subject": ""}, ctx)
        assert out.is_error is True

    @pytest.mark.asyncio
    async def test_update_task_tool(self, store: TaskStore, ctx: ToolContext) -> None:
        t1 = store.create("a")
        t2 = store.create("b")
        tool = UpdateTaskTool(store)
        out = await tool.execute({"task_id": t2.id, "addBlockedBy": [t1.id]}, ctx)
        assert out.is_error is False
        assert t1.id in out.content

    @pytest.mark.asyncio
    async def test_list_tasks_tool_empty(self, store: TaskStore, ctx: ToolContext) -> None:
        tool = ListTasksTool(store)
        out = await tool.execute({}, ctx)
        assert "No tasks" in out.content

    @pytest.mark.asyncio
    async def test_list_tasks_tool_with_items(self, store: TaskStore, ctx: ToolContext) -> None:
        store.create("a")
        store.create("b")
        tool = ListTasksTool(store)
        out = await tool.execute({}, ctx)
        assert "[ ]" in out.content
        assert "a" in out.content
        assert "b" in out.content

    @pytest.mark.asyncio
    async def test_get_task_tool(self, store: TaskStore, ctx: ToolContext) -> None:
        t = store.create("find me", "desc")
        tool = GetTaskTool(store)
        out = await tool.execute({"task_id": t.id}, ctx)
        assert out.is_error is False
        data = json.loads(out.content)
        assert data["subject"] == "find me"

    @pytest.mark.asyncio
    async def test_get_task_nonexistent(self, store: TaskStore, ctx: ToolContext) -> None:
        tool = GetTaskTool(store)
        out = await tool.execute({"task_id": TASK_ID}, ctx)
        assert out.is_error is True

    @pytest.mark.asyncio
    async def test_claim_task_tool(self, store: TaskStore, ctx: ToolContext) -> None:
        t = store.create("work")
        tool = ClaimTaskTool(store)
        out = await tool.execute({"task_id": t.id}, ctx)
        assert out.is_error is False
        assert "Claimed" in out.content
        assert store.load(t.id).status == "in_progress"

    @pytest.mark.asyncio
    async def test_complete_task_tool_reports_unblocked(
        self, store: TaskStore, ctx: ToolContext
    ) -> None:
        t1 = store.create("schema")
        t2 = store.create("api")
        store.update_dependencies(t2.id, [t1.id])
        store.claim(t1.id)
        tool = CompleteTaskTool(store)
        out = await tool.execute({"task_id": t1.id}, ctx)
        assert out.is_error is False
        assert "Unblocked" in out.content

    @pytest.mark.asyncio
    async def test_complete_task_tool_no_unblock(
        self, store: TaskStore, ctx: ToolContext
    ) -> None:
        t = store.create("solo")
        store.claim(t.id)
        tool = CompleteTaskTool(store)
        out = await tool.execute({"task_id": t.id}, ctx)
        assert out.is_error is False
        assert "Unblocked" not in out.content

    def test_tool_names_and_schemas(self, store: TaskStore) -> None:
        tools = [
            CreateTaskTool(store),
            UpdateTaskTool(store),
            ListTasksTool(store),
            GetTaskTool(store),
            ClaimTaskTool(store),
            CompleteTaskTool(store),
        ]
        names = [t.name for t in tools]
        assert names == [
            "create_task", "update_task", "list_tasks",
            "get_task", "claim_task", "complete_task",
        ]
        for t in tools:
            assert isinstance(t.input_schema(), dict)
