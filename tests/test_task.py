"""task 工具集测试：TaskStore 持久化、依赖图、生命周期、状态机、6 工具转发。"""

from __future__ import annotations

import json
import time
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
    TaskStatus,
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
        store.claim(t2.id, owner="agent-1")
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
        store.claim(t1.id, owner="agent-1")
        store.complete(t1.id, owner="agent-1")
        assert store.can_start(t2.id)

    def test_claim_blocked_by_incomplete_dep(self, store: TaskStore) -> None:
        t1 = store.create("schema")
        t2 = store.create("api")
        store.update_dependencies(t2.id, [t1.id])
        with pytest.raises(ValueError, match="blocked"):
            store.claim(t2.id, owner="agent-1")

    def test_claim_sets_in_progress_and_owner(self, store: TaskStore) -> None:
        t = store.create("work")
        claimed = store.claim(t.id, owner="agent-1")
        assert claimed.status == "in_progress"
        assert claimed.owner == "agent-1"

    def test_claim_non_pending_rejected(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        with pytest.raises(ValueError, match="cannot claim"):
            store.claim(t.id, owner="agent-1")

    def test_complete_owner_mismatch(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        with pytest.raises(ValueError, match="owned by"):
            store.complete(t.id, owner="agent-2")

    def test_complete_non_in_progress_rejected(self, store: TaskStore) -> None:
        t = store.create("work")
        with pytest.raises(ValueError, match="cannot complete"):
            store.complete(t.id, owner="agent-1")

    def test_complete_sets_completed(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        done = store.complete(t.id, owner="agent-1")
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
        store.claim(t1.id, owner="test")
        tool = CompleteTaskTool(store)
        out = await tool.execute({"task_id": t1.id}, ctx)
        assert out.is_error is False
        assert "Unblocked" in out.content

    @pytest.mark.asyncio
    async def test_complete_task_tool_no_unblock(
        self, store: TaskStore, ctx: ToolContext
    ) -> None:
        t = store.create("solo")
        store.claim(t.id, owner="test")
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


class TestTaskStateMachine:
    """P1-04: Task 状态机重构 — 新增 fail/timeout/cancel/retry + 迁移校验。"""

    def test_fail_from_in_progress(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        failed = store.fail(t.id, owner="agent-1", error="boom")
        assert failed.status == TaskStatus.FAILED.value
        assert failed.error == "boom"

    def test_fail_default_error(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        failed = store.fail(t.id, owner="agent-1")
        assert failed.error == "task failed"

    def test_fail_non_in_progress_rejected(self, store: TaskStore) -> None:
        t = store.create("work")
        with pytest.raises(ValueError, match="cannot fail"):
            store.fail(t.id, owner="agent-1")

    def test_fail_owner_mismatch(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        with pytest.raises(ValueError, match="owned by"):
            store.fail(t.id, owner="agent-2")

    def test_timeout_from_in_progress(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        timed = store.timeout(t.id, owner="agent-1")
        assert timed.status == TaskStatus.TIMEOUT.value
        assert "timed out" in (timed.error or "")

    def test_timeout_non_in_progress_rejected(self, store: TaskStore) -> None:
        t = store.create("work")
        with pytest.raises(ValueError, match="cannot timeout"):
            store.timeout(t.id, owner="agent-1")

    def test_cancel_from_pending(self, store: TaskStore) -> None:
        t = store.create("work")
        cancelled = store.cancel(t.id, reason="not needed")
        assert cancelled.status == TaskStatus.CANCELLED.value
        assert cancelled.error == "not needed"

    def test_cancel_from_in_progress(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        cancelled = store.cancel(t.id)
        assert cancelled.status == TaskStatus.CANCELLED.value

    def test_cancel_by_owner(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        cancelled = store.cancel(t.id, owner="agent-1")
        assert cancelled.status == TaskStatus.CANCELLED.value

    def test_cancel_by_non_owner_rejected(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        with pytest.raises(ValueError, match="owned by"):
            store.cancel(t.id, owner="agent-2")

    def test_cancel_completed_rejected(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        store.complete(t.id, owner="agent-1")
        with pytest.raises(ValueError, match="cannot transition"):
            store.cancel(t.id)

    def test_cancel_cancelled_rejected(self, store: TaskStore) -> None:
        t = store.create("work")
        store.cancel(t.id)
        with pytest.raises(ValueError, match="cannot transition"):
            store.cancel(t.id)

    def test_retry_from_failed(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        store.fail(t.id, owner="agent-1")
        retried = store.retry(t.id)
        assert retried.status == TaskStatus.PENDING.value
        assert retried.attempt == 1
        assert retried.owner is None
        assert retried.error is None

    def test_retry_from_timeout(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        store.timeout(t.id, owner="agent-1")
        retried = store.retry(t.id)
        assert retried.status == TaskStatus.PENDING.value
        assert retried.attempt == 1

    def test_retry_increments_attempt(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        store.fail(t.id, owner="agent-1")
        store.retry(t.id)
        store.claim(t.id, owner="agent-1")
        store.fail(t.id, owner="agent-1")
        retried = store.retry(t.id)
        assert retried.attempt == 2

    def test_retry_non_failed_rejected(self, store: TaskStore) -> None:
        t = store.create("work")
        with pytest.raises(ValueError, match="can only retry"):
            store.retry(t.id)

    def test_retry_in_progress_rejected(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        with pytest.raises(ValueError, match="can only retry"):
            store.retry(t.id)

    def test_version_increments_on_transition(self, store: TaskStore) -> None:
        t = store.create("work")
        assert t.version == 0
        claimed = store.claim(t.id, owner="agent-1")
        assert claimed.version == 1
        done = store.complete(t.id, owner="agent-1")
        assert done.version == 2

    def test_version_increments_on_fail(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        failed = store.fail(t.id, owner="agent-1")
        assert failed.version == 2

    def test_retry_version_increments_twice(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        store.fail(t.id, owner="agent-1")
        retried = store.retry(t.id)
        assert retried.version == 4

    @pytest.mark.parametrize(
        "terminal",
        [TaskStatus.COMPLETED.value, TaskStatus.CANCELLED.value],
    )
    def test_terminal_no_outgoing(
        self, store: TaskStore, terminal: str
    ) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        if terminal == TaskStatus.COMPLETED.value:
            store.complete(t.id, owner="agent-1")
        else:
            store.cancel(t.id)
        with pytest.raises(ValueError, match="cannot transition"):
            store.cancel(t.id)

    def test_complete_after_fail_rejected(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        store.fail(t.id, owner="agent-1")
        with pytest.raises(ValueError, match="cannot complete"):
            store.complete(t.id, owner="agent-1")

    def test_claim_after_fail_rejected(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        store.fail(t.id, owner="agent-1")
        with pytest.raises(ValueError, match="cannot claim"):
            store.claim(t.id, owner="agent-1")

    def test_claim_after_retry_succeeds(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        store.fail(t.id, owner="agent-1")
        store.retry(t.id)
        claimed = store.claim(t.id, owner="agent-2")
        assert claimed.status == TaskStatus.IN_PROGRESS.value
        assert claimed.owner == "agent-2"


class TestLegacyCompat:
    """P1-04: 旧数据兼容 — 缺少新字段的 JSON 能正确加载。"""

    def test_load_legacy_missing_new_fields(
        self, store: TaskStore, tmp_path: Path
    ) -> None:
        legacy = {
            "id": "task_11223344",
            "subject": "legacy",
            "description": "",
            "status": "pending",
            "owner": None,
            "blockedBy": [],
        }
        path = tmp_path / ".tasks" / "task_11223344.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(legacy))
        task = store.load("task_11223344")
        assert task.attempt == 0
        assert task.max_attempts is None
        assert task.owner_agent_id is None
        assert task.deadline_at is None
        assert task.error is None
        assert task.result_ref is None
        assert task.version == 0

    def test_load_legacy_in_progress(
        self, store: TaskStore, tmp_path: Path
    ) -> None:
        legacy = {
            "id": "task_11223344",
            "subject": "legacy",
            "description": "",
            "status": "in_progress",
            "owner": "agent-1",
            "blockedBy": [],
        }
        path = tmp_path / ".tasks" / "task_11223344.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(legacy))
        task = store.load("task_11223344")
        assert task.status == TaskStatus.IN_PROGRESS.value
        assert task.owner == "agent-1"

    def test_save_roundtrip_preserves_new_fields(
        self, store: TaskStore
    ) -> None:
        t = store.create("roundtrip")
        store.claim(t.id, owner="agent-1")
        store.fail(t.id, owner="agent-1", error="test error")
        loaded = store.load(t.id)
        assert loaded.status == TaskStatus.FAILED.value
        assert loaded.error == "test error"
        assert loaded.version == 2


class TestClaimAtomicityAndOwner:
    """P1-12: Task Claim 原子性与 Owner 统一。"""

    def test_claim_sets_owner_agent_id(self, store: TaskStore) -> None:
        t = store.create("work")
        claimed = store.claim(t.id, owner="agent-7")
        assert claimed.owner == "agent-7"
        assert claimed.owner_agent_id == "agent-7"

    def test_claim_requires_owner(self, store: TaskStore) -> None:
        t = store.create("work")
        with pytest.raises(TypeError, match="owner"):
            store.claim(t.id)  # type: ignore[call-arg]

    def test_complete_requires_owner(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        with pytest.raises(TypeError, match="owner"):
            store.complete(t.id)  # type: ignore[call-arg]

    def test_fail_requires_owner(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        with pytest.raises(TypeError, match="owner"):
            store.fail(t.id)  # type: ignore[call-arg]

    def test_timeout_requires_owner(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        with pytest.raises(TypeError, match="owner"):
            store.timeout(t.id)  # type: ignore[call-arg]

    def test_non_owner_complete_rejected(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        with pytest.raises(ValueError, match="owned by"):
            store.complete(t.id, owner="agent-2")

    def test_non_owner_fail_rejected(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        with pytest.raises(ValueError, match="owned by"):
            store.fail(t.id, owner="agent-2", error="nope")

    def test_non_owner_timeout_rejected(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        with pytest.raises(ValueError, match="owned by"):
            store.timeout(t.id, owner="agent-2")

    def test_non_owner_cancel_rejected(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        with pytest.raises(ValueError, match="owned by"):
            store.cancel(t.id, owner="agent-2")

    def test_cancel_without_owner_on_unclaimed(self, store: TaskStore) -> None:
        t = store.create("work")
        cancelled = store.cancel(t.id, reason="not needed")
        assert cancelled.status == TaskStatus.CANCELLED.value

    def test_cancel_by_owner(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        cancelled = store.cancel(t.id, owner="agent-1")
        assert cancelled.status == TaskStatus.CANCELLED.value

    def test_claim_sets_lease_expires_at(self, store: TaskStore) -> None:
        t = store.create("work")
        claimed = store.claim(t.id, owner="agent-1", lease_seconds=120.0)
        assert claimed.lease_expires_at is not None

    def test_claim_default_lease_300s(self, store: TaskStore) -> None:
        t = store.create("work")
        claimed = store.claim(t.id, owner="agent-1")
        assert claimed.lease_expires_at is not None

    def test_lease_persisted_to_disk(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1", lease_seconds=60.0)
        loaded = store.load(t.id)
        assert loaded.lease_expires_at is not None

    def test_renew_lease_extends(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1", lease_seconds=1.0)
        renewed = store.renew_lease(t.id, owner="agent-1", lease_seconds=300.0)
        assert renewed.lease_expires_at is not None

    def test_renew_lease_non_owner_rejected(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        with pytest.raises(ValueError, match="owned by"):
            store.renew_lease(t.id, owner="agent-2")

    def test_renew_lease_non_in_progress_rejected(
        self, store: TaskStore
    ) -> None:
        t = store.create("work")
        with pytest.raises(ValueError, match="cannot renew lease"):
            store.renew_lease(t.id, owner="agent-1")

    def test_reclaim_after_lease_expired(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1", lease_seconds=0.01)
        time.sleep(0.02)
        re_claimed = store.claim(t.id, owner="agent-2")
        assert re_claimed.owner_agent_id == "agent-2"

    def test_reclaim_before_lease_expired_rejected(
        self, store: TaskStore
    ) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1", lease_seconds=300.0)
        with pytest.raises(ValueError, match="in_progress"):
            store.claim(t.id, owner="agent-2")

    def test_retry_clears_lease(self, store: TaskStore) -> None:
        t = store.create("work")
        store.claim(t.id, owner="agent-1")
        store.fail(t.id, owner="agent-1")
        retried = store.retry(t.id)
        assert retried.lease_expires_at is None

    def test_concurrent_claim_only_one_succeeds(
        self, store: TaskStore
    ) -> None:
        import threading

        t = store.create("shared work")
        results: list[bool] = []
        lock = threading.Lock()

        def try_claim(agent: str) -> None:
            try:
                store.claim(t.id, owner=agent)
                with lock:
                    results.append(True)
            except Exception:
                with lock:
                    results.append(False)

        threads = [
            threading.Thread(target=try_claim, args=(f"agent-{i}",))
            for i in range(10)
        ]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert results.count(True) == 1
        assert results.count(False) == 9

    def test_load_legacy_missing_lease(
        self, store: TaskStore, tmp_path: Path
    ) -> None:
        legacy = {
            "id": "task_11223344",
            "subject": "legacy",
            "description": "",
            "status": "pending",
            "owner": None,
            "blockedBy": [],
        }
        path = tmp_path / ".tasks" / "task_11223344.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(legacy))
        task = store.load("task_11223344")
        assert task.lease_expires_at is None
