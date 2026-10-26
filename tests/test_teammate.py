"""多 Agent 团队测试：MessageBus / PlanStore / Worktree / TeammateManager / 工具。"""

from __future__ import annotations

import asyncio

import pytest
from mcc_core import ToolContext
from mcc_harness import (
    MessageBus,
    PlanStore,
    TeammateManager,
    WorktreeError,
    create_worktree,
    format_team_events,
    remove_worktree,
)
from mcc_harness.plan import PlanError, PlanRequest
from mcc_harness.teammate import (
    LEAD_NAME,
    CreateWorktreeTool,
    ListTeammatesTool,
    RequestShutdownTool,
    ReviewPlanTool,
    SendTeammateMessageTool,
    SendToLeadTool,
    SpawnTeammateTool,
    SubmitPlanTool,
    TeammateRuntime,
    TeammateSpec,
)
from mcc_llm.mock import MockLlm, single_text_turn
from mcc_tools import ToolRegistry

# -- MessageBus --


class TestMessageBus:
    async def test_send_and_consume(self, tmp_path):
        bus = MessageBus(tmp_path)
        await bus.send("alice", "bob", "hello")
        assert await bus.peek("bob")
        msgs = await bus.consume("bob")
        assert len(msgs) == 1
        assert msgs[0].sender == "alice"
        assert msgs[0].recipient == "bob"
        assert msgs[0].content == "hello"
        assert not await bus.peek("bob")

    async def test_consume_empty(self, tmp_path):
        bus = MessageBus(tmp_path)
        assert await bus.consume("nobody") == []

    async def test_destructive_read(self, tmp_path):
        bus = MessageBus(tmp_path)
        await bus.send("a", "b", "msg1")
        await bus.send("a", "b", "msg2")
        msgs = await bus.consume("b")
        assert len(msgs) == 2
        assert await bus.consume("b") == []

    async def test_independent_mailboxes(self, tmp_path):
        bus = MessageBus(tmp_path)
        await bus.send("x", "alice", "for alice")
        await bus.send("x", "bob", "for bob")
        alice_msgs = await bus.consume("alice")
        bob_msgs = await bus.consume("bob")
        assert len(alice_msgs) == 1 and len(bob_msgs) == 1
        assert alice_msgs[0].content == "for alice"
        assert bob_msgs[0].content == "for bob"

    async def test_message_kinds_and_meta(self, tmp_path):
        bus = MessageBus(tmp_path)
        await bus.send("lead", "worker", "do task", kind="assignment", meta={"task_id": "t1"})
        msgs = await bus.consume("worker")
        assert msgs[0].kind == "assignment"
        assert msgs[0].meta == {"task_id": "t1"}

    async def test_concurrent_send(self, tmp_path):
        bus = MessageBus(tmp_path)
        await asyncio.gather(
            *[bus.send("sender", "box", f"msg{i}") for i in range(20)]
        )
        msgs = await bus.consume("box")
        assert len(msgs) == 20

    async def test_file_persistence(self, tmp_path):
        bus1 = MessageBus(tmp_path)
        await bus1.send("a", "b", "persisted")
        bus2 = MessageBus(tmp_path)
        msgs = await bus2.consume("b")
        assert len(msgs) == 1
        assert msgs[0].content == "persisted"


# -- PlanStore --


class TestPlanStore:
    async def test_create_and_get(self):
        store = PlanStore()
        req = await store.create("worker", "my plan", task_id="t1")
        assert req.status == "pending"
        assert req.sender == "worker"
        fetched = await store.get(req.request_id)
        assert fetched is not None
        assert fetched.plan == "my plan"

    async def test_approve(self):
        store = PlanStore()
        req = await store.create("worker", "plan")
        reviewed = await store.review(req.request_id, approve=True, feedback="ok")
        assert reviewed.status == "approved"
        assert reviewed.feedback == "ok"

    async def test_reject(self):
        store = PlanStore()
        req = await store.create("worker", "plan")
        reviewed = await store.review(req.request_id, approve=False, feedback="redo")
        assert reviewed.status == "rejected"

    async def test_review_unknown(self):
        store = PlanStore()
        with pytest.raises(PlanError):
            await store.review("nope", approve=True)

    async def test_double_review(self):
        store = PlanStore()
        req = await store.create("worker", "plan")
        await store.review(req.request_id, approve=True)
        with pytest.raises(PlanError):
            await store.review(req.request_id, approve=False)

    async def test_wait_for_review(self):
        store = PlanStore()
        req = await store.create("worker", "plan")

        async def approve_later():
            await asyncio.sleep(0.05)
            await store.review(req.request_id, approve=True, feedback="good")

        asyncio.create_task(approve_later())
        reviewed = await store.wait_for_review(req.request_id)
        assert reviewed.status == "approved"

    async def test_pending_for(self):
        store = PlanStore()
        await store.create("worker1", "plan1")
        await store.create("worker2", "plan2")
        pending = await store.pending_for("worker1")
        assert pending is not None
        assert pending.sender == "worker1"
        assert await store.pending_for("nobody") is None

    async def test_list_pending(self):
        store = PlanStore()
        r1 = await store.create("w1", "p1")
        r2 = await store.create("w2", "p2")
        await store.review(r1.request_id, approve=True)
        pending = await store.list_pending()
        assert len(pending) == 1
        assert pending[0].request_id == r2.request_id


# -- Worktree --


class TestWorktree:
    def test_create_and_remove_non_git(self, tmp_path):
        wt = create_worktree("wt1", "task_001", tmp_path)
        assert wt.exists()
        assert (wt / ".task_id").read_text() == "task_001"
        remove_worktree("wt1", tmp_path)
        assert not wt.exists()

    def test_duplicate_create(self, tmp_path):
        create_worktree("wt1", "task_001", tmp_path)
        with pytest.raises(WorktreeError):
            create_worktree("wt1", "task_002", tmp_path)

    def test_remove_nonexistent(self, tmp_path):
        with pytest.raises(WorktreeError):
            remove_worktree("nope", tmp_path)


# -- TeammateManager + TeammateRuntime --


def _make_manager(tmp_path, llm=None, max_teammates=4):
    registry = ToolRegistry()
    bus = MessageBus(tmp_path)
    plan_store = PlanStore()
    return TeammateManager(
        llm=llm or MockLlm([single_text_turn("done")]),
        registry=registry,
        bus=bus,
        plan_store=plan_store,
        base_dir=tmp_path,
        max_teammates=max_teammates,
    )


class TestTeammateManager:
    async def test_spawn_and_list(self, tmp_path):
        mgr = _make_manager(tmp_path)
        await mgr.spawn("worker1", "coder", "write hello.py")
        infos = mgr.list_active()
        assert len(infos) == 1
        assert infos[0].name == "worker1"
        assert infos[0].role == "coder"
        await mgr.shutdown_all()

    async def test_max_teammates(self, tmp_path):
        mgr = _make_manager(tmp_path, max_teammates=2)
        await mgr.spawn("w1", "coder", "task1")
        await mgr.spawn("w2", "coder", "task2")
        with pytest.raises(ValueError, match="max teammates"):
            await mgr.spawn("w3", "coder", "task3")
        await mgr.shutdown_all()

    async def test_duplicate_spawn(self, tmp_path):
        mgr = _make_manager(tmp_path)
        await mgr.spawn("worker", "coder", "task")
        with pytest.raises(ValueError, match="already active"):
            await mgr.spawn("worker", "coder", "task2")
        await mgr.shutdown_all()

    async def test_send_message(self, tmp_path):
        mgr = _make_manager(tmp_path)
        await mgr.spawn("worker", "coder", "initial task")
        result = await mgr.send_message("worker", "new instruction")
        assert "sent" in result.lower()
        await mgr.shutdown_all()

    async def test_send_to_nonexistent(self, tmp_path):
        mgr = _make_manager(tmp_path)
        with pytest.raises(ValueError, match="no active"):
            await mgr.send_message("ghost", "msg")

    async def test_shutdown(self, tmp_path):
        mgr = _make_manager(tmp_path)
        await mgr.spawn("worker", "coder", "task")
        result = await mgr.request_shutdown("worker")
        assert "shut down" in result.lower()
        assert mgr.list_active() == []

    async def test_consume_lead_inbox(self, tmp_path):
        mgr = _make_manager(tmp_path)
        await mgr.spawn("worker", "coder", "do something")
        await asyncio.sleep(0.3)
        inbox = await mgr.consume_lead_inbox()
        assert len(inbox) >= 1
        assert any(m.kind == "work_done" for m in inbox)
        await mgr.shutdown_all()

    async def test_shutdown_all(self, tmp_path):
        mgr = _make_manager(tmp_path, max_teammates=5)
        await mgr.spawn("w1", "coder", "t1")
        await mgr.spawn("w2", "coder", "t2")
        await mgr.shutdown_all()
        assert mgr.list_active() == []


class TestTeammateRuntime:
    async def test_lifecycle(self, tmp_path):
        bus = MessageBus(tmp_path)
        plan_store = PlanStore()
        registry = ToolRegistry()
        llm = MockLlm([single_text_turn("completed")])
        spec = TeammateSpec(
            name="rt_test", role="coder", system="test", model="claude-haiku-4-5",
        )
        rt = TeammateRuntime(
            spec=spec, llm=llm, registry=registry, bus=bus, plan_store=plan_store,
            base_dir=tmp_path,
        )
        await rt.start("do work")
        assert rt.alive
        await asyncio.sleep(0.3)
        msgs = await bus.consume(LEAD_NAME)
        assert any(m.kind == "work_done" for m in msgs)
        await rt.shutdown()
        assert not rt.alive

    async def test_notify_wakes_idle(self, tmp_path):
        bus = MessageBus(tmp_path)
        plan_store = PlanStore()
        registry = ToolRegistry()
        llm = MockLlm([
            single_text_turn("first done"),
            single_text_turn("second done"),
        ])
        spec = TeammateSpec(
            name="idle_test", role="coder", system="test", model="claude-haiku-4-5",
        )
        rt = TeammateRuntime(
            spec=spec, llm=llm, registry=registry, bus=bus, plan_store=plan_store,
            base_dir=tmp_path,
        )
        await rt.start("first task")
        await asyncio.sleep(0.3)
        await bus.send(LEAD_NAME, "idle_test", "second task", kind="message")
        rt.notify()
        await asyncio.sleep(0.3)
        msgs = await bus.consume(LEAD_NAME)
        assert len(msgs) >= 2
        await rt.shutdown()


# -- format_team_events --


class TestFormatTeamEvents:
    def test_empty(self):
        assert format_team_events([]) == "[team events]\n"

    def test_single(self):
        from mcc_harness.mailbox import Message

        msgs = [Message(sender="worker", recipient="lead", content="done", kind="work_done")]
        result = format_team_events(msgs)
        assert "[team events]" in result
        assert "WORK_DONE" in result
        assert "worker" in result
        assert "done" in result

    def test_multiple(self):
        from mcc_harness.mailbox import Message

        msgs = [
            Message(sender="w1", recipient="lead", content="msg1", kind="message"),
            Message(sender="w2", recipient="lead", content="msg2", kind="error"),
        ]
        result = format_team_events(msgs)
        assert "msg1" in result
        assert "msg2" in result
        assert "MESSAGE" in result
        assert "ERROR" in result


# -- Lead 工具 --


def _ctx(tmp_path):
    return ToolContext(cwd=tmp_path, session_id="test", depth=0)


class TestLeadTools:
    async def test_spawn_teammate_tool(self, tmp_path):
        mgr = _make_manager(tmp_path)
        tool = SpawnTeammateTool(mgr)
        ctx = _ctx(tmp_path)
        result = await tool.execute(
            {"name": "worker", "role": "coder", "prompt": "do stuff"}, ctx
        )
        assert not result.is_error
        assert "worker" in result.content
        await mgr.shutdown_all()

    async def test_spawn_teammate_missing_fields(self, tmp_path):
        mgr = _make_manager(tmp_path)
        tool = SpawnTeammateTool(mgr)
        ctx = _ctx(tmp_path)
        result = await tool.execute({"name": "worker"}, ctx)
        assert result.is_error

    async def test_list_teammates_tool(self, tmp_path):
        mgr = _make_manager(tmp_path)
        await mgr.spawn("w1", "coder", "task")
        tool = ListTeammatesTool(mgr)
        ctx = _ctx(tmp_path)
        result = await tool.execute({}, ctx)
        assert not result.is_error
        assert "w1" in result.content
        await mgr.shutdown_all()

    async def test_list_teammates_empty(self, tmp_path):
        mgr = _make_manager(tmp_path)
        tool = ListTeammatesTool(mgr)
        ctx = _ctx(tmp_path)
        result = await tool.execute({}, ctx)
        assert "No active" in result.content

    async def test_send_message_tool(self, tmp_path):
        mgr = _make_manager(tmp_path)
        await mgr.spawn("worker", "coder", "task")
        tool = SendTeammateMessageTool(mgr)
        ctx = _ctx(tmp_path)
        result = await tool.execute({"to": "worker", "content": "hello"}, ctx)
        assert not result.is_error
        await mgr.shutdown_all()

    async def test_request_shutdown_tool(self, tmp_path):
        mgr = _make_manager(tmp_path)
        await mgr.spawn("worker", "coder", "task")
        tool = RequestShutdownTool(mgr)
        ctx = _ctx(tmp_path)
        result = await tool.execute({"teammate": "worker"}, ctx)
        assert not result.is_error
        assert mgr.list_active() == []

    async def test_review_plan_tool(self, tmp_path):
        mgr = _make_manager(tmp_path)
        plan_store = mgr.plan_store
        req = await plan_store.create("worker", "my plan")
        tool = ReviewPlanTool(plan_store, mgr.bus)
        ctx = _ctx(tmp_path)
        result = await tool.execute(
            {"request_id": req.request_id, "approve": True, "feedback": "good"}, ctx
        )
        assert not result.is_error
        assert "approved" in result.content

    async def test_create_worktree_tool(self, tmp_path):
        tool = CreateWorktreeTool(tmp_path)
        ctx = _ctx(tmp_path)
        result = await tool.execute(
            {"name": "wt_test", "task_id": "task_001"}, ctx
        )
        assert not result.is_error
        assert "Worktree created" in result.content
        remove_worktree("wt_test", tmp_path)


# -- Teammate 专属工具 --


class TestTeammateTools:
    async def test_send_to_lead(self, tmp_path):
        bus = MessageBus(tmp_path)
        tool = SendToLeadTool(bus, "worker")
        ctx = _ctx(tmp_path)
        result = await tool.execute({"content": "progress update"}, ctx)
        assert not result.is_error
        msgs = await bus.consume(LEAD_NAME)
        assert len(msgs) == 1
        assert msgs[0].content == "progress update"

    async def test_send_to_lead_empty(self, tmp_path):
        bus = MessageBus(tmp_path)
        tool = SendToLeadTool(bus, "worker")
        ctx = _ctx(tmp_path)
        result = await tool.execute({"content": ""}, ctx)
        assert result.is_error

    async def test_submit_plan_approved(self, tmp_path):
        bus = MessageBus(tmp_path)
        plan_store = PlanStore()
        tool = SubmitPlanTool(plan_store, bus, "worker")
        ctx = _ctx(tmp_path)

        async def approve_later():
            await asyncio.sleep(0.05)
            pending = await plan_store.list_pending()
            if pending:
                await plan_store.review(pending[0].request_id, approve=True, feedback="ok")

        asyncio.create_task(approve_later())
        result = await tool.execute({"plan": "I will write tests"}, ctx)
        assert not result.is_error
        assert "approved" in result.content.lower()

    async def test_submit_plan_rejected(self, tmp_path):
        bus = MessageBus(tmp_path)
        plan_store = PlanStore()
        tool = SubmitPlanTool(plan_store, bus, "worker")
        ctx = _ctx(tmp_path)

        async def reject_later():
            await asyncio.sleep(0.05)
            pending = await plan_store.list_pending()
            if pending:
                await plan_store.review(pending[0].request_id, approve=False, feedback="no")

        asyncio.create_task(reject_later())
        result = await tool.execute({"plan": "bad plan"}, ctx)
        assert result.is_error
        assert "rejected" in result.content.lower()
