"""多 Agent 团队测试：MessageBus / PlanStore / Worktree / TeammateManager / 工具。"""

from __future__ import annotations

import asyncio
import json
import time

import pytest
from mcc_core import ToolContext
from mcc_harness import (
    Message,
    MessageBus,
    PlanStore,
    TeammateManager,
    WorktreeError,
    create_worktree,
    format_team_events,
    remove_worktree,
)
from mcc_harness.plan import PlanError, PlanRequest, PlanStatus
from mcc_harness.teammate import (
    _TEAMMATE_TRANSITIONS,
    LEAD_NAME,
    CreateWorktreeTool,
    CurrentExecution,
    ListTeammatesTool,
    RequestShutdownTool,
    ReviewPlanTool,
    SendTeammateMessageTool,
    SendToLeadTool,
    SpawnTeammateTool,
    SubmitPlanTool,
    TeammateRuntime,
    TeammateSpec,
    TeammateStateError,
    TeammateStatus,
    TransitionRecord,
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


class TestMessageBusReliability:
    """claim/ack/nack 可靠投递协议测试。"""

    async def test_claim_returns_pending(self, tmp_path):
        bus = MessageBus(tmp_path)
        await bus.send("a", "b", "msg1")
        await bus.send("a", "b", "msg2")
        claimed = await bus.claim("b")
        assert len(claimed) == 2
        assert all(m.status == "claimed" for m in claimed)
        assert all(m.claim_token is not None for m in claimed)

    async def test_claim_empty(self, tmp_path):
        bus = MessageBus(tmp_path)
        assert await bus.claim("nobody") == []

    async def test_claim_only_pending(self, tmp_path):
        bus = MessageBus(tmp_path)
        await bus.send("a", "b", "pending1")
        claimed1 = await bus.claim("b")
        await bus.send("a", "b", "pending2")
        claimed2 = await bus.claim("b")
        assert len(claimed1) == 1 and claimed1[0].content == "pending1"
        assert len(claimed2) == 1 and claimed2[0].content == "pending2"

    async def test_ack_removes_message(self, tmp_path):
        bus = MessageBus(tmp_path)
        await bus.send("a", "b", "hello")
        claimed = await bus.claim("b")
        ok = await bus.ack("b", claimed[0].message_id, claimed[0].claim_token)
        assert ok
        assert await bus.claim("b") == []

    async def test_ack_wrong_token(self, tmp_path):
        bus = MessageBus(tmp_path)
        await bus.send("a", "b", "hello")
        claimed = await bus.claim("b")
        ok = await bus.ack("b", claimed[0].message_id, "wrong-token")
        assert not ok
        assert await bus.peek("b")

    async def test_ack_nonexistent(self, tmp_path):
        bus = MessageBus(tmp_path)
        ok = await bus.ack("b", "no-such-id", "no-token")
        assert not ok

    async def test_nack_retry(self, tmp_path):
        bus = MessageBus(tmp_path)
        await bus.send("a", "b", "fail-me", max_attempts=3)
        claimed = await bus.claim("b")
        result = await bus.nack("b", claimed[0].message_id, claimed[0].claim_token, "boom")
        assert result == "retry"
        re_claimed = await bus.claim("b")
        assert len(re_claimed) == 1
        assert re_claimed[0].attempt == 1

    async def test_nack_dead(self, tmp_path):
        bus = MessageBus(tmp_path)
        await bus.send("a", "b", "doomed", max_attempts=1)
        claimed = await bus.claim("b")
        result = await bus.nack("b", claimed[0].message_id, claimed[0].claim_token, "fatal")
        assert result == "dead"
        dead = await bus.dead_letters("b")
        assert len(dead) == 1
        assert dead[0].status == "dead"
        assert await bus.claim("b") == []

    async def test_nack_reasons_recorded(self, tmp_path):
        bus = MessageBus(tmp_path)
        await bus.send("a", "b", "flaky", max_attempts=2)
        claimed = await bus.claim("b")
        await bus.nack("b", claimed[0].message_id, claimed[0].claim_token, "timeout")
        claimed2 = await bus.claim("b")
        await bus.nack("b", claimed2[0].message_id, claimed2[0].claim_token, "conn-reset")
        dead = await bus.dead_letters("b")
        assert dead[0].meta["nack_reasons"] == ["timeout", "conn-reset"]

    async def test_recover_stale_claim(self, tmp_path):
        bus = MessageBus(tmp_path)
        await bus.send("a", "b", "stale")
        claimed = await bus.claim("b")
        assert len(claimed) == 1
        recovered = await bus.recover("b", claim_timeout=0.0)
        assert recovered == 1
        re_claimed = await bus.claim("b")
        assert len(re_claimed) == 1
        assert re_claimed[0].content == "stale"

    async def test_recover_no_stale(self, tmp_path):
        bus = MessageBus(tmp_path)
        await bus.send("a", "b", "fresh")
        claimed = await bus.claim("b")
        assert len(claimed) == 1
        recovered = await bus.recover("b", claim_timeout=60.0)
        assert recovered == 0

    async def test_dead_letters_empty(self, tmp_path):
        bus = MessageBus(tmp_path)
        await bus.send("a", "b", "ok")
        assert await bus.dead_letters("b") == []

    async def test_send_idempotent(self, tmp_path):
        bus = MessageBus(tmp_path)
        msg1 = await bus.send("a", "b", "once", message_id="fixed-id")
        msg2 = await bus.send("a", "b", "once", message_id="fixed-id")
        assert msg1.message_id == msg2.message_id
        claimed = await bus.claim("b")
        assert len(claimed) == 1

    async def test_send_idempotent_allows_after_dead(self, tmp_path):
        bus = MessageBus(tmp_path)
        await bus.send("a", "b", "retry-me", message_id="mid", max_attempts=1)
        claimed = await bus.claim("b")
        await bus.nack("b", claimed[0].message_id, claimed[0].claim_token, "fail")
        msg2 = await bus.send("a", "b", "retry-me", message_id="mid")
        assert msg2.status == "pending"
        re_claimed = await bus.claim("b")
        assert len(re_claimed) == 1

    async def test_expired_message_skipped(self, tmp_path):
        bus = MessageBus(tmp_path)
        past = time.time() - 10
        await bus.send("a", "b", "old", expires_at=past)
        await bus.send("a", "b", "new")
        claimed = await bus.claim("b")
        assert len(claimed) == 1
        assert claimed[0].content == "new"

    async def test_claim_max_messages(self, tmp_path):
        bus = MessageBus(tmp_path)
        for i in range(5):
            await bus.send("a", "b", f"msg{i}")
        claimed = await bus.claim("b", max_messages=2)
        assert len(claimed) == 2
        claimed2 = await bus.claim("b", max_messages=2)
        assert len(claimed2) == 2
        claimed3 = await bus.claim("b", max_messages=2)
        assert len(claimed3) == 1

    async def test_consume_skips_bad_lines(self, tmp_path):
        bus = MessageBus(tmp_path)
        await bus.send("a", "b", "good1")
        path = bus._path("b")
        with path.open("a", encoding="utf-8") as f:
            f.write("{this is not valid json}\n")
        await bus.send("a", "b", "good2")
        msgs = await bus.consume("b")
        contents = [m.content for m in msgs]
        assert "good1" in contents
        assert "good2" in contents
        assert len(msgs) == 2

    async def test_claim_recover_on_claim(self, tmp_path):
        bus = MessageBus(tmp_path)
        await bus.send("a", "b", "auto-recover", max_attempts=3)
        claimed = await bus.claim("b", claim_timeout=0.0)
        assert len(claimed) == 1
        claimed2 = await bus.claim("b", claim_timeout=0.0)
        assert len(claimed2) == 1
        assert claimed2[0].attempt == 0

    async def test_full_ack_nack_cycle(self, tmp_path):
        bus = MessageBus(tmp_path)
        await bus.send("a", "b", "task1", max_attempts=2)
        await bus.send("a", "b", "task2", max_attempts=2)
        claimed = await bus.claim("b")
        assert len(claimed) == 2
        t1, t2 = claimed[0], claimed[1]
        await bus.ack("b", t1.message_id, t1.claim_token)
        await bus.nack("b", t2.message_id, t2.claim_token, "error")
        re_claimed = await bus.claim("b")
        assert len(re_claimed) == 1
        assert re_claimed[0].content == t2.content
        await bus.ack("b", re_claimed[0].message_id, re_claimed[0].claim_token)
        assert await bus.claim("b") == []

    async def test_message_new_fields_defaults(self, tmp_path):
        bus = MessageBus(tmp_path)
        msg = await bus.send("a", "b", "check")
        assert msg.message_id is not None
        assert msg.status == "pending"
        assert msg.attempt == 0
        assert msg.max_attempts == 3
        assert msg.priority == 0
        assert msg.correlation_id is None
        assert msg.task_id is None
        assert msg.expires_at is None

    async def test_legacy_message_compat(self, tmp_path):
        path = tmp_path / ".mailboxes"
        path.mkdir()
        legacy = {"sender": "old", "recipient": "b", "content": "legacy", "kind": "message"}
        with (path / "b.jsonl").open("w", encoding="utf-8") as f:
            f.write(json.dumps(legacy) + "\n")
        bus = MessageBus(tmp_path)
        msgs = await bus.consume("b")
        assert len(msgs) == 1
        assert msgs[0].content == "legacy"
        assert msgs[0].status == "pending"
        assert msgs[0].message_id is not None


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


class TestPlanStoreReliability:
    """PlanStore 超时、取消、幂等与持久化测试。"""

    async def test_wait_for_review_explicit_timeout(self):
        store = PlanStore()
        req = await store.create("worker", "plan")
        with pytest.raises(PlanError, match="timed out"):
            await store.wait_for_review(req.request_id, timeout=0.05)
        fetched = await store.get(req.request_id)
        assert fetched.status == PlanStatus.EXPIRED

    async def test_wait_for_review_deadline_timeout(self):
        store = PlanStore()
        deadline = time.time() + 0.05
        req = await store.create("worker", "plan", deadline=deadline)
        with pytest.raises(PlanError, match="timed out"):
            await store.wait_for_review(req.request_id)
        fetched = await store.get(req.request_id)
        assert fetched.status == PlanStatus.EXPIRED

    async def test_wait_for_review_succeeds_before_timeout(self):
        store = PlanStore()
        req = await store.create("worker", "plan")

        async def approve_later():
            await asyncio.sleep(0.02)
            await store.review(req.request_id, approve=True)

        asyncio.create_task(approve_later())
        reviewed = await store.wait_for_review(req.request_id, timeout=1.0)
        assert reviewed.status == PlanStatus.APPROVED

    async def test_cancel(self):
        store = PlanStore()
        req = await store.create("worker", "plan")
        cancelled = await store.cancel(req.request_id, reason="aborted")
        assert cancelled.status == PlanStatus.CANCELLED
        assert cancelled.feedback == "aborted"

    async def test_cancel_non_pending(self):
        store = PlanStore()
        req = await store.create("worker", "plan")
        await store.review(req.request_id, approve=True)
        with pytest.raises(PlanError):
            await store.cancel(req.request_id)

    async def test_cancel_unknown(self):
        store = PlanStore()
        with pytest.raises(PlanError):
            await store.cancel("nope")

    async def test_cancel_wakes_waiter(self):
        store = PlanStore()
        req = await store.create("worker", "plan")

        async def cancel_later():
            await asyncio.sleep(0.02)
            await store.cancel(req.request_id, reason="superseded")

        asyncio.create_task(cancel_later())
        result = await store.wait_for_review(req.request_id, timeout=1.0)
        assert result.status == PlanStatus.CANCELLED

    async def test_expire_overdue(self):
        store = PlanStore()
        past = time.time() - 1
        await store.create("w1", "p1", deadline=past)
        await store.create("w2", "p2", deadline=time.time() + 100)
        expired = await store.expire_overdue()
        assert len(expired) == 1
        assert expired[0].status == PlanStatus.EXPIRED

    async def test_expire_overdue_no_deadline(self):
        store = PlanStore()
        await store.create("w1", "p1")
        expired = await store.expire_overdue()
        assert expired == []

    async def test_expire_overdue_empty(self, tmp_path):
        store = PlanStore()
        assert await store.expire_overdue() == []

    async def test_review_idempotent_same_approve(self):
        store = PlanStore()
        req = await store.create("worker", "plan")
        r1 = await store.review(req.request_id, approve=True, feedback="ok")
        r2 = await store.review(req.request_id, approve=True, feedback="ok")
        assert r1.status == r2.status == PlanStatus.APPROVED

    async def test_review_idempotent_same_reject(self):
        store = PlanStore()
        req = await store.create("worker", "plan")
        r1 = await store.review(req.request_id, approve=False, feedback="no")
        r2 = await store.review(req.request_id, approve=False, feedback="no")
        assert r1.status == r2.status == PlanStatus.REJECTED

    async def test_review_approved_by(self):
        store = PlanStore()
        req = await store.create("worker", "plan")
        reviewed = await store.review(req.request_id, approve=True, approved_by="lead")
        assert reviewed.approved_by == "lead"
        assert reviewed.approved_at is not None

    async def test_plan_version_default(self):
        store = PlanStore()
        req = await store.create("worker", "plan")
        assert req.plan_version == 1

    async def test_create_with_deadline(self):
        store = PlanStore()
        deadline = time.time() + 60
        req = await store.create("worker", "plan", deadline=deadline)
        assert req.deadline == deadline

    async def test_create_with_worktree_id(self):
        store = PlanStore()
        req = await store.create("worker", "plan", worktree_id="wt_abc")
        assert req.worktree_id == "wt_abc"

    async def test_persistence_save_load(self, tmp_path):
        store1 = PlanStore(base_dir=tmp_path)
        req = await store1.create("worker", "my plan", task_id="t1")
        await store1.review(req.request_id, approve=True, feedback="good", approved_by="lead")
        store2 = PlanStore(base_dir=tmp_path)
        fetched = await store2.get(req.request_id)
        assert fetched is not None
        assert fetched.plan == "my plan"
        assert fetched.status == PlanStatus.APPROVED
        assert fetched.approved_by == "lead"

    async def test_persistence_restore_pending(self, tmp_path):
        store1 = PlanStore(base_dir=tmp_path)
        req = await store1.create("worker", "pending plan")
        store2 = PlanStore(base_dir=tmp_path)
        fetched = await store2.get(req.request_id)
        assert fetched is not None
        assert fetched.status == PlanStatus.PENDING
        pending = await store2.list_pending()
        assert len(pending) == 1

    async def test_persistence_restore_reviewed(self, tmp_path):
        store1 = PlanStore(base_dir=tmp_path)
        r1 = await store1.create("w1", "p1")
        r2 = await store1.create("w2", "p2")
        await store1.review(r1.request_id, approve=True)
        await store1.review(r2.request_id, approve=False, feedback="redo")
        store2 = PlanStore(base_dir=tmp_path)
        f1 = await store2.get(r1.request_id)
        f2 = await store2.get(r2.request_id)
        assert f1.status == PlanStatus.APPROVED
        assert f2.status == PlanStatus.REJECTED
        assert f2.feedback == "redo"
        assert await store2.list_pending() == []

    async def test_persistence_counter_continues(self, tmp_path):
        store1 = PlanStore(base_dir=tmp_path)
        req1 = await store1.create("w1", "p1")
        store2 = PlanStore(base_dir=tmp_path)
        req2 = await store2.create("w2", "p2")
        assert req2.request_id != req1.request_id

    async def test_persistence_expire_overdue(self, tmp_path):
        store1 = PlanStore(base_dir=tmp_path)
        past = time.time() - 1
        await store1.create("w1", "p1", deadline=past)
        store2 = PlanStore(base_dir=tmp_path)
        expired = await store2.expire_overdue()
        assert len(expired) == 1

    async def test_status_enum_values(self):
        assert PlanStatus.PENDING == "pending"
        assert PlanStatus.APPROVED == "approved"
        assert PlanStatus.REJECTED == "rejected"
        assert PlanStatus.EXPIRED == "expired"
        assert PlanStatus.CANCELLED == "cancelled"

    async def test_review_expired_raises(self):
        store = PlanStore()
        req = await store.create("worker", "plan", deadline=time.time() - 1)
        with pytest.raises(PlanError):
            await store.wait_for_review(req.request_id, timeout=0.01)
        with pytest.raises(PlanError):
            await store.review(req.request_id, approve=True)

    async def test_no_persistence_by_default(self):
        store = PlanStore()
        assert store._base_dir is None


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
        msgs = [Message(sender="worker", recipient="lead", content="done", kind="work_done")]
        result = format_team_events(msgs)
        assert "[team events]" in result
        assert "WORK_DONE" in result
        assert "worker" in result
        assert "done" in result

    def test_multiple(self):
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


# -- P1-07: Teammate 生命周期状态机 --


class TestTeammateStatusMatrix:
    """状态迁移矩阵校验：合法迁移允许，非法迁移拒绝。"""

    @pytest.mark.parametrize("frm,to", [
        (TeammateStatus.CREATED, TeammateStatus.STARTING),
        (TeammateStatus.CREATED, TeammateStatus.SHUTTING_DOWN),
        (TeammateStatus.STARTING, TeammateStatus.IDLE),
        (TeammateStatus.STARTING, TeammateStatus.ASSIGNED),
        (TeammateStatus.IDLE, TeammateStatus.ASSIGNED),
        (TeammateStatus.ASSIGNED, TeammateStatus.PLANNING),
        (TeammateStatus.ASSIGNED, TeammateStatus.RUNNING),
        (TeammateStatus.PLANNING, TeammateStatus.WAITING_APPROVAL),
        (TeammateStatus.PLANNING, TeammateStatus.FAILED),
        (TeammateStatus.WAITING_APPROVAL, TeammateStatus.RUNNING),
        (TeammateStatus.WAITING_APPROVAL, TeammateStatus.IDLE),
        (TeammateStatus.RUNNING, TeammateStatus.REPORTING),
        (TeammateStatus.RUNNING, TeammateStatus.FAILED),
        (TeammateStatus.RUNNING, TeammateStatus.TIMEOUT),
        (TeammateStatus.RUNNING, TeammateStatus.CANCELLED),
        (TeammateStatus.REPORTING, TeammateStatus.IDLE),
        (TeammateStatus.FAILED, TeammateStatus.IDLE),
        (TeammateStatus.FAILED, TeammateStatus.RESTARTING),
        (TeammateStatus.TIMEOUT, TeammateStatus.RESTARTING),
        (TeammateStatus.CANCELLED, TeammateStatus.IDLE),
        (TeammateStatus.RESTARTING, TeammateStatus.IDLE),
        (TeammateStatus.SHUTTING_DOWN, TeammateStatus.SHUTDOWN),
    ])
    def test_legal_transition_allowed(self, frm, to):
        assert to in _TEAMMATE_TRANSITIONS[frm]

    @pytest.mark.parametrize("frm,to", [
        (TeammateStatus.CREATED, TeammateStatus.IDLE),
        (TeammateStatus.CREATED, TeammateStatus.RUNNING),
        (TeammateStatus.IDLE, TeammateStatus.RUNNING),
        (TeammateStatus.IDLE, TeammateStatus.PLANNING),
        (TeammateStatus.RUNNING, TeammateStatus.IDLE),
        (TeammateStatus.RUNNING, TeammateStatus.ASSIGNED),
        (TeammateStatus.SHUTDOWN, TeammateStatus.IDLE),
        (TeammateStatus.SHUTDOWN, TeammateStatus.SHUTTING_DOWN),
        (TeammateStatus.IDLE, TeammateStatus.CREATED),
        (TeammateStatus.REPORTING, TeammateStatus.RUNNING),
    ])
    def test_illegal_transition_blocked(self, frm, to):
        assert to not in _TEAMMATE_TRANSITIONS[frm]

    def test_shutdown_is_terminal(self):
        assert _TEAMMATE_TRANSITIONS[TeammateStatus.SHUTDOWN] == frozenset()

    def test_all_statuses_covered(self):
        assert set(_TEAMMATE_TRANSITIONS) == set(TeammateStatus)

    def test_no_self_transition(self):
        for frm, targets in _TEAMMATE_TRANSITIONS.items():
            assert frm not in targets


class TestTeammateTransitionMethod:
    """_transition 方法：校验、更新、记录。"""

    def _make_rt(self, tmp_path):
        bus = MessageBus(tmp_path)
        plan_store = PlanStore()
        registry = ToolRegistry()
        llm = MockLlm([single_text_turn("done")])
        spec = TeammateSpec(name="t", role="coder", system="s", model="m")
        return TeammateRuntime(
            spec=spec, llm=llm, registry=registry, bus=bus, plan_store=plan_store,
        )

    def test_illegal_raises(self, tmp_path):
        rt = self._make_rt(tmp_path)
        with pytest.raises(TeammateStateError, match="cannot transition"):
            rt._transition(TeammateStatus.IDLE)

    def test_legal_updates_status(self, tmp_path):
        rt = self._make_rt(tmp_path)
        rt._transition(TeammateStatus.STARTING, reason="start")
        assert rt.status == "starting"

    def test_records_transition(self, tmp_path):
        rt = self._make_rt(tmp_path)
        rt._transition(TeammateStatus.STARTING, reason="start")
        rec = rt._last_transition
        assert rec.from_status == TeammateStatus.CREATED
        assert rec.to_status == TeammateStatus.STARTING
        assert rec.reason == "start"
        assert rec.timestamp > 0

    def test_info_exposes_last_transition(self, tmp_path):
        rt = self._make_rt(tmp_path)
        rt._transition(TeammateStatus.STARTING, reason="start")
        info = rt.info()
        assert info.last_transition == "created->starting (start)"

    def test_safe_shutdown_graceful(self, tmp_path):
        rt = self._make_rt(tmp_path)
        rt._safe_shutdown()
        assert rt.status == "shutdown"
        assert rt._last_transition.reason == "graceful exit"

    def test_safe_shutdown_forced(self, tmp_path):
        rt = self._make_rt(tmp_path)
        rt._force_stop = True
        rt._safe_shutdown()
        assert rt.status == "shutdown"
        assert rt._last_transition.reason == "force terminated"

    def test_safe_shutdown_idempotent(self, tmp_path):
        rt = self._make_rt(tmp_path)
        rt._safe_shutdown()
        rt._safe_shutdown()
        assert rt.status == "shutdown"


class TestCurrentExecution:
    """CurrentExecution：active 判定与 clear。"""

    def test_default_inactive(self):
        ce = CurrentExecution()
        assert not ce.active
        assert ce.task_id is None
        assert ce.assignment is None
        assert ce.work_version == 0

    def test_active_with_task(self):
        ce = CurrentExecution(task_id="t1", assignment="work")
        assert ce.active

    def test_active_with_assignment_only(self):
        ce = CurrentExecution(assignment="work")
        assert ce.active

    def test_clear_resets_all(self):
        ce = CurrentExecution(
            task_id="t1", assignment="work", work_version=3, started_at=1.0,
        )
        ce.clear()
        assert ce.task_id is None
        assert ce.assignment is None
        assert ce.work_version == 0
        assert ce.started_at is None
        assert not ce.active

    def test_cancel_token_default_none(self):
        ce = CurrentExecution()
        assert ce.cancel_token is None


class TestTeammateRecovery:
    """P1-07: 异常退出后状态识别与 CurrentExecution 清空。"""

    async def test_normal_completion_clears_current(self, tmp_path):
        bus = MessageBus(tmp_path)
        plan_store = PlanStore()
        registry = ToolRegistry()
        llm = MockLlm([single_text_turn("done")])
        spec = TeammateSpec(name="rec", role="coder", system="s", model="m")
        rt = TeammateRuntime(
            spec=spec, llm=llm, registry=registry, bus=bus, plan_store=plan_store,
        )
        await rt.start("do work")
        await asyncio.sleep(0.3)
        await rt.shutdown()
        assert rt.status == "shutdown"
        assert not rt._current.active

    async def test_graceful_shutdown_via_manager(self, tmp_path):
        mgr = _make_manager(tmp_path)
        await mgr.spawn("w", "coder", "task")
        await asyncio.sleep(0.3)
        await mgr.request_shutdown("w")
        assert mgr.list_active() == []


# -- P1-01: 任务结果分发 --


class ErrorLlm:
    """始终抛出异常的 LLM。"""

    async def complete(self, req):
        raise RuntimeError("LLM boom")

    async def stream(self, req):
        raise RuntimeError("LLM boom")
        yield  # make async generator


class SlowLlm:
    """响应慢的 LLM，用于触发超时。"""

    async def complete(self, req):
        await asyncio.sleep(10)
        from mcc_llm.mock import single_text_turn
        return single_text_turn("late")

    async def stream(self, req):
        await asyncio.sleep(10)
        yield  # make async generator


class TestTaskOutcomeDispatch:
    """验证任务结果正确分发：成功→COMPLETED，异常→FAILED，超时→TIMEOUT。"""

    def _make_rt(self, tmp_path, llm, timeout_secs=300.0, task_store=None):
        from mcc_tools import TaskStore as _TS

        bus = MessageBus(tmp_path)
        plan_store = PlanStore()
        registry = ToolRegistry()
        ts = task_store or _TS(tmp_path / ".tasks")
        spec = TeammateSpec(
            name="dispatch", role="coder", system="test",
            model="claude-haiku-4-5", timeout_secs=timeout_secs,
        )
        rt = TeammateRuntime(
            spec=spec, llm=llm, registry=registry, bus=bus,
            plan_store=plan_store, task_store=ts, base_dir=tmp_path,
        )
        return rt, bus, ts

    async def test_success_completes_task(self, tmp_path):
        from mcc_tools import TaskStatus

        llm = MockLlm([single_text_turn("done")])
        rt, bus, ts = self._make_rt(tmp_path, llm)
        task = ts.create("test task")
        ts.claim(task.id, owner="dispatch")

        await rt.start("do work", task_id=task.id)
        await asyncio.sleep(0.3)
        msgs = await bus.consume(LEAD_NAME)
        await rt.shutdown()

        assert any(m.kind == "work_done" for m in msgs)
        final = ts.load(task.id)
        assert final.status == TaskStatus.COMPLETED.value

    async def test_exception_fails_task(self, tmp_path):
        from mcc_tools import TaskStatus

        llm = ErrorLlm()
        rt, bus, ts = self._make_rt(tmp_path, llm)
        task = ts.create("test task")
        ts.claim(task.id, owner="dispatch")

        await rt.start("do work", task_id=task.id)
        await asyncio.sleep(0.3)
        msgs = await bus.consume(LEAD_NAME)
        await rt.shutdown()

        assert any(m.kind == "work_failed" for m in msgs)
        final = ts.load(task.id)
        assert final.status == TaskStatus.FAILED.value
        assert final.error is not None
        assert "LLM boom" in final.error

    async def test_timeout_times_out_task(self, tmp_path):
        from mcc_tools import TaskStatus

        llm = SlowLlm()
        rt, bus, ts = self._make_rt(tmp_path, llm, timeout_secs=0.05)
        task = ts.create("test task")
        ts.claim(task.id, owner="dispatch")

        await rt.start("do work", task_id=task.id)
        await asyncio.sleep(0.3)
        msgs = await bus.consume(LEAD_NAME)
        await rt.shutdown()

        assert any(m.kind == "work_timeout" for m in msgs)
        final = ts.load(task.id)
        assert final.status == TaskStatus.TIMEOUT.value

    async def test_no_complete_on_failure(self, tmp_path):
        """回归测试：异常路径不得调用 complete()。"""
        from mcc_tools import TaskStatus

        llm = ErrorLlm()
        rt, bus, ts = self._make_rt(tmp_path, llm)
        task = ts.create("regression task")
        ts.claim(task.id, owner="dispatch")

        await rt.start("do work", task_id=task.id)
        await asyncio.sleep(0.3)
        await rt.shutdown()

        final = ts.load(task.id)
        assert final.status != TaskStatus.COMPLETED.value
        assert final.status == TaskStatus.FAILED.value


# -- P1-05: 运行中指令注入与取消控制 --


def _p1_05_tool_use_turn(tool_id: str, tool_name: str, tool_input: dict) -> list:
    """构造工具调用轮次（stop_reason=tool_use 使循环继续）。"""
    from mcc_core import Usage
    from mcc_llm import MessageStop, ToolUseInputDelta, ToolUseStart

    return [
        ToolUseStart(id=tool_id, name=tool_name),
        ToolUseInputDelta(partial_json=json.dumps(tool_input)),
        MessageStop(stop_reason="tool_use", usage=Usage()),
    ]


class _RecordingLlm:
    """包装 MockLlm，记录每次 stream 调用收到的 messages。"""

    def __init__(self, mock: MockLlm) -> None:
        self._mock = mock
        self.calls: list[list] = []

    @property
    def call_count(self) -> int:
        return self._mock.call_count

    async def complete(self, req):
        return await self._mock.complete(req)

    async def stream(self, req):
        self.calls.append(req.messages)
        async for ev in self._mock.stream(req):
            yield ev


def _user_texts(messages: list) -> list[str]:
    """从 messages 中提取所有 user 角色的文本块。"""
    from mcc_core import Role

    return [
        b.text
        for m in messages
        if m.role == Role.user
        for b in m.content
        if hasattr(b, "text")
    ]


class TestP105InstructionInjectAndCancel:
    """P1-05: 运行中指令注入与取消控制。"""

    # === Agent 级别：run_streaming 控制参数 ===

    async def test_instruction_queue_injects_user_message(self, tmp_path):
        """instruction_queue 中的内容在迭代边界注入为 user 消息。"""
        from mcc_core import ToolContext
        from mcc_harness.agent import AgentLoopBuilder
        from mcc_tools import default_registry

        llm = _RecordingLlm(MockLlm([single_text_turn("done")]))
        ctx = ToolContext(cwd=tmp_path, session_id="p105", depth=0)
        registry = default_registry()
        agent = (
            AgentLoopBuilder(llm, registry, ctx)
            .model("claude-haiku-4-5-20251001")
            .build()
        )

        iq: asyncio.Queue[str] = asyncio.Queue()
        iq.put_nowait("pivot to new approach")

        queue: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("do work", queue, instruction_queue=iq)

        assert llm.call_count == 1
        texts = _user_texts(llm.calls[0])
        assert any("do work" in t for t in texts)
        assert any("[instruction] pivot to new approach" in t for t in texts)

    async def test_multiple_instructions_combined(self, tmp_path):
        """多条指令合并为一条 user 消息注入。"""
        from mcc_core import ToolContext
        from mcc_harness.agent import AgentLoopBuilder
        from mcc_tools import default_registry

        llm = _RecordingLlm(MockLlm([single_text_turn("done")]))
        ctx = ToolContext(cwd=tmp_path, session_id="p105", depth=0)
        registry = default_registry()
        agent = (
            AgentLoopBuilder(llm, registry, ctx)
            .model("claude-haiku-4-5-20251001")
            .build()
        )

        iq: asyncio.Queue[str] = asyncio.Queue()
        iq.put_nowait("step 1")
        iq.put_nowait("step 2")

        queue: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("do work", queue, instruction_queue=iq)

        texts = _user_texts(llm.calls[0])
        assert any("[instruction] step 1" in t and "[instruction] step 2" in t for t in texts)

    async def test_instruction_queue_injects_between_turns(self, tmp_path):
        """指令在工具调用轮次之间注入：第二轮 LLM 调用应包含指令。"""
        from mcc_core import ToolContext, ToolOutput
        from mcc_harness.agent import AgentLoopBuilder
        from mcc_tools import default_registry

        class _InjectTool:
            name = "inject_instruction"
            description = "inject instruction into queue"

            def __init__(self, queue, text):
                self._queue = queue
                self._text = text

            def input_schema(self):
                return {"type": "object", "properties": {}}

            async def execute(self, input, ctx):
                self._queue.put_nowait(self._text)
                return ToolOutput.ok("injected")

        iq: asyncio.Queue[str] = asyncio.Queue()
        turn1 = _p1_05_tool_use_turn("t1", "inject_instruction", {})
        turn2 = single_text_turn("done")
        llm = _RecordingLlm(MockLlm([turn1, turn2]))
        ctx = ToolContext(cwd=tmp_path, session_id="p105", depth=0)
        registry = default_registry()
        registry.register(_InjectTool(iq, "switch strategy"))
        agent = (
            AgentLoopBuilder(llm, registry, ctx)
            .model("claude-haiku-4-5-20251001")
            .build()
        )

        queue: asyncio.Queue = asyncio.Queue()
        await agent.run_streaming("do work", queue, instruction_queue=iq)

        assert llm.call_count == 2
        texts = _user_texts(llm.calls[1])
        assert any("[instruction] switch strategy" in t for t in texts)

    async def test_cancel_event_returns_cancelled(self, tmp_path):
        """cancel_event 置位后 run_streaming 返回 "(cancelled)"。"""
        from mcc_core import ToolContext, ToolOutput
        from mcc_harness.agent import AgentLoopBuilder
        from mcc_tools import default_registry

        class _CancelTrigger:
            name = "trigger_cancel"
            description = "trigger cancel event"

            def __init__(self, event):
                self._event = event

            def input_schema(self):
                return {"type": "object", "properties": {}}

            async def execute(self, input, ctx):
                self._event.set()
                return ToolOutput.ok("cancelled")

        cancel_event = asyncio.Event()
        turn1 = _p1_05_tool_use_turn("t1", "trigger_cancel", {})
        turn2 = single_text_turn("done")
        llm = MockLlm([turn1, turn2])
        ctx = ToolContext(cwd=tmp_path, session_id="p105", depth=0)
        registry = default_registry()
        registry.register(_CancelTrigger(cancel_event))
        agent = (
            AgentLoopBuilder(llm, registry, ctx)
            .model("claude-haiku-4-5-20251001")
            .build()
        )

        queue: asyncio.Queue = asyncio.Queue()
        run = await agent.run_streaming(
            "do work", queue, cancel_event=cancel_event
        )

        assert run.final_text == "(cancelled)"

    async def test_pause_event_blocks_until_resumed(self, tmp_path):
        """pause_event 清零后阻塞，置位后继续执行。"""
        from mcc_core import ToolContext, ToolOutput
        from mcc_core.event import Notice
        from mcc_harness.agent import AgentLoopBuilder
        from mcc_tools import default_registry

        class _PauseTrigger:
            name = "trigger_pause"
            description = "trigger pause event"

            def __init__(self, event):
                self._event = event

            def input_schema(self):
                return {"type": "object", "properties": {}}

            async def execute(self, input, ctx):
                self._event.clear()
                asyncio.get_event_loop().call_later(0.1, self._event.set)
                return ToolOutput.ok("paused")

        pause_event = asyncio.Event()
        pause_event.set()
        turn1 = _p1_05_tool_use_turn("t1", "trigger_pause", {})
        turn2 = single_text_turn("done")
        llm = MockLlm([turn1, turn2])
        ctx = ToolContext(cwd=tmp_path, session_id="p105", depth=0)
        registry = default_registry()
        registry.register(_PauseTrigger(pause_event))
        agent = (
            AgentLoopBuilder(llm, registry, ctx)
            .model("claude-haiku-4-5-20251001")
            .build()
        )

        queue: asyncio.Queue = asyncio.Queue()
        run = await agent.run_streaming(
            "do work", queue, pause_event=pause_event
        )

        assert run.final_text == "done"
        events: list = []
        while not queue.empty():
            events.append(queue.get_nowait())
        notices = [e for e in events if isinstance(e, Notice)]
        assert any("[paused]" in n.text for n in notices)
        assert any("[resumed]" in n.text for n in notices)

    # === Teammate 级别：消息路由与队列 ===

    def _make_rt_p1_05(self, tmp_path, llm, timeout_secs=300.0):
        from mcc_tools import TaskStore as _TS
        from mcc_tools import default_registry

        bus = MessageBus(tmp_path)
        plan_store = PlanStore()
        registry = default_registry()
        ts = _TS(tmp_path / ".tasks")
        spec = TeammateSpec(
            name="p105worker",
            role="coder",
            system="test",
            model="claude-haiku-4-5",
            timeout_secs=timeout_secs,
            tools=["run_bash"],
        )
        rt = TeammateRuntime(
            spec=spec,
            llm=llm,
            registry=registry,
            bus=bus,
            plan_store=plan_store,
            task_store=ts,
            base_dir=tmp_path,
        )
        return rt, bus, ts

    async def test_assignment_does_not_overwrite_current(self, tmp_path):
        """运行中收到新 assignment 时排队，不覆盖当前任务。"""
        from mcc_tools import TaskStatus

        turn1 = _p1_05_tool_use_turn("t1", "run_bash", {"command": "echo A"})
        turn2 = single_text_turn("task A done")
        turn3 = single_text_turn("task B done")
        llm = MockLlm([turn1, turn2, turn3])
        rt, bus, ts = self._make_rt_p1_05(tmp_path, llm)

        task_a = ts.create("task A")
        ts.claim(task_a.id, owner="p105worker")

        await rt.start("task A", task_id=task_a.id)
        await asyncio.sleep(0.1)

        await bus.send(LEAD_NAME, "p105worker", "task B", kind="assignment")
        rt.notify()

        await asyncio.sleep(1.0)
        msgs = await bus.consume(LEAD_NAME)
        await rt.shutdown()

        work_done_msgs = [m for m in msgs if m.kind == "work_done"]
        assert len(work_done_msgs) >= 1
        assert any("task A done" in m.content for m in work_done_msgs)
        final_a = ts.load(task_a.id)
        assert final_a.status == TaskStatus.COMPLETED.value

    async def test_cancel_sets_cancelled_status(self, tmp_path):
        """取消信号导致任务进入 CANCELLED 状态。"""
        from mcc_core import ToolOutput
        from mcc_tools import TaskStatus, default_registry
        from mcc_tools import TaskStore as _TS

        started = asyncio.Event()

        class _SignalAndWait:
            name = "signal_and_wait"
            description = "signal start then wait"

            def input_schema(self):
                return {"type": "object", "properties": {}}

            async def execute(self, input, ctx):
                started.set()
                await asyncio.sleep(0.2)
                return ToolOutput.ok("done")

        turn1 = _p1_05_tool_use_turn("t1", "signal_and_wait", {})
        turn2 = single_text_turn("done")
        llm = MockLlm([turn1, turn2])

        bus = MessageBus(tmp_path)
        plan_store = PlanStore()
        registry = default_registry()
        registry.register(_SignalAndWait())
        ts = _TS(tmp_path / ".tasks")
        spec = TeammateSpec(
            name="p105worker",
            role="coder",
            system="test",
            model="claude-haiku-4-5",
            tools=["signal_and_wait"],
        )
        rt = TeammateRuntime(
            spec=spec, llm=llm, registry=registry, bus=bus,
            plan_store=plan_store, task_store=ts, base_dir=tmp_path,
        )

        task = ts.create("cancellable task")
        ts.claim(task.id, owner="p105worker")

        await rt.start("do work", task_id=task.id)
        await started.wait()
        assert rt._current.cancel_token is not None
        rt._current.cancel_token.set()

        await asyncio.sleep(0.5)
        msgs = await bus.consume(LEAD_NAME)
        await rt.shutdown()

        assert any(m.kind == "work_cancelled" for m in msgs)
        final = ts.load(task.id)
        assert final.status == TaskStatus.CANCELLED.value

    # === TeammateManager 新方法 ===

    async def test_send_instruction_returns_confirmation(self, tmp_path):
        """send_instruction 返回确认消息。"""
        mgr = _make_manager(tmp_path)
        await mgr.spawn("worker", "coder", "initial task")
        result = await mgr.send_instruction("worker", "new approach")
        assert "Instruction sent" in result
        await mgr.shutdown_all()

    async def test_assign_task_returns_confirmation(self, tmp_path):
        """assign_task 返回确认消息。"""
        mgr = _make_manager(tmp_path)
        await mgr.spawn("worker", "coder", "initial task")
        result = await mgr.assign_task("worker", "new task")
        assert "Task assigned" in result
        await mgr.shutdown_all()

    async def test_cancel_teammate_returns_confirmation(self, tmp_path):
        """cancel_teammate 返回确认消息。"""
        mgr = _make_manager(tmp_path)
        await mgr.spawn("worker", "coder", "initial task")
        result = await mgr.cancel_teammate("worker")
        assert "Cancel signal sent" in result
        await mgr.shutdown_all()

    async def test_pause_and_resume_return_confirmation(self, tmp_path):
        """pause_teammate / resume_teammate 返回确认消息。"""
        mgr = _make_manager(tmp_path)
        await mgr.spawn("worker", "coder", "initial task")
        assert "Pause signal sent" in await mgr.pause_teammate("worker")
        assert "Resume signal sent" in await mgr.resume_teammate("worker")
        await mgr.shutdown_all()

    async def test_send_instruction_nonexistent_raises(self, tmp_path):
        """向不存在的 teammate 发指令抛 ValueError。"""
        mgr = _make_manager(tmp_path)
        with pytest.raises(ValueError, match="no active"):
            await mgr.send_instruction("ghost", "content")

    async def test_assign_task_nonexistent_raises(self, tmp_path):
        """向不存在的 teammate 分配任务抛 ValueError。"""
        mgr = _make_manager(tmp_path)
        with pytest.raises(ValueError, match="no active"):
            await mgr.assign_task("ghost", "content")

    async def test_cancel_teammate_nonexistent_raises(self, tmp_path):
        """取消不存在的 teammate 抛 ValueError。"""
        mgr = _make_manager(tmp_path)
        with pytest.raises(ValueError, match="no active"):
            await mgr.cancel_teammate("ghost")

    async def test_pause_teammate_nonexistent_raises(self, tmp_path):
        """暂停不存在的 teammate 抛 ValueError。"""
        mgr = _make_manager(tmp_path)
        with pytest.raises(ValueError, match="no active"):
            await mgr.pause_teammate("ghost")

    async def test_resume_teammate_nonexistent_raises(self, tmp_path):
        """恢复不存在的 teammate 抛 ValueError。"""
        mgr = _make_manager(tmp_path)
        with pytest.raises(ValueError, match="no active"):
            await mgr.resume_teammate("ghost")
