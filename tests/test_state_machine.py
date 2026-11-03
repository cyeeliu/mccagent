"""AgentStateMachine 测试：迁移矩阵、非法迁移、守卫、订阅、并发、审计。

P2-02 验收标准：
- 状态机可独立单测。
- 覆盖迁移矩阵、非法迁移和并发更新。
- 状态迁移记录 reason、timestamp、actor 和 trace_id。
"""

from __future__ import annotations

import threading
from contextlib import suppress

import pytest
from mcc_harness import (
    AgentState,
    AgentStateMachine,
    CallbackGuard,
    GuardRejectedError,
    IllegalTransitionError,
    RuntimeMode,
    StateTransition,
)

# -- 基础 fixture --


@pytest.fixture
def ephemeral_sm() -> AgentStateMachine:
    return AgentStateMachine("agent-1", mode=RuntimeMode.EPHEMERAL)


@pytest.fixture
def persistent_sm() -> AgentStateMachine:
    return AgentStateMachine("agent-2", mode=RuntimeMode.PERSISTENT)


# -- Ephemeral 迁移矩阵 --


class TestEphemeralTransitions:
    """Ephemeral 模式合法迁移矩阵全覆盖。"""

    @pytest.mark.parametrize("target", [
        AgentState.RUNNING,
        AgentState.CANCELLED,
        AgentState.FAILED,
    ])
    def test_from_created(self, ephemeral_sm, target):
        ephemeral_sm.transition(target, reason="test")
        assert ephemeral_sm.state == target

    @pytest.mark.parametrize("target", [
        AgentState.PAUSED,
        AgentState.COMPLETED,
        AgentState.FAILED,
        AgentState.CANCELLED,
        AgentState.TIMEOUT,
        AgentState.BUDGET_EXCEEDED,
    ])
    def test_from_running(self, ephemeral_sm, target):
        ephemeral_sm.transition(AgentState.RUNNING)
        ephemeral_sm.transition(target, reason="test")
        assert ephemeral_sm.state == target

    @pytest.mark.parametrize("target", [
        AgentState.RUNNING,
        AgentState.CANCELLED,
    ])
    def test_from_paused(self, ephemeral_sm, target):
        ephemeral_sm.transition(AgentState.RUNNING)
        ephemeral_sm.transition(AgentState.PAUSED)
        ephemeral_sm.transition(target, reason="resume")
        assert ephemeral_sm.state == target

    @pytest.mark.parametrize("terminal", [
        AgentState.COMPLETED,
        AgentState.FAILED,
        AgentState.CANCELLED,
        AgentState.TIMEOUT,
        AgentState.BUDGET_EXCEEDED,
    ])
    def test_terminal_is_stuck(self, ephemeral_sm, terminal):
        ephemeral_sm.transition(AgentState.RUNNING)
        ephemeral_sm.transition(terminal)
        assert ephemeral_sm.is_terminal
        assert ephemeral_sm.allowed_targets() == frozenset()


# -- Persistent 迁移矩阵 --


class TestPersistentTransitions:
    """Persistent 模式合法迁移矩阵全覆盖。"""

    def test_created_to_starting(self, persistent_sm):
        persistent_sm.transition(AgentState.STARTING)
        assert persistent_sm.state == AgentState.STARTING

    def test_created_to_shutdown(self, persistent_sm):
        persistent_sm.transition(AgentState.SHUTDOWN)
        assert persistent_sm.state == AgentState.SHUTDOWN
        assert persistent_sm.is_terminal

    @pytest.mark.parametrize("target", [
        AgentState.IDLE,
        AgentState.RUNNING,
        AgentState.FAILED,
        AgentState.SHUTDOWN,
    ])
    def test_from_starting(self, persistent_sm, target):
        persistent_sm.transition(AgentState.STARTING)
        persistent_sm.transition(target)
        assert persistent_sm.state == target

    @pytest.mark.parametrize("target", [
        AgentState.RUNNING,
        AgentState.SHUTDOWN,
    ])
    def test_from_idle(self, persistent_sm, target):
        persistent_sm.transition(AgentState.STARTING)
        persistent_sm.transition(AgentState.IDLE)
        persistent_sm.transition(target)
        assert persistent_sm.state == target

    @pytest.mark.parametrize("target", [
        AgentState.IDLE,
        AgentState.PAUSED,
        AgentState.COMPLETED,
        AgentState.FAILED,
        AgentState.CANCELLED,
        AgentState.TIMEOUT,
        AgentState.BUDGET_EXCEEDED,
        AgentState.STOPPING,
    ])
    def test_from_running(self, persistent_sm, target):
        persistent_sm.transition(AgentState.STARTING)
        persistent_sm.transition(AgentState.RUNNING)
        persistent_sm.transition(target)
        assert persistent_sm.state == target

    def test_full_lifecycle(self, persistent_sm):
        persistent_sm.transition(AgentState.STARTING, reason="start")
        persistent_sm.transition(AgentState.IDLE, reason="idle")
        persistent_sm.transition(AgentState.RUNNING, reason="assigned")
        persistent_sm.transition(AgentState.PAUSED, reason="pause")
        persistent_sm.transition(AgentState.RUNNING, reason="resume")
        persistent_sm.transition(AgentState.COMPLETED, reason="done")
        persistent_sm.transition(AgentState.IDLE, reason="recover")
        persistent_sm.transition(AgentState.RUNNING, reason="new task")
        persistent_sm.transition(AgentState.STOPPING, reason="shutdown req")
        persistent_sm.transition(AgentState.SHUTDOWN, reason="exit")
        assert persistent_sm.state == AgentState.SHUTDOWN
        assert persistent_sm.is_terminal

    def test_recover_from_failure(self, persistent_sm):
        persistent_sm.transition(AgentState.STARTING)
        persistent_sm.transition(AgentState.RUNNING)
        persistent_sm.transition(AgentState.FAILED, reason="crash")
        persistent_sm.transition(AgentState.IDLE, reason="recover")
        assert persistent_sm.state == AgentState.IDLE
        assert not persistent_sm.is_terminal

    def test_budget_exceeded_to_stopping(self, persistent_sm):
        persistent_sm.transition(AgentState.STARTING)
        persistent_sm.transition(AgentState.RUNNING)
        persistent_sm.transition(AgentState.BUDGET_EXCEEDED)
        persistent_sm.transition(AgentState.STOPPING)
        persistent_sm.transition(AgentState.SHUTDOWN)
        assert persistent_sm.is_terminal


# -- 非法迁移 --


class TestIllegalTransitions:
    """非法迁移必须显式失败。"""

    def test_ephemal_cannot_starting(self, ephemeral_sm):
        with pytest.raises(IllegalTransitionError, match="cannot transition"):
            ephemeral_sm.transition(AgentState.STARTING)

    def test_ephemal_cannot_idle(self, ephemeral_sm):
        with pytest.raises(IllegalTransitionError):
            ephemeral_sm.transition(AgentState.IDLE)

    def test_ephemal_cannot_shutdown_directly(self, ephemeral_sm):
        with pytest.raises(IllegalTransitionError):
            ephemeral_sm.transition(AgentState.SHUTDOWN)

    def test_persistent_cannot_created_to_running(self, persistent_sm):
        with pytest.raises(IllegalTransitionError, match="cannot transition"):
            persistent_sm.transition(AgentState.RUNNING)

    def test_persistent_cannot_idle_to_idle(self, persistent_sm):
        persistent_sm.transition(AgentState.STARTING)
        persistent_sm.transition(AgentState.IDLE)
        with pytest.raises(IllegalTransitionError):
            persistent_sm.transition(AgentState.IDLE)

    def test_terminal_rejects_all(self, ephemeral_sm):
        ephemeral_sm.transition(AgentState.RUNNING)
        ephemeral_sm.transition(AgentState.COMPLETED)
        for target in AgentState:
            if target == AgentState.COMPLETED:
                continue
            with pytest.raises(IllegalTransitionError):
                ephemeral_sm.transition(target)

    def test_error_message_contains_agent_id(self, ephemeral_sm):
        with pytest.raises(IllegalTransitionError, match="agent-1"):
            ephemeral_sm.transition(AgentState.SHUTDOWN)

    def test_error_message_contains_states(self, ephemeral_sm):
        with pytest.raises(IllegalTransitionError, match="created.*shutdown"):
            ephemeral_sm.transition(AgentState.SHUTDOWN)


# -- 审计记录 --


class TestAuditRecord:
    """状态迁移记录 reason、timestamp、actor 和 trace_id。"""

    def test_record_fields(self, ephemeral_sm):
        record = ephemeral_sm.transition(
            AgentState.RUNNING,
            reason="start work",
            actor="scheduler",
            trace_id="req-abc123",
        )
        assert record.from_state == AgentState.CREATED
        assert record.to_state == AgentState.RUNNING
        assert record.reason == "start work"
        assert record.actor == "scheduler"
        assert record.trace_id == "req-abc123"
        assert record.timestamp > 0

    def test_record_is_frozen(self, ephemeral_sm):
        record = ephemeral_sm.transition(AgentState.RUNNING)
        with pytest.raises(AttributeError):
            record.reason = "mutated"  # type: ignore[misc]

    def test_default_optional_fields(self, ephemeral_sm):
        record = ephemeral_sm.transition(AgentState.RUNNING)
        assert record.reason == ""
        assert record.actor == ""
        assert record.trace_id == ""

    def test_history_tracks_all_transitions(self, persistent_sm):
        persistent_sm.transition(AgentState.STARTING, reason="r1")
        persistent_sm.transition(AgentState.IDLE, reason="r2")
        persistent_sm.transition(AgentState.RUNNING, reason="r3")
        history = persistent_sm.history
        assert len(history) == 3
        assert history[0].to_state == AgentState.STARTING
        assert history[1].to_state == AgentState.IDLE
        assert history[2].to_state == AgentState.RUNNING
        assert [h.reason for h in history] == ["r1", "r2", "r3"]

    def test_history_is_copy(self, ephemeral_sm):
        ephemeral_sm.transition(AgentState.RUNNING)
        hist = ephemeral_sm.history
        hist.clear()
        assert len(ephemeral_sm.history) == 1

    def test_last_transition(self, ephemeral_sm):
        assert ephemeral_sm.last_transition is None
        ephemeral_sm.transition(AgentState.RUNNING, reason="start")
        last = ephemeral_sm.last_transition
        assert last is not None
        assert last.to_state == AgentState.RUNNING
        ephemeral_sm.transition(AgentState.COMPLETED, reason="done")
        last = ephemeral_sm.last_transition
        assert last is not None
        assert last.to_state == AgentState.COMPLETED

    def test_timestamps_are_monotonic(self, persistent_sm):
        import time
        t0 = time.monotonic()
        persistent_sm.transition(AgentState.STARTING)
        persistent_sm.transition(AgentState.IDLE)
        persistent_sm.transition(AgentState.RUNNING)
        history = persistent_sm.history
        assert history[0].timestamp >= t0
        assert history[1].timestamp >= history[0].timestamp
        assert history[2].timestamp >= history[1].timestamp


# -- 事件订阅 --


class TestSubscription:
    """状态变更事件订阅。"""

    def test_subscriber_receives_record(self, ephemeral_sm):
        received: list[StateTransition] = []
        ephemeral_sm.subscribe(received.append)
        ephemeral_sm.transition(AgentState.RUNNING, reason="start")
        assert len(received) == 1
        assert received[0].to_state == AgentState.RUNNING
        assert received[0].reason == "start"

    def test_multiple_subscribers(self, ephemeral_sm):
        events_a: list[StateTransition] = []
        events_b: list[StateTransition] = []
        ephemeral_sm.subscribe(events_a.append)
        ephemeral_sm.subscribe(events_b.append)
        ephemeral_sm.transition(AgentState.RUNNING)
        ephemeral_sm.transition(AgentState.COMPLETED)
        assert len(events_a) == 2
        assert len(events_b) == 2

    def test_unsubscribe(self, ephemeral_sm):
        received: list[StateTransition] = []
        unsubscribe = ephemeral_sm.subscribe(received.append)
        ephemeral_sm.transition(AgentState.RUNNING)
        assert len(received) == 1
        unsubscribe()
        ephemeral_sm.transition(AgentState.COMPLETED)
        assert len(received) == 1

    def test_unsubscribe_idempotent(self, ephemeral_sm):
        received: list[StateTransition] = []
        unsubscribe = ephemeral_sm.subscribe(received.append)
        unsubscribe()
        unsubscribe()

    def test_subscriber_called_after_state_change(self, ephemeral_sm):
        seen_states: list[AgentState] = []

        def callback(record: StateTransition) -> None:
            seen_states.append(ephemeral_sm.state)

        ephemeral_sm.subscribe(callback)
        ephemeral_sm.transition(AgentState.RUNNING)
        assert seen_states == [AgentState.RUNNING]


# -- 守卫 --


class TestGuards:
    """迁移守卫。"""

    def test_guard_allows(self, ephemeral_sm):
        guard = CallbackGuard(
            predicate=lambda cur, tgt: True,
            description="always allow",
        )
        ephemeral_sm.add_guard(AgentState.CREATED, AgentState.RUNNING, guard)
        ephemeral_sm.transition(AgentState.RUNNING)
        assert ephemeral_sm.state == AgentState.RUNNING

    def test_guard_rejects(self, ephemeral_sm):
        guard = CallbackGuard(
            predicate=lambda cur, tgt: False,
            description="always deny",
        )
        ephemeral_sm.add_guard(AgentState.CREATED, AgentState.RUNNING, guard)
        with pytest.raises(GuardRejectedError, match="guard rejected"):
            ephemeral_sm.transition(AgentState.RUNNING)
        assert ephemeral_sm.state == AgentState.CREATED

    def test_guard_error_message(self, ephemeral_sm):
        guard = CallbackGuard(predicate=lambda cur, tgt: False)
        ephemeral_sm.add_guard(AgentState.CREATED, AgentState.RUNNING, guard)
        with pytest.raises(GuardRejectedError, match="created.*running.*agent-1"):
            ephemeral_sm.transition(AgentState.RUNNING)

    def test_multiple_guards_all_must_pass(self, ephemeral_sm):
        ephemeral_sm.add_guard(
            AgentState.CREATED, AgentState.RUNNING,
            CallbackGuard(predicate=lambda cur, tgt: True),
        )
        ephemeral_sm.add_guard(
            AgentState.CREATED, AgentState.RUNNING,
            CallbackGuard(predicate=lambda cur, tgt: False),
        )
        with pytest.raises(GuardRejectedError):
            ephemeral_sm.transition(AgentState.RUNNING)

    def test_guard_does_not_affect_other_transitions(self, ephemeral_sm):
        ephemeral_sm.add_guard(
            AgentState.CREATED, AgentState.RUNNING,
            CallbackGuard(predicate=lambda cur, tgt: False),
        )
        ephemeral_sm.transition(AgentState.CANCELLED)
        assert ephemeral_sm.state == AgentState.CANCELLED

    def test_guard_checked_after_transition_table(self, ephemeral_sm):
        guard = CallbackGuard(predicate=lambda cur, tgt: False)
        ephemeral_sm.add_guard(AgentState.CREATED, AgentState.SHUTDOWN, guard)
        with pytest.raises(IllegalTransitionError):
            ephemeral_sm.transition(AgentState.SHUTDOWN)

    def test_guard_rejection_does_not_record_history(self, ephemeral_sm):
        guard = CallbackGuard(predicate=lambda cur, tgt: False)
        ephemeral_sm.add_guard(AgentState.CREATED, AgentState.RUNNING, guard)
        with pytest.raises(GuardRejectedError):
            ephemeral_sm.transition(AgentState.RUNNING)
        assert len(ephemeral_sm.history) == 0
        assert ephemeral_sm.state == AgentState.CREATED


# -- 并发更新 --


class TestConcurrency:
    """并发更新安全性。"""

    def test_concurrent_transitions_one_wins(self):
        custom = {
            AgentState.CREATED: frozenset({
                AgentState.COMPLETED,
                AgentState.CANCELLED,
                AgentState.FAILED,
            }),
            AgentState.COMPLETED: frozenset(),
            AgentState.CANCELLED: frozenset(),
            AgentState.FAILED: frozenset(),
            AgentState.RUNNING: frozenset(),
            AgentState.PAUSED: frozenset(),
            AgentState.TIMEOUT: frozenset(),
            AgentState.BUDGET_EXCEEDED: frozenset(),
            AgentState.IDLE: frozenset(),
            AgentState.STARTING: frozenset(),
            AgentState.STOPPING: frozenset(),
            AgentState.SHUTDOWN: frozenset(),
        }
        sm = AgentStateMachine(
            "concurrent-agent",
            mode=RuntimeMode.EPHEMERAL,
            transitions=custom,
        )
        results: list[str] = []
        errors: list[Exception] = []

        def attempt_transition(target: AgentState) -> None:
            try:
                sm.transition(target, reason="concurrent")
                results.append(target.value)
            except (IllegalTransitionError, GuardRejectedError) as e:
                errors.append(e)

        threads = [
            threading.Thread(target=attempt_transition, args=(AgentState.COMPLETED,)),
            threading.Thread(target=attempt_transition, args=(AgentState.CANCELLED,)),
            threading.Thread(target=attempt_transition, args=(AgentState.FAILED,)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 1
        assert len(errors) == 2
        assert sm.state in {
            AgentState.COMPLETED,
            AgentState.CANCELLED,
            AgentState.FAILED,
        }
        assert sm.is_terminal

    def test_concurrent_same_target_idempotent_state(self):
        sm = AgentStateMachine("concurrent-agent-2", mode=RuntimeMode.EPHEMERAL)
        success_count = 0
        lock = threading.Lock()

        def attempt() -> None:
            nonlocal success_count
            try:
                sm.transition(AgentState.RUNNING)
                with lock:
                    success_count += 1
            except IllegalTransitionError:
                pass

        threads = [threading.Thread(target=attempt) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert success_count == 1
        assert sm.state == AgentState.RUNNING

    def test_concurrent_subscribe_safe(self):
        sm = AgentStateMachine("concurrent-sub", mode=RuntimeMode.EPHEMERAL)
        received: list[StateTransition] = []
        lock = threading.Lock()

        def safe_append(record: StateTransition) -> None:
            with lock:
                received.append(record)

        sm.subscribe(safe_append)

        def do_transition() -> None:
            with suppress(IllegalTransitionError):
                sm.transition(AgentState.RUNNING)

        threads = [threading.Thread(target=do_transition) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(received) == 1
        assert sm.state == AgentState.RUNNING


# -- 自定义迁移表 --


class TestCustomTransitions:
    """自定义迁移表。"""

    def test_custom_table(self):
        custom = {
            AgentState.CREATED: frozenset({AgentState.RUNNING}),
            AgentState.RUNNING: frozenset({AgentState.COMPLETED}),
            AgentState.COMPLETED: frozenset(),
        }
        sm = AgentStateMachine(
            "custom-agent",
            mode=RuntimeMode.EPHEMERAL,
            transitions=custom,
        )
        sm.transition(AgentState.RUNNING)
        sm.transition(AgentState.COMPLETED)
        assert sm.is_terminal

    def test_custom_table_rejects_undefined(self):
        custom = {
            AgentState.CREATED: frozenset({AgentState.RUNNING}),
            AgentState.RUNNING: frozenset(),
        }
        sm = AgentStateMachine(
            "custom-agent",
            mode=RuntimeMode.EPHEMERAL,
            transitions=custom,
        )
        with pytest.raises(IllegalTransitionError):
            sm.transition(AgentState.CANCELLED)


# -- 属性查询 --


class TestProperties:
    """状态机属性查询。"""

    def test_agent_id(self, ephemeral_sm):
        assert ephemeral_sm.agent_id == "agent-1"

    def test_initial_state(self, ephemeral_sm):
        assert ephemeral_sm.state == AgentState.CREATED

    def test_mode(self, ephemeral_sm, persistent_sm):
        assert ephemeral_sm.mode == RuntimeMode.EPHEMERAL
        assert persistent_sm.mode == RuntimeMode.PERSISTENT

    def test_not_terminal_initially(self, ephemeral_sm):
        assert not ephemeral_sm.is_terminal

    def test_can_transition(self, ephemeral_sm):
        assert ephemeral_sm.can_transition(AgentState.RUNNING)
        assert ephemeral_sm.can_transition(AgentState.CANCELLED)
        assert not ephemeral_sm.can_transition(AgentState.IDLE)

    def test_allowed_targets(self, ephemeral_sm):
        targets = ephemeral_sm.allowed_targets()
        assert AgentState.RUNNING in targets
        assert AgentState.CANCELLED in targets
        assert AgentState.IDLE not in targets

    def test_empty_history_initially(self, ephemeral_sm):
        assert ephemeral_sm.history == []
        assert ephemeral_sm.last_transition is None
