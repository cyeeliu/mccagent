"""Phase 1: mcc-core/runtime/ 领域模型测试。

覆盖：字段校验、序列化往返、非法迁移、ID 唯一性、事件关联。
"""

from __future__ import annotations

from datetime import UTC, datetime, timezone
from decimal import Decimal

import pytest
from mcc_core.runtime import (
    INTERACTIVE_CAPABILITIES,
    PERSISTENT_CAPABILITIES,
    TASK_CAPABILITIES,
    AgentDefinition,
    ArtifactRef,
    DomainError,
    ExecutionProfile,
    IdempotencyConflictError,
    InvalidTransitionError,
    ModelProfile,
    PermissionProfile,
    RuntimeCapabilities,
    RuntimeErrorInfo,
    RuntimeEvent,
    RuntimeEventType,
    RuntimeHealth,
    RuntimeKind,
    RuntimeNotFoundError,
    RuntimeNotReadyError,
    RuntimeResult,
    RuntimeStatus,
    TaskContext,
    TaskHandle,
    TaskStatus,
    TaskTimeoutError,
    TaskValidationError,
    TokenUsage,
    ToolProfile,
    UnsupportedOperationError,
    is_runtime_terminal,
    is_task_terminal,
    is_valid_runtime_transition,
    is_valid_task_transition,
)
from mcc_core.runtime.event import RuntimeEvent as RE
from mcc_core.runtime.task import RuntimeTask


class TestRuntimeStatus:
    @pytest.mark.parametrize(
        "status,expected",
        [
            (RuntimeStatus.CREATED, False),
            (RuntimeStatus.STARTING, False),
            (RuntimeStatus.READY, False),
            (RuntimeStatus.PAUSING, False),
            (RuntimeStatus.PAUSED, False),
            (RuntimeStatus.STOPPING, False),
            (RuntimeStatus.STOPPED, True),
            (RuntimeStatus.FAILED, True),
        ],
    )
    def test_terminal(self, status, expected):
        assert is_runtime_terminal(status) is expected

    @pytest.mark.parametrize(
        "source,target,expected",
        [
            (RuntimeStatus.CREATED, RuntimeStatus.STARTING, True),
            (RuntimeStatus.CREATED, RuntimeStatus.READY, False),
            (RuntimeStatus.STARTING, RuntimeStatus.READY, True),
            (RuntimeStatus.STARTING, RuntimeStatus.FAILED, True),
            (RuntimeStatus.READY, RuntimeStatus.PAUSING, True),
            (RuntimeStatus.READY, RuntimeStatus.STOPPING, True),
            (RuntimeStatus.PAUSING, RuntimeStatus.PAUSED, True),
            (RuntimeStatus.PAUSED, RuntimeStatus.READY, True),
            (RuntimeStatus.STOPPING, RuntimeStatus.STOPPED, True),
            (RuntimeStatus.STOPPED, RuntimeStatus.READY, False),
            (RuntimeStatus.FAILED, RuntimeStatus.READY, False),
        ],
    )
    def test_transition(self, source, target, expected):
        assert is_valid_runtime_transition(source, target) is expected


class TestTaskStatus:
    @pytest.mark.parametrize(
        "status,expected",
        [
            (TaskStatus.PENDING, False),
            (TaskStatus.QUEUED, False),
            (TaskStatus.DISPATCHED, False),
            (TaskStatus.ACCEPTED, False),
            (TaskStatus.RUNNING, False),
            (TaskStatus.SUCCEEDED, True),
            (TaskStatus.FAILED, True),
            (TaskStatus.CANCELLED, True),
            (TaskStatus.TIMED_OUT, True),
        ],
    )
    def test_terminal(self, status, expected):
        assert is_task_terminal(status) is expected

    @pytest.mark.parametrize(
        "source,target,expected",
        [
            (TaskStatus.PENDING, TaskStatus.QUEUED, True),
            (TaskStatus.QUEUED, TaskStatus.DISPATCHED, True),
            (TaskStatus.DISPATCHED, TaskStatus.ACCEPTED, True),
            (TaskStatus.ACCEPTED, TaskStatus.RUNNING, True),
            (TaskStatus.RUNNING, TaskStatus.SUCCEEDED, True),
            (TaskStatus.RUNNING, TaskStatus.FAILED, True),
            (TaskStatus.RUNNING, TaskStatus.CANCELLED, True),
            (TaskStatus.RUNNING, TaskStatus.TIMED_OUT, True),
            (TaskStatus.SUCCEEDED, TaskStatus.RUNNING, False),
            (TaskStatus.FAILED, TaskStatus.RUNNING, False),
            (TaskStatus.PENDING, TaskStatus.RUNNING, False),
        ],
    )
    def test_transition(self, source, target, expected):
        assert is_valid_task_transition(source, target) is expected


class TestRuntimeTask:
    def _make_task(self, **overrides):
        defaults = {
            "task_id": "task-1",
            "attempt_id": "attempt-1",
            "agent_id": "agent-1",
            "objective": "Complete the mission",
        }
        defaults.update(overrides)
        return RuntimeTask(**defaults)

    def test_basic_creation(self):
        task = self._make_task()
        assert task.task_id == "task-1"
        assert task.attempt_id == "attempt-1"
        assert task.agent_id == "agent-1"
        assert task.objective == "Complete the mission"

    def test_auto_idempotency_key(self):
        task = self._make_task()
        assert task.idempotency_key == "task-1:attempt-1"

    def test_explicit_idempotency_key(self):
        task = self._make_task(idempotency_key="custom-key")
        assert task.idempotency_key == "custom-key"

    def test_default_created_at(self):
        before = datetime.now(UTC)
        task = self._make_task()
        after = datetime.now(UTC)
        assert before <= task.created_at <= after

    @pytest.mark.parametrize(
        "field,value",
        [
            ("task_id", ""),
            ("attempt_id", ""),
            ("agent_id", ""),
            ("objective", ""),
        ],
    )
    def test_empty_field_rejected(self, field, value):
        with pytest.raises(TaskValidationError):
            self._make_task(**{field: value})

    def test_long_objective_rejected(self):
        with pytest.raises(TaskValidationError, match="objective 长度超限"):
            self._make_task(objective="x" * 20_000)

    def test_frozen(self):
        task = self._make_task()
        with pytest.raises(AttributeError):
            task.task_id = "changed"

    def test_to_dict(self):
        task = self._make_task(
            run_id="run-1",
            assignment_id="asg-1",
            timeout_seconds=30.0,
            input={"key": "value"},
        )
        d = task.to_dict()
        assert d["task_id"] == "task-1"
        assert d["attempt_id"] == "attempt-1"
        assert d["run_id"] == "run-1"
        assert d["input"] == {"key": "value"}
        assert "created_at" in d
        assert "context" in d

    def test_task_context(self):
        ctx = TaskContext(
            session_id="s1",
            working_directory="/tmp",
            permissions=frozenset({"read", "write"}),
        )
        task = self._make_task(context=ctx)
        assert task.context.session_id == "s1"
        assert task.context.working_directory == "/tmp"
        assert "read" in task.context.permissions


class TestTaskHandle:
    def test_creation(self):
        handle = TaskHandle(
            task_id="t1",
            attempt_id="a1",
            runtime_id="r1",
        )
        assert handle.task_id == "t1"
        assert handle.attempt_id == "a1"
        assert handle.runtime_id == "r1"

    def test_to_dict(self):
        handle = TaskHandle(task_id="t1", attempt_id="a1", runtime_id="r1")
        d = handle.to_dict()
        assert d["task_id"] == "t1"
        assert "accepted_at" in d


class TestRuntimeResult:
    def _make_result(self, **overrides):
        defaults = {
            "task_id": "t1",
            "attempt_id": "a1",
            "runtime_id": "r1",
            "agent_id": "ag1",
            "status": TaskStatus.SUCCEEDED,
            "summary": "Done",
        }
        defaults.update(overrides)
        return RuntimeResult(**defaults)

    def test_success(self):
        result = self._make_result()
        assert result.is_success is True
        assert result.is_terminal is True

    def test_failed(self):
        result = self._make_result(status=TaskStatus.FAILED)
        assert result.is_success is False
        assert result.is_terminal is True

    def test_non_terminal(self):
        result = self._make_result(status=TaskStatus.RUNNING)
        assert result.is_terminal is False

    def test_with_usage(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        result = self._make_result(usage=usage)
        assert result.usage.total_tokens == 150

    def test_with_error(self):
        error = RuntimeErrorInfo(code="tool_failed", message="Tool failed", retryable=True)
        result = self._make_result(status=TaskStatus.FAILED, error=error)
        assert result.error.code == "tool_failed"
        assert result.error.retryable is True

    def test_with_artifacts(self):
        artifacts = (
            ArtifactRef(uri="file:///tmp/out.txt", mime_type="text/plain"),
            ArtifactRef(uri="file:///tmp/report.json"),
        )
        result = self._make_result(artifacts=artifacts)
        assert len(result.artifacts) == 2

    def test_with_cost(self):
        result = self._make_result(cost_usd=Decimal("0.05"))
        assert result.cost_usd == Decimal("0.05")

    def test_to_dict(self):
        result = self._make_result(
            usage=TokenUsage(input_tokens=10),
            error=RuntimeErrorInfo(code="err", message="msg"),
        )
        d = result.to_dict()
        assert d["status"] == "succeeded"
        assert d["usage"]["input_tokens"] == 10
        assert d["error"]["code"] == "err"

    def test_frozen(self):
        result = self._make_result()
        with pytest.raises(AttributeError):
            result.summary = "changed"


class TestRuntimeEvent:
    def test_creation(self):
        event = RuntimeEvent(
            event_id="e1",
            event_type=RuntimeEventType.TASK_STARTED,
            runtime_id="r1",
            agent_id="ag1",
            task_id="t1",
            attempt_id="a1",
            sequence=1,
        )
        assert event.event_type == RuntimeEventType.TASK_STARTED
        assert event.sequence == 1

    def test_to_dict(self):
        event = RuntimeEvent(
            event_id="e1",
            event_type=RuntimeEventType.RUNTIME_READY,
            runtime_id="r1",
        )
        d = event.to_dict()
        assert d["event_type"] == "runtime_ready"
        assert d["runtime_id"] == "r1"

    @pytest.mark.parametrize("etype", list(RuntimeEventType))
    def test_all_event_types_serializable(self, etype):
        event = RuntimeEvent(event_id="e", event_type=etype)
        d = event.to_dict()
        assert d["event_type"] == etype.value


class TestRuntimeHealth:
    def test_healthy(self):
        from mcc_core.runtime.health import HealthState

        health = RuntimeHealth(runtime_id="r1", state=HealthState.HEALTHY)
        assert health.is_healthy is True

    def test_unhealthy(self):
        from mcc_core.runtime.health import HealthState

        health = RuntimeHealth(
            runtime_id="r1",
            state=HealthState.UNHEALTHY,
            last_error="OOM",
        )
        assert health.is_healthy is False
        assert health.last_error == "OOM"


class TestRuntimeCapabilities:
    def test_interactive(self):
        assert INTERACTIVE_CAPABILITIES.supports_pause is False
        assert INTERACTIVE_CAPABILITIES.max_concurrency == 1

    def test_persistent(self):
        assert PERSISTENT_CAPABILITIES.supports_pause is True
        assert PERSISTENT_CAPABILITIES.supports_multiple_tasks is True
        assert PERSISTENT_CAPABILITIES.queue_capacity == 32

    def test_task(self):
        assert TASK_CAPABILITIES.supports_pause is False
        assert TASK_CAPABILITIES.max_concurrency == 1

    def test_can(self):
        caps = RuntimeCapabilities(supports_pause=True)
        assert caps.can("pause") is True
        assert caps.can("resume") is False

    def test_frozen(self):
        with pytest.raises(AttributeError):
            INTERACTIVE_CAPABILITIES.supports_pause = True


class TestAgentDefinition:
    def test_creation(self):
        defn = AgentDefinition(
            agent_id="ag1",
            name="Worker",
            role="worker",
            description="A worker",
            model_profile=ModelProfile(model_id="claude-3.5"),
            capabilities=frozenset({"code", "test"}),
            runtime_kind=RuntimeKind.PERSISTENT,
        )
        assert defn.agent_id == "ag1"
        assert defn.runtime_kind == RuntimeKind.PERSISTENT
        assert "code" in defn.capabilities

    def test_defaults(self):
        defn = AgentDefinition(
            agent_id="ag1",
            name="Agent",
            role="agent",
            description="",
            model_profile=ModelProfile(model_id="gpt-4"),
        )
        assert defn.runtime_kind == RuntimeKind.INTERACTIVE
        assert defn.capabilities == frozenset()

    def test_frozen(self):
        defn = AgentDefinition(
            agent_id="ag1",
            name="Agent",
            role="agent",
            description="",
            model_profile=ModelProfile(model_id="gpt-4"),
        )
        with pytest.raises(AttributeError):
            defn.name = "changed"

    def test_to_dict(self):
        defn = AgentDefinition(
            agent_id="ag1",
            name="Worker",
            role="worker",
            description="A worker",
            model_profile=ModelProfile(model_id="claude-3.5"),
            capabilities=frozenset({"code"}),
            runtime_kind=RuntimeKind.TASK,
        )
        d = defn.to_dict()
        assert d["agent_id"] == "ag1"
        assert d["runtime_kind"] == "task"
        assert d["capabilities"] == ["code"]


class TestErrors:
    def test_invalid_transition(self):
        err = InvalidTransitionError("stopped", "ready")
        assert err.code == "invalid_transition"
        assert err.retryable is False
        assert err.details["source"] == "stopped"

    def test_runtime_not_ready(self):
        err = RuntimeNotReadyError("r1", "starting")
        assert err.code == "runtime_not_ready"
        assert err.retryable is True

    def test_task_validation(self):
        err = TaskValidationError("bad field")
        assert err.code == "task_validation_error"

    def test_idempotency_conflict(self):
        err = IdempotencyConflictError("key-1")
        assert err.code == "idempotency_conflict"

    def test_unsupported_operation(self):
        err = UnsupportedOperationError("pause", "r1")
        assert err.code == "unsupported_operation"

    def test_runtime_not_found(self):
        err = RuntimeNotFoundError("r1")
        assert err.code == "runtime_not_found"

    def test_task_timeout(self):
        err = TaskTimeoutError("t1", "a1")
        assert err.code == "task_timeout"
        assert err.retryable is True

    def test_all_inherit_domain_error(self):
        assert issubclass(InvalidTransitionError, DomainError)
        assert issubclass(RuntimeNotReadyError, DomainError)
        assert issubclass(TaskValidationError, DomainError)
        assert issubclass(IdempotencyConflictError, DomainError)
        assert issubclass(UnsupportedOperationError, DomainError)
        assert issubclass(RuntimeNotFoundError, DomainError)
        assert issubclass(TaskTimeoutError, DomainError)
