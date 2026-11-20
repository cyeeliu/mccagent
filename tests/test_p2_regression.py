"""P2-01~P2-06 回归测试：能力声明/协议统一/校验/事件/架构/文档。

每个测试类对应一个 P2 问题。修复前确定性失败，修复后通过。
充分使用 pytest parametrize/raises/fixture/mark，用真实集成测试而非注入 mock。
"""

from __future__ import annotations

import ast
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from mcc_core.runtime.enums import RuntimeEventType, TaskStatus
from mcc_core.runtime.errors import TaskValidationError
from mcc_core.runtime.event import RuntimeEvent
from mcc_core.runtime.result import RuntimeResult
from mcc_core.runtime.task import RuntimeTask, TaskContext

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGES = REPO_ROOT / "packages"


# ---------------------------------------------------------------------------
# 辅助：记录 execute 调用参数的测试 Executor
# ---------------------------------------------------------------------------


class _RecordingExecutor:
    """记录 execute 调用参数的测试 Executor，返回成功 RuntimeResult。"""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def execute(self, task: RuntimeTask, **kwargs: Any) -> RuntimeResult:
        self.calls.append(kwargs)
        return RuntimeResult(
            task_id=task.task_id,
            attempt_id=task.attempt_id,
            runtime_id="test-rt",
            agent_id=task.agent_id,
            status=TaskStatus.SUCCEEDED,
            summary="ok",
        )

    async def close(self) -> None:
        pass


def _make_task(**overrides: Any) -> RuntimeTask:
    """构造 RuntimeTask 工厂，提供合理默认值。"""
    defaults: dict[str, Any] = {
        "task_id": "task-1",
        "attempt_id": "attempt-1",
        "agent_id": "agent-1",
        "objective": "Complete the mission",
    }
    defaults.update(overrides)
    return RuntimeTask(**defaults)


# ---------------------------------------------------------------------------
# P2-01: PersistentRuntimeV2 capabilities 从实例配置生成
# ---------------------------------------------------------------------------


class TestP201CapabilitiesFromInstance:
    """P2-01: capabilities 应由 Runtime 实例真实配置生成，而非全局常量。"""

    @pytest.mark.parametrize(
        ("max_concurrency", "queue_capacity"),
        [
            (4, 64),
            (8, 128),
            (2, 0),
            (3, 16),
        ],
    )
    def test_capabilities_reflect_instance_config(
        self,
        max_concurrency: int,
        queue_capacity: int,
    ) -> None:
        from mcc_harness.runtime.persistent import PersistentRuntimeV2

        rt = PersistentRuntimeV2(
            executor=MagicMock(),
            max_concurrency=max_concurrency,
            queue_capacity=queue_capacity,
        )
        caps = rt.capabilities
        assert caps.max_concurrency == max_concurrency, (
            f"capabilities.max_concurrency 应反映实例配置 {max_concurrency}，"
            f"实际为 {caps.max_concurrency}"
        )
        assert caps.queue_capacity == queue_capacity, (
            f"capabilities.queue_capacity 应反映实例配置 {queue_capacity}，"
            f"实际为 {caps.queue_capacity}"
        )

    def test_capabilities_not_global_constant(self) -> None:
        """两个不同配置的 Runtime 应有不同 capabilities。"""
        from mcc_harness.runtime.persistent import PersistentRuntimeV2

        rt_a = PersistentRuntimeV2(
            executor=MagicMock(),
            max_concurrency=2,
            queue_capacity=10,
        )
        rt_b = PersistentRuntimeV2(
            executor=MagicMock(),
            max_concurrency=5,
            queue_capacity=20,
        )
        assert rt_a.capabilities != rt_b.capabilities


# ---------------------------------------------------------------------------
# P2-02: AgentExecutor Protocol emit/cancellation 统一可选参数
# ---------------------------------------------------------------------------


class TestP202ProtocolOptionalParams:
    """P2-02: Protocol 中 emit/cancellation 应为可选参数。"""

    def test_protocol_execute_emit_has_default(self) -> None:
        proto_file = PACKAGES / "mcc-core/src/mcc_core/runtime/protocols.py"
        tree = ast.parse(proto_file.read_text(encoding="utf-8"))
        execute_node = _find_method(tree, "AgentExecutor", "execute")
        assert execute_node is not None, "AgentExecutor.execute 未找到"
        kw_names = [a.arg for a in execute_node.args.kwonlyargs]
        kw_defaults = execute_node.args.kw_defaults
        assert "emit" in kw_names, "execute 应有 emit 参数"
        idx = kw_names.index("emit")
        assert kw_defaults[idx] is not None, "emit 应有默认值（可选参数）"

    def test_protocol_execute_cancellation_has_default(self) -> None:
        proto_file = PACKAGES / "mcc-core/src/mcc_core/runtime/protocols.py"
        tree = ast.parse(proto_file.read_text(encoding="utf-8"))
        execute_node = _find_method(tree, "AgentExecutor", "execute")
        assert execute_node is not None
        kw_names = [a.arg for a in execute_node.args.kwonlyargs]
        kw_defaults = execute_node.args.kw_defaults
        assert "cancellation" in kw_names, "execute 应有 cancellation 参数"
        idx = kw_names.index("cancellation")
        assert kw_defaults[idx] is not None, "cancellation 应有默认值（可选参数）"


class TestP202WorkerPoolPassesParams:
    """P2-02: WorkerPool 应统一传入 emit 和 cancellation。"""

    @pytest.mark.asyncio
    async def test_worker_pool_passes_emit_and_cancellation(self) -> None:
        from mcc_harness.runtime.bounded_task_queue import BoundedTaskQueue
        from mcc_harness.runtime.worker_pool import WorkerPool

        executor = _RecordingExecutor()
        queue = BoundedTaskQueue(max_size=4)
        pool = WorkerPool(queue, executor, max_workers=1)
        pool.start()

        task = _make_task()
        await queue.put(task)
        await asyncio.sleep(0.3)
        await pool.stop()

        assert len(executor.calls) == 1, f"应执行 1 次，实际 {len(executor.calls)} 次"
        call = executor.calls[0]
        assert "emit" in call, "WorkerPool 未传入 emit 参数"
        assert "cancellation" in call, "WorkerPool 未传入 cancellation 参数"


class TestP202TaskRuntimePassesParams:
    """P2-02: TaskRuntime 应统一传入 emit 和 cancellation。"""

    @pytest.mark.asyncio
    async def test_task_runtime_passes_emit_and_cancellation(self) -> None:
        from mcc_harness.runtime.task_runtime import TaskRuntime

        executor = _RecordingExecutor()
        rt = TaskRuntime(executor)
        await rt.start()

        task = _make_task()
        await rt.submit(task)
        await asyncio.sleep(0.3)
        await rt.stop()

        assert len(executor.calls) == 1
        call = executor.calls[0]
        assert "emit" in call, "TaskRuntime 未传入 emit 参数"
        assert "cancellation" in call, "TaskRuntime 未传入 cancellation 参数"


class TestP202InteractiveRuntimePassesParams:
    """P2-02: InteractiveRuntimeV2 应统一传入 emit 和 cancellation。"""

    @pytest.mark.asyncio
    async def test_interactive_runtime_passes_emit_and_cancellation(self) -> None:
        from mcc_harness.runtime.interactive import InteractiveRuntimeV2

        rt = InteractiveRuntimeV2(agent=MagicMock())
        rt._executor = _RecordingExecutor()
        await rt.start()

        task = _make_task()
        await rt.submit(task)
        await asyncio.sleep(0.3)
        await rt.stop()

        assert len(rt._executor.calls) == 1
        call = rt._executor.calls[0]
        assert "emit" in call, "InteractiveRuntimeV2 未传入 emit 参数"
        assert "cancellation" in call, "InteractiveRuntimeV2 未传入 cancellation 参数"


# ---------------------------------------------------------------------------
# P2-03: RuntimeTask 校验增强
# ---------------------------------------------------------------------------


class TestP203TimeoutValidation:
    """P2-03: timeout_seconds 必须为正数。"""

    @pytest.mark.parametrize("timeout", [0, -1, -0.1, -100])
    def test_non_positive_timeout_rejected(self, timeout: float) -> None:
        with pytest.raises(TaskValidationError, match="timeout"):
            _make_task(timeout_seconds=timeout)

    @pytest.mark.parametrize("timeout", [None, 0.1, 1, 30.0, 999])
    def test_valid_timeout_allowed(self, timeout: float | None) -> None:
        task = _make_task(timeout_seconds=timeout)
        assert task.timeout_seconds == timeout


class TestP203Created_atTimezone:
    """P2-03: created_at 必须带时区信息。"""

    def test_naive_datetime_rejected(self) -> None:
        with pytest.raises(TaskValidationError, match="时区"):
            _make_task(created_at=datetime.now())

    def test_timezone_aware_datetime_allowed(self) -> None:
        task = _make_task(created_at=datetime.now(UTC))
        assert task.created_at.tzinfo is not None


class TestP203ObjectiveWhitespace:
    """P2-03: objective 空白字符串应拒绝。"""

    @pytest.mark.parametrize("objective", ["   ", "\t\n", " \t \n "])
    def test_whitespace_objective_rejected(self, objective: str) -> None:
        with pytest.raises(TaskValidationError):
            _make_task(objective=objective)

    def test_normal_objective_allowed(self) -> None:
        task = _make_task(objective="  do something  ")
        assert task.objective == "  do something  "


class TestP203JsonSerializable:
    """P2-03: input/metadata 必须可 JSON 序列化。"""

    def test_non_serializable_input_rejected(self) -> None:
        with pytest.raises(TaskValidationError, match="序列化"):
            _make_task(input={"key": object()})

    def test_non_serializable_metadata_rejected(self) -> None:
        with pytest.raises(TaskValidationError, match="序列化"):
            _make_task(metadata={"key": set()})

    @pytest.mark.parametrize(
        "data",
        [
            {"key": "value"},
            {"num": 42, "list": [1, 2, 3]},
            {"nested": {"a": True, "b": None}},
            {},
        ],
    )
    def test_serializable_input_allowed(self, data: dict) -> None:
        task = _make_task(input=data)
        assert task.input == data


class TestP203IdFormatValidation:
    """P2-03: task_id/attempt_id/agent_id 格式校验。"""

    @pytest.mark.parametrize(
        "field",
        ["task_id", "attempt_id", "agent_id"],
    )
    def test_control_char_rejected(self, field: str) -> None:
        with pytest.raises(TaskValidationError, match="控制字符"):
            _make_task(**{field: "bad\x00id"})

    @pytest.mark.parametrize(
        "field",
        ["task_id", "attempt_id", "agent_id"],
    )
    def test_excessive_length_rejected(self, field: str) -> None:
        with pytest.raises(TaskValidationError, match="长度"):
            _make_task(**{field: "x" * 512})

    @pytest.mark.parametrize(
        "field",
        ["task_id", "attempt_id", "agent_id"],
    )
    def test_normal_id_allowed(self, field: str) -> None:
        task = _make_task(**{field: "normal-id-123"})
        assert getattr(task, field) == "normal-id-123"


class TestP203IdempotencyKeyValidation:
    """P2-03: idempotency_key 长度限制。"""

    def test_excessive_length_rejected(self) -> None:
        with pytest.raises(TaskValidationError, match="idempotency"):
            _make_task(idempotency_key="x" * 1024)

    def test_normal_key_allowed(self) -> None:
        task = _make_task(idempotency_key="normal-key")
        assert task.idempotency_key == "normal-key"


class TestP203EnvironmentValidation:
    """P2-03: environment key/value 长度限制。"""

    def test_long_key_rejected(self) -> None:
        ctx = TaskContext(environment={"k" * 512: "v"})
        with pytest.raises(TaskValidationError, match="environment"):
            _make_task(context=ctx)

    def test_long_value_rejected(self) -> None:
        ctx = TaskContext(environment={"k": "v" * 16384})
        with pytest.raises(TaskValidationError, match="environment"):
            _make_task(context=ctx)

    def test_normal_environment_allowed(self) -> None:
        ctx = TaskContext(environment={"PATH": "/usr/bin", "HOME": "/root"})
        task = _make_task(context=ctx)
        assert task.context.environment["PATH"] == "/usr/bin"


# ---------------------------------------------------------------------------
# P2-04: RuntimeEvent 增加关联字段
# ---------------------------------------------------------------------------


class TestP204EventFields:
    """P2-04: RuntimeEvent 增加 causation_id/result_ref/schema_version/trace_id/parent_event_id。"""

    def test_new_fields_exist_with_defaults(self) -> None:
        ev = RuntimeEvent(
            event_id="evt-1",
            event_type=RuntimeEventType.TASK_STARTED,
        )
        assert hasattr(ev, "causation_id")
        assert hasattr(ev, "result_ref")
        assert hasattr(ev, "schema_version")
        assert hasattr(ev, "trace_id")
        assert hasattr(ev, "parent_event_id")
        assert ev.causation_id is None
        assert ev.result_ref is None
        assert ev.trace_id is None
        assert ev.parent_event_id is None

    def test_new_fields_settable(self) -> None:
        ev = RuntimeEvent(
            event_id="evt-1",
            event_type=RuntimeEventType.TASK_STARTED,
            causation_id="cau-1",
            result_ref="res-1",
            schema_version=2,
            trace_id="trace-1",
            parent_event_id="evt-0",
        )
        assert ev.causation_id == "cau-1"
        assert ev.result_ref == "res-1"
        assert ev.schema_version == 2
        assert ev.trace_id == "trace-1"
        assert ev.parent_event_id == "evt-0"

    @pytest.mark.parametrize(
        "field",
        ["causation_id", "result_ref", "schema_version", "trace_id", "parent_event_id"],
    )
    def test_to_dict_includes_new_field(self, field: str) -> None:
        ev = RuntimeEvent(
            event_id="evt-1",
            event_type=RuntimeEventType.TASK_STARTED,
        )
        d = ev.to_dict()
        assert field in d, f"to_dict 缺少字段 {field}"


# ---------------------------------------------------------------------------
# P2-05: 架构 AST 规则
# ---------------------------------------------------------------------------


class TestP205ArchitectureRules:
    """P2-05: 架构 AST 规则检查禁止模式。"""

    def test_orchestrator_no_submit_task_wait_task(self) -> None:
        """TeamOrchestrator 不得调用 submit_task/wait_task。"""
        orch_file = PACKAGES / "mcc-harness/src/mcc_harness/team/orchestrator.py"
        source = orch_file.read_text(encoding="utf-8")
        tree = ast.parse(source)
        violations: list[str] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("submit_task", "wait_task")
            ):
                violations.append(node.func.attr)
        assert not violations, f"TeamOrchestrator 调用了禁止方法: {violations}"

    def test_no_hasattr_capability_probing(self) -> None:
        """不得使用 hasattr 探测 capabilities。"""
        violations: list[str] = []
        for pkg in ("mcc-core", "mcc-harness", "mcc-cli"):
            src = PACKAGES / pkg / "src"
            if not src.exists():
                continue
            for f in sorted(src.rglob("*.py")):
                tree = ast.parse(f.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "hasattr"
                        and len(node.args) >= 2
                        and isinstance(node.args[1], ast.Constant)
                        and isinstance(node.args[1].value, str)
                        and "capabilit" in node.args[1].value.lower()
                    ):
                        violations.append(str(f.relative_to(REPO_ROOT)))
        assert not violations, f"发现 hasattr 能力探测: {violations}"

    def test_subagent_executor_prefers_factory_path(self) -> None:
        """SubagentExecutor 应仅走 agent_factory 装配路径，无 runner 回退。"""
        exec_file = PACKAGES / "mcc-harness/src/mcc_harness/execution/subagent_executor.py"
        source = exec_file.read_text(encoding="utf-8")
        factory_check = "self._agent_factory is None or self._factory_context is None"
        assert factory_check in source, "SubagentExecutor 应检查 _agent_factory 和 _factory_context"
        assert "self._runner.run" not in source, "SubagentExecutor 不应有 _runner.run 回退"


# ---------------------------------------------------------------------------
# P2-06: 文档校准
# ---------------------------------------------------------------------------


class TestP206DocCalibration:
    """P2-06: runtime-current-state.md 应反映当前真实状态。"""

    @pytest.fixture
    def doc_content(self) -> str:
        doc = REPO_ROOT / "docs/architecture/runtime-current-state.md"
        assert doc.exists(), f"文档不存在: {doc}"
        return doc.read_text(encoding="utf-8")

    def test_doc_mentions_persistence_complete(self, doc_content: str) -> None:
        """文档应反映 Attempt/Assignment 持久化已完成。"""
        assert "持久化" in doc_content
        assert "已完成" in doc_content or "完成" in doc_content

    def test_doc_mentions_adapter_retained(self, doc_content: str) -> None:
        """文档应反映旧执行器仍通过 Adapter 运行。"""
        assert "Adapter" in doc_content or "适配" in doc_content

    def test_doc_no_false_deletion_claim(self, doc_content: str) -> None:
        """文档不应声称旧执行引擎已完全删除。"""
        false_claims = [
            "旧执行引擎删除完成",
            "执行引擎已删除",
            "旧执行器已删除",
        ]
        for claim in false_claims:
            assert claim not in doc_content, f"文档包含虚假声明: {claim}"

    def test_doc_has_completion_status_section(self, doc_content: str) -> None:
        """文档应有完成状态汇总节。"""
        assert "P2" in doc_content or "收尾" in doc_content or "完成状态" in doc_content


# ---------------------------------------------------------------------------
# AST 辅助函数
# ---------------------------------------------------------------------------


def _find_method(
    tree: ast.AST,
    class_name: str,
    method_name: str,
) -> ast.AsyncFunctionDef | ast.FunctionDef | None:
    """在 AST 中查找指定类的指定方法。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == method_name
                ):
                    return item
    return None
