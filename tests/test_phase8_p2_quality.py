"""Phase 8 P2 质量守卫测试：P2-01 ~ P2-05。

P2-01: RuntimeCapabilities 由实例真实配置生成。
P2-02: AgentExecutor Protocol 统一 execute 签名。
P2-03: RuntimeTask 校验完整。
P2-04: RuntimeEvent 字段完整。
P2-05: 架构 AST 守卫增强。
"""

from __future__ import annotations

import ast
import inspect
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGES = REPO_ROOT / "packages"


# =========================================================================
# P2-01: RuntimeCapabilities 由实例真实配置生成
# =========================================================================


class TestP2_01CapabilitiesFromInstance:
    """P2-01: Runtime 的 capabilities 应反映实例真实配置。"""

    def test_persistent_runtime_capabilities_reflect_config(self):
        """PersistentRuntimeV2 的 max_concurrency/queue_capacity 应来自构造参数。"""
        from mcc_harness.runtime.persistent import PersistentRuntimeV2

        class _NoOpExecutor:
            async def execute(self, task: Any, **kwargs: Any) -> Any:
                return None

            async def close(self) -> None:
                pass

        rt = PersistentRuntimeV2(
            _NoOpExecutor(),
            max_concurrency=4,
            queue_capacity=100,
        )
        caps = rt.capabilities
        assert caps.max_concurrency == 4, (
            f"expected max_concurrency=4 from config, got {caps.max_concurrency}"
        )
        assert caps.queue_capacity == 100, (
            f"expected queue_capacity=100 from config, got {caps.queue_capacity}"
        )

    def test_persistent_runtime_default_concurrency(self):
        """PersistentRuntimeV2 默认 max_concurrency=1。"""
        from mcc_harness.runtime.persistent import PersistentRuntimeV2

        class _NoOpExecutor:
            async def execute(self, task: Any, **kwargs: Any) -> Any:
                return None

            async def close(self) -> None:
                pass

        rt = PersistentRuntimeV2(_NoOpExecutor())
        assert rt.capabilities.max_concurrency == 1

    def test_interactive_runtime_capabilities(self):
        """InteractiveRuntimeV2 capabilities max_concurrency=1（单对话）。"""
        from mcc_harness import InteractiveRuntimeV2

        class _FakeAgent:
            async def run(self, messages, *, tools=None, system=None, model=None):
                yield

        rt = InteractiveRuntimeV2(_FakeAgent(), runtime_id="ir-1")
        assert rt.capabilities.max_concurrency == 1
        assert rt.capabilities.supports_cancel_task is True

    def test_capabilities_has_cancel_task(self):
        """RuntimeCapabilities 应有 supports_cancel_task 字段。"""
        from mcc_core.runtime.capabilities import RuntimeCapabilities

        caps = RuntimeCapabilities()
        assert hasattr(caps, "supports_cancel_task")
        assert caps.supports_cancel_task is True  # 默认 True


# =========================================================================
# P2-02: AgentExecutor Protocol 统一 execute 签名
# =========================================================================


class TestP2_02ExecutorProtocol:
    """P2-02: AgentExecutor.execute 统一签名 (task, *, emit, cancellation)。"""

    def test_executor_protocol_has_unified_signature(self):
        """AgentExecutor.execute 应有 emit 和 cancellation 关键字参数。"""
        from mcc_core.runtime.protocols import AgentExecutor

        sig = inspect.signature(AgentExecutor.execute)
        params = sig.parameters
        assert "task" in params, "execute should have 'task' parameter"
        assert "emit" in params, "execute should have 'emit' keyword parameter"
        assert "cancellation" in params, "execute should have 'cancellation' keyword parameter"
        # emit 和 cancellation 应为关键字参数
        assert params["emit"].kind == inspect.Parameter.KEYWORD_ONLY
        assert params["cancellation"].kind == inspect.Parameter.KEYWORD_ONLY

    def test_event_emitter_protocol_exists(self):
        """EventEmitter Protocol 应存在。"""
        from mcc_core.runtime.protocols import EventEmitter

        assert hasattr(EventEmitter, "emit")

    def test_cancellation_token_protocol_exists(self):
        """CancellationToken Protocol 应存在。"""
        from mcc_core.runtime.protocols import CancellationToken

        assert hasattr(CancellationToken, "is_cancelled")
        assert hasattr(CancellationToken, "cancel")
        assert hasattr(CancellationToken, "wait")


# =========================================================================
# P2-03: RuntimeTask 校验完整
# =========================================================================


class TestP2_03RuntimeTaskValidation:
    """P2-03: RuntimeTask 应校验各种边界情况。"""

    def _make_task(self, **overrides: Any):
        from mcc_core.runtime.task import RuntimeTask

        defaults = {
            "task_id": "task-1",
            "attempt_id": "att-1",
            "agent_id": "agent-1",
            "objective": "do something",
        }
        defaults.update(overrides)
        return RuntimeTask(**defaults)

    def test_empty_objective_rejected(self):
        """空白 objective 应被拒绝。"""
        from mcc_core.runtime.errors import TaskValidationError

        with pytest.raises(TaskValidationError, match="空白"):
            self._make_task(objective="   ")

    def test_timeout_must_be_positive(self):
        """timeout_seconds <= 0 应被拒绝。"""
        from mcc_core.runtime.errors import TaskValidationError

        with pytest.raises(TaskValidationError, match="正数"):
            self._make_task(timeout_seconds=0)
        with pytest.raises(TaskValidationError, match="正数"):
            self._make_task(timeout_seconds=-1)

    def test_created_at_must_have_timezone(self):
        """created_at 不带时区应被拒绝。"""
        from mcc_core.runtime.errors import TaskValidationError

        naive = datetime(2026, 1, 1, 12, 0, 0)
        with pytest.raises(TaskValidationError, match="时区"):
            self._make_task(created_at=naive)

    def test_input_must_be_json_serializable(self):
        """input 不可 JSON 序列化应被拒绝。"""
        from mcc_core.runtime.errors import TaskValidationError

        with pytest.raises(TaskValidationError, match="JSON"):
            self._make_task(input={"bad": {1, 2, 3}})

    def test_metadata_must_be_json_serializable(self):
        """metadata 不可 JSON 序列化应被拒绝。"""
        from mcc_core.runtime.errors import TaskValidationError

        with pytest.raises(TaskValidationError, match="JSON"):
            self._make_task(metadata={"bad": object()})

    def test_environment_key_length_limit(self):
        """environment key 超长应被拒绝。"""
        from mcc_core.runtime.errors import TaskValidationError
        from mcc_core.runtime.task import TaskContext

        long_key = "k" * 300
        with pytest.raises(TaskValidationError, match="environment key"):
            self._make_task(
                context=TaskContext(environment={long_key: "v"}),
            )

    def test_environment_value_length_limit(self):
        """environment value 超长应被拒绝。"""
        from mcc_core.runtime.errors import TaskValidationError
        from mcc_core.runtime.task import TaskContext

        long_value = "v" * 10000
        with pytest.raises(TaskValidationError, match="environment value"):
            self._make_task(
                context=TaskContext(environment={"K": long_value}),
            )

    def test_id_length_limit(self):
        """task_id 超长应被拒绝。"""
        from mcc_core.runtime.errors import TaskValidationError

        with pytest.raises(TaskValidationError, match="长度超限"):
            self._make_task(task_id="t" * 300)

    def test_control_char_in_id_rejected(self):
        """task_id 包含控制字符应被拒绝。"""
        from mcc_core.runtime.errors import TaskValidationError

        with pytest.raises(TaskValidationError, match="控制字符"):
            self._make_task(task_id="task\x00bad")

    def test_idempotency_key_length_limit(self):
        """idempotency_key 超长应被拒绝。"""
        from mcc_core.runtime.errors import TaskValidationError

        with pytest.raises(TaskValidationError, match="idempotency_key"):
            self._make_task(idempotency_key="k" * 600)

    def test_valid_task_accepted(self):
        """合法 task 应被接受。"""
        task = self._make_task(
            timeout_seconds=30.0,
            created_at=datetime.now(UTC),
            input={"key": "value"},
            metadata={"meta": 42},
        )
        assert task.timeout_seconds == 30.0


# =========================================================================
# P2-04: RuntimeEvent 字段完整
# =========================================================================


class TestP2_04RuntimeEventFields:
    """P2-04: RuntimeEvent 应有完整关联字段。"""

    def test_event_has_correlation_id(self):
        """RuntimeEvent 应有 correlation_id 字段。"""
        from mcc_core.runtime.enums import RuntimeEventType
        from mcc_core.runtime.event import RuntimeEvent

        event = RuntimeEvent(
            event_id="evt-1",
            event_type=RuntimeEventType.TASK_STARTED,
            correlation_id="corr-1",
        )
        assert event.correlation_id == "corr-1"

    def test_event_has_causation_id(self):
        """RuntimeEvent 应有 causation_id 字段。"""
        from mcc_core.runtime.enums import RuntimeEventType
        from mcc_core.runtime.event import RuntimeEvent

        event = RuntimeEvent(
            event_id="evt-1",
            event_type=RuntimeEventType.TASK_STARTED,
            causation_id="cause-1",
        )
        assert event.causation_id == "cause-1"

    def test_event_has_result_ref(self):
        """RuntimeEvent 应有 result_ref 字段。"""
        from mcc_core.runtime.enums import RuntimeEventType
        from mcc_core.runtime.event import RuntimeEvent

        event = RuntimeEvent(
            event_id="evt-1",
            event_type=RuntimeEventType.TASK_COMPLETED,
            result_ref="result-1",
        )
        assert event.result_ref == "result-1"

    def test_event_has_schema_version(self):
        """RuntimeEvent 应有 schema_version 字段，默认为 1。"""
        from mcc_core.runtime.enums import RuntimeEventType
        from mcc_core.runtime.event import RuntimeEvent

        event = RuntimeEvent(
            event_id="evt-1",
            event_type=RuntimeEventType.TASK_STARTED,
        )
        assert event.schema_version == 1

    def test_event_has_trace_id(self):
        """RuntimeEvent 应有 trace_id 字段。"""
        from mcc_core.runtime.enums import RuntimeEventType
        from mcc_core.runtime.event import RuntimeEvent

        event = RuntimeEvent(
            event_id="evt-1",
            event_type=RuntimeEventType.TASK_STARTED,
            trace_id="trace-1",
        )
        assert event.trace_id == "trace-1"

    def test_event_has_parent_event_id(self):
        """RuntimeEvent 应有 parent_event_id 字段。"""
        from mcc_core.runtime.enums import RuntimeEventType
        from mcc_core.runtime.event import RuntimeEvent

        event = RuntimeEvent(
            event_id="evt-1",
            event_type=RuntimeEventType.TASK_STARTED,
            parent_event_id="parent-evt-1",
        )
        assert event.parent_event_id == "parent-evt-1"

    def test_event_to_dict_includes_all_fields(self):
        """to_dict 应包含所有新增字段。"""
        from mcc_core.runtime.enums import RuntimeEventType
        from mcc_core.runtime.event import RuntimeEvent

        event = RuntimeEvent(
            event_id="evt-1",
            event_type=RuntimeEventType.TASK_STARTED,
            correlation_id="corr-1",
            causation_id="cause-1",
            result_ref="ref-1",
            trace_id="trace-1",
            parent_event_id="parent-1",
        )
        d = event.to_dict()
        assert d["correlation_id"] == "corr-1"
        assert d["causation_id"] == "cause-1"
        assert d["result_ref"] == "ref-1"
        assert d["schema_version"] == 1
        assert d["trace_id"] == "trace-1"
        assert d["parent_event_id"] == "parent-1"


# =========================================================================
# P2-05: 架构 AST 守卫增强
# =========================================================================


def _parse_source(filepath: Path) -> ast.AST:
    return ast.parse(filepath.read_text(encoding="utf-8"))


def _package_files(package: str) -> list[Path]:
    src = PACKAGES / package / "src"
    if not src.exists():
        return []
    return sorted(src.rglob("*.py"))


class TestP2_05ArchitectureAstGuards:
    """P2-05: 增强 AST 架构守卫。"""

    def test_no_teammate_executor_adapter_in_persistent(self):
        """persistent.py 中不得存在 _TeammateExecutorAdapter 类。"""
        f = PACKAGES / "mcc-harness/src/mcc_harness/runtime/persistent.py"
        tree = _parse_source(f)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                assert node.name != "_TeammateExecutorAdapter", (
                    "_TeammateExecutorAdapter should be removed from persistent.py"
                )

    def test_no_submit_task_wait_task_in_orchestrator(self):
        """TeamOrchestrator 不得调用 submit_task/wait_task。"""
        f = PACKAGES / "mcc-harness/src/mcc_harness/team/orchestrator.py"
        tree = _parse_source(f)
        violations: list[str] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("submit_task", "wait_task")
            ):
                violations.append(node.func.attr)
        assert not violations, f"TeamOrchestrator 调用了禁止方法: {violations}"

    def test_no_hasattr_capability_probing(self):
        """不得使用 hasattr 探测 capabilities。"""
        violations: list[str] = []
        for pkg in ("mcc-core", "mcc-harness", "mcc-cli"):
            for f in _package_files(pkg):
                tree = _parse_source(f)
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

    def test_subagent_executor_factory_before_runner(self):
        """SubagentExecutor 应仅走 factory 装配路径，无 runner 回退。"""
        f = PACKAGES / "mcc-harness/src/mcc_harness/execution/subagent_executor.py"
        source = f.read_text(encoding="utf-8")
        factory_check = "self._agent_factory is None or self._factory_context is None"
        assert factory_check in source, "SubagentExecutor 应检查 _agent_factory 和 _factory_context"
        assert "self._runner.run" not in source, "SubagentExecutor 不应有 _runner.run 回退"

    def test_orchestrator_no_agent_loop_constructor(self):
        """TeamOrchestrator 不得直接构造 AgentLoop。"""
        f = PACKAGES / "mcc-harness/src/mcc_harness/team/orchestrator.py"
        tree = _parse_source(f)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "AgentLoop"
            ):
                pytest.fail("TeamOrchestrator directly constructs AgentLoop")
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "AgentLoop"
            ):
                pytest.fail("TeamOrchestrator directly constructs AgentLoop")

    def test_core_no_reverse_dependencies(self):
        """mcc-core 不得 import mcc-harness/mcc-cli/mcc-tools。"""
        forbidden_prefixes = ("mcc_harness", "mcc_cli", "mcc_tools", "mcc_eval", "mcc_tui")
        violations: list[str] = []
        for f in _package_files("mcc-core"):
            try:
                tree = _parse_source(f)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        for prefix in forbidden_prefixes:
                            if alias.name == prefix or alias.name.startswith(prefix + "."):
                                violations.append(f"{f.name} imports {alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module:
                    for prefix in forbidden_prefixes:
                        if node.module == prefix or node.module.startswith(prefix + "."):
                            violations.append(f"{f.name} imports {node.module}")
        assert not violations, f"mcc-core reverse-depends on upper packages:\n{violations}"

    def test_harness_no_cli_dependency(self):
        """mcc-harness 不得 import mcc-cli。"""
        violations: list[str] = []
        for f in _package_files("mcc-harness"):
            try:
                tree = _parse_source(f)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "mcc_cli" or alias.name.startswith("mcc_cli."):
                            violations.append(f"{f.name} imports {alias.name}")
                elif (
                    isinstance(node, ast.ImportFrom)
                    and node.module
                    and (node.module == "mcc_cli" or node.module.startswith("mcc_cli."))
                ):
                    violations.append(f"{f.name} imports {node.module}")
        assert not violations, f"mcc-harness depends on mcc-cli:\n{violations}"

    def test_cli_no_direct_agent_loop_constructor(self):
        """mcc-cli 不得直接构造 AgentLoop 或 TeammateRuntime。"""
        for class_name in ("AgentLoop", "TeammateRuntime"):
            violations: list[str] = []
            for f in _package_files("mcc-cli"):
                try:
                    tree = _parse_source(f)
                except SyntaxError:
                    continue
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == class_name
                    ):
                        violations.append(str(f.relative_to(REPO_ROOT)))
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == class_name
                    ):
                        violations.append(str(f.relative_to(REPO_ROOT)))
            assert not violations, f"mcc-cli directly constructs {class_name}:\n{violations}"
