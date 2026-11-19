"""P1-04 最终拆除测试：验证旧路径已删除、新路径可用。

确定性断言：
- _TeammateExecutorAdapter 已从 persistent 模块删除；
- PersistentRuntimeV2.from_teammate 已删除；
- PersistentRuntimeV2.from_executor 存在且为 classmethod；
- from_executor 接受可选 teammate/bus/lead_name 参数；
- 所有生产调用方使用 from_executor 而非 from_teammate。
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import pytest


class TestOldPathRemoved:
    """旧路径 _TeammateExecutorAdapter / from_teammate 已彻底删除。"""

    def test_teammate_executor_adapter_not_exported(self) -> None:
        import mcc_harness.runtime.persistent as persistent_mod

        assert not hasattr(persistent_mod, "_TeammateExecutorAdapter"), (
            "_TeammateExecutorAdapter 应已从 persistent 模块删除"
        )

    def test_from_teammate_not_defined(self) -> None:
        from mcc_harness.runtime.persistent import PersistentRuntimeV2

        assert not hasattr(PersistentRuntimeV2, "from_teammate"), (
            "PersistentRuntimeV2.from_teammate 应已删除"
        )

    def test_no_from_teammate_in_source(self) -> None:
        """源文件中不应出现 from_teammate 定义或调用。"""
        repo = Path(__file__).resolve().parent.parent
        for py in repo.rglob("*.py"):
            if ".venv" in str(py) or "__pycache__" in str(py):
                continue
            if py.name == Path(__file__).name:
                continue
            text = py.read_text(encoding="utf-8")
            assert ".from_teammate(" not in text, f"{py} 仍包含 .from_teammate() 调用"
            assert "def from_teammate(" not in text, f"{py} 仍定义 from_teammate"

    def test_no_teammate_executor_adapter_in_source(self) -> None:
        """源文件中不应出现 _TeammateExecutorAdapter 引用。"""
        repo = Path(__file__).resolve().parent.parent
        for py in repo.rglob("*.py"):
            if ".venv" in str(py) or "__pycache__" in str(py):
                continue
            if py.name == Path(__file__).name:
                continue
            text = py.read_text(encoding="utf-8")
            assert "_TeammateExecutorAdapter" not in text, f"{py} 仍引用 _TeammateExecutorAdapter"


class TestNewPathAvailable:
    """新路径 from_executor 可用且功能完整。"""

    def test_from_executor_exists(self) -> None:
        from mcc_harness.runtime.persistent import PersistentRuntimeV2

        assert hasattr(PersistentRuntimeV2, "from_executor")
        assert callable(PersistentRuntimeV2.from_executor)

    def test_from_executor_is_classmethod(self) -> None:
        from mcc_harness.runtime.persistent import PersistentRuntimeV2

        assert isinstance(
            inspect.getattr_static(PersistentRuntimeV2, "from_executor"),
            classmethod,
        )

    @pytest.mark.parametrize(
        "param_name",
        ["teammate", "bus", "lead_name"],
    )
    def test_from_executor_accepts_optional_lifecycle_params(
        self,
        param_name: str,
    ) -> None:
        """from_executor 签名包含可选 teammate/bus/lead_name 参数。"""
        from mcc_harness.runtime.persistent import PersistentRuntimeV2

        sig = inspect.signature(PersistentRuntimeV2.from_executor)
        assert param_name in sig.parameters, f"from_executor 应接受 {param_name} 参数"


class TestFromExecutorWithTeammate:
    """from_executor 接受 teammate 参数时正确设置内部状态。"""

    async def test_from_executor_with_teammate_sets_fields(self) -> None:
        from mcc_harness.runtime.persistent import PersistentRuntimeV2

        class StubExecutor:
            async def execute(self, task: Any, **kw: Any) -> Any: ...

            async def close(self) -> None: ...

        class StubTeammate:
            spec = type("S", (), {"name": "worker"})()

            def notify(self) -> None: ...

            @property
            def alive(self) -> bool:
                return False

            async def start(self, *a: Any, **k: Any) -> None: ...

            async def shutdown(self, *a: Any, **k: Any) -> None: ...

            def info(self) -> Any:
                return None

        class StubBus:
            async def send(self, *a: Any, **k: Any) -> None: ...

        pr = PersistentRuntimeV2.from_executor(
            StubExecutor(),
            agent_id="worker",
            teammate=StubTeammate(),
            bus=StubBus(),
            lead_name="lead",
        )
        assert pr._teammate is not None
        assert pr._bus is not None
        assert pr._lead_name == "lead"

    async def test_from_executor_without_teammate_has_none(self) -> None:
        from mcc_harness.runtime.persistent import PersistentRuntimeV2

        class StubExecutor:
            async def execute(self, task: Any, **kw: Any) -> Any: ...

            async def close(self) -> None: ...

        pr = PersistentRuntimeV2.from_executor(
            StubExecutor(),
            agent_id="worker",
        )
        assert pr._teammate is None
        assert pr._bus is None


class TestProductionCallersUseFromExecutor:
    """生产代码中所有 PersistentRuntimeV2 构造均通过 from_executor。"""

    def test_teammate_spawn_uses_from_executor(self) -> None:
        """teammate.py spawn 方法使用 from_executor。"""
        import mcc_harness.teammate as teammate_mod

        source = inspect.getsource(teammate_mod.TeammateManager.spawn)
        assert "from_executor" in source, "TeammateManager.spawn 应使用 from_executor"
        assert "from_teammate" not in source

    def test_runtime_factory_uses_from_executor(self) -> None:
        """runtime/factory.py create_persistent_v2 使用 from_executor。"""
        import mcc_harness.runtime.factory as factory_mod

        source = inspect.getsource(factory_mod.RuntimeFactory.create_persistent_v2)
        assert "from_executor" in source, "RuntimeFactory.create_persistent_v2 应使用 from_executor"

    def test_execution_factory_uses_from_executor(self) -> None:
        """execution/factory.py create PERSISTENT 分支使用 from_executor。"""
        import mcc_harness.execution.factory as exec_factory_mod

        source = inspect.getsource(exec_factory_mod.DefaultAgentFactory.create)
        assert "from_executor" in source, (
            "DefaultAgentFactory.create PERSISTENT 分支应使用 from_executor"
        )

    def test_teammate_runtime_has_build_agent_executor(self) -> None:
        """TeammateRuntime 暴露 build_agent_executor 公共方法。"""
        from mcc_harness.teammate import TeammateRuntime

        assert hasattr(TeammateRuntime, "build_agent_executor")
        assert callable(TeammateRuntime.build_agent_executor)
