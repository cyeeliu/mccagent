"""静态架构测试：验证模块间依赖方向符合分层约束。

禁止的依赖关系：
  mcc-cli -> TeammateRuntime / AgentLoop（直接构造）
  TeamOrchestrator -> AgentLoop / LlmProvider（直接构造）
  mcc-core -> mcc-harness / mcc-cli / mcc-tools（反向依赖）

允许的依赖关系：
  mcc-cli -> ApplicationService -> AgentFactory / TeamOrchestrator
  RuntimeCoordinator -> AgentRuntime
  TeamOrchestrator -> TaskScheduler / TaskDispatcher / TaskStore
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGES = REPO_ROOT / "packages"


def _collect_imports(filepath: Path) -> set[str]:
    """提取 Python 文件中所有顶层和函数内 import 的模块名。"""
    try:
        tree = ast.parse(filepath.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _contains_constructor(filepath: Path, class_name: str) -> bool:
    """检查文件中是否直接构造了指定类（出现 ClassName( 模式）。"""
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == class_name
        ):
            return True
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == class_name
        ):
            return True
    return False


def _package_files(package: str) -> list[Path]:
    """递归列出包内所有 .py 文件。"""
    src = PACKAGES / package / "src"
    if not src.exists():
        return []
    return sorted(src.rglob("*.py"))


def _file_to_module(filepath: Path) -> str:
    """将文件路径转换为模块点路径。"""
    rel = filepath.relative_to(PACKAGES)
    parts = list(rel.parts)
    parts[-1] = parts[-1][:-3]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


# -- 反向依赖测试：mcc-core 不得依赖上层包 --


class TestNoReverseDependencies:
    """mcc-core 是最底层包，不得 import mcc-harness/mcc-cli/mcc-tools/mcc-eval。"""

    @pytest.mark.parametrize("forbidden", [
        "mcc_harness",
        "mcc_cli",
        "mcc_tools",
        "mcc_eval",
        "mcc_tui",
    ])
    def test_core_does_not_import(self, forbidden):
        violations: list[str] = []
        for f in _package_files("mcc-core"):
            mods = _collect_imports(f)
            for m in mods:
                if m == forbidden or m.startswith(forbidden + "."):
                    violations.append(f"{_file_to_module(f)} imports {m}")
        assert not violations, (
            f"mcc-core reverse-depends on {forbidden}:\n" + "\n".join(violations)
        )


# -- CLI 不直接构造 AgentLoop --


class TestCliLayerConstraints:
    """CLI 层不得直接构造 AgentLoop / TeammateRuntime。"""

    @pytest.mark.parametrize("class_name", ["AgentLoop", "TeammateRuntime"])
    def test_cli_no_direct_constructor(self, class_name):
        violations: list[str] = []
        for f in _package_files("mcc-cli"):
            if _contains_constructor(f, class_name):
                violations.append(str(f.relative_to(REPO_ROOT)))
        assert not violations, (
            f"mcc-cli directly constructs {class_name}:\n"
            + "\n".join(violations)
        )


# -- TeamOrchestrator 不直接构造 AgentLoop / LLM --


class TestOrchestratorConstraints:
    """TeamOrchestrator 不得直接构造 AgentLoop 或 LLM Provider。"""

    @pytest.fixture
    def orchestrator_file(self) -> Path:
        f = PACKAGES / "mcc-harness/src/mcc_harness/team/orchestrator.py"
        assert f.exists(), f"orchestrator.py not found at {f}"
        return f

    @pytest.mark.parametrize("class_name", ["AgentLoop", "LlmProvider"])
    def test_no_direct_constructor(self, orchestrator_file, class_name):
        assert not _contains_constructor(orchestrator_file, class_name), (
            f"TeamOrchestrator directly constructs {class_name}"
        )

    def test_no_agent_loop_import(self, orchestrator_file):
        mods = _collect_imports(orchestrator_file)
        agent_loop_imports = [m for m in mods if "AgentLoop" in m]
        assert not agent_loop_imports, (
            f"TeamOrchestrator imports AgentLoop: {agent_loop_imports}"
        )


# -- Runtime 不直接构造 AgentLoop --


class TestRuntimeConstraints:
    """AgentRuntime 实现层不得在 Protocol 模块中构造 AgentLoop。"""

    def test_protocol_no_agent_loop_constructor(self):
        proto_file = PACKAGES / "mcc-core/src/mcc_core/runtime/protocols.py"
        assert not _contains_constructor(proto_file, "AgentLoop"), (
            "AgentRuntime Protocol directly constructs AgentLoop"
        )

    def test_base_no_agent_loop_constructor(self):
        base_file = PACKAGES / "mcc-harness/src/mcc_harness/runtime/base.py"
        assert not _contains_constructor(base_file, "AgentLoop"), (
            "runtime/base.py directly constructs AgentLoop"
        )


# -- 分层依赖方向测试 --


class TestLayering:
    """验证包间分层依赖方向。"""

    def test_harness_does_not_import_cli(self):
        violations: list[str] = []
        for f in _package_files("mcc-harness"):
            mods = _collect_imports(f)
            for m in mods:
                if m == "mcc_cli" or m.startswith("mcc_cli."):
                    violations.append(f"{_file_to_module(f)} imports {m}")
        assert not violations, (
            "mcc-harness depends on mcc-cli:\n" + "\n".join(violations)
        )

    def test_core_does_not_import_harness(self):
        violations: list[str] = []
        for f in _package_files("mcc-core"):
            mods = _collect_imports(f)
            for m in mods:
                if m == "mcc_harness" or m.startswith("mcc_harness."):
                    violations.append(f"{_file_to_module(f)} imports {m}")
        assert not violations, (
            "mcc-core depends on mcc-harness:\n" + "\n".join(violations)
        )
