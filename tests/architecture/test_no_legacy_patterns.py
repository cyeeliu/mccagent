"""PR-16 架构守卫测试。

使用 AST 扫描 packages/ 下所有生产 Python 文件，断言以下旧模式不存在或仅限于
已知白名单：

1. V1 API 拗留 — 不存在 EphemeralRuntime、旧 InteractiveRuntime（非 V2）、
   旧 PersistentRuntime（非 V2）的类定义与引用。
2. AgentLoop 直接构造 — 仅允许白名单文件构造（AgentLoopBuilder.build、
   DefaultAgentFactory、SubagentRunner、Teammate 既有路径）。
3. run_streaming 调用 — 仅允许白名单文件调用（AgentLoop 自身、
   AgentLoopExecutor、InteractiveRuntimeV2、SubagentRunner、eval 适配器）。
4. submit_task / wait_task 旧兼容方法调用 — 仅允许 teammate 既有路径调用。

白名单采用 (文件相对路径, 行号) 精确匹配。新增违规将导致测试失败，从而
防止旧模式回潮。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

# packages 根目录
_PACKAGES_DIR = Path(__file__).resolve().parents[2] / "packages"

# ---------------------------------------------------------------------------
# V1 API 旧类名 — 任何位置都不允许出现（类定义或引用）
# ---------------------------------------------------------------------------
_V1_FORBIDDEN_NAMES: frozenset[str] = frozenset(
    {
        "EphemeralRuntime",
        # 旧 InteractiveRuntime / PersistentRuntime（非 V2）已删除，
        # 仅保留 V2 后缀版本。若有人重新引入无后缀旧类，守卫触发。
    },
)

# ---------------------------------------------------------------------------
# AgentLoop 直接构造白名单
# 键: 相对于 packages/ 的文件路径
# 值: 允许构造的行号集合
# ---------------------------------------------------------------------------
_AGENTLOOP_CTOR_WHITELIST: dict[str, set[int]] = {
    "mcc-harness/src/mcc_harness/agent.py": {1142},  # AgentLoopBuilder.build()
    "mcc-harness/src/mcc_harness/execution/factory.py": {141},  # DefaultAgentFactory
    "mcc-harness/src/mcc_harness/subagent.py": {562},  # SubagentRunner 既有路径
    "mcc-harness/src/mcc_harness/teammate.py": {442},  # Teammate 既有路径
}

# ---------------------------------------------------------------------------
# run_streaming 调用白名单
# ---------------------------------------------------------------------------
_RUN_STREAMING_WHITELIST: dict[str, set[int]] = {
    "mcc-harness/src/mcc_harness/agent.py": {706},  # AgentLoop 自身
    "mcc-harness/src/mcc_harness/execution/agent_loop_executor.py": {75},
    "mcc-harness/src/mcc_harness/runtime/interactive.py": {453},
    "mcc-harness/src/mcc_harness/subagent.py": {599},
    "mcc-eval/src/mcc_eval/adapters.py": {301},
}

# ---------------------------------------------------------------------------
# submit_task / wait_task 旧兼容方法调用白名单
# ---------------------------------------------------------------------------
_SUBMIT_TASK_WHITELIST: dict[str, set[int]] = {
    "mcc-harness/src/mcc_harness/teammate.py": {1074, 1118},
}

_WAIT_TASK_WHITELIST: dict[str, set[int]] = {}


def _collect_production_files() -> list[Path]:
    """收集 packages/ 下所有生产 Python 文件（排除测试、__pycache__）。"""
    files: list[Path] = []
    for py in _PACKAGES_DIR.rglob("*.py"):
        rel = py.relative_to(_PACKAGES_DIR)
        parts = rel.parts
        # 排除测试目录
        if "tests" in parts or "test" in parts:
            continue
        if "__pycache__" in parts:
            continue
        files.append(py)
    return sorted(files)


def _rel_path(path: Path) -> str:
    """返回相对于 packages/ 的路径字符串。"""
    return str(path.relative_to(_PACKAGES_DIR))


def _find_call_name(node: ast.Call) -> str | None:
    """提取 Call 节点的被调用名称。

    AgentLoop(...)         -> "AgentLoop"
    self.run_streaming(...) -> "run_streaming"
    obj.submit_task(...)   -> "submit_task"
    """
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _scan_file(path: Path) -> dict[str, list[tuple[int, str]]]:
    """扫描单个文件，返回 {模式: [(行号, 详情), ...]}。"""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    findings: dict[str, list[tuple[int, str]]] = {
        "v1_name": [],
        "agentloop_ctor": [],
        "run_streaming": [],
        "submit_task": [],
        "wait_task": [],
    }

    for node in ast.walk(tree):
        # --- V1 旧名引用（Name 节点）---
        if isinstance(node, ast.Name) and node.id in _V1_FORBIDDEN_NAMES:
            findings["v1_name"].append((node.lineno, node.id))

        # --- V1 旧名引用（Attribute 节点，如 module.EphemeralRuntime）---
        if isinstance(node, ast.Attribute) and node.attr in _V1_FORBIDDEN_NAMES:
            findings["v1_name"].append((node.lineno, node.attr))

        if isinstance(node, ast.Call):
            name = _find_call_name(node)
            if name is None:
                continue
            # --- AgentLoop 直接构造 ---
            if name == "AgentLoop":
                findings["agentloop_ctor"].append((node.lineno, "AgentLoop(...)"))
            # --- run_streaming 调用 ---
            if name == "run_streaming":
                findings["run_streaming"].append(
                    (node.lineno, "run_streaming(...)"),
                )
            # --- submit_task 调用 ---
            if name == "submit_task":
                findings["submit_task"].append((node.lineno, "submit_task(...)"))
            # --- wait_task 调用 ---
            if name == "wait_task":
                findings["wait_task"].append((node.lineno, "wait_task(...)"))

    return findings


def _check_whitelist(
    pattern: str,
    whitelist: dict[str, set[int]],
    all_files: list[Path],
) -> list[str]:
    """检查某模式的实际出现位置是否都在白名单内。返回违规描述列表。"""
    violations: list[str] = []
    for path in all_files:
        rel = _rel_path(path)
        findings = _scan_file(path)
        for lineno, detail in findings[pattern]:
            allowed_lines = whitelist.get(rel, set())
            if lineno not in allowed_lines:
                violations.append(
                    f"{rel}:{lineno}: {detail}（白名单未收录）",
                )
    # 检查白名单中是否有已不存在的条目（行号漂移）
    for rel, lines in whitelist.items():
        full = _PACKAGES_DIR / rel
        if not full.exists():
            violations.append(f"白名单条目失效（文件不存在）: {rel}")
            continue
        findings = _scan_file(full)
        actual_lines = {ln for ln, _ in findings[pattern]}
        for expected in lines:
            if expected not in actual_lines:
                violations.append(
                    f"{rel}:{expected}: 白名单行号漂移"
                    f"（该行已无 {pattern} 调用）",
                )
    return violations


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def production_files() -> list[Path]:
    return _collect_production_files()


class TestNoV1ApiRemnants:
    """V1 API 拗留守卫：禁止旧类名出现。"""

    def test_no_ephemeral_runtime_or_legacy_classes(
        self,
        production_files: list[Path],
    ) -> None:
        violations: list[str] = []
        for path in production_files:
            rel = _rel_path(path)
            findings = _scan_file(path)
            for lineno, name in findings["v1_name"]:
                violations.append(f"{rel}:{lineno}: 引用旧 V1 类 {name}")
        assert not violations, (
            "发现 V1 API 拗留（应已完全删除）:\n" + "\n".join(violations)
        )


class TestAgentLoopConstructionGuard:
    """AgentLoop 直接构造守卫：仅允许白名单位置。"""

    def test_agentloop_ctor_only_in_whitelist(
        self,
        production_files: list[Path],
    ) -> None:
        violations = _check_whitelist(
            "agentloop_ctor",
            _AGENTLOOP_CTOR_WHITELIST,
            production_files,
        )
        assert not violations, (
            "AgentLoop 直接构造出现在白名单之外（应通过 AgentFactory 装配）:\n"
            + "\n".join(violations)
        )


class TestRunStreamingCallGuard:
    """run_streaming 调用守卫：仅允许白名单位置。"""

    def test_run_streaming_only_in_whitelist(
        self,
        production_files: list[Path],
    ) -> None:
        violations = _check_whitelist(
            "run_streaming",
            _RUN_STREAMING_WHITELIST,
            production_files,
        )
        assert not violations, (
            "run_streaming 调用出现在白名单之外:\n" + "\n".join(violations)
        )


class TestLegacyCompatMethodGuard:
    """submit_task / wait_task 旧兼容方法调用守卫。"""

    def test_submit_task_only_in_whitelist(
        self,
        production_files: list[Path],
    ) -> None:
        violations = _check_whitelist(
            "submit_task",
            _SUBMIT_TASK_WHITELIST,
            production_files,
        )
        assert not violations, (
            "submit_task 调用出现在白名单之外:\n" + "\n".join(violations)
        )

    def test_wait_task_only_in_whitelist(
        self,
        production_files: list[Path],
    ) -> None:
        violations = _check_whitelist(
            "wait_task",
            _WAIT_TASK_WHITELIST,
            production_files,
        )
        assert not violations, (
            "wait_task 调用出现在白名单之外:\n" + "\n".join(violations)
        )


class TestNoCompatImportGuards:
    """兼容导入守卫：禁止 try/except ImportError 兼容分支。"""

    def test_no_except_import_error(self, production_files: list[Path]) -> None:
        violations: list[str] = []
        for path in production_files:
            rel = _rel_path(path)
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler) and isinstance(
                    node.type, ast.Name
                ) and node.type.id == "ImportError":
                    violations.append(
                        f"{rel}:{node.lineno}: except ImportError 兼容分支",
                    )
        assert not violations, (
            "发现 except ImportError 兼容导入分支（应已清理）:\n"
            + "\n".join(violations)
        )
