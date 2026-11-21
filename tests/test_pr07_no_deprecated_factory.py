"""PR-07: wiring.py 和 events.py 不再调用已废弃的 Factory 方法。

P1-07 要求 AgentFactory 成为唯一创建入口。build_interactive_runtime_v2
和 create_interactive_runtime 已标记 DeprecationWarning，wiring.py
和 events.py 应直接构造 InteractiveRuntimeV2，不再经过废弃方法。

修复后验证：
- wiring.py 源码不包含 build_interactive_runtime_v2 调用
- events.py 源码不包含 create_interactive_runtime 调用
"""

from __future__ import annotations

import re
from pathlib import Path


def _read_source(rel_path: str) -> str:
    root = Path(__file__).resolve().parent.parent
    return (root / rel_path).read_text(encoding="utf-8")


class TestPr07NoDeprecatedFactoryCalls:
    """生产代码不再调用已废弃的 Factory 方法。"""

    def test_wiring_no_build_interactive_runtime_v2(self):
        """wiring.py 不调用 build_interactive_runtime_v2。"""
        source = _read_source("packages/mcc-cli/src/mcc_cli/wiring.py")
        # 排除注释和字符串中的提及
        calls = re.findall(r"\.build_interactive_runtime_v2\s*\(", source)
        assert calls == [], (
            f"wiring.py should not call build_interactive_runtime_v2, "
            f"found {len(calls)} call(s)"
        )

    def test_events_no_create_interactive_runtime(self):
        """events.py 不调用 create_interactive_runtime。"""
        source = _read_source("packages/mcc-cli/src/mcc_cli/events.py")
        calls = re.findall(r"\.create_interactive_runtime\s*\(", source)
        assert calls == [], (
            f"events.py should not call create_interactive_runtime, "
            f"found {len(calls)} call(s)"
        )

    def test_wiring_uses_interactive_runtime_v2(self):
        """wiring.py 直接构造 InteractiveRuntimeV2。"""
        source = _read_source("packages/mcc-cli/src/mcc_cli/wiring.py")
        assert "InteractiveRuntimeV2" in source, (
            "wiring.py should directly construct InteractiveRuntimeV2"
        )
