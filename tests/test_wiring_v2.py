"""wiring 层 V2 Runtime 集成测试。

验证 build_source 始终创建 InteractiveRuntimeV2（特性开关已移除，
V2 Runtime 为唯一路径）。
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestBuildSourceV2:
    """build_source 始终使用 V2 Runtime。"""

    def test_creates_interactive_runtime_v2(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        from mcc_cli import _build_source
        from mcc_harness.runtime.interactive import InteractiveRuntimeV2

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        source = _build_source(tmp_path, None, True, None)
        assert isinstance(source.runtime, InteractiveRuntimeV2)

    def test_v2_runtime_registered_in_coordinator(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        from mcc_cli import _build_source

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        source = _build_source(tmp_path, None, True, None)
        registered = source.coordinator.list_all()
        assert any(r.runtime_id == source.runtime.runtime_id for r in registered)

    def test_v2_runtime_has_agent(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        from mcc_cli import _build_source

        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        source = _build_source(tmp_path, None, True, None)
        assert source.runtime.agent is source.agent
