"""PR-09: TaskStoreAdapter 损坏 fail closed + 异步写入（P1-03 + P1-11）。

P1-03/P1-11 要求：
1. JSON 损坏时 fail closed（抛异常阻止启动），不静默 return
2. 所有磁盘 IO 放入 asyncio.to_thread，不阻塞事件循环

修复前 _load_attempts/_load_assignments/_load_run_ids 在 JSON
解析失败时直接 return（fail open），_save_* 在 async 方法中
同步调用阻塞事件循环。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcc_core.runtime import InMemoryTaskStore
from mcc_harness.runtime import TaskStoreAdapter


class TestFailClosedOnCorruptJson:
    """JSON 损坏时 fail closed，不静默初始化空 Store。"""

    def test_corrupt_attempts_json_raises(self, tmp_path: Path):
        """attempts.json 损坏时初始化抛异常。"""
        persist_dir = tmp_path / ".runtime" / "attempts"
        persist_dir.mkdir(parents=True)
        (persist_dir / "attempts.json").write_text("{invalid json", encoding="utf-8")

        with pytest.raises((json.JSONDecodeError, ValueError, OSError)):
            TaskStoreAdapter(InMemoryTaskStore(), persist_dir=persist_dir)

    def test_corrupt_assignments_json_raises(self, tmp_path: Path):
        """assignments.json 损坏时初始化抛异常。"""
        persist_dir = tmp_path / ".runtime" / "attempts"
        persist_dir.mkdir(parents=True)
        (persist_dir / "assignments.json").write_text("not json", encoding="utf-8")

        with pytest.raises((json.JSONDecodeError, ValueError, OSError)):
            TaskStoreAdapter(InMemoryTaskStore(), persist_dir=persist_dir)

    def test_corrupt_run_ids_json_raises(self, tmp_path: Path):
        """run_ids.json 损坏时初始化抛异常。"""
        persist_dir = tmp_path / ".runtime" / "attempts"
        persist_dir.mkdir(parents=True)
        (persist_dir / "run_ids.json").write_text("<<<broken>>>", encoding="utf-8")

        with pytest.raises((json.JSONDecodeError, ValueError, OSError)):
            TaskStoreAdapter(InMemoryTaskStore(), persist_dir=persist_dir)


class TestAsyncDiskWrite:
    """_save_* 方法通过 asyncio.to_thread 执行，不阻塞事件循环。"""

    def test_save_methods_use_to_thread(self):
        """源码中 _save_* 调用点使用 asyncio.to_thread。"""
        import inspect

        source = inspect.getsource(TaskStoreAdapter)
        # 不应在 async 方法中直接调用 self._save_attempts()
        # 而应通过 asyncio.to_thread 包裹
        lines = source.splitlines()
        direct_calls = [
            line.strip()
            for line in lines
            if "self._save_attempts()" in line
            and "asyncio.to_thread" not in line
            and "def _save_attempts" not in line
        ]
        assert direct_calls == [], (
            f"_save_attempts should be called via asyncio.to_thread, "
            f"found direct calls: {direct_calls}"
        )

        direct_calls = [
            line.strip()
            for line in lines
            if "self._save_assignments()" in line
            and "asyncio.to_thread" not in line
            and "def _save_assignments" not in line
        ]
        assert direct_calls == [], (
            f"_save_assignments should be called via asyncio.to_thread, "
            f"found direct calls: {direct_calls}"
        )

        direct_calls = [
            line.strip()
            for line in lines
            if "self._save_run_ids()" in line
            and "asyncio.to_thread" not in line
            and "def _save_run_ids" not in line
        ]
        assert direct_calls == [], (
            f"_save_run_ids should be called via asyncio.to_thread, "
            f"found direct calls: {direct_calls}"
        )
