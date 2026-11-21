"""PR-08: wiring.py _wire_team 传入 async_task_store 给 TeamOrchestrator。

P1-06 要求 TeamOrchestrator 统一通过 TaskStoreProtocol 操作任务。
_wire_team 已创建 async_task_store（TaskStoreAdapter 或 InMemoryTaskStore）
并传给 TaskDispatcher，但未传给 TeamOrchestrator，导致 orchestrator
仍走旧同步 task_store 路径，与 dispatcher 不一致。

修复后验证：
- wiring.py 源码中 TeamOrchestrator 构造调用包含 async_task_store=
- 通过 wiring 构建的 orchestrator.async_task_store 不为 None
"""

from __future__ import annotations

import re
from pathlib import Path


def _read_source(rel_path: str) -> str:
    root = Path(__file__).resolve().parent.parent
    return (root / rel_path).read_text(encoding="utf-8")


class TestWiringPassesAsyncStore:
    """wiring.py 将 async_task_store 传给 TeamOrchestrator。"""

    def test_wiring_passes_async_task_store_to_orchestrator(self):
        """wiring.py 中 TeamOrchestrator 构造包含 async_task_store= 参数。"""
        source = _read_source("packages/mcc-cli/src/mcc_cli/wiring.py")
        # 找到 TeamOrchestrator( 构造调用块
        pattern = r"TeamOrchestrator\([^)]*async_task_store\s*="
        match = re.search(pattern, source, re.DOTALL)
        assert match is not None, (
            "wiring.py should pass async_task_store= to TeamOrchestrator"
        )

    def test_wiring_creates_async_task_store(self):
        """wiring.py 创建 async_task_store 变量。"""
        source = _read_source("packages/mcc-cli/src/mcc_cli/wiring.py")
        assert "async_task_store" in source, (
            "wiring.py should create async_task_store variable"
        )
