"""PR-01 修复: TaskStoreAdapter 状态映射不折叠。

验证：
- TaskStatus.QUEUED 映射为 "queued" 而非 "pending"
- TaskStatus.DISPATCHED 映射为 "dispatched" 而非 "claimed"
- TaskStatus.ACCEPTED 映射为 "accepted" 而非 "claimed"
- TaskStatus.RUNNING 映射为 "running" 而非 "in_progress"
- 反向映射不丢失信息："accepted" 映射回 TaskStatus.ACCEPTED
"""

from __future__ import annotations

import pytest
from mcc_core.runtime import TaskStatus


class TestPR01StatusNoFolding:
    """PR-01 修复: TaskStoreAdapter 状态映射不折叠。"""

    def test_enum_to_status_not_folded(self) -> None:
        """_ENUM_TO_STATUS 不折叠状态。"""
        from mcc_harness.runtime.task_store_adapter import _ENUM_TO_STATUS

        assert _ENUM_TO_STATUS[TaskStatus.QUEUED] == "queued", (
            f"QUEUED 应映射为 'queued'，当前为 '{_ENUM_TO_STATUS[TaskStatus.QUEUED]}'"
        )
        assert _ENUM_TO_STATUS[TaskStatus.DISPATCHED] == "dispatched", (
            f"DISPATCHED 应映射为 'dispatched'，当前为 '{_ENUM_TO_STATUS[TaskStatus.DISPATCHED]}'"
        )
        assert _ENUM_TO_STATUS[TaskStatus.ACCEPTED] == "accepted", (
            f"ACCEPTED 应映射为 'accepted'，当前为 '{_ENUM_TO_STATUS[TaskStatus.ACCEPTED]}'"
        )
        assert _ENUM_TO_STATUS[TaskStatus.RUNNING] == "running", (
            f"RUNNING 应映射为 'running'，当前为 '{_ENUM_TO_STATUS[TaskStatus.RUNNING]}'"
        )

    def test_status_to_enum_not_lossy(self) -> None:
        """_STATUS_TO_ENUM 反向映射不丢失信息。"""
        from mcc_harness.runtime.task_store_adapter import _STATUS_TO_ENUM

        assert _STATUS_TO_ENUM["queued"] is TaskStatus.QUEUED
        assert _STATUS_TO_ENUM["dispatched"] is TaskStatus.DISPATCHED
        assert _STATUS_TO_ENUM["accepted"] is TaskStatus.ACCEPTED
        assert _STATUS_TO_ENUM["running"] is TaskStatus.RUNNING

    def test_round_trip_no_information_loss(self) -> None:
        """enum → string → enum 往返不丢失信息。"""
        from mcc_harness.runtime.task_store_adapter import (
            _ENUM_TO_STATUS,
            _STATUS_TO_ENUM,
        )

        for status in [
            TaskStatus.QUEUED,
            TaskStatus.DISPATCHED,
            TaskStatus.ACCEPTED,
            TaskStatus.RUNNING,
        ]:
            mapped = _ENUM_TO_STATUS[status]
            restored = _STATUS_TO_ENUM[mapped]
            assert restored is status, (
                f"往返映射丢失信息: {status.name} -> '{mapped}' -> {restored.name}"
            )
