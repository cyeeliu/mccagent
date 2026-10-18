"""任务调度状态机与租约常量。

run 生命周期：queued → leased → running → passed/failed/error/timeout；
租约到期未续由 reaper 回收，重试超过上限的 run 直接置为 error。
"""

from __future__ import annotations

RUN_STATUSES: frozenset[str] = frozenset(
    {"queued", "leased", "running", "passed", "failed", "error", "timeout"}
)
ACTIVE_STATUSES: frozenset[str] = frozenset({"leased", "running"})
TERMINAL_STATUSES: frozenset[str] = frozenset({"passed", "failed", "error", "timeout"})

BATCH_STATUSES: frozenset[str] = frozenset({"pending", "running", "done", "cancelled"})

MAX_RETRIES = 3
