"""Runtime 领域标识类型。

所有 ID 均为 str 别名，便于序列化和日志关联。
ID 语义和生命周期见迁移方案 §3.1。
"""

from __future__ import annotations

AgentId = str
RuntimeId = str
RunId = str
TaskId = str
AssignmentId = str
AttemptId = str
EventId = str
SessionId = str
CorrelationId = str

__all__ = [
    "AgentId",
    "RuntimeId",
    "RunId",
    "TaskId",
    "AssignmentId",
    "AttemptId",
    "EventId",
    "SessionId",
    "CorrelationId",
]
