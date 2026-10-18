"""会话持久化与记忆。

JSONL 会话流水（录制、恢复、列表、清理）与
Markdown 长期记忆存储及 Agent 工具。
"""

from __future__ import annotations

from .memory import (
    MarkdownMemoryStore,
    ReadMemoryTool,
    SaveMemoryTool,
    default_memory_base,
    memory_tools,
)
from .models import SessionMeta, TurnSnapshot
from .recorder import (
    TITLE_MAX_CHARS,
    TITLE_MAX_TOKENS,
    TITLE_MODEL,
    SessionNotFoundError,
    SessionRecorder,
    build_turn_snapshot,
    default_sessions_base,
    list_sessions,
    load_session,
    maybe_generate_title,
    purge_sessions,
)

__all__ = [
    "MarkdownMemoryStore",
    "ReadMemoryTool",
    "SaveMemoryTool",
    "SessionMeta",
    "SessionNotFoundError",
    "SessionRecorder",
    "TITLE_MAX_CHARS",
    "TITLE_MAX_TOKENS",
    "TITLE_MODEL",
    "TurnSnapshot",
    "build_turn_snapshot",
    "default_memory_base",
    "default_sessions_base",
    "list_sessions",
    "load_session",
    "maybe_generate_title",
    "memory_tools",
    "purge_sessions",
]
__version__ = "0.1.0"
