"""会话录制器：JSONL 流水落盘 + 元数据维护 + 恢复与列表。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mcc_core import ContentBlock, Message, Role, TextBlock, ToolResultBlock, Usage
from mcc_llm import CompleteRequest

from .models import SessionMeta, TurnSnapshot


def default_sessions_base() -> Path:
    """默认会话目录（调用时求值，跟随当前 HOME）。"""
    return Path.home() / ".mcc" / "sessions"

TITLE_MODEL = "claude-haiku-4-5"
"""标题生成模型的最终兜底值。

调用方（CLI 层）应按"环境变量 MODEL > 配置 summarize > 配置 main"
解析后显式传入；仅当未传时使用此默认。
"""


def _utcnow() -> datetime:
    return datetime.now(UTC)


class SessionRecorder:
    """会话录制器。

    以 JSONL 追加每轮快照，元数据（轮数、成本、状态）随每次
    record 重写。cost_usd 必须由调用方（AgentLoop）算好写入
    TurnSnapshot 后传入，本类只做累加，避免二次估算造成定价漂移。
    """

    def __init__(
        self,
        id: str,
        jsonl_path: Path,
        meta_path: Path,
        meta: SessionMeta,
    ) -> None:
        self.id = id
        self._jsonl_path = jsonl_path
        self._meta_path = meta_path
        self.meta = meta
        self._lock = asyncio.Lock()

    @classmethod
    def open_sync(
        cls, id: str, cwd: Path, base_dir: Path | None = None
    ) -> SessionRecorder:
        """同步打开（或恢复）一个会话，供 CLI 构建期调用。"""
        base = base_dir or default_sessions_base()
        dir_ = base / _utcnow().strftime("%Y-%m-%d")
        dir_.mkdir(parents=True, exist_ok=True)

        jsonl_path = dir_ / f"{id}.jsonl"
        meta_path = dir_ / f"{id}.meta.json"

        if meta_path.exists():
            meta = SessionMeta.model_validate_json(
                meta_path.read_text(encoding="utf-8")
            )
            meta.status = "running"
        else:
            meta = SessionMeta(
                id=id,
                started_at=_utcnow(),
                cwd=str(cwd),
                turns=0,
                cost_usd=0.0,
                status="running",
                title=None,
            )
        meta_path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
        return cls(id, jsonl_path, meta_path, meta)

    @classmethod
    async def open(
        cls, id: str, cwd: Path, base_dir: Path | None = None
    ) -> SessionRecorder:
        """打开（或恢复）一个会话。

        按"当天日期/session id"定位文件；meta 已存在时沿用原
        元数据（保留累计轮数与成本），否则新建。
        """
        return await asyncio.to_thread(cls.open_sync, id, cwd, base_dir)

    async def record(self, turn: TurnSnapshot) -> None:
        """追加一轮快照并更新元数据。"""
        line = turn.model_dump_json() + "\n"
        async with self._lock:
            await asyncio.to_thread(self._append_line, line)
            self.meta.turns += 1
            self.meta.cost_usd += turn.cost_usd
            self.meta.status = "running"
            await asyncio.to_thread(
                self._meta_path.write_text,
                self.meta.model_dump_json(indent=2),
                "utf-8",
            )

    def _append_line(self, line: str) -> None:
        with self._jsonl_path.open("a", encoding="utf-8") as f:
            f.write(line)

    async def save_meta(self) -> None:
        """把当前元数据落盘（供 title 等外部更新后调用）。"""
        async with self._lock:
            await asyncio.to_thread(
                self._meta_path.write_text,
                self.meta.model_dump_json(indent=2),
                "utf-8",
            )

    async def finalize(self, status: str = "ended") -> None:
        """写入终态（ended / errored）。"""
        async with self._lock:
            self.meta.status = status
            await asyncio.to_thread(
                self._meta_path.write_text,
                self.meta.model_dump_json(indent=2),
                "utf-8",
            )


class SessionNotFoundError(Exception):
    """指定 id 的会话不存在。"""


def _find_meta_path(id: str, base_dir: Path | None) -> Path | None:
    base = base_dir or default_sessions_base()
    if not base.is_dir():
        return None
    for day in base.iterdir():
        if not day.is_dir():
            continue
        candidate = day / f"{id}.meta.json"
        if candidate.exists():
            return candidate
    return None


def load_session(
    id: str, base_dir: Path | None = None
) -> tuple[SessionMeta, list[Message]]:
    """加载会话：返回元数据与从 JSONL 重建的消息列表。

    每行快照都携带完整 request_messages，逐行重放得到末态对话；
    assistant 块与工具结果按原顺序追加。
    """
    meta_path = _find_meta_path(id, base_dir)
    if meta_path is None:
        raise SessionNotFoundError(f"session not found: {id}")

    meta = SessionMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
    jsonl_path = meta_path.with_name(f"{id}.jsonl")

    messages: list[Message] = []
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            turn = TurnSnapshot.model_validate_json(line)
            messages = list(turn.request_messages)
            messages.append(
                Message(role=Role.assistant, content=list(turn.assistant_blocks))
            )
            if turn.tool_outputs:
                messages.append(
                    Message(
                        role=Role.tool,
                        content=[
                            ToolResultBlock(
                                tool_use_id=tid, content=out, is_error=is_err
                            )
                            for tid, out, is_err in turn.tool_outputs
                        ],
                    )
                )
    return meta, messages


def list_sessions(base_dir: Path | None = None) -> list[SessionMeta]:
    """列出所有会话元数据，按开始时间降序。"""
    base = base_dir or default_sessions_base()
    metas: list[SessionMeta] = []
    if not base.is_dir():
        return metas
    for day in base.iterdir():
        if not day.is_dir():
            continue
        for f in day.glob("*.meta.json"):
            try:
                metas.append(
                    SessionMeta.model_validate_json(f.read_text(encoding="utf-8"))
                )
            except Exception:
                continue
    metas.sort(key=lambda m: m.started_at, reverse=True)
    return metas


def purge_sessions(older_than_days: int, base_dir: Path | None = None) -> int:
    """删除 started_at 早于指定天数的会话文件，返回删除的会话数。"""
    base = base_dir or default_sessions_base()
    if not base.is_dir() or older_than_days <= 0:
        return 0

    cutoff = _utcnow() - timedelta(days=older_than_days)
    removed = 0
    for meta in list_sessions(base):
        if meta.started_at >= cutoff:
            continue
        meta_path = _find_meta_path(meta.id, base)
        if meta_path is None:
            continue
        for suffix in (".meta.json", ".jsonl", ".messages.json"):
            p = meta_path.with_name(f"{meta.id}{suffix}")
            p.unlink(missing_ok=True)
        removed += 1
    return removed


async def maybe_generate_title(
    meta: SessionMeta,
    llm,
    first_turn: TurnSnapshot,
    model: str = TITLE_MODEL,
) -> str | None:
    """首轮结束后生成简短会话标题。

    已有标题或首条用户消息为空时跳过；生成失败不抛出。
    """
    if meta.title or meta.turns < 1:
        return meta.title

    first_user = ""
    for m in first_turn.request_messages:
        if m.role == Role.user:
            for b in m.content:
                if isinstance(b, TextBlock) and b.text.strip():
                    first_user = b.text.strip()
                    break
        if first_user:
            break
    if not first_user:
        return None

    try:
        resp = await llm.complete(
            CompleteRequest(
                model=model,
                max_tokens=32,
                messages=[
                    Message.user(
                        f"用 10 个中文字以内概括这次对话主题，只输出标题：\n{first_user}"
                    )
                ],
                temperature=0.0,
            )
        )
    except Exception:
        return None

    text = ""
    for b in resp.content:
        if isinstance(b, TextBlock) and b.text.strip():
            text = b.text.strip()
            break
    if text:
        meta.title = text
    return meta.title


def build_turn_snapshot(
    iteration: int,
    request_messages: list[Message],
    assistant_blocks: list[ContentBlock],
    tool_outputs: list[tuple[str, str, bool]],
    usage: Usage,
    model: str,
    cost_usd: float,
) -> TurnSnapshot:
    """构造一轮快照的便捷工厂（内部做列表快照，避免共享可变引用）。"""
    return TurnSnapshot(
        ts=_utcnow(),
        iteration=iteration,
        request_messages=list(request_messages),
        assistant_blocks=list(assistant_blocks),
        tool_outputs=list(tool_outputs),
        usage=usage,
        model=model,
        cost_usd=cost_usd,
    )


__all__ = [
    "default_sessions_base",
    "SessionNotFoundError",
    "SessionRecorder",
    "TITLE_MODEL",
    "build_turn_snapshot",
    "load_session",
    "list_sessions",
    "maybe_generate_title",
    "purge_sessions",
]
