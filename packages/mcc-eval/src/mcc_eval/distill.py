"""失败回流：把 mcc 生产会话固化为评测 case 草稿。

纪律：每一个生产 bad case，第一反应是固化成 eval case——真实失败
的分布就是业务分布，拍脑袋编的测试题不是。mcc 会话流水
（``~/.mcc/sessions/<day>/<id>.jsonl``）天然记录了任务描述与完整
轨迹，本模块把其中的失败会话提取为 case 草稿，落到 ``evals/cases/``
目录（单 case 单文件，可 review / diff / 版本化）。

产物是「草稿」：task 取自会话首条用户消息，expectations 与 setup
留空——判定器与环境快照需要人工补齐校准后才算入库。回流的意义
在于把失败沉淀成可重放的输入，不在于自动生成完整评测。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mcc_eval.datasets import Case


@dataclass
class DistillOutcome:
    """一次回流的结果：case 草稿、落盘位置、是否新建。"""

    case: Case
    path: Path
    created: bool
    source_status: str


def _first_user_text(jsonl_path: Path) -> str | None:
    """从会话流水提取首条用户消息的文本（任务描述）。

    每行快照都携带完整 request_messages，末行即末态；首条 user
    消息是本次会话的原始任务。无用户文本（如 resume 续跑）时
    返回 None。
    """
    lines = [ln for ln in jsonl_path.read_text(encoding="utf-8").splitlines() if ln]
    if not lines:
        return None
    snapshot = json.loads(lines[-1])
    for msg in snapshot.get("request_messages", []):
        if msg.get("role") != "user":
            continue
        texts = [
            block.get("text", "")
            for block in msg.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        joined = "\n".join(t for t in texts if t).strip()
        if joined:
            return joined
    return None


def distill_sessions(
    sessions_dir: Path,
    *,
    statuses: tuple[str, ...] = ("errored",),
    limit: int = 20,
    out_dir: Path | None = None,
) -> list[DistillOutcome]:
    """扫描会话目录，把匹配状态的会话固化为 case 草稿。

    - ``statuses``：会话 meta 的 status 过滤（默认只收失败会话）
    - ``limit``：最多处理多少个会话（按开始时间倒序）
    - ``out_dir``：草稿落盘目录，默认 ``evals/cases``（相对 cwd）

    同名草稿已存在时跳过（幂等），保证人工修订过的 case
    不会被重复回流覆盖。
    """
    out_dir = out_dir or Path("evals/cases")
    if not sessions_dir.is_dir():
        return []

    metas: list[tuple[str, str, Path, str]] = []  # (started_at, status, meta_path, id)
    for day in sessions_dir.iterdir():
        if not day.is_dir():
            continue
        for meta_path in day.glob("*.meta.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            sid = str(meta.get("id") or meta_path.name.removesuffix(".meta.json"))
            started = str(meta.get("started_at") or "")
            status = str(meta.get("status") or "running")
            if status in statuses:
                metas.append((started, status, meta_path, sid))
    metas.sort(reverse=True)

    outcomes: list[DistillOutcome] = []
    for _, status, meta_path, sid in metas[: max(0, limit)]:
        jsonl_path = meta_path.with_name(f"{sid}.jsonl")
        task = _first_user_text(jsonl_path) if jsonl_path.exists() else None
        if not task:
            continue

        case = Case(case_id=f"distill-{sid}", task=task, expectations=[])
        path = out_dir / f"{case.case_id}.json"
        created = not path.exists()
        if created:
            out_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps([case.to_dict()], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        outcomes.append(
            DistillOutcome(case=case, path=path, created=created, source_status=status)
        )
    return outcomes
