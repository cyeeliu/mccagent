"""输入补全：候选项模型与前缀过滤。

TUI 输入框以 ``/`` 开头时弹出候选浮层；候选由调用方（CLI 装配
层）注入——TUI 不反向依赖 slash 注册表，只消费纯数据。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class CompletionItem:
    """一个补全候选：insert 为补全后的文本，label/hint 供浮层展示。"""

    insert: str
    label: str
    hint: str = ""


CompletionProvider = Callable[[str], list[CompletionItem]]


def filter_completions(
    items: list[CompletionItem], text: str
) -> list[CompletionItem]:
    """前缀过滤：输入以 / 开头且无空格时返回匹配候选。

    - 非 / 开头或已含空格（命令参数输入中）→ 关闭浮层
    - 前缀大小写不敏感匹配 insert
    """
    if not text.startswith("/") or " " in text:
        return []
    prefix = text.lower()
    return [item for item in items if item.insert.lower().startswith(prefix)]


def make_provider(
    commands: list[tuple[str, str]], skill_names: list[str] | None = None
) -> CompletionProvider:
    """把 slash 命令表与 skill 名合成为补全 provider。"""

    items = [
        CompletionItem(insert=f"/{name}", label=f"/{name}", hint=desc)
        for name, desc in commands
    ] + [
        CompletionItem(insert=f"/{name}", label=f"/{name}", hint="skill")
        for name in (skill_names or [])
    ]

    def _provider(prefix: str) -> list[CompletionItem]:
        return filter_completions(items, prefix)

    return _provider
