"""Slash 命令框架：本地命令注册表（不进 AgentLoop）与内置命令。

两种 slash 语义共存：
- 本地命令（本注册表）：/model /config /help /status /goal ——
  直接执行，结果以 Notice 事件回显；
- 内容替换（slash_resolver）：/skill-name → SKILL.md 注入 prompt
  后进 AgentLoop（现有机制，不受影响）。

拦截点在 AgentEventSource.run 开头；TUI 零改动。
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class SlashResult:
    """本地命令的执行结果。

    ``text`` 非空 → Notice 回显；``prompt`` 非空 → 替换用户输入
    继续走后续链路（如 /goal set 后把 goal 文本送进 AgentLoop）。
    """

    text: str | None = None
    prompt: str | None = None


SlashHandler = Callable[[Any, str], "SlashResult | Awaitable[SlashResult]"]


class SlashRegistry:
    """slash 命令注册表：名字 → (描述, handler)。"""

    def __init__(self) -> None:
        self._commands: dict[str, tuple[str, SlashHandler]] = {}

    def register(
        self, name: str, description: str, handler: SlashHandler
    ) -> None:
        self._commands[name] = (description, handler)

    def parse(self, prompt: str) -> tuple[str, str] | None:
        """解析 "/name args" 形式；非 slash 输入返回 None。"""
        stripped = prompt.strip()
        if not stripped.startswith("/"):
            return None
        head, _, rest = stripped[1:].partition(" ")
        if head not in self._commands:
            return None
        return head, rest.strip()

    def list_commands(self) -> list[tuple[str, str]]:
        return [
            (name, desc) for name, (desc, _h) in sorted(self._commands.items())
        ]

    async def dispatch(self, prompt: str, source: Any) -> SlashResult | None:
        """命中注册命令则执行并返回结果；未命中返回 None。"""
        parsed = self.parse(prompt)
        if parsed is None:
            return None
        name, args = parsed
        handler = self._commands[name][1]
        result = handler(source, args)
        if inspect.isawaitable(result):
            result = await result
        return result


# -- 内置命令 --

# /config set 白名单：key → (settings 嵌套路径, 值转换函数, 生效语义)
_CONFIG_KEYS: dict[str, tuple[list[str], Callable[[str], Any], str]] = {
    "model.main": (["model", "main"], str, "即刻生效（下轮调用）"),
    "model.subagent": (["model", "subagent"], str, "下个会话生效"),
    "model.summarize": (["model", "summarize"], str, "下个会话生效"),
    "llm.base_url": (["llm", "base_url"], str, "下个会话生效"),
    "llm.api_key": (["llm", "api_key"], str, "下个会话生效"),
    "budget.max_usd_per_session": (
        ["budget", "max_usd_per_session"], float, "即刻生效",
    ),
    "permissions.mode": (["permissions", "mode"], str, "下个会话生效"),
    "sandbox.mode": (["sandbox", "mode"], str, "下个会话生效"),
    "reliability.turn_timeout_secs": (
        ["reliability", "turn_timeout_secs"], float, "即刻生效",
    ),
}

_SECRET_KEYS = {"llm.api_key"}


def _mask_secret(value: Any) -> str:
    """凭据展示脱敏：只露前缀与末四位。"""
    text = str(value or "")
    if len(text) <= 8:
        return "***" if text else "(unset)"
    return f"{text[:6]}…{text[-4:]}"


def _help(source: Any, args: str) -> SlashResult:
    lines = ["内置命令:"]
    for name, desc in source.slash_registry.list_commands():
        lines.append(f"  /{name}  {desc}")
    catalog = getattr(source, "skills_catalog", None)
    if catalog:
        lines.append("Skills（输入 /skill-name 加载）:")
        for entry in catalog.splitlines():
            lines.append(f"  {entry}")
    return SlashResult(text="\n".join(lines))


def _status(source: Any, args: str) -> SlashResult:
    agent = source.agent
    config = getattr(source, "config", None)
    model = getattr(agent, "model", "?")
    sid = getattr(source, "session_short_id", None) or "?"
    turns = len(getattr(agent, "history", []) or [])
    base_url = getattr(config.llm, "base_url", None) if config else None
    lines = [
        f"session: {sid}",
        f"model: {model}",
        f"history messages: {turns}",
    ]
    if base_url:
        lines.append(f"llm gateway: {base_url}")
    return SlashResult(text="\n".join(lines))


def _model(source: Any, args: str) -> SlashResult:
    from mcc_config import update_home_setting

    config = getattr(source, "config", None)
    parts = args.split()
    if not parts:
        current = getattr(source.agent, "model", "?")
        lines = [f"main: {current}"]
        if config is not None:
            if config.model.subagent:
                lines.append(f"subagent: {config.model.subagent}")
            if config.model.summarize:
                lines.append(f"summarize: {config.model.summarize}")
        lines.append("用法: /model <id> 或 /model <main|subagent|summarize> <id>")
        return SlashResult(text="\n".join(lines))

    if len(parts) == 1:
        kind, model_id = "main", parts[0]
    else:
        kind, model_id = parts[0], parts[1]
    if kind not in ("main", "subagent", "summarize"):
        return SlashResult(text=f"未知模型类型: {kind}（可选 main/subagent/summarize）")

    if kind == "main":
        source.agent.model = model_id
        note = "即刻生效"
    else:
        note = "下个会话生效"
    update_home_setting(["model", kind], model_id)
    if config is not None:
        setattr(config.model, kind, model_id)
    return SlashResult(text=f"model.{kind} = {model_id}（{note}，已写入配置）")


def _config(source: Any, args: str) -> SlashResult:
    from mcc_config import update_home_setting

    config = getattr(source, "config", None)
    if config is None:
        return SlashResult(text="配置不可用")
    parts = args.split(maxsplit=2)

    if not parts:
        return SlashResult(text=_config_summary(config))

    if parts[0] == "get":
        if len(parts) < 2:
            return SlashResult(text="用法: /config get <key>")
        return _config_get(config, parts[1])

    if parts[0] == "set":
        if len(parts) < 3:
            return SlashResult(text="用法: /config set <key> <value>")
        key, raw_value = parts[1], parts[2]
        entry = _CONFIG_KEYS.get(key)
        if entry is None:
            keys = "、".join(sorted(_CONFIG_KEYS))
            return SlashResult(text=f"不支持修改 {key}。可修改项：{keys}")
        keys, convert, note = entry
        try:
            value = convert(raw_value)
        except (ValueError, TypeError):
            return SlashResult(text=f"值 {raw_value!r} 不是合法的 {key} 类型")
        update_home_setting(keys, value)
        _apply_runtime(config, key, value)
        if key == "model.main":
            source.agent.model = str(value)
        shown = _mask_secret(value) if key in _SECRET_KEYS else str(value)
        return SlashResult(text=f"{key} = {shown}（{note}，已写入配置）")

    return SlashResult(text="用法: /config | /config get <key> | /config set <key> <value>")


def _apply_runtime(config: Any, key: str, value: Any) -> None:
    """把 /config set 的值同步到运行中 Config 对象（即刻生效项）。"""
    mapping: dict[str, tuple[Any, str]] = {
        "model.main": (config.model, "main"),
        "model.subagent": (config.model, "subagent"),
        "model.summarize": (config.model, "summarize"),
        "llm.base_url": (config.llm, "base_url"),
        "llm.api_key": (config.llm, "api_key"),
        "budget.max_usd_per_session": (config.budget, "max_usd_per_session"),
        "permissions.mode": (config.permissions, "mode"),
        "sandbox.mode": (config.sandbox, "mode"),
        "reliability.turn_timeout_secs": (config.reliability, "turn_timeout_secs"),
    }
    target = mapping.get(key)
    if target is not None:
        setattr(target[0], target[1], value)


def _config_get(config: Any, key: str) -> SlashResult:
    mapping = {
        "model.main": config.model.main,
        "model.subagent": config.model.subagent,
        "model.summarize": config.model.summarize,
        "llm.base_url": config.llm.base_url,
        "llm.api_key": config.llm.api_key,
        "budget.max_usd_per_session": config.budget.max_usd_per_session,
        "permissions.mode": config.permissions.mode,
        "sandbox.mode": config.sandbox.mode,
        "reliability.turn_timeout_secs": config.reliability.turn_timeout_secs,
    }
    if key not in mapping:
        return SlashResult(text=f"未知配置项: {key}")
    value = mapping[key]
    shown = _mask_secret(value) if key in _SECRET_KEYS else value
    return SlashResult(text=f"{key} = {shown if shown is not None else '(unset)'}")


def _config_summary(config: Any) -> str:
    lines = [
        "当前生效配置：",
        f"  model.main = {config.model.main}",
        f"  model.subagent = {config.model.subagent or '(回落 main)'}",
        f"  model.summarize = {config.model.summarize or '(回落 main)'}",
        f"  llm.base_url = {config.llm.base_url or '(默认)'}",
        f"  llm.api_key = {_mask_secret(config.llm.api_key)}",
        f"  budget.max_usd_per_session = {config.budget.max_usd_per_session}",
        f"  permissions.mode = {config.permissions.mode or 'default'}",
        f"  sandbox.mode = {config.sandbox.mode}",
        f"  reliability.turn_timeout_secs = {config.reliability.turn_timeout_secs}",
        "",
        "修改: /config set <key> <value>（/config 查看全表）",
    ]
    return "\n".join(lines)


def _goal_handler(source: Any, args: str) -> SlashResult:
    """/goal 迁移进注册表：语义不变（set 时 prompt 替换为 goal 文本）。"""
    controller = source.goal_controller
    if controller is None:
        return SlashResult(text="goal loop 未启用")
    stripped = args.strip()
    if not stripped:
        return SlashResult(
            text=f"goal: {controller.status()}",
        )
    if stripped.lower() in ("clear", "reset"):
        controller.clear()
        return SlashResult(text="goal cleared")
    controller.set_goal(stripped)
    return SlashResult(text=f"goal set: {stripped}", prompt=stripped)


def build_default_registry() -> SlashRegistry:
    """注册内置命令的默认注册表。"""
    registry = SlashRegistry()
    registry.register("help", "列出内置命令与 skills", _help)
    registry.register("status", "当前会话状态（session/model/历史）", _status)
    registry.register("model", "查看或切换模型", _model)
    registry.register("config", "查看或修改配置", _config)
    registry.register("goal", "设置/查看/清除 goal loop", _goal_handler)
    return registry
