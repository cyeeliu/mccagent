"""工具调用权限门控。

规则格式：Bash(cmd_prefix:*) / Read(**/*.py) / Write(src/**) / Edit(**)
deny 优先于 allow，两者均无匹配时走 mode 默认策略。
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass

from mcc_config import PermissionConfig

# ---------- Decision / Action ----------


@dataclass(frozen=True)
class Allow:
    """允许执行。"""


@dataclass(frozen=True)
class Deny:
    """拒绝执行。"""

    reason: str


@dataclass(frozen=True)
class Ask:
    """需要交互确认。"""

    message: str


Decision = Allow | Deny | Ask


@dataclass(frozen=True)
class BashAction:
    """Bash 命令动作。"""

    cmd: str


@dataclass(frozen=True)
class PathAction:
    """文件路径动作。"""

    path: str


Action = BashAction | PathAction


@dataclass(frozen=True)
class PermissionRequest:
    """权限检查请求。"""

    category: str
    action: Action


# ---------- PermissionChecker ----------


class PermissionChecker:
    """权限检查器：deny 优先，allow 次之，无匹配走 mode 默认。"""

    def __init__(self, cfg: PermissionConfig) -> None:
        self._allow_bash: list[str] = []
        self._deny_bash: list[str] = []
        self._allow_paths: list[str] = []
        self._deny_paths: list[str] = []
        self._mode = cfg.mode or "default"

        for rule in cfg.allow:
            _route(rule, self._allow_bash, self._allow_paths)
        for rule in cfg.deny:
            _route(rule, self._deny_bash, self._deny_paths)

    def check(self, req: PermissionRequest) -> Decision:
        match req.action:
            case BashAction(cmd=cmd):
                c = cmd.lstrip()
                if any(bash_word_match(c, p) for p in self._deny_bash):
                    return Deny("denied by deny rule")
                if any(bash_word_match(c, p) for p in self._allow_bash):
                    return Allow()
            case PathAction(path=path):
                if any(fnmatch.fnmatch(path, g) for g in self._deny_paths):
                    return Deny("denied by deny rule")
                if any(fnmatch.fnmatch(path, g) for g in self._allow_paths):
                    return Allow()

        if self._mode == "bypassPermissions":
            return Allow()

        match (req.action, req.category):
            case (PathAction(), "Read"):
                return Allow()
            case (PathAction(), "Write" | "Edit") if self._mode == "acceptEdits":
                return Allow()
            case _:
                return Ask("confirmation required")

    def allow_always(self, category: str, action: Action) -> None:
        """运行时追加 allow 规则（用户选择"总是允许"后调用）。

        Bash 以完整命令作为前缀（精确匹配该命令），
        Path 以完整路径作为 glob（精确匹配该路径）。
        """
        match action:
            case BashAction(cmd=cmd):
                prefix = cmd.strip()
                if prefix and prefix not in self._allow_bash:
                    self._allow_bash.append(prefix)
            case PathAction(path=path):
                if path and path not in self._allow_paths:
                    self._allow_paths.append(path)


def bash_word_match(cmd: str, prefix: str) -> bool:
    """在词边界处匹配 bash 命令前缀。

    Bash(rm:*) 生成前缀 "rm"：
    - "rm -rf /" → 匹配（rm 后跟空格）
    - "rmdir /tmp" → 不匹配（rm 是不同命令的前缀）
    """
    if not cmd.startswith(prefix):
        return False
    after = cmd[len(prefix):]
    return after == "" or after[0].isspace() or after[0] == "/"


def _route(rule: str, bash_prefix: list[str], path_glob: list[str]) -> None:
    """解析规则字符串，分发到 bash 前缀列表或路径 glob 列表。"""
    idx = rule.find("(")
    if idx < 0 or not rule.endswith(")"):
        return
    cat = rule[:idx]
    inner = rule[idx + 1 : -1]
    if cat == "Bash":
        p = inner.removesuffix(":*")
        bash_prefix.append(p)
    elif cat in ("Read", "Write", "Edit"):
        path_glob.append(inner)
