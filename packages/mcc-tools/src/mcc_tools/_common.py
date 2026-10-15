"""工具实现共享的辅助函数。"""

from __future__ import annotations

from pathlib import Path

import pathspec
from pathspec import GitIgnoreSpec


def resolve(cwd: Path, p: str) -> Path:
    """将相对路径解析为相对于 cwd 的绝对路径。"""
    pb = Path(p)
    if pb.is_absolute():
        return pb
    return cwd / pb


def cap_str(s: str, max_bytes: int) -> str:
    """截断字符串到指定字节数，在 UTF-8 字符边界处截断。"""
    encoded = s.encode("utf-8")
    if len(encoded) <= max_bytes:
        return s
    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return f"{truncated}\n… [output truncated at {max_bytes} bytes]"


def load_gitignore(root: Path) -> pathspec.PathSpec | None:
    """加载 root 目录下的 .gitignore，返回 PathSpec 或 None。"""
    gitignore = root / ".gitignore"
    if not gitignore.is_file():
        return None
    lines = gitignore.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return None
    return GitIgnoreSpec.from_lines(lines)


def is_ignored(path: Path, root: Path, spec: pathspec.PathSpec | None) -> bool:
    """检查 path 相对于 root 是否被 gitignore 规则匹配。"""
    if spec is None:
        return False
    try:
        rel = path.relative_to(root)
    except ValueError:
        return False
    return spec.match_file(str(rel))
