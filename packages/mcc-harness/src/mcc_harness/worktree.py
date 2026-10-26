"""Task-bound git worktree 隔离。

为 teammate 创建独立工作目录，避免并行编辑冲突。
工作目录是 git repo 时用 ``git worktree add``（基于当前 HEAD
创建独立分支）；非 git 时降级为普通子目录。
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class WorktreeError(Exception):
    """worktree 操作错误。"""


def _is_git_repo(cwd: Path) -> bool:
    try:
        subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=cwd,
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return True


def create_worktree(name: str, task_id: str, base_dir: Path) -> Path:
    """创建并绑定一个 task worktree，返回工作目录路径。

    git repo 时用 ``git worktree add -b task/{name}`` 创建独立分支；
    非 git 时降级为普通子目录。在 worktree 根写入 ``.task_id`` 标记。
    """
    wt_dir = base_dir / ".worktrees" / name
    if wt_dir.exists():
        raise WorktreeError(f"worktree already exists: {name}")

    if _is_git_repo(base_dir):
        branch = f"task/{name}"
        try:
            subprocess.run(
                ["git", "worktree", "add", "-b", branch, str(wt_dir)],
                cwd=base_dir,
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            raise WorktreeError(
                f"git worktree add failed: {e.stderr.decode().strip()}"
            ) from e
    else:
        wt_dir.mkdir(parents=True)

    (wt_dir / ".task_id").write_text(task_id, encoding="utf-8")
    return wt_dir


def remove_worktree(name: str, base_dir: Path) -> None:
    """移除一个 task worktree。"""
    wt_dir = base_dir / ".worktrees" / name
    if not wt_dir.exists():
        raise WorktreeError(f"worktree not found: {name}")

    if _is_git_repo(base_dir):
        try:
            subprocess.run(
                ["git", "worktree", "remove", str(wt_dir), "--force"],
                cwd=base_dir,
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            raise WorktreeError(
                f"git worktree remove failed: {e.stderr.decode().strip()}"
            ) from e
    else:
        shutil.rmtree(wt_dir)


def worktree_path(name: str, base_dir: Path) -> Path:
    """返回 worktree 路径（不检查是否存在）。"""
    return base_dir / ".worktrees" / name


__all__ = [
    "WorktreeError",
    "create_worktree",
    "remove_worktree",
    "worktree_path",
]
