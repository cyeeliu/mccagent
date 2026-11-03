"""Task-bound git worktree 隔离。

为 teammate 创建独立工作目录，避免并行编辑冲突。
工作目录是 git repo 时用 ``git worktree add``（基于当前 HEAD
创建独立分支）；非 git 时降级为普通子目录。

清理策略区分 clean 与 dirty 工作区，dirty 工作区不得被静默删除。
"""

from __future__ import annotations

import enum
import shutil
import subprocess
from pathlib import Path


class WorktreeError(Exception):
    """worktree 操作错误。"""


class CleanupPolicy(enum.Enum):
    """worktree 清理策略。

    - SAFE: dirty 工作区拒绝清理（默认）
    - ARCHIVE: 先归档未提交变更再清理
    - FORCE: 无条件强制清理
    """

    SAFE = "safe"
    ARCHIVE = "archive"
    FORCE = "force"


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


def is_dirty(wt_dir: Path) -> bool:
    """检查 worktree 是否有未提交变更。

    检测已修改、已暂存和未跟踪文件。``.task_id`` 标记文件不视为 dirty。
    非 git 目录检查是否存在非 ``.task_id`` 内容。
    """
    if not wt_dir.exists():
        return False
    if not _is_git_repo(wt_dir):
        return any(p.name != ".task_id" for p in wt_dir.iterdir())
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=wt_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return True
    lines = [
        line for line in result.stdout.strip().splitlines()
        if not line.endswith(".task_id")
    ]
    return bool(lines)


def archive_worktree(
    name: str, base_dir: Path, archive_dir: Path | None = None,
) -> Path:
    """将 worktree 的未提交变更归档为 patch 文件，返回 patch 路径。

    git worktree 保存 ``git diff HEAD`` 和未跟踪文件列表。
    非 git 目录打包整个目录为 tar 归档。
    归档失败时 raise WorktreeError，调用方应保留工作区。
    """
    wt_dir = base_dir / ".worktrees" / name
    if not wt_dir.exists():
        raise WorktreeError(f"worktree not found: {name}")

    if archive_dir is None:
        archive_dir = base_dir / ".worktree_archives"
    archive_dir.mkdir(parents=True, exist_ok=True)

    task_id = "unknown"
    task_id_file = wt_dir / ".task_id"
    if task_id_file.exists():
        task_id = task_id_file.read_text(encoding="utf-8").strip()

    if _is_git_repo(wt_dir):
        patch_path = archive_dir / f"{name}.patch"
        try:
            diff_result = subprocess.run(
                ["git", "diff", "HEAD"],
                cwd=wt_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            untracked_result = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                cwd=wt_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode().strip() if e.stderr else str(e)
            raise WorktreeError(f"git diff failed: {err}") from e

        header = f"# Worktree archive: {name}\n# Task ID: {task_id}\n# Untracked files:\n"
        for line in untracked_result.stdout.strip().splitlines():
            header += f"#   {line}\n"
        patch_path.write_text(header + diff_result.stdout, encoding="utf-8")
        return patch_path
    else:
        archive_path = archive_dir / f"{name}.tar"
        shutil.make_archive(str(archive_path.with_suffix("")), "tar", wt_dir)
        return archive_path


def create_worktree(name: str, task_id: str, base_dir: Path) -> Path:
    """创建并绑定一个 task worktree，返回工作目录路径。

    git repo 时用 ``git worktree add -b task/{name}`` 创建独立分支；
    非 git 时降级为普通子目录。在 worktree 根写入 ``.task_id`` 标记。

    创建失败时 raise WorktreeError，调用方必须处理，不得回退共享目录。
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


def remove_worktree(
    name: str,
    base_dir: Path,
    policy: CleanupPolicy = CleanupPolicy.SAFE,
) -> Path | None:
    """移除一个 task worktree。

    清理策略：
    - SAFE: dirty 工作区拒绝清理（raise WorktreeError），clean 正常清理
    - ARCHIVE: dirty 时先归档变更再清理，归档失败保留工作区
    - FORCE: 无条件强制清理

    返回归档 patch 路径（ARCHIVE 模式且 dirty 时），或 None。
    """
    wt_dir = base_dir / ".worktrees" / name
    if not wt_dir.exists():
        raise WorktreeError(f"worktree not found: {name}")

    dirty = is_dirty(wt_dir)
    archive_path: Path | None = None

    if dirty:
        if policy == CleanupPolicy.SAFE:
            raise WorktreeError(
                f"worktree {name} has uncommitted changes; "
                f"use policy=ARCHIVE or FORCE to override"
            )
        if policy == CleanupPolicy.ARCHIVE:
            archive_path = archive_worktree(name, base_dir)

    task_id_file = wt_dir / ".task_id"
    if task_id_file.exists():
        task_id_file.unlink()

    if _is_git_repo(base_dir):
        cmd = ["git", "worktree", "remove", str(wt_dir)]
        if dirty or policy == CleanupPolicy.FORCE:
            cmd.append("--force")
        try:
            subprocess.run(
                cmd,
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

    return archive_path


def worktree_path(name: str, base_dir: Path) -> Path:
    """返回 worktree 路径（不检查是否存在）。"""
    return base_dir / ".worktrees" / name


__all__ = [
    "CleanupPolicy",
    "WorktreeError",
    "archive_worktree",
    "create_worktree",
    "is_dirty",
    "remove_worktree",
    "worktree_path",
]
