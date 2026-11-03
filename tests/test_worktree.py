"""Worktree 保护测试：P1-02 脏检测 + P1-03 Fail Closed。

用真实 git repo 验证 clean/dirty 检测、清理策略和归档功能。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from mcc_harness.worktree import (
    CleanupPolicy,
    WorktreeError,
    archive_worktree,
    create_worktree,
    is_dirty,
    remove_worktree,
    worktree_path,
)


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    return repo


@pytest.fixture
def non_git_base(tmp_path: Path) -> Path:
    base = tmp_path / "workspace"
    base.mkdir()
    return base


# -- is_dirty --


class TestIsDirty:
    def test_clean_git_worktree(self, git_repo: Path):
        wt = create_worktree("wt1", "task_001", git_repo)
        assert not is_dirty(wt)

    def test_modified_tracked_file(self, git_repo: Path):
        wt = create_worktree("wt1", "task_001", git_repo)
        (wt / "README.md").write_text("changed\n", encoding="utf-8")
        assert is_dirty(wt)

    def test_staged_file(self, git_repo: Path):
        wt = create_worktree("wt1", "task_001", git_repo)
        (wt / "README.md").write_text("changed\n", encoding="utf-8")
        _git(wt, "add", "README.md")
        assert is_dirty(wt)

    def test_untracked_file(self, git_repo: Path):
        wt = create_worktree("wt1", "task_001", git_repo)
        (wt / "new_file.txt").write_text("new\n", encoding="utf-8")
        assert is_dirty(wt)

    def test_task_id_marker_not_dirty(self, git_repo: Path):
        wt = create_worktree("wt1", "task_001", git_repo)
        assert not is_dirty(wt)

    def test_non_git_empty_dir(self, non_git_base: Path):
        wt = create_worktree("wt1", "task_001", non_git_base)
        assert not is_dirty(wt)

    def test_non_git_with_content(self, non_git_base: Path):
        wt = create_worktree("wt1", "task_001", non_git_base)
        (wt / "file.txt").write_text("content\n", encoding="utf-8")
        assert is_dirty(wt)

    def test_nonexistent_dir(self, tmp_path: Path):
        assert not is_dirty(tmp_path / "nope")


# -- create_worktree --


class TestCreateWorktree:
    def test_create_git(self, git_repo: Path):
        wt = create_worktree("wt1", "task_001", git_repo)
        assert wt.exists()
        assert (wt / ".task_id").read_text() == "task_001"
        assert wt == worktree_path("wt1", git_repo)

    def test_create_non_git(self, non_git_base: Path):
        wt = create_worktree("wt1", "task_001", non_git_base)
        assert wt.exists()
        assert (wt / ".task_id").read_text() == "task_001"

    def test_duplicate_create(self, non_git_base: Path):
        create_worktree("wt1", "task_001", non_git_base)
        with pytest.raises(WorktreeError, match="already exists"):
            create_worktree("wt1", "task_002", non_git_base)

    def test_git_worktree_add_failure(self, git_repo: Path):
        _git(git_repo, "branch", "task/wt1")
        with pytest.raises(WorktreeError, match="git worktree add failed"):
            create_worktree("wt1", "task_001", git_repo)


# -- remove_worktree: SAFE policy --


class TestRemoveSafePolicy:
    def test_clean_git_removes_without_force(self, git_repo: Path):
        wt = create_worktree("wt1", "task_001", git_repo)
        result = remove_worktree("wt1", git_repo, policy=CleanupPolicy.SAFE)
        assert result is None
        assert not wt.exists()

    def test_clean_non_git_removes(self, non_git_base: Path):
        wt = create_worktree("wt1", "task_001", non_git_base)
        result = remove_worktree("wt1", non_git_base, policy=CleanupPolicy.SAFE)
        assert result is None
        assert not wt.exists()

    def test_dirty_modified_rejected(self, git_repo: Path):
        wt = create_worktree("wt1", "task_001", git_repo)
        (wt / "README.md").write_text("changed\n", encoding="utf-8")
        with pytest.raises(WorktreeError, match="uncommitted changes"):
            remove_worktree("wt1", git_repo, policy=CleanupPolicy.SAFE)
        assert wt.exists()

    def test_dirty_staged_rejected(self, git_repo: Path):
        wt = create_worktree("wt1", "task_001", git_repo)
        (wt / "README.md").write_text("changed\n", encoding="utf-8")
        _git(wt, "add", "README.md")
        with pytest.raises(WorktreeError, match="uncommitted changes"):
            remove_worktree("wt1", git_repo, policy=CleanupPolicy.SAFE)
        assert wt.exists()

    def test_dirty_untracked_rejected(self, git_repo: Path):
        wt = create_worktree("wt1", "task_001", git_repo)
        (wt / "new.txt").write_text("new\n", encoding="utf-8")
        with pytest.raises(WorktreeError, match="uncommitted changes"):
            remove_worktree("wt1", git_repo, policy=CleanupPolicy.SAFE)
        assert wt.exists()

    def test_dirty_non_git_rejected(self, non_git_base: Path):
        wt = create_worktree("wt1", "task_001", non_git_base)
        (wt / "file.txt").write_text("content\n", encoding="utf-8")
        with pytest.raises(WorktreeError, match="uncommitted changes"):
            remove_worktree("wt1", non_git_base, policy=CleanupPolicy.SAFE)
        assert wt.exists()

    def test_remove_nonexistent(self, non_git_base: Path):
        with pytest.raises(WorktreeError, match="not found"):
            remove_worktree("nope", non_git_base)


# -- remove_worktree: ARCHIVE policy --


class TestRemoveArchivePolicy:
    def test_dirty_git_archives_then_removes(self, git_repo: Path):
        wt = create_worktree("wt1", "task_001", git_repo)
        (wt / "README.md").write_text("changed\n", encoding="utf-8")
        (wt / "new.txt").write_text("new file\n", encoding="utf-8")

        archive_path = remove_worktree("wt1", git_repo, policy=CleanupPolicy.ARCHIVE)
        assert archive_path is not None
        assert archive_path.exists()
        assert archive_path.suffix == ".patch"
        content = archive_path.read_text(encoding="utf-8")
        assert "Worktree archive: wt1" in content
        assert "Task ID: task_001" in content
        assert "new.txt" in content
        assert not wt.exists()

    def test_clean_git_archive_no_patch(self, git_repo: Path):
        wt = create_worktree("wt1", "task_001", git_repo)
        result = remove_worktree("wt1", git_repo, policy=CleanupPolicy.ARCHIVE)
        assert result is None
        assert not wt.exists()

    def test_dirty_non_git_archives_tar(self, non_git_base: Path):
        wt = create_worktree("wt1", "task_001", non_git_base)
        (wt / "file.txt").write_text("content\n", encoding="utf-8")

        archive_path = remove_worktree(
            "wt1", non_git_base, policy=CleanupPolicy.ARCHIVE,
        )
        assert archive_path is not None
        assert archive_path.exists()
        assert archive_path.suffix == ".tar"
        assert not wt.exists()

    def test_archive_custom_dir(self, git_repo: Path, tmp_path: Path):
        wt = create_worktree("wt1", "task_001", git_repo)
        (wt / "README.md").write_text("changed\n", encoding="utf-8")

        custom_archive = tmp_path / "custom_archives"
        archive_path = archive_worktree("wt1", git_repo, archive_dir=custom_archive)
        assert archive_path.exists()
        assert custom_archive.exists()
        remove_worktree("wt1", git_repo, policy=CleanupPolicy.FORCE)


# -- remove_worktree: FORCE policy --


class TestRemoveForcePolicy:
    def test_dirty_git_force_removes(self, git_repo: Path):
        wt = create_worktree("wt1", "task_001", git_repo)
        (wt / "README.md").write_text("changed\n", encoding="utf-8")
        result = remove_worktree("wt1", git_repo, policy=CleanupPolicy.FORCE)
        assert result is None
        assert not wt.exists()

    def test_dirty_non_git_force_removes(self, non_git_base: Path):
        wt = create_worktree("wt1", "task_001", non_git_base)
        (wt / "file.txt").write_text("content\n", encoding="utf-8")
        result = remove_worktree("wt1", non_git_base, policy=CleanupPolicy.FORCE)
        assert result is None
        assert not wt.exists()

    def test_clean_force_removes(self, git_repo: Path):
        wt = create_worktree("wt1", "task_001", git_repo)
        remove_worktree("wt1", git_repo, policy=CleanupPolicy.FORCE)
        assert not wt.exists()


# -- archive_worktree --


class TestArchiveWorktree:
    def test_archive_nonexistent(self, git_repo: Path):
        with pytest.raises(WorktreeError, match="not found"):
            archive_worktree("nope", git_repo)

    def test_archive_preserves_worktree(self, git_repo: Path):
        wt = create_worktree("wt1", "task_001", git_repo)
        (wt / "README.md").write_text("changed\n", encoding="utf-8")

        archive_path = archive_worktree("wt1", git_repo)
        assert archive_path.exists()
        assert wt.exists()

        remove_worktree("wt1", git_repo, policy=CleanupPolicy.FORCE)


# -- 默认策略为 SAFE --


class TestDefaultPolicy:
    def test_default_is_safe(self, git_repo: Path):
        wt = create_worktree("wt1", "task_001", git_repo)
        (wt / "README.md").write_text("changed\n", encoding="utf-8")
        with pytest.raises(WorktreeError, match="uncommitted changes"):
            remove_worktree("wt1", git_repo)
        assert wt.exists()
        remove_worktree("wt1", git_repo, policy=CleanupPolicy.FORCE)
