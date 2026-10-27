"""权限检查器测试。"""

from __future__ import annotations

import pytest
from mcc_config import PermissionConfig
from mcc_harness import (
    Allow,
    Ask,
    BashAction,
    Deny,
    PathAction,
    PermissionChecker,
    PermissionRequest,
    bash_word_match,
)


def _cfg(allow: list[str], deny: list[str], mode: str | None = None) -> PermissionConfig:
    return PermissionConfig(mode=mode, allow=allow, deny=deny)


@pytest.fixture
def checker_factory():
    """创建 PermissionChecker 的工厂 fixture。"""
    def _make(allow: list[str], deny: list[str], mode: str | None = None):
        return PermissionChecker(_cfg(allow, deny, mode))
    return _make


def _bash_req(cmd: str) -> PermissionRequest:
    return PermissionRequest(category="Bash", action=BashAction(cmd=cmd))


def _path_req(category: str, path: str) -> PermissionRequest:
    return PermissionRequest(category=category, action=PathAction(path=path))


# ---------- deny / allow 优先级 ----------


def test_deny_overrides_allow(checker_factory):
    checker = checker_factory(["Bash(git:*)"], ["Bash(rm:*)"])
    assert isinstance(checker.check(_bash_req("git status")), Allow)
    assert isinstance(checker.check(_bash_req("rm -rf /")), Deny)


def test_bypass_permissions_mode(checker_factory):
    checker = checker_factory([], [], "bypassPermissions")
    assert isinstance(checker.check(_bash_req("anything")), Allow)


def test_read_always_allowed_by_default(checker_factory):
    checker = checker_factory([], [], None)
    assert isinstance(checker.check(_path_req("Read", "/some/file.py")), Allow)


def test_write_asks_in_default_mode(checker_factory):
    checker = checker_factory([], [], None)
    assert isinstance(checker.check(_path_req("Write", "/some/file.py")), Ask)


def test_write_allowed_in_accept_edits_mode(checker_factory):
    checker = checker_factory([], [], "acceptEdits")
    assert isinstance(checker.check(_path_req("Write", "/any/file.py")), Allow)


def test_edit_allowed_in_accept_edits_mode(checker_factory):
    checker = checker_factory([], [], "acceptEdits")
    assert isinstance(checker.check(_path_req("Edit", "/any/file.py")), Allow)


def test_path_allow_glob(checker_factory):
    checker = checker_factory(["Read(src/**/*.py)"], [], None)
    assert isinstance(checker.check(_path_req("Read", "src/main/mod.py")), Allow)


def test_unknown_bash_asks_in_default_mode(checker_factory):
    checker = checker_factory([], [], None)
    assert isinstance(checker.check(_bash_req("curl https://example.com")), Ask)


# ---------- bash_word_match 词边界 ----------


@pytest.mark.parametrize(
    "cmd,prefix,expected",
    [
        ("rm -rf /", "rm", True),
        ("rm", "rm", True),
        ("rmdir /tmp", "rm", False),
        ("rm -i file", "rm", True),
        ("git status", "git", True),
        ("git-upload-pack", "git", False),
        ("git", "git", True),
        ("rm/file", "rm", True),
        ("curl-evil", "curl", False),
    ],
    ids=[
        "rm-with-space",
        "rm-bare",
        "rmdir-not-rm",
        "rm-with-flag",
        "git-with-space",
        "git-hyphen-not-match",
        "git-bare",
        "rm-with-slash",
        "curl-hyphen-not-match",
    ],
)
def test_bash_word_boundary(cmd: str, prefix: str, expected: bool):
    assert bash_word_match(cmd, prefix) is expected


def test_bash_deny_requires_word_boundary(checker_factory):
    """Bash(rm:*) 不能匹配 rmdir。"""
    checker = checker_factory([], ["Bash(rm:*)"])
    assert isinstance(checker.check(_bash_req("rm -rf /")), Deny)
    result = checker.check(_bash_req("rmdir /tmp/foo"))
    assert not isinstance(result, Deny), "rmdir should not match rm prefix rule"


def test_bash_allow_requires_word_boundary(checker_factory):
    """Bash(git:*) 不能匹配 git-upload-pack。"""
    checker = checker_factory(["Bash(git:*)"], [])
    result = checker.check(_bash_req("git-upload-pack"))
    assert not isinstance(result, Allow), "git-upload-pack should not match git allow rule"


# ---------- Decision 类型 ----------


def test_deny_carries_reason(checker_factory):
    checker = checker_factory([], ["Bash(rm:*)"])
    result = checker.check(_bash_req("rm -rf ./tmp"))
    assert isinstance(result, Deny)
    assert "denied" in result.reason


def test_ask_carries_message(checker_factory):
    checker = checker_factory([], [], None)
    result = checker.check(_bash_req("curl http://example.com"))
    assert isinstance(result, Ask)
    assert "confirmation" in result.message
