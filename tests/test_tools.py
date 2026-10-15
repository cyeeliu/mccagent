"""内建工具的核心逻辑测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
from mcc_core import ToolContext
from mcc_tools import (
    EditFileTool,
    GlobTool,
    GrepTool,
    ListDirTool,
    ReadFileTool,
    RunBashTool,
    ToolRegistry,
    WriteFileTool,
    default_registry,
)


def _ctx(dir: Path) -> ToolContext:
    return ToolContext(cwd=dir, session_id="test", depth=0)


# ---------- ToolRegistry ----------


def test_default_registry_has_all_tools():
    reg = default_registry()
    assert sorted(reg.names()) == [
        "edit_file",
        "glob",
        "grep",
        "list_dir",
        "read_file",
        "run_bash",
        "write_file",
    ]


def test_as_api_schema_sorted():
    reg = default_registry()
    schema = reg.as_api_schema()
    assert isinstance(schema, list)
    names = [s["name"] for s in schema]
    assert names == sorted(names)
    assert "input_schema" in schema[0]
    assert "description" in schema[0]


def test_subset():
    reg = default_registry()
    sub = reg.subset(["read_file", "grep"])
    assert sorted(sub.names()) == ["grep", "read_file"]
    assert sub.get("write_file") is None


def test_subset_unknown_ignored():
    reg = default_registry()
    sub = reg.subset(["read_file", "nonexistent"])
    assert sub.names() == ["read_file"]


# ---------- write_file + read_file ----------


async def test_write_then_read(tmp_path: Path):
    ctx = _ctx(tmp_path)
    tool = WriteFileTool()

    out = await tool.execute({"path": "hello.txt", "content": "line1\nline2\n"}, ctx)
    assert not out.is_error, out.content

    rtool = ReadFileTool()
    out = await rtool.execute({"path": "hello.txt"}, ctx)
    assert not out.is_error, out.content
    assert "line1" in out.content
    assert "     1\t" in out.content


async def test_write_refuses_overwrite(tmp_path: Path):
    ctx = _ctx(tmp_path)
    (tmp_path / "exists.txt").write_text("content")
    tool = WriteFileTool()

    out = await tool.execute({"path": "exists.txt", "content": "new"}, ctx)
    assert out.is_error
    assert "refusing" in out.content


async def test_write_creates_parent_dirs(tmp_path: Path):
    ctx = _ctx(tmp_path)
    tool = WriteFileTool()

    out = await tool.execute({"path": "sub/dir/f.txt", "content": "hi"}, ctx)
    assert not out.is_error, out.content
    assert (tmp_path / "sub" / "dir" / "f.txt").read_text() == "hi"


async def test_read_file_offset_limit(tmp_path: Path):
    ctx = _ctx(tmp_path)
    (tmp_path / "f.txt").write_text("\n".join(f"line{i}" for i in range(1, 21)))
    tool = ReadFileTool()

    out = await tool.execute({"path": "f.txt", "offset": 5, "limit": 3}, ctx)
    assert not out.is_error
    assert "line6" in out.content
    assert "line8" in out.content
    assert "line9" not in out.content
    assert "more lines" in out.content


async def test_read_file_not_found(tmp_path: Path):
    ctx = _ctx(tmp_path)
    tool = ReadFileTool()

    out = await tool.execute({"path": "nope.txt"}, ctx)
    assert out.is_error


# ---------- edit_file ----------


async def test_edit_unique(tmp_path: Path):
    ctx = _ctx(tmp_path)
    (tmp_path / "f.py").write_text("def main() {}")
    tool = EditFileTool()

    out = await tool.execute(
        {"path": "f.py", "old_string": "def main()", "new_string": "def run()"}, ctx
    )
    assert not out.is_error, out.content
    assert "def run()" in (tmp_path / "f.py").read_text()


async def test_edit_not_unique_fails(tmp_path: Path):
    ctx = _ctx(tmp_path)
    (tmp_path / "dup.txt").write_text("a\na\n")
    tool = EditFileTool()

    out = await tool.execute(
        {"path": "dup.txt", "old_string": "a", "new_string": "b"}, ctx
    )
    assert out.is_error
    assert "2 times" in out.content


async def test_edit_replace_all(tmp_path: Path):
    ctx = _ctx(tmp_path)
    (tmp_path / "dup.txt").write_text("a\na\n")
    tool = EditFileTool()

    out = await tool.execute(
        {"path": "dup.txt", "old_string": "a", "new_string": "b", "replace_all": True}, ctx
    )
    assert not out.is_error, out.content
    assert (tmp_path / "dup.txt").read_text() == "b\nb\n"


async def test_edit_not_found(tmp_path: Path):
    ctx = _ctx(tmp_path)
    (tmp_path / "f.txt").write_text("hello")
    tool = EditFileTool()

    out = await tool.execute(
        {"path": "f.txt", "old_string": "xyz", "new_string": "abc"}, ctx
    )
    assert out.is_error
    assert "not found" in out.content


async def test_edit_identical_strings(tmp_path: Path):
    ctx = _ctx(tmp_path)
    (tmp_path / "f.txt").write_text("hello")
    tool = EditFileTool()

    out = await tool.execute(
        {"path": "f.txt", "old_string": "hello", "new_string": "hello"}, ctx
    )
    assert out.is_error
    assert "identical" in out.content


# ---------- run_bash ----------


async def test_run_bash_echo(tmp_path: Path):
    ctx = _ctx(tmp_path)
    tool = RunBashTool()

    out = await tool.execute({"command": "echo hello-enterprise"}, ctx)
    assert not out.is_error, out.content
    assert "hello-enterprise" in out.content


async def test_run_bash_timeout(tmp_path: Path):
    ctx = _ctx(tmp_path)
    tool = RunBashTool()

    out = await tool.execute({"command": "sleep 60", "timeout_secs": 1}, ctx)
    assert out.is_error
    assert "timed out" in out.content


async def test_run_bash_nonzero_exit(tmp_path: Path):
    ctx = _ctx(tmp_path)
    tool = RunBashTool()

    out = await tool.execute({"command": "exit 42"}, ctx)
    assert out.is_error
    assert "42" in out.content


async def test_run_bash_stderr(tmp_path: Path):
    ctx = _ctx(tmp_path)
    tool = RunBashTool()

    out = await tool.execute({"command": "echo err-msg >&2"}, ctx)
    assert not out.is_error
    assert "err-msg" in out.content
    assert "stderr" in out.content


async def test_run_bash_no_output(tmp_path: Path):
    ctx = _ctx(tmp_path)
    tool = RunBashTool()

    out = await tool.execute({"command": "true"}, ctx)
    assert not out.is_error
    assert "no output" in out.content


# ---------- grep ----------


async def test_grep_basic(tmp_path: Path):
    (tmp_path / "src.py").write_text("def main() {}\ndef helper() {}")
    ctx = _ctx(tmp_path)
    tool = GrepTool()

    out = await tool.execute({"pattern": r"def \w+\(\)"}, ctx)
    assert not out.is_error, out.content
    assert "src.py:1:" in out.content
    assert "def main()" in out.content


async def test_grep_no_match(tmp_path: Path):
    (tmp_path / "f.txt").write_text("hello world")
    ctx = _ctx(tmp_path)
    tool = GrepTool()

    out = await tool.execute({"pattern": "xyz_not_found"}, ctx)
    assert not out.is_error
    assert "No matches" in out.content


async def test_grep_case_insensitive(tmp_path: Path):
    (tmp_path / "f.txt").write_text("Hello World")
    ctx = _ctx(tmp_path)
    tool = GrepTool()

    out = await tool.execute({"pattern": "hello", "case_insensitive": True}, ctx)
    assert not out.is_error
    assert "Hello" in out.content


async def test_grep_invalid_regex(tmp_path: Path):
    ctx = _ctx(tmp_path)
    tool = GrepTool()

    out = await tool.execute({"pattern": "[invalid"}, ctx)
    assert out.is_error
    assert "invalid regex" in out.content


async def test_grep_glob_filter(tmp_path: Path):
    (tmp_path / "a.py").write_text("target")
    (tmp_path / "b.txt").write_text("target")
    ctx = _ctx(tmp_path)
    tool = GrepTool()

    out = await tool.execute({"pattern": "target", "glob": "*.py"}, ctx)
    assert not out.is_error
    assert "a.py" in out.content
    assert "b.txt" not in out.content


# ---------- glob ----------


async def test_glob_basic(tmp_path: Path):
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.py").write_text("y")
    (tmp_path / "c.txt").write_text("z")
    ctx = _ctx(tmp_path)
    tool = GlobTool()

    out = await tool.execute({"pattern": "*.py"}, ctx)
    assert not out.is_error, out.content
    assert "a.py" in out.content
    assert "b.py" in out.content
    assert "c.txt" not in out.content


async def test_glob_recursive(tmp_path: Path):
    sub = tmp_path / "src" / "deep"
    sub.mkdir(parents=True)
    (sub / "f.py").write_text("x")
    ctx = _ctx(tmp_path)
    tool = GlobTool()

    out = await tool.execute({"pattern": "**/*.py"}, ctx)
    assert not out.is_error
    assert "f.py" in out.content


async def test_glob_no_match(tmp_path: Path):
    ctx = _ctx(tmp_path)
    tool = GlobTool()

    out = await tool.execute({"pattern": "*.xyz"}, ctx)
    assert not out.is_error
    assert "No files match" in out.content


# ---------- list_dir ----------


async def test_list_dir_basic(tmp_path: Path):
    (tmp_path / "file.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "inner.py").write_text("y")
    ctx = _ctx(tmp_path)
    tool = ListDirTool()

    out = await tool.execute({"path": "."}, ctx)
    assert not out.is_error, out.content
    assert "file.txt" in out.content
    assert "sub/" in out.content


async def test_list_dir_max_depth(tmp_path: Path):
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "f.txt").write_text("x")
    ctx = _ctx(tmp_path)
    tool = ListDirTool()

    out = await tool.execute({"path": ".", "max_depth": 2}, ctx)
    assert not out.is_error
    assert "a/" in out.content
    assert "b/" in out.content
    assert "f.txt" not in out.content


async def test_list_dir_respects_gitignore(tmp_path: Path):
    (tmp_path / ".gitignore").write_text("ignored.txt\n")
    (tmp_path / "visible.txt").write_text("x")
    (tmp_path / "ignored.txt").write_text("y")
    ctx = _ctx(tmp_path)
    tool = ListDirTool()

    out = await tool.execute({"path": "."}, ctx)
    assert not out.is_error
    assert "visible.txt" in out.content
    assert "ignored.txt" not in out.content


# ---------- schema validation ----------


def test_all_tools_have_valid_schema():
    reg = default_registry()
    for schema in reg.as_api_schema():
        assert schema["name"]
        assert schema["description"]
        assert schema["input_schema"]["type"] == "object"
        assert "properties" in schema["input_schema"]


def test_tool_registry_get_returns_tool():
    reg = default_registry()
    tool = reg.get("read_file")
    assert tool is not None
    assert tool.name == "read_file"
    assert isinstance(tool.description, str)
    assert isinstance(tool.input_schema(), dict)


def test_registry_register_overwrites():
    reg = ToolRegistry()
    reg.register(ReadFileTool())
    reg.register(ReadFileTool())
    assert reg.names() == ["read_file"]
