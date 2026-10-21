"""内建工具的核心逻辑测试。"""

from __future__ import annotations

import re
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
from mcc_tools._common import cap_str, resolve

EXPECTED_TOOL_NAMES = [
    "compact",
    "edit_file",
    "glob",
    "grep",
    "list_dir",
    "read_file",
    "run_bash",
    "write_file",
]


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    """以 tmp_path 为工作目录的 ToolContext。"""
    return ToolContext(cwd=tmp_path, session_id="test", depth=0)


# ---------- ToolRegistry ----------


def test_default_registry_has_all_tools():
    reg = default_registry()
    assert sorted(reg.names()) == EXPECTED_TOOL_NAMES


@pytest.mark.parametrize("tool_name", EXPECTED_TOOL_NAMES)
def test_registry_has_tool(tool_name: str):
    reg = default_registry()
    tool = reg.get(tool_name)
    assert tool is not None
    assert tool.name == tool_name


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


def test_registry_get_nonexistent_returns_none():
    reg = default_registry()
    assert reg.get("nonexistent") is None


def test_registry_register_overwrites():
    reg = ToolRegistry()
    reg.register(ReadFileTool())
    reg.register(ReadFileTool())
    assert reg.names() == ["read_file"]


# ---------- _common helpers ----------


def test_resolve_absolute_path(tmp_path: Path):
    abs_path = Path("/tmp/some_file")
    assert resolve(tmp_path, str(abs_path)) == abs_path


def test_resolve_relative_path(tmp_path: Path):
    result = resolve(tmp_path, "sub/file.txt")
    assert result == tmp_path / "sub" / "file.txt"


@pytest.mark.parametrize(
    "text,max_bytes,should_truncate",
    [
        ("hello", 100, False),
        ("hello", 3, True),
        ("你好世界", 100, False),
        ("你好世界", 6, True),
    ],
    ids=["ascii-short", "ascii-trunc", "unicode-short", "unicode-trunc"],
)
def test_cap_str_truncation(text: str, max_bytes: int, should_truncate: bool):
    result = cap_str(text, max_bytes)
    if should_truncate:
        assert "truncated" in result
    else:
        assert result == text


# ---------- write_file + read_file ----------


@pytest.mark.asyncio
async def test_write_then_read(ctx: ToolContext, tmp_path: Path):
    tool = WriteFileTool()
    out = await tool.execute({"path": "hello.txt", "content": "line1\nline2\n"}, ctx)
    assert not out.is_error, out.content

    rtool = ReadFileTool()
    out = await rtool.execute({"path": "hello.txt"}, ctx)
    assert not out.is_error, out.content
    assert "line1" in out.content
    assert "     1\t" in out.content


@pytest.mark.asyncio
async def test_write_refuses_overwrite(ctx: ToolContext, tmp_path: Path):
    (tmp_path / "exists.txt").write_text("content")
    tool = WriteFileTool()

    out = await tool.execute({"path": "exists.txt", "content": "new"}, ctx)
    assert out.is_error
    assert "refusing" in out.content


@pytest.mark.asyncio
async def test_write_creates_parent_dirs(ctx: ToolContext, tmp_path: Path):
    tool = WriteFileTool()

    out = await tool.execute({"path": "sub/dir/f.txt", "content": "hi"}, ctx)
    assert not out.is_error, out.content
    assert (tmp_path / "sub" / "dir" / "f.txt").read_text() == "hi"


@pytest.mark.asyncio
async def test_write_missing_path_returns_error(ctx: ToolContext):
    tool = WriteFileTool()
    out = await tool.execute({"content": "hi"}, ctx)
    assert out.is_error
    assert "path" in out.content


@pytest.mark.asyncio
async def test_write_missing_content_returns_error(ctx: ToolContext):
    tool = WriteFileTool()
    out = await tool.execute({"path": "f.txt"}, ctx)
    assert out.is_error
    assert "content" in out.content


@pytest.mark.asyncio
async def test_read_file_offset_limit(ctx: ToolContext, tmp_path: Path):
    (tmp_path / "f.txt").write_text("\n".join(f"line{i}" for i in range(1, 21)))
    tool = ReadFileTool()

    out = await tool.execute({"path": "f.txt", "offset": 5, "limit": 3}, ctx)
    assert not out.is_error
    assert "line6" in out.content
    assert "line8" in out.content
    assert "line9" not in out.content
    assert "more lines" in out.content


@pytest.mark.asyncio
async def test_read_file_not_found(ctx: ToolContext):
    tool = ReadFileTool()
    out = await tool.execute({"path": "nope.txt"}, ctx)
    assert out.is_error


@pytest.mark.asyncio
async def test_read_file_missing_path_returns_error(ctx: ToolContext):
    tool = ReadFileTool()
    out = await tool.execute({}, ctx)
    assert out.is_error
    assert "path" in out.content


# ---------- edit_file ----------


@pytest.mark.asyncio
async def test_edit_unique(ctx: ToolContext, tmp_path: Path):
    (tmp_path / "f.py").write_text("def main() {}")
    tool = EditFileTool()

    out = await tool.execute(
        {"path": "f.py", "old_string": "def main()", "new_string": "def run()"}, ctx
    )
    assert not out.is_error, out.content
    assert "def run()" in (tmp_path / "f.py").read_text()


@pytest.mark.asyncio
async def test_edit_not_unique_fails(ctx: ToolContext, tmp_path: Path):
    (tmp_path / "dup.txt").write_text("a\na\n")
    tool = EditFileTool()

    out = await tool.execute(
        {"path": "dup.txt", "old_string": "a", "new_string": "b"}, ctx
    )
    assert out.is_error
    assert "2 times" in out.content


@pytest.mark.asyncio
async def test_edit_replace_all(ctx: ToolContext, tmp_path: Path):
    (tmp_path / "dup.txt").write_text("a\na\n")
    tool = EditFileTool()

    out = await tool.execute(
        {"path": "dup.txt", "old_string": "a", "new_string": "b", "replace_all": True}, ctx
    )
    assert not out.is_error, out.content
    assert (tmp_path / "dup.txt").read_text() == "b\nb\n"


@pytest.mark.asyncio
async def test_edit_not_found(ctx: ToolContext, tmp_path: Path):
    (tmp_path / "f.txt").write_text("hello")
    tool = EditFileTool()

    out = await tool.execute(
        {"path": "f.txt", "old_string": "xyz", "new_string": "abc"}, ctx
    )
    assert out.is_error
    assert "not found" in out.content


@pytest.mark.asyncio
async def test_edit_identical_strings(ctx: ToolContext, tmp_path: Path):
    (tmp_path / "f.txt").write_text("hello")
    tool = EditFileTool()

    out = await tool.execute(
        {"path": "f.txt", "old_string": "hello", "new_string": "hello"}, ctx
    )
    assert out.is_error
    assert "identical" in out.content


@pytest.mark.asyncio
async def test_edit_file_not_found(ctx: ToolContext):
    tool = EditFileTool()
    out = await tool.execute(
        {"path": "nope.txt", "old_string": "a", "new_string": "b"}, ctx
    )
    assert out.is_error


# ---------- run_bash ----------


@pytest.mark.asyncio
async def test_run_bash_echo(ctx: ToolContext):
    tool = RunBashTool()
    out = await tool.execute({"command": "echo hello-enterprise"}, ctx)
    assert not out.is_error, out.content
    assert "hello-enterprise" in out.content


@pytest.mark.slow
@pytest.mark.asyncio
async def test_run_bash_timeout(ctx: ToolContext):
    tool = RunBashTool()
    out = await tool.execute({"command": "sleep 60", "timeout_secs": 1}, ctx)
    assert out.is_error
    assert "timed out" in out.content


@pytest.mark.asyncio
async def test_run_bash_nonzero_exit(ctx: ToolContext):
    tool = RunBashTool()
    out = await tool.execute({"command": "exit 42"}, ctx)
    assert out.is_error
    assert "42" in out.content


@pytest.mark.asyncio
async def test_run_bash_stderr(ctx: ToolContext):
    tool = RunBashTool()
    out = await tool.execute({"command": "echo err-msg >&2"}, ctx)
    assert not out.is_error
    assert "err-msg" in out.content
    assert "stderr" in out.content


@pytest.mark.asyncio
async def test_run_bash_no_output(ctx: ToolContext):
    tool = RunBashTool()
    out = await tool.execute({"command": "true"}, ctx)
    assert not out.is_error
    assert "no output" in out.content


@pytest.mark.asyncio
async def test_run_bash_missing_command_returns_error(ctx: ToolContext):
    tool = RunBashTool()
    out = await tool.execute({}, ctx)
    assert out.is_error
    assert "command" in out.content


# ---------- grep ----------


@pytest.mark.asyncio
async def test_grep_basic(ctx: ToolContext, tmp_path: Path):
    (tmp_path / "src.py").write_text("def main() {}\ndef helper() {}")
    tool = GrepTool()

    out = await tool.execute({"pattern": r"def \w+\(\)"}, ctx)
    assert not out.is_error, out.content
    assert "src.py:1:" in out.content
    assert "def main()" in out.content


@pytest.mark.asyncio
async def test_grep_no_match(ctx: ToolContext, tmp_path: Path):
    (tmp_path / "f.txt").write_text("hello world")
    tool = GrepTool()

    out = await tool.execute({"pattern": "xyz_not_found"}, ctx)
    assert not out.is_error
    assert "No matches" in out.content


@pytest.mark.asyncio
async def test_grep_case_insensitive(ctx: ToolContext, tmp_path: Path):
    (tmp_path / "f.txt").write_text("Hello World")
    tool = GrepTool()

    out = await tool.execute({"pattern": "hello", "case_insensitive": True}, ctx)
    assert not out.is_error
    assert "Hello" in out.content


@pytest.mark.asyncio
async def test_grep_invalid_regex_returns_error(ctx: ToolContext):
    tool = GrepTool()
    out = await tool.execute({"pattern": "[invalid"}, ctx)
    assert out.is_error
    assert "invalid regex" in out.content


def test_invalid_regex_raises_re_error():
    with pytest.raises(re.error):
        re.compile("[invalid")


@pytest.mark.asyncio
async def test_grep_glob_filter(ctx: ToolContext, tmp_path: Path):
    (tmp_path / "a.py").write_text("target")
    (tmp_path / "b.txt").write_text("target")
    tool = GrepTool()

    out = await tool.execute({"pattern": "target", "glob": "*.py"}, ctx)
    assert not out.is_error
    assert "a.py" in out.content
    assert "b.txt" not in out.content


@pytest.mark.asyncio
async def test_grep_missing_pattern_returns_error(ctx: ToolContext):
    tool = GrepTool()
    out = await tool.execute({}, ctx)
    assert out.is_error
    assert "pattern" in out.content


# ---------- glob ----------


@pytest.mark.asyncio
async def test_glob_basic(ctx: ToolContext, tmp_path: Path):
    (tmp_path / "a.py").write_text("x")
    (tmp_path / "b.py").write_text("y")
    (tmp_path / "c.txt").write_text("z")
    tool = GlobTool()

    out = await tool.execute({"pattern": "*.py"}, ctx)
    assert not out.is_error, out.content
    assert "a.py" in out.content
    assert "b.py" in out.content
    assert "c.txt" not in out.content


@pytest.mark.asyncio
async def test_glob_recursive(ctx: ToolContext, tmp_path: Path):
    sub = tmp_path / "src" / "deep"
    sub.mkdir(parents=True)
    (sub / "f.py").write_text("x")
    tool = GlobTool()

    out = await tool.execute({"pattern": "**/*.py"}, ctx)
    assert not out.is_error
    assert "f.py" in out.content


@pytest.mark.asyncio
async def test_glob_no_match(ctx: ToolContext):
    tool = GlobTool()
    out = await tool.execute({"pattern": "*.xyz"}, ctx)
    assert not out.is_error
    assert "No files match" in out.content


@pytest.mark.asyncio
async def test_glob_missing_pattern_returns_error(ctx: ToolContext):
    tool = GlobTool()
    out = await tool.execute({}, ctx)
    assert out.is_error
    assert "pattern" in out.content


# ---------- list_dir ----------


@pytest.mark.asyncio
async def test_list_dir_basic(ctx: ToolContext, tmp_path: Path):
    (tmp_path / "file.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "inner.py").write_text("y")
    tool = ListDirTool()

    out = await tool.execute({"path": "."}, ctx)
    assert not out.is_error, out.content
    assert "file.txt" in out.content
    assert "sub/" in out.content


@pytest.mark.asyncio
async def test_list_dir_max_depth(ctx: ToolContext, tmp_path: Path):
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "f.txt").write_text("x")
    tool = ListDirTool()

    out = await tool.execute({"path": ".", "max_depth": 2}, ctx)
    assert not out.is_error
    assert "a/" in out.content
    assert "b/" in out.content
    assert "f.txt" not in out.content


@pytest.mark.asyncio
async def test_list_dir_respects_gitignore(ctx: ToolContext, tmp_path: Path):
    (tmp_path / ".gitignore").write_text("ignored.txt\n")
    (tmp_path / "visible.txt").write_text("x")
    (tmp_path / "ignored.txt").write_text("y")
    tool = ListDirTool()

    out = await tool.execute({"path": "."}, ctx)
    assert not out.is_error
    assert "visible.txt" in out.content
    assert "ignored.txt" not in out.content


@pytest.mark.asyncio
async def test_list_dir_missing_path_returns_error(ctx: ToolContext):
    tool = ListDirTool()
    out = await tool.execute({}, ctx)
    assert out.is_error
    assert "path" in out.content


@pytest.mark.asyncio
async def test_list_dir_not_a_directory(ctx: ToolContext, tmp_path: Path):
    (tmp_path / "file.txt").write_text("x")
    tool = ListDirTool()
    out = await tool.execute({"path": "file.txt"}, ctx)
    assert out.is_error


# ---------- schema validation ----------


@pytest.mark.parametrize("tool_name", EXPECTED_TOOL_NAMES)
def test_tool_has_valid_schema(tool_name: str):
    reg = default_registry()
    tool = reg.get(tool_name)
    assert tool is not None
    schema = tool.input_schema()
    assert schema["type"] == "object"
    assert "properties" in schema


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
