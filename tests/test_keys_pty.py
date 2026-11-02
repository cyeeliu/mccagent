"""KeyReader PTY 集成测试：验证真实终端字节流下 escape sequence 解析。

用 os.openpty 创建 PTY 对，向 master 端写入 ANSI escape sequence，
从 slave 端经 KeyReader 读取，验证方向键等被正确解析而非误判为 ESC。
"""

from __future__ import annotations

import asyncio
import os
import pty
from contextlib import ExitStack

import pytest
from mcc_tui.keys import Key, KeyReader


@pytest.fixture
def pty_pair():
    with ExitStack() as stack:
        master_fd, slave_fd = pty.openpty()
        stack.callback(os.close, master_fd)
        stack.callback(os.close, slave_fd)
        slave_path = os.ttyname(slave_fd)
        slave_stream = stack.enter_context(open(slave_path))
        yield master_fd, slave_stream


async def _read_one(pty_pair, data: bytes) -> Key | str:
    master_fd, slave_stream = pty_pair
    with KeyReader(stream=slave_stream) as reader:
        os.write(master_fd, data)
        await asyncio.sleep(0.05)
        key, _ = await reader.read_key()
        return key


@pytest.mark.parametrize(
    ("name", "seq", "expected"),
    [
        ("up", b"\x1b[A", Key.up),
        ("down", b"\x1b[B", Key.down),
        ("right", b"\x1b[C", Key.right),
        ("left", b"\x1b[D", Key.left),
        ("home", b"\x1b[H", Key.home),
        ("end", b"\x1b[F", Key.end),
        ("page_up", b"\x1b[5~", Key.page_up),
        ("page_down", b"\x1b[6~", Key.page_down),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_escape_sequence_parsed(pty_pair, name, seq, expected):
    result = asyncio.run(_read_one(pty_pair, seq))
    assert result == expected


def test_pure_escape_returns_escape(pty_pair):
    result = asyncio.run(_read_one(pty_pair, b"\x1b"))
    assert result == Key.escape


@pytest.mark.parametrize(
    ("seq", "expected"),
    [
        (b"a", "a"),
        (b"\r", Key.enter),
        (b"\n", Key.enter),
        (b"\t", Key.tab),
        (b"\x7f", Key.backspace),
        (b"\x03", Key.ctrl_c),
        (b"\x04", Key.ctrl_d),
    ],
    ids=["char", "cr", "lf", "tab", "backspace", "ctrl_c", "ctrl_d"],
)
def test_single_byte_keys(pty_pair, seq, expected):
    result = asyncio.run(_read_one(pty_pair, seq))
    assert result == expected
