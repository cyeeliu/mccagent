"""终端按键读取：cbreak 模式 + ANSI escape sequence 解析。

在 TTY 环境下设置 cbreak 以逐字符读取，非 TTY 环境降级为行读取。
"""

from __future__ import annotations

import asyncio
import os
import select
import sys
import termios
import tty
from enum import Enum
from typing import TextIO


class Key(Enum):
    enter = "enter"
    backspace = "backspace"
    escape = "escape"
    ctrl_c = "ctrl_c"
    ctrl_d = "ctrl_d"
    up = "up"
    down = "down"
    left = "left"
    right = "right"
    home = "home"
    end = "end"
    page_up = "page_up"
    page_down = "page_down"
    tab = "tab"


class KeyReader:
    """管理终端 cbreak 状态，提供异步按键读取。"""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdin
        self._fd: int | None = None
        self._old_settings: list[int] | None = None
        self._is_tty = hasattr(self._stream, "fileno") and self._stream.isatty()

    def __enter__(self) -> KeyReader:
        if self._is_tty:
            self._fd = self._stream.fileno()
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            attrs = termios.tcgetattr(self._fd)
            attrs[3] &= ~termios.ISIG
            termios.tcsetattr(self._fd, termios.TCSANOW, attrs)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    async def read_key(self) -> tuple[Key | str, None]:
        if not self._is_tty:
            line = await asyncio.to_thread(self._stream.readline)
            return (line.rstrip("\n"), None) if line else (Key.ctrl_d, None)

        ch = await asyncio.to_thread(self._read_char)
        if ch == "":
            return Key.ctrl_d, None
        if ch == "\x03":
            return Key.ctrl_c, None
        if ch == "\x04":
            return Key.ctrl_d, None
        if ch in ("\r", "\n"):
            return Key.enter, None
        if ch == "\t":
            return Key.tab, None
        if ch == "\x7f":
            return Key.backspace, None
        if ch == "\x1b":
            seq = await asyncio.to_thread(self._read_escape)
            return self._decode_escape(seq), None
        return ch, None

    def _read_char(self) -> str:
        # 直接从 fd 读取，绕过 TextIO 缓冲层，确保 select 和 read
        # 在同一层面操作，避免缓冲层吞掉 escape sequence 后续字节。
        ch = os.read(self._fd, 1)
        if not ch:
            return ""
        first = ch[0]
        if first < 0x80:
            return ch.decode("latin-1")
        # UTF-8 多字节字符：首字节前导 1 数量决定总长度
        if first >= 0xF0:
            need = 4
        elif first >= 0xE0:
            need = 3
        elif first >= 0xC0:
            need = 2
        else:
            # 10xxxxxx 续字节不应作为首字节，防御性返回
            return ch.decode("latin-1")
        # 终端多字节字符各字节同时到达，循环确保读满
        buf = ch
        while len(buf) < need:
            extra = os.read(self._fd, need - len(buf))
            if not extra:
                break
            buf += extra
        return buf.decode("utf-8", errors="replace")

    def _read_escape(self) -> str:
        # 纯 ESC 键无后续字符；方向键等 escape sequence 在几毫秒内
        # 连续到达。短超时内无可读字符则判定为单独的 ESC，避免
        # read(1) 阻塞导致 ESC 按键要凑满 4 次循环才返回。
        # 用 fd 而非 stream 做 select，与 os.read 同层操作。
        rlist, _, _ = select.select([self._fd], [], [], 0.05)
        if not rlist:
            return ""
        seq = ""
        for _ in range(4):
            ch = os.read(self._fd, 1)
            if not ch:
                break
            c = ch.decode("latin-1")
            seq += c
            if c.isalpha() or c == "~":
                break
        return seq

    @staticmethod
    def _decode_escape(seq: str) -> Key | str:
        return {
            "[A": Key.up,
            "[B": Key.down,
            "[C": Key.right,
            "[D": Key.left,
            "[H": Key.home,
            "[F": Key.end,
            "[5~": Key.page_up,
            "[6~": Key.page_down,
        }.get(seq, Key.escape)
