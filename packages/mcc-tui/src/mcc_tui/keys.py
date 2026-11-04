"""终端按键读取：cbreak 模式 + ANSI escape sequence 解析。

在 TTY 环境下设置 cbreak 以逐字符读取，非 TTY 环境降级为行读取。
通过唤醒管道解决阻塞读取与 asyncio 任务取消之间的生命周期问题：
close() 向管道写入数据使 select 立即返回，线程不再阻塞在 os.read。
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import select
import sys
import termios
import tty
from enum import Enum
from typing import TextIO

_ESCAPE_TIMEOUT = 0.05
_ESCAPE_MAX_LEN = 4


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
    """管理终端 cbreak 状态，提供异步按键读取。

    内部维护唤醒管道（os.pipe），使阻塞在 select 的读取线程
    能被 close() 立即唤醒，避免退出时线程残留。
    """

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdin
        self._fd: int | None = None
        self._old_settings: list[int] | None = None
        self._is_tty = hasattr(self._stream, "fileno") and self._stream.isatty()
        self._wake_r: int | None = None
        self._wake_w: int | None = None
        self._closed = False

    def __enter__(self) -> KeyReader:
        if self._is_tty:
            self._fd = self._stream.fileno()
            self._old_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            attrs = termios.tcgetattr(self._fd)
            attrs[3] &= ~termios.ISIG
            termios.tcsetattr(self._fd, termios.TCSANOW, attrs)
            self._wake_r, self._wake_w = os.pipe()
            os.set_blocking(self._wake_r, False)
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
        if self._fd is not None and self._old_settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
        for fd in (self._wake_r, self._wake_w):
            if fd is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)
        self._wake_r = None
        self._wake_w = None

    def close(self) -> None:
        """唤醒可能阻塞的读取线程。

        向唤醒管道写入数据使 select 立即返回，
        从而使 asyncio.to_thread 中的线程不再阻塞。
        不关闭管道 FD，避免与 _read_char 中的 select 产生竞态。
        FD 由 __exit__ 统一关闭。
        """
        if self._closed:
            return
        self._closed = True
        if self._wake_w is not None:
            with contextlib.suppress(OSError):
                os.write(self._wake_w, b"\x00")

    @property
    def _wake_fds(self) -> list[int]:
        """select 需要监听的 FD 列表（stdin + 唤醒管道）。"""
        fds: list[int] = []
        if self._fd is not None:
            fds.append(self._fd)
        if self._wake_r is not None:
            fds.append(self._wake_r)
        return fds

    def _is_wakeup(self, rlist: list[int]) -> bool:
        """检查 select 返回的可读 FD 是否包含唤醒管道。"""
        return self._wake_r is not None and self._wake_r in rlist

    async def read_key(self) -> tuple[Key | str, None]:
        if self._closed:
            return Key.ctrl_d, None
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
        # 先 select 监听 stdin + 唤醒管道，被 close() 唤醒时返回空字符串，
        # 避免 os.read 无限阻塞导致退出时线程残留。
        rlist, _, _ = select.select(self._wake_fds, [], [])
        if self._is_wakeup(rlist):
            return ""
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
        # 终端多字节字符各字节同时到达，循环确保读满；
        # 后续字节也用 select 带超时，避免异常终端分段发送时阻塞
        buf = ch
        while len(buf) < need:
            rlist, _, _ = select.select([self._fd], [], [], _ESCAPE_TIMEOUT)
            if not rlist:
                break
            extra = os.read(self._fd, need - len(buf))
            if not extra:
                break
            buf += extra
        return buf.decode("utf-8", errors="replace")

    def _read_escape(self) -> str:
        # 纯 ESC 键无后续字符；方向键等 escape sequence 在几毫秒内
        # 连续到达。短超时内无可读字符则判定为单独的 ESC，避免
        # read(1) 阻塞导致 ESC 按键要凑满 4 次循环才返回。
        # 每次读取前都 select 带超时，避免分段到达时无限阻塞。
        rlist, _, _ = select.select([self._fd], [], [], _ESCAPE_TIMEOUT)
        if not rlist:
            return ""
        seq = ""
        for _ in range(_ESCAPE_MAX_LEN):
            rlist, _, _ = select.select([self._fd], [], [], _ESCAPE_TIMEOUT)
            if not rlist:
                break
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
