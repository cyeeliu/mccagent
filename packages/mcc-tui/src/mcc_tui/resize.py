"""终端尺寸变化监听。

POSIX 平台通过 SIGWINCH 信号即时响应；不支持 SIGWINCH 的平台
（如 Windows）退化为轮询 os.get_terminal_size()。
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal


class ResizeWatcher:
    """监听终端尺寸变化，通过异步队列通知调用方。"""

    def __init__(self, poll_interval: float = 0.5) -> None:
        self._queue: asyncio.Queue[bool] = asyncio.Queue()
        self._poll_interval = poll_interval
        self._poll_task: asyncio.Task | None = None
        self._signal_installed = False
        self._last_size: tuple[int, int] | None = None

    async def __aenter__(self) -> ResizeWatcher:
        self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.stop()

    def start(self) -> None:
        """开始监听：优先注册 SIGWINCH，不可用则启动轮询。"""
        loop = asyncio.get_event_loop()
        sig = getattr(signal, "SIGWINCH", None)
        if sig is not None:
            try:
                loop.add_signal_handler(sig, self._on_resize)
                self._signal_installed = True
            except (NotImplementedError, RuntimeError, ValueError):
                pass
        if not self._signal_installed:
            self._poll_task = asyncio.create_task(self._poll_loop())

    def stop(self) -> None:
        """停止监听：移除信号处理器或取消轮询任务。"""
        if self._signal_installed:
            sig = getattr(signal, "SIGWINCH", None)
            if sig is not None:
                loop = asyncio.get_event_loop()
                with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
                    loop.remove_signal_handler(sig)
            self._signal_installed = False
        if self._poll_task is not None and not self._poll_task.done():
            self._poll_task.cancel()
            self._poll_task = None

    def _on_resize(self) -> None:
        """SIGWINCH 回调：向队列放入信号。"""
        self._queue.put_nowait(True)

    async def _poll_loop(self) -> None:
        """轮询终端尺寸变化（SIGWINCH 不可用时的兜底）。"""
        try:
            while True:
                await asyncio.sleep(self._poll_interval)
                try:
                    size = os.get_terminal_size()
                except OSError:
                    continue
                current = (size.columns, size.lines)
                if self._last_size is None or current != self._last_size:
                    self._last_size = current
                    self._queue.put_nowait(True)
        except asyncio.CancelledError:
            pass

    async def wait(self) -> None:
        """阻塞直到检测到终端尺寸变化。"""
        await self._queue.get()

    @property
    def using_signal(self) -> bool:
        """是否通过 SIGWINCH 信号监听（而非轮询）。"""
        return self._signal_installed
