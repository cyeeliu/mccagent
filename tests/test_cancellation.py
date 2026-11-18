"""Phase 2: CancellationToken 测试。

覆盖：初始状态、取消信号、reason 传递、asyncio.Event 桥接、并发 wait。
"""

from __future__ import annotations

import asyncio

import pytest
from mcc_harness.execution.cancellation import CancellationToken


class TestCancellationTokenInitialState:
    def test_not_cancelled_on_creation(self):
        token = CancellationToken()
        assert token.is_cancelled is False

    def test_reason_none_on_creation(self):
        token = CancellationToken()
        assert token.reason is None

    def test_asyncio_event_not_set_on_creation(self):
        token = CancellationToken()
        assert token.to_asyncio_event().is_set() is False


class TestCancellationTokenCancel:
    def test_cancel_sets_flag(self):
        token = CancellationToken()
        token.cancel()
        assert token.is_cancelled is True

    def test_cancel_without_reason(self):
        token = CancellationToken()
        token.cancel()
        assert token.reason is None

    def test_cancel_with_reason(self):
        token = CancellationToken()
        token.cancel(reason="user_request")
        assert token.reason == "user_request"

    def test_cancel_sets_asyncio_event(self):
        token = CancellationToken()
        token.cancel()
        assert token.to_asyncio_event().is_set() is True

    def test_cancel_idempotent(self):
        token = CancellationToken()
        token.cancel(reason="first")
        token.cancel(reason="second")
        assert token.is_cancelled is True
        assert token.reason == "second"


class TestCancellationTokenWait:
    @pytest.mark.asyncio
    async def test_wait_returns_immediately_if_already_cancelled(self):
        token = CancellationToken()
        token.cancel()
        await asyncio.wait_for(token.wait(), timeout=0.1)

    @pytest.mark.asyncio
    async def test_wait_blocks_until_cancel(self):
        token = CancellationToken()

        async def cancel_after_delay():
            await asyncio.sleep(0.05)
            token.cancel()

        await asyncio.gather(
            asyncio.wait_for(token.wait(), timeout=1.0),
            cancel_after_delay(),
        )

    @pytest.mark.asyncio
    async def test_wait_timeout_raises(self):
        token = CancellationToken()
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(token.wait(), timeout=0.05)


class TestCancellationTokenAsyncioBridge:
    @pytest.mark.asyncio
    async def test_same_event_reference(self):
        token = CancellationToken()
        event1 = token.to_asyncio_event()
        event2 = token.to_asyncio_event()
        assert event1 is event2

    @pytest.mark.asyncio
    async def test_event_set_propagates_to_token(self):
        token = CancellationToken()
        token.to_asyncio_event().set()
        assert token.is_cancelled is True

    @pytest.mark.asyncio
    async def test_token_cancel_propagates_to_event(self):
        token = CancellationToken()
        event = token.to_asyncio_event()
        token.cancel()
        assert event.is_set() is True
