"""Tests for orchestrator.timeout_manager."""

from __future__ import annotations

import asyncio

import pytest

from orchestrator.timeout_manager import TimeoutManager


@pytest.fixture
def tm() -> TimeoutManager:
    return TimeoutManager()


class TestExecuteWithTimeout:
    @pytest.mark.asyncio
    async def test_success_within_timeout(self, tm: TimeoutManager) -> None:
        async def fast():
            return 42

        result = await tm.execute_with_timeout(fast(), timeout=1.0, agent_name="a")
        assert result == 42

    @pytest.mark.asyncio
    async def test_timeout_exceeded_raises(self, tm: TimeoutManager) -> None:
        async def slow():
            await asyncio.sleep(5)
            return "never"

        with pytest.raises(asyncio.TimeoutError):
            await tm.execute_with_timeout(slow(), timeout=0.05, agent_name="a")

    @pytest.mark.asyncio
    async def test_zero_timeout_raises_valueerror(self, tm: TimeoutManager) -> None:
        async def noop():
            return None

        with pytest.raises(ValueError):
            await tm.execute_with_timeout(noop(), timeout=0, agent_name="a")

    @pytest.mark.asyncio
    async def test_timeout_stats_tracked(self, tm: TimeoutManager) -> None:
        async def slow():
            await asyncio.sleep(5)

        with pytest.raises(asyncio.TimeoutError):
            await tm.execute_with_timeout(slow(), timeout=0.05, agent_name="slow_agent")

        stats = tm.get_timeout_stats()
        assert stats["slow_agent"] == 1

    @pytest.mark.asyncio
    async def test_multiple_timeouts_counted(self, tm: TimeoutManager) -> None:
        async def slow():
            await asyncio.sleep(5)

        for _ in range(3):
            with pytest.raises(asyncio.TimeoutError):
                await tm.execute_with_timeout(
                    slow(), timeout=0.02, agent_name="agent_x"
                )
        assert tm.get_timeout_stats()["agent_x"] == 3

    @pytest.mark.asyncio
    async def test_reset_stats(self, tm: TimeoutManager) -> None:
        async def slow():
            await asyncio.sleep(5)

        with pytest.raises(asyncio.TimeoutError):
            await tm.execute_with_timeout(slow(), timeout=0.02, agent_name="a")
        tm.reset_stats()
        assert tm.get_timeout_stats() == {}

    @pytest.mark.asyncio
    async def test_nested_timeouts(self, tm: TimeoutManager) -> None:
        """Inner timeout fires before outer."""

        async def inner():
            await asyncio.sleep(5)

        async def outer():
            return await tm.execute_with_timeout(
                inner(), timeout=0.03, agent_name="inner"
            )

        with pytest.raises(asyncio.TimeoutError):
            await tm.execute_with_timeout(outer(), timeout=1.0, agent_name="outer")

        # Inner timeout fired
        assert tm.get_timeout_stats().get("inner", 0) == 1
