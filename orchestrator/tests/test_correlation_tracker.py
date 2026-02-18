"""Tests for orchestrator.correlation_tracker."""

from __future__ import annotations

import uuid

import pytest

from orchestrator.correlation_tracker import CorrelationTracker


class TestGenerateCorrelationId:
    def test_generates_valid_uuid(self) -> None:
        tracker = CorrelationTracker()
        cid = tracker.generate_correlation_id()
        uuid.UUID(cid)  # validates format

    def test_unique_ids(self) -> None:
        tracker = CorrelationTracker()
        ids = {tracker.generate_correlation_id() for _ in range(100)}
        assert len(ids) == 100


class TestSetGetCorrelationId:
    def test_set_and_get(self) -> None:
        tracker = CorrelationTracker()
        cid = str(uuid.uuid4())
        tracker.set_correlation_id(cid)
        assert tracker.get_correlation_id() == cid

    def test_invalid_uuid_raises(self) -> None:
        tracker = CorrelationTracker()
        with pytest.raises(ValueError, match="Invalid"):
            tracker.set_correlation_id("not-a-uuid")


class TestPropagateToAgent:
    @pytest.mark.asyncio
    async def test_propagate_passes_correlation_id(self) -> None:
        tracker = CorrelationTracker()
        received_kwargs = {}

        def fake_agent(*args, **kwargs):
            received_kwargs.update(kwargs)
            return "result"

        cid = str(uuid.uuid4())
        result = await tracker.propagate_to_agent(fake_agent, cid, "input")
        assert result == "result"
        assert received_kwargs["correlation_id"] == cid

    @pytest.mark.asyncio
    async def test_propagate_async_callable(self) -> None:
        tracker = CorrelationTracker()

        async def async_agent(*args, **kwargs):
            return "async_result"

        cid = str(uuid.uuid4())
        result = await tracker.propagate_to_agent(async_agent, cid)
        assert result == "async_result"
