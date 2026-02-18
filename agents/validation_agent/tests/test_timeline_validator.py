"""Tests for timeline_validator.py — Algorithm 6."""

from __future__ import annotations

import pytest

from agents.root_cause_agent.schema import (
    EvidenceSourceAgent,
    Severity,
    TimelineEvent,
)
from agents.validation_agent.core.timeline_validator import (
    _check_chronological_order,
    _check_service_coverage,
    validate_timeline,
)
from agents.validation_agent.schema import PropagationStep


def _event(ts: str, service: str = "") -> TimelineEvent:
    """Helper to create a timeline event."""
    return TimelineEvent(
        timestamp=ts,
        source=EvidenceSourceAgent.LOG_AGENT,
        event="test event",
        service=service,
        severity=Severity.MEDIUM,
    )


class TestChronologicalOrder:
    """Tests for ordering check."""

    def test_correct_order(self) -> None:
        events = [
            _event("2026-01-01T00:00:00Z"),
            _event("2026-01-01T00:01:00Z"),
            _event("2026-01-01T00:02:00Z"),
        ]
        assert _check_chronological_order(events) == 1.0

    def test_reversed_order(self) -> None:
        events = [
            _event("2026-01-01T00:02:00Z"),
            _event("2026-01-01T00:01:00Z"),
            _event("2026-01-01T00:00:00Z"),
        ]
        assert _check_chronological_order(events) == 0.0

    def test_single_event(self) -> None:
        assert _check_chronological_order([_event("2026-01-01T00:00:00Z")]) == 1.0

    def test_empty_list(self) -> None:
        assert _check_chronological_order([]) == 1.0

    def test_partial_order(self) -> None:
        events = [
            _event("2026-01-01T00:00:00Z"),
            _event("2026-01-01T00:02:00Z"),
            _event("2026-01-01T00:01:00Z"),  # out of order
        ]
        score = _check_chronological_order(events)
        assert 0.0 < score < 1.0


class TestServiceCoverage:
    """Tests for service coverage check."""

    def test_full_coverage(self) -> None:
        events = [
            _event("2026-01-01T00:00:00Z", "database"),
            _event("2026-01-01T00:01:00Z", "payment-service"),
        ]
        chain = [
            PropagationStep(
                from_service="database",
                to_service="payment-service",
            )
        ]
        assert _check_service_coverage(events, chain) == 1.0

    def test_no_coverage(self) -> None:
        events = [
            _event("2026-01-01T00:00:00Z", "unrelated-service"),
        ]
        chain = [
            PropagationStep(
                from_service="database",
                to_service="payment-service",
            )
        ]
        assert _check_service_coverage(events, chain) == 0.0

    def test_empty_chain(self) -> None:
        events = [_event("2026-01-01T00:00:00Z", "database")]
        assert _check_service_coverage(events, []) == 1.0


class TestValidateTimeline:
    """Tests for complete timeline validation."""

    def test_correct_timeline(self) -> None:
        events = [
            _event("2026-01-01T00:00:00Z", "database"),
            _event("2026-01-01T00:00:05Z", "payment-service"),
        ]
        chain = [
            PropagationStep(
                from_service="database",
                to_service="payment-service",
                delay_seconds=5.0,
            )
        ]
        accuracy = validate_timeline(events, chain)
        assert accuracy >= 0.9

    def test_wrong_order(self) -> None:
        events = [
            _event("2026-01-01T00:01:00Z", "payment-service"),
            _event("2026-01-01T00:00:00Z", "database"),
        ]
        chain = [
            PropagationStep(
                from_service="database",
                to_service="payment-service",
            )
        ]
        accuracy = validate_timeline(events, chain)
        # Ordering is wrong → lower score
        assert accuracy < 1.0

    def test_missing_events(self) -> None:
        """Timeline only covers part of propagation chain."""
        events = [
            _event("2026-01-01T00:00:00Z", "database"),
        ]
        chain = [
            PropagationStep(
                from_service="database",
                to_service="payment-service",
            ),
            PropagationStep(
                from_service="payment-service",
                to_service="notification-service",
            ),
        ]
        accuracy = validate_timeline(events, chain)
        assert accuracy < 1.0

    def test_empty_timeline(self) -> None:
        chain = [
            PropagationStep(
                from_service="database",
                to_service="payment-service",
            )
        ]
        assert validate_timeline([], chain) == 0.0

    def test_no_ground_truth_chain(self) -> None:
        events = [
            _event("2026-01-01T00:00:00Z", "database"),
            _event("2026-01-01T00:01:00Z", "payment-service"),
        ]
        accuracy = validate_timeline(events, [])
        assert accuracy == 1.0  # Can only check ordering

    def test_custom_weights(self) -> None:
        events = [
            _event("2026-01-01T00:00:00Z", "database"),
            _event("2026-01-01T00:01:00Z", "payment-service"),
        ]
        chain = [
            PropagationStep(
                from_service="database",
                to_service="payment-service",
            )
        ]
        accuracy = validate_timeline(
            events, chain, order_weight=0.8, timing_weight=0.2
        )
        assert 0.0 <= accuracy <= 1.0
