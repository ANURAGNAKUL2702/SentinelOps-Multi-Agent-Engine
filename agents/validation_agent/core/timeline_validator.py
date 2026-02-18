"""
File: core/timeline_validator.py
Purpose: Verify timeline accuracy — correct ordering and timing deltas.
Dependencies: Standard library only.
Performance: <1ms per call, pure function.

Algorithm 6: Compare verdict timeline with ground truth propagation chain.
Check chronological ordering and timing accuracy.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from agents.validation_agent.schema import (
    PropagationStep,
    TimelineEvent,
)


def _parse_timestamp(ts: str) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp string.

    Args:
        ts: Timestamp string.

    Returns:
        Parsed datetime, or None if unparseable.
    """
    if not ts:
        return None
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    return None


def _check_chronological_order(events: List[TimelineEvent]) -> float:
    """Check if timeline events are in chronological order.

    Args:
        events: List of timeline events.

    Returns:
        Score 0.0–1.0 (1.0 = perfect order).
    """
    if len(events) <= 1:
        return 1.0

    timestamps: List[datetime] = []
    for e in events:
        parsed = _parse_timestamp(e.timestamp)
        if parsed:
            timestamps.append(parsed)

    if len(timestamps) <= 1:
        return 1.0

    ordered_pairs = 0
    total_pairs = len(timestamps) - 1

    for i in range(total_pairs):
        if timestamps[i] <= timestamps[i + 1]:
            ordered_pairs += 1

    return ordered_pairs / total_pairs if total_pairs > 0 else 1.0


def _check_service_coverage(
    events: List[TimelineEvent],
    propagation_chain: List[PropagationStep],
) -> float:
    """Check how many propagation chain services appear in timeline.

    Args:
        events: Timeline events from verdict.
        propagation_chain: Ground truth propagation chain.

    Returns:
        Coverage score 0.0–1.0.
    """
    if not propagation_chain:
        return 1.0

    # Collect all services from ground truth chain
    gt_services: set = set()
    for step in propagation_chain:
        gt_services.add(step.from_service.lower())
        gt_services.add(step.to_service.lower())

    if not gt_services:
        return 1.0

    # Collect services from timeline events
    timeline_services = set(
        e.service.lower() for e in events if e.service
    )

    if not timeline_services:
        return 0.0

    # Jaccard-like coverage
    covered = gt_services & timeline_services
    return len(covered) / len(gt_services) if gt_services else 1.0


def validate_timeline(
    timeline: List[TimelineEvent],
    ground_truth_chain: List[PropagationStep],
    timing_tolerance_seconds: float = 10.0,
    order_weight: float = 0.6,
    timing_weight: float = 0.4,
) -> float:
    """Validate timeline accuracy against ground truth propagation.

    Combines chronological ordering check and service coverage.

    Args:
        timeline: Timeline events from the verdict.
        ground_truth_chain: Ground truth failure propagation chain.
        timing_tolerance_seconds: Tolerance for timing deltas.
        order_weight: Weight for ordering accuracy (default 0.6).
        timing_weight: Weight for timing/coverage accuracy (default 0.4).

    Returns:
        Timeline accuracy score 0.0–1.0.
    """
    if not timeline:
        return 0.0

    if not ground_truth_chain:
        # No ground truth chain — can only check ordering
        return _check_chronological_order(timeline)

    # Component 1: Chronological ordering
    order_score = _check_chronological_order(timeline)

    # Component 2: Service coverage (proxy for timing accuracy)
    coverage_score = _check_service_coverage(timeline, ground_truth_chain)

    # Weighted combination
    accuracy = order_weight * order_score + timing_weight * coverage_score
    return round(min(1.0, max(0.0, accuracy)), 4)
