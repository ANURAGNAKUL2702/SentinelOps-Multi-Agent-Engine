"""
File: core/timeline_reconstructor.py
Purpose: Algorithm 7 — Merge timestamps from all agents, sort chronologically, dedup.
Dependencies: Schema models only.
Performance: <2ms, O(n log n) where n = events.

Collects all timestamped events from 4 agents, merges them into a
single timeline, sorts chronologically, and removes duplicates
that are <1 second apart + same service.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Tuple

from agents.root_cause_agent.config import RootCauseAgentConfig
from agents.root_cause_agent.schema import (
    EvidenceSourceAgent,
    RootCauseAgentInput,
    Severity,
    TimelineEvent,
)
from agents.root_cause_agent.telemetry import get_logger

logger = get_logger("root_cause_agent.timeline_reconstructor")


class TimelineReconstructor:
    """Reconstructs a chronological incident timeline from all agents.

    Pipeline::

        4 agents → collect events → sort → dedup (<1s apart) → trim → timeline

    Args:
        config: Agent configuration.
    """

    def __init__(
        self, config: Optional[RootCauseAgentConfig] = None
    ) -> None:
        self._config = config or RootCauseAgentConfig()

    def reconstruct(
        self,
        input_data: RootCauseAgentInput,
        correlation_id: str = "",
    ) -> List[TimelineEvent]:
        """Build chronological timeline from all agent timestamps.

        Args:
            input_data: Root cause agent input.
            correlation_id: Request correlation ID.

        Returns:
            Sorted, deduplicated list of TimelineEvent.
        """
        events: List[TimelineEvent] = []

        # ── Collect from log agent ──────────────────────────────
        events.extend(self._log_events(input_data))

        # ── Collect from metrics agent ──────────────────────────
        events.extend(self._metrics_events(input_data))

        # ── Collect from dependency agent ───────────────────────
        events.extend(self._dependency_events(input_data))

        # ── Collect from hypothesis agent ───────────────────────
        events.extend(self._hypothesis_events(input_data))

        # ── Sort chronologically ────────────────────────────────
        events = self._sort_events(events)

        # ── Deduplicate close events ────────────────────────────
        threshold = self._config.timeline_dedup.close_event_threshold_seconds
        events = self._deduplicate(events, threshold)

        # ── Trim to max ─────────────────────────────────────────
        max_events = self._config.limits.max_timeline_events
        if len(events) > max_events:
            events = events[:max_events]

        logger.debug(
            f"Timeline reconstructed: {len(events)} events",
            extra={
                "correlation_id": correlation_id,
                "layer": "timeline_reconstruction",
            },
        )

        return events

    def _log_events(
        self, input_data: RootCauseAgentInput
    ) -> List[TimelineEvent]:
        """Extract timeline events from log findings."""
        events: List[TimelineEvent] = []
        lf = input_data.log_findings

        if lf.timestamp:
            for svc in lf.suspicious_services:
                events.append(TimelineEvent(
                    timestamp=lf.timestamp,
                    source=EvidenceSourceAgent.LOG_AGENT,
                    event=f"Error spike detected in {svc}",
                    service=svc,
                    severity=Severity.HIGH,
                ))

            for pattern in lf.error_patterns:
                events.append(TimelineEvent(
                    timestamp=lf.timestamp,
                    source=EvidenceSourceAgent.LOG_AGENT,
                    event=f"Error pattern: {pattern}",
                    service="",
                    severity=Severity.MEDIUM,
                ))

        return events

    def _metrics_events(
        self, input_data: RootCauseAgentInput
    ) -> List[TimelineEvent]:
        """Extract timeline events from metrics findings."""
        events: List[TimelineEvent] = []
        mf = input_data.metrics_findings

        if mf.timestamp:
            for anom in mf.anomalies:
                svc = anom.get("service", "unknown")
                metric = anom.get("metric", "unknown")
                sev_str = anom.get("severity", "medium")
                try:
                    sev = Severity(sev_str)
                except (ValueError, KeyError):
                    sev = Severity.MEDIUM

                events.append(TimelineEvent(
                    timestamp=mf.timestamp,
                    source=EvidenceSourceAgent.METRICS_AGENT,
                    event=f"Metric anomaly: {metric} in {svc}",
                    service=svc,
                    severity=sev,
                ))

        return events

    def _dependency_events(
        self, input_data: RootCauseAgentInput
    ) -> List[TimelineEvent]:
        """Extract timeline events from dependency findings."""
        events: List[TimelineEvent] = []
        df = input_data.dependency_findings

        if df.timestamp:
            if df.blast_radius > 0:
                events.append(TimelineEvent(
                    timestamp=df.timestamp,
                    source=EvidenceSourceAgent.DEPENDENCY_AGENT,
                    event=f"Blast radius: {df.blast_radius} services affected",
                    service="",
                    severity=Severity.CRITICAL,
                ))

            for bn in df.bottlenecks:
                events.append(TimelineEvent(
                    timestamp=df.timestamp,
                    source=EvidenceSourceAgent.DEPENDENCY_AGENT,
                    event=f"Bottleneck detected: {bn}",
                    service=bn,
                    severity=Severity.HIGH,
                ))

        return events

    def _hypothesis_events(
        self, input_data: RootCauseAgentInput
    ) -> List[TimelineEvent]:
        """Extract timeline events from hypothesis findings."""
        events: List[TimelineEvent] = []
        hf = input_data.hypothesis_findings

        if hf.timestamp and hf.top_hypothesis:
            events.append(TimelineEvent(
                timestamp=hf.timestamp,
                source=EvidenceSourceAgent.HYPOTHESIS_AGENT,
                event=f"Hypothesis: {hf.top_hypothesis}",
                service="",
                severity=Severity.HIGH,
            ))

        return events

    def _sort_events(
        self, events: List[TimelineEvent]
    ) -> List[TimelineEvent]:
        """Sort events chronologically by timestamp.

        Args:
            events: Unsorted events.

        Returns:
            Sorted events.
        """
        def parse_ts(ev: TimelineEvent) -> datetime:
            try:
                return datetime.fromisoformat(
                    ev.timestamp.replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                return datetime.min.replace(tzinfo=timezone.utc)

        return sorted(events, key=parse_ts)

    def _deduplicate(
        self,
        events: List[TimelineEvent],
        threshold_seconds: float,
    ) -> List[TimelineEvent]:
        """Remove events that are within threshold of each other + same service.

        Args:
            events: Sorted events.
            threshold_seconds: Time window for dedup.

        Returns:
            Deduplicated events.
        """
        if not events:
            return events

        result: List[TimelineEvent] = [events[0]]
        for ev in events[1:]:
            prev = result[-1]
            # Same service + close timestamp = duplicate
            if (
                ev.service == prev.service
                and ev.source == prev.source
                and self._time_diff_seconds(prev.timestamp, ev.timestamp)
                    < threshold_seconds
            ):
                continue
            result.append(ev)

        return result

    def _time_diff_seconds(self, ts1: str, ts2: str) -> float:
        """Compute absolute time difference in seconds between two timestamps.

        Args:
            ts1: First ISO-8601 timestamp.
            ts2: Second ISO-8601 timestamp.

        Returns:
            Absolute difference in seconds.
        """
        try:
            dt1 = datetime.fromisoformat(ts1.replace("Z", "+00:00"))
            dt2 = datetime.fromisoformat(ts2.replace("Z", "+00:00"))
            return abs((dt2 - dt1).total_seconds())
        except (ValueError, TypeError):
            return float("inf")  # can't compare = not duplicates
