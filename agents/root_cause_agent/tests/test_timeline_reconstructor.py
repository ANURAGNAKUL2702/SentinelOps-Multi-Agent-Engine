"""
Tests for core/timeline_reconstructor.py — Algorithm 7.
"""

import pytest

from agents.root_cause_agent.core.timeline_reconstructor import TimelineReconstructor
from agents.root_cause_agent.schema import (
    DependencyAgentFindings,
    EvidenceSourceAgent,
    HypothesisFindings,
    LogAgentFindings,
    MetricsAgentFindings,
    RootCauseAgentInput,
    Severity,
)


def _make_input() -> RootCauseAgentInput:
    return RootCauseAgentInput(
        log_findings=LogAgentFindings(
            suspicious_services=["svc-a"],
            error_patterns=["timeout"],
            confidence=0.8,
            timestamp="2024-01-01T00:00:01Z",
        ),
        metrics_findings=MetricsAgentFindings(
            anomalies=[
                {"service": "svc-a", "metric": "latency", "z_score": 3.0, "severity": "high"},
            ],
            confidence=0.75,
            timestamp="2024-01-01T00:00:02Z",
        ),
        dependency_findings=DependencyAgentFindings(
            bottlenecks=["svc-a"],
            blast_radius=2,
            confidence=0.85,
            timestamp="2024-01-01T00:00:03Z",
        ),
        hypothesis_findings=HypothesisFindings(
            top_hypothesis="DB overload",
            top_confidence=0.88,
            confidence=0.88,
            timestamp="2024-01-01T00:00:04Z",
        ),
    )


class TestTimelineReconstructor:
    def test_reconstruct_basic(self):
        recon = TimelineReconstructor()
        events = recon.reconstruct(_make_input())
        assert len(events) > 0

    def test_events_sorted_chronologically(self):
        recon = TimelineReconstructor()
        events = recon.reconstruct(_make_input())
        timestamps = [e.timestamp for e in events]
        assert timestamps == sorted(timestamps)

    def test_events_from_multiple_sources(self):
        recon = TimelineReconstructor()
        events = recon.reconstruct(_make_input())
        sources = {e.source for e in events}
        assert len(sources) >= 2

    def test_deduplication(self):
        # Create input with two events from same source at same time
        inp = RootCauseAgentInput(
            log_findings=LogAgentFindings(
                suspicious_services=["svc-a", "svc-a"],  # duplicate service
                confidence=0.8,
                timestamp="2024-01-01T00:00:00Z",
            ),
        )
        recon = TimelineReconstructor()
        events = recon.reconstruct(inp)
        # Same service, same source, same timestamp — should dedup
        svc_a_log_events = [
            e for e in events
            if e.service == "svc-a" and e.source == EvidenceSourceAgent.LOG_AGENT
        ]
        assert len(svc_a_log_events) == 1

    def test_empty_input(self):
        recon = TimelineReconstructor()
        events = recon.reconstruct(RootCauseAgentInput())
        assert events == []

    def test_max_events_respected(self):
        from agents.root_cause_agent.config import RootCauseAgentConfig, VerdictLimits
        config = RootCauseAgentConfig(limits=VerdictLimits(max_timeline_events=3))
        recon = TimelineReconstructor(config)
        events = recon.reconstruct(_make_input())
        assert len(events) <= 3

    def test_events_have_severity(self):
        recon = TimelineReconstructor()
        events = recon.reconstruct(_make_input())
        for ev in events:
            assert isinstance(ev.severity, Severity)
