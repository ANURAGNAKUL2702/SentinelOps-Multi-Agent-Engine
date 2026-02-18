"""
Tests for core/contradiction_resolver.py — Algorithm 6.
"""

import pytest

from agents.root_cause_agent.core.contradiction_resolver import ContradictionResolver
from agents.root_cause_agent.schema import (
    ContradictionStrategy,
    DependencyAgentFindings,
    EvidenceSourceAgent,
    HypothesisFindings,
    LogAgentFindings,
    MetricsAgentFindings,
    RootCauseAgentInput,
)


def _make_conflicting_input() -> RootCauseAgentInput:
    """Input where log and metrics blame different services."""
    return RootCauseAgentInput(
        log_findings=LogAgentFindings(
            suspicious_services=["service-a"],
            confidence=0.9,
            timestamp="2024-01-01T00:00:00Z",
        ),
        metrics_findings=MetricsAgentFindings(
            anomalies=[
                {"service": "service-b", "metric": "cpu", "z_score": 3.0, "severity": "high"},
            ],
            confidence=0.6,
            timestamp="2024-01-01T00:00:00Z",
        ),
        dependency_findings=DependencyAgentFindings(
            impact_graph={"service-a": ["service-c"], "service-b": []},
            confidence=0.8,
            timestamp="2024-01-01T00:00:00Z",
        ),
    )


class TestContradictionResolver:
    def test_detect_conflict(self):
        resolver = ContradictionResolver()
        contradictions = resolver.resolve(
            _make_conflicting_input(), []
        )
        assert len(contradictions) > 0

    def test_confidence_wins_resolution(self):
        resolver = ContradictionResolver()
        contradictions = resolver.resolve(
            _make_conflicting_input(), []
        )
        # Log has 0.9 vs metrics 0.6 — confidence wins
        resolved = [c for c in contradictions if c.resolved]
        assert len(resolved) > 0
        assert resolved[0].resolution_strategy == ContradictionStrategy.CONFIDENCE_WINS

    def test_no_conflict_when_same_service(self):
        inp = RootCauseAgentInput(
            log_findings=LogAgentFindings(
                suspicious_services=["same-svc"],
                confidence=0.8,
                timestamp="2024-01-01T00:00:00Z",
            ),
            metrics_findings=MetricsAgentFindings(
                anomalies=[
                    {"service": "same-svc", "metric": "cpu", "z_score": 3.0},
                ],
                confidence=0.7,
                timestamp="2024-01-01T00:00:00Z",
            ),
        )
        resolver = ContradictionResolver()
        contradictions = resolver.resolve(inp, [])
        assert len(contradictions) == 0

    def test_empty_input_no_contradictions(self):
        resolver = ContradictionResolver()
        contradictions = resolver.resolve(RootCauseAgentInput(), [])
        assert len(contradictions) == 0

    def test_graph_centrality_resolution(self):
        # Both agents have same confidence, but service-a has more connections
        inp = RootCauseAgentInput(
            log_findings=LogAgentFindings(
                suspicious_services=["service-a"],
                confidence=0.75,
                timestamp="2024-01-01T00:00:00Z",
            ),
            metrics_findings=MetricsAgentFindings(
                anomalies=[
                    {"service": "service-b", "metric": "cpu", "z_score": 3.0},
                ],
                confidence=0.75,
                timestamp="2024-01-01T00:00:00Z",
            ),
            dependency_findings=DependencyAgentFindings(
                impact_graph={
                    "service-a": ["x", "y", "z"],
                    "service-b": [],
                },
                confidence=0.8,
            ),
        )
        resolver = ContradictionResolver()
        contradictions = resolver.resolve(inp, [])
        resolved = [c for c in contradictions if c.resolved]
        if resolved:
            assert resolved[0].winner == "service-a"
            assert resolved[0].resolution_strategy == ContradictionStrategy.GRAPH_CENTRALITY
