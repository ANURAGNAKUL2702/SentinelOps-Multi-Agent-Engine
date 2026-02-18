"""
Tests for fallback.py — Deterministic fallback analysis.
"""

import pytest

from agents.root_cause_agent.config import RootCauseAgentConfig
from agents.root_cause_agent.fallback import DeterministicFallback
from agents.root_cause_agent.schema import (
    DependencyAgentFindings,
    HypothesisFindings,
    LogAgentFindings,
    MetricsAgentFindings,
    RootCauseAgentInput,
)


def _make_input() -> RootCauseAgentInput:
    return RootCauseAgentInput(
        log_findings=LogAgentFindings(
            suspicious_services=["payment-service"],
            error_patterns=["ConnectionTimeout"],
            confidence=0.8,
            timestamp="2024-01-01T00:00:00Z",
        ),
        metrics_findings=MetricsAgentFindings(
            anomalies=[
                {"service": "payment-service", "metric": "latency",
                 "z_score": 4.2, "severity": "high"},
            ],
            confidence=0.75,
            timestamp="2024-01-01T00:00:01Z",
        ),
        dependency_findings=DependencyAgentFindings(
            impact_graph={"payment-service": ["order-service"]},
            critical_paths=[["payment-service", "order-service"]],
            bottlenecks=["payment-service"],
            blast_radius=2,
            affected_services=["order-service"],
            confidence=0.85,
            timestamp="2024-01-01T00:00:02Z",
        ),
        hypothesis_findings=HypothesisFindings(
            ranked_hypotheses=[
                {"theory": "DB connection pool exhaustion",
                 "confidence": 0.88,
                 "evidence_supporting": ["metric_spike", "error_log"]},
                {"theory": "Network partition",
                 "confidence": 0.4,
                 "evidence_supporting": ["timeout_pattern"]},
            ],
            top_hypothesis="DB connection pool exhaustion",
            top_confidence=0.88,
            causal_chains=[
                {"chain": ["db_overload", "timeout", "service_down"]}
            ],
            category="database",
            confidence=0.88,
            timestamp="2024-01-01T00:00:03Z",
        ),
    )


class TestDeterministicFallback:
    def test_analyze_returns_verdict(self):
        fb = DeterministicFallback()
        verdict = fb.analyze(_make_input())
        assert verdict.agent == "root_cause_agent"
        assert verdict.root_cause != ""
        assert verdict.confidence > 0

    def test_analyze_has_evidence_trail(self):
        fb = DeterministicFallback()
        verdict = fb.analyze(_make_input())
        assert len(verdict.evidence_trail) > 0

    def test_analyze_has_causal_chain(self):
        fb = DeterministicFallback()
        verdict = fb.analyze(_make_input())
        assert len(verdict.causal_chain) > 0

    def test_analyze_has_timeline(self):
        fb = DeterministicFallback()
        verdict = fb.analyze(_make_input())
        assert len(verdict.timeline) > 0

    def test_analyze_has_impact(self):
        fb = DeterministicFallback()
        verdict = fb.analyze(_make_input())
        assert verdict.impact is not None

    def test_analyze_uses_fallback_flag(self):
        fb = DeterministicFallback()
        verdict = fb.analyze(_make_input())
        assert verdict.metadata is not None
        assert verdict.metadata.used_fallback is True
        assert verdict.metadata.used_llm is False

    def test_analyze_classification_source(self):
        fb = DeterministicFallback()
        verdict = fb.analyze(_make_input())
        assert verdict.classification_source == "deterministic"

    def test_analyze_reasoning_not_empty(self):
        fb = DeterministicFallback()
        verdict = fb.analyze(_make_input())
        assert len(verdict.reasoning) > 0

    def test_analyze_category_from_hypothesis(self):
        fb = DeterministicFallback()
        verdict = fb.analyze(_make_input())
        assert verdict.category == "database"

    def test_analyze_empty_input(self):
        fb = DeterministicFallback()
        verdict = fb.analyze(RootCauseAgentInput())
        assert verdict.agent == "root_cause_agent"
