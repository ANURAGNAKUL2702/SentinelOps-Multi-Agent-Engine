"""
Tests for core/evidence_synthesizer.py — Algorithm 1.
"""

import pytest

from agents.root_cause_agent.config import RootCauseAgentConfig
from agents.root_cause_agent.core.evidence_synthesizer import EvidenceSynthesizer
from agents.root_cause_agent.schema import (
    DependencyAgentFindings,
    EvidenceSourceAgent,
    EvidenceType,
    HypothesisFindings,
    LogAgentFindings,
    MetricsAgentFindings,
    RootCauseAgentInput,
)


def _make_input(**kwargs) -> RootCauseAgentInput:
    """Build a RootCauseAgentInput with sensible defaults."""
    defaults = dict(
        log_findings=LogAgentFindings(
            suspicious_services=["payment-service"],
            error_patterns=["ConnectionTimeout"],
            confidence=0.8,
            timestamp="2024-01-01T00:00:00Z",
        ),
        metrics_findings=MetricsAgentFindings(
            anomalies=[
                {"service": "payment-service", "metric": "latency", "z_score": 4.2, "severity": "high"},
            ],
            confidence=0.75,
            timestamp="2024-01-01T00:00:01Z",
        ),
        dependency_findings=DependencyAgentFindings(
            impact_graph={"payment-service": ["order-service", "notification-service"]},
            bottlenecks=["payment-service"],
            blast_radius=3,
            affected_services=["order-service", "notification-service"],
            confidence=0.85,
            timestamp="2024-01-01T00:00:02Z",
        ),
        hypothesis_findings=HypothesisFindings(
            ranked_hypotheses=[
                {"theory": "DB connection pool exhaustion", "confidence": 0.88,
                 "evidence_supporting": ["metric_spike", "error_log"]},
            ],
            top_hypothesis="DB connection pool exhaustion",
            top_confidence=0.88,
            causal_chains=[{"chain": ["db_overload", "timeout", "service_down"]}],
            confidence=0.88,
            timestamp="2024-01-01T00:00:03Z",
        ),
    )
    defaults.update(kwargs)
    return RootCauseAgentInput(**defaults)


class TestEvidenceSynthesizer:
    def test_synthesize_all_agents(self):
        synth = EvidenceSynthesizer()
        result = synth.synthesize(_make_input())
        assert len(result.evidence_trail) > 0
        assert len(result.sources_present) == 4
        assert result.agreement_score > 0

    def test_synthesize_returns_primary_service(self):
        synth = EvidenceSynthesizer()
        result = synth.synthesize(_make_input())
        assert result.primary_service == "payment-service"

    def test_synthesize_empty_input(self):
        synth = EvidenceSynthesizer()
        result = synth.synthesize(RootCauseAgentInput())
        assert len(result.evidence_trail) == 0
        assert result.agreement_score == 0.0

    def test_synthesize_single_agent(self):
        inp = RootCauseAgentInput(
            log_findings=LogAgentFindings(
                suspicious_services=["svc-a"],
                confidence=0.7,
                timestamp="2024-01-01T00:00:00Z",
            ),
        )
        synth = EvidenceSynthesizer()
        result = synth.synthesize(inp)
        assert EvidenceSourceAgent.LOG_AGENT in result.sources_present

    def test_synthesize_caps_evidence(self):
        config = RootCauseAgentConfig()
        # Using default max_evidence_items=500, so this should not cap
        synth = EvidenceSynthesizer(config)
        result = synth.synthesize(_make_input())
        assert len(result.evidence_trail) <= config.performance.max_evidence_items

    def test_agreement_score_range(self):
        synth = EvidenceSynthesizer()
        result = synth.synthesize(_make_input())
        assert 0.0 <= result.agreement_score <= 1.0

    def test_evidence_has_timestamps(self):
        synth = EvidenceSynthesizer()
        result = synth.synthesize(_make_input())
        for ev in result.evidence_trail:
            assert ev.timestamp != ""

    def test_evidence_types_present(self):
        synth = EvidenceSynthesizer()
        result = synth.synthesize(_make_input())
        types = {ev.evidence_type for ev in result.evidence_trail}
        assert EvidenceType.DIRECT in types
