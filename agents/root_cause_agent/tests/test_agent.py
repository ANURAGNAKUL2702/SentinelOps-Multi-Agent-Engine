"""
Tests for agent.py — RootCauseAgent orchestrator.
"""

import pytest

from agents.root_cause_agent.agent import RootCauseAgent
from agents.root_cause_agent.config import RootCauseAgentConfig, FeatureFlags
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
        incident_id="test-001",
    )


class TestRootCauseAgent:
    def test_analyze_basic(self):
        agent = RootCauseAgent()
        verdict = agent.analyze(_make_input())
        assert verdict.agent == "root_cause_agent"
        assert verdict.root_cause != ""
        assert verdict.confidence > 0

    def test_analyze_with_validation(self):
        config = RootCauseAgentConfig(
            features=FeatureFlags(enable_validation=True)
        )
        agent = RootCauseAgent(config)
        verdict = agent.analyze(_make_input())
        assert verdict.validation is not None

    def test_analyze_without_validation(self):
        config = RootCauseAgentConfig(
            features=FeatureFlags(enable_validation=False)
        )
        agent = RootCauseAgent(config)
        verdict = agent.analyze(_make_input())
        assert verdict.validation is None

    def test_analyze_returns_evidence(self):
        agent = RootCauseAgent()
        verdict = agent.analyze(_make_input())
        assert len(verdict.evidence_trail) > 0

    def test_analyze_returns_causal_chain(self):
        agent = RootCauseAgent()
        verdict = agent.analyze(_make_input())
        assert len(verdict.causal_chain) > 0

    def test_analyze_pipeline_latency_recorded(self):
        agent = RootCauseAgent()
        verdict = agent.analyze(_make_input())
        assert verdict.pipeline_latency_ms > 0

    def test_analyze_with_llm_mock(self):
        config = RootCauseAgentConfig(
            features=FeatureFlags(use_llm=True)
        )
        agent = RootCauseAgent(config)
        verdict = agent.analyze(_make_input())
        assert verdict.agent == "root_cause_agent"

    def test_telemetry_accessible(self):
        agent = RootCauseAgent()
        agent.analyze(_make_input())
        snapshot = agent.telemetry.snapshot()
        assert snapshot["counters"]["analyses_total"] >= 1

    def test_config_accessible(self):
        agent = RootCauseAgent()
        assert agent.config is not None

    def test_empty_input_doesnt_crash(self):
        agent = RootCauseAgent()
        verdict = agent.analyze(RootCauseAgentInput())
        assert verdict.agent == "root_cause_agent"
