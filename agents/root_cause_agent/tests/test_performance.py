"""
Tests for performance — latency budgets.
"""

import time
import pytest

from agents.root_cause_agent.agent import RootCauseAgent
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
            error_patterns=["ConnectionTimeout", "NullPointer"],
            log_anomalies=[{"description": "spike in errors"}],
            confidence=0.8,
            timestamp="2024-01-01T00:00:00Z",
        ),
        metrics_findings=MetricsAgentFindings(
            anomalies=[
                {"service": "payment-service", "metric": "latency",
                 "z_score": 4.2, "severity": "high"},
                {"service": "order-service", "metric": "errors",
                 "z_score": 3.1, "severity": "medium"},
            ],
            correlations=[{"description": "latency-error correlation"}],
            confidence=0.75,
            timestamp="2024-01-01T00:00:01Z",
        ),
        dependency_findings=DependencyAgentFindings(
            impact_graph={
                "payment-service": ["order-service", "notification-service"],
                "order-service": ["inventory-service"],
            },
            critical_paths=[["payment-service", "order-service", "inventory-service"]],
            bottlenecks=["payment-service"],
            blast_radius=4,
            affected_services=["order-service", "notification-service", "inventory-service"],
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
                {"chain": ["db_overload", "timeout", "service_down"]},
            ],
            category="database",
            confidence=0.88,
            timestamp="2024-01-01T00:00:03Z",
        ),
    )


class TestPerformance:
    def test_fallback_under_100ms(self):
        """Deterministic fallback must complete in <100ms."""
        fb = DeterministicFallback()
        inp = _make_input()

        # Warm up
        fb.analyze(inp)

        start = time.perf_counter()
        verdict = fb.analyze(inp)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, f"Fallback took {elapsed_ms:.1f}ms (budget: 100ms)"
        assert verdict.confidence > 0

    def test_agent_deterministic_under_100ms(self):
        """Agent in deterministic mode must complete in <100ms."""
        agent = RootCauseAgent()
        inp = _make_input()

        # Warm up
        agent.analyze(inp)

        start = time.perf_counter()
        verdict = agent.analyze(inp)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, f"Agent took {elapsed_ms:.1f}ms (budget: 100ms)"

    def test_multiple_runs_consistent(self):
        """Verify latency consistency across 10 runs."""
        fb = DeterministicFallback()
        inp = _make_input()
        fb.analyze(inp)  # warm up

        timings = []
        for _ in range(10):
            start = time.perf_counter()
            fb.analyze(inp)
            timings.append((time.perf_counter() - start) * 1000)

        avg = sum(timings) / len(timings)
        assert avg < 50, f"Average {avg:.1f}ms exceeds 50ms target"

    def test_evidence_synthesis_under_5ms(self):
        """Evidence synthesis should complete in <5ms."""
        from agents.root_cause_agent.core.evidence_synthesizer import EvidenceSynthesizer
        synth = EvidenceSynthesizer()
        inp = _make_input()

        # Warm up
        synth.synthesize(inp)

        start = time.perf_counter()
        synth.synthesize(inp)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 5, f"Synthesis took {elapsed_ms:.1f}ms (budget: 5ms)"
