"""
Tests for performance requirements.

Covers:
  - Fallback pipeline < 100ms
  - Evidence aggregation < 35ms
  - Pattern matching < 10ms
  - Ranking < 10ms
  - Validation < 10ms
  - Large evidence set performance
"""

from __future__ import annotations

import time
import pytest

from agents.hypothesis_agent.agent import HypothesisAgent
from agents.hypothesis_agent.config import (
    FeatureFlags,
    HypothesisAgentConfig,
)
from agents.hypothesis_agent.core.evidence_aggregator import (
    EvidenceAggregator,
)
from agents.hypothesis_agent.core.hypothesis_ranker import (
    HypothesisRanker,
)
from agents.hypothesis_agent.core.pattern_matcher import (
    PatternMatcher,
)
from agents.hypothesis_agent.fallback import FallbackGenerator
from agents.hypothesis_agent.schema import (
    AggregatedEvidence,
    DependencyFindings,
    EvidenceItem,
    EvidenceSource,
    EvidenceStrength,
    Hypothesis,
    HypothesisAgentInput,
    IncidentCategory,
    LogFindings,
    MetricFindings,
    PatternMatch,
    PatternName,
    Severity,
)
from agents.hypothesis_agent.validator import OutputValidator


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════


def _rich_input() -> HypothesisAgentInput:
    return HypothesisAgentInput(
        log_findings=LogFindings(
            suspicious_services=[
                {
                    "service": f"svc-{i}",
                    "error_count": 50 + i * 10,
                    "severity_hint": "high",
                    "error_keywords_detected": ["timeout", "error"],
                }
                for i in range(5)
            ],
            total_error_logs=250,
            system_wide_spike=True,
            database_errors_detected=True,
            potential_upstream_failure=True,
        ),
        metric_findings=MetricFindings(
            anomalous_metrics=[
                {
                    "metric_name": f"metric_{i}",
                    "zscore": 3.0 + i,
                    "severity": "high",
                    "anomaly_type": "spike",
                }
                for i in range(5)
            ],
            resource_saturation=True,
            cascading_degradation=True,
        ),
        dependency_findings=DependencyFindings(
            failed_service="payment-service",
            is_cascading=True,
            blast_radius_count=10,
            single_points_of_failure=["gateway"],
            bottleneck_services=["api-gw", "lb"],
            graph_has_cycles=True,
        ),
    )


# ═══════════════════════════════════════════════════════════════
#  TESTS: End-to-End Fallback Performance
# ═══════════════════════════════════════════════════════════════


class TestFallbackPerformance:
    """Fallback pipeline must complete in < 100ms."""

    def test_fallback_pipeline_under_100ms(self):
        config = HypothesisAgentConfig(
            features=FeatureFlags(use_llm=False),
        )
        agent = HypothesisAgent(config)
        inp = _rich_input()

        # Warm up
        agent.analyze(inp)

        start = time.perf_counter()
        output = agent.analyze(inp)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, (
            f"Fallback pipeline took {elapsed_ms:.2f}ms (limit: 100ms)"
        )
        assert len(output.hypotheses) >= 3


# ═══════════════════════════════════════════════════════════════
#  TESTS: Phase-Level Performance
# ═══════════════════════════════════════════════════════════════


class TestPhaseLevelPerformance:
    """Test individual phase performance budgets."""

    def test_evidence_aggregation_under_35ms(self):
        agg = EvidenceAggregator()
        inp = _rich_input()

        # Warm up
        agg.aggregate(inp)

        start = time.perf_counter()
        result = agg.aggregate(inp)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 35, (
            f"Evidence aggregation took {elapsed_ms:.2f}ms"
        )
        assert result.total_evidence_count > 0

    def test_pattern_matching_under_10ms(self):
        agg = EvidenceAggregator()
        matcher = PatternMatcher()
        inp = _rich_input()
        evidence = agg.aggregate(inp)

        # Warm up
        matcher.match(evidence)

        start = time.perf_counter()
        matches = matcher.match(evidence)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 10, (
            f"Pattern matching took {elapsed_ms:.2f}ms"
        )

    def test_ranking_under_10ms(self):
        ranker = HypothesisRanker()
        evidence = AggregatedEvidence(
            evidence_items=[
                EvidenceItem(
                    source=EvidenceSource.LOG_AGENT,
                    description="test",
                    strength=EvidenceStrength.STRONG,
                )
            ],
            total_evidence_count=1,
        )
        hypotheses = [
            Hypothesis(
                theory=f"Theory {i}",
                likelihood_score=0.5,
                evidence_supporting=["e1"],
            )
            for i in range(5)
        ]

        # Warm up
        ranker.rank(hypotheses, evidence, [], [])

        start = time.perf_counter()
        ranked = ranker.rank(hypotheses, evidence, [], [])
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 10, (
            f"Ranking took {elapsed_ms:.2f}ms"
        )

    def test_validation_under_10ms(self):
        from agents.hypothesis_agent.schema import (
            HypothesisAgentOutput,
        )

        validator = OutputValidator()
        output = HypothesisAgentOutput(
            incident_id="inc-1",
            hypotheses=[
                Hypothesis(
                    theory=f"Theory {i}",
                    likelihood_score=0.9 - i * 0.2,
                    evidence_supporting=["e1"],
                )
                for i in range(3)
            ],
            confidence_score=0.7,
            hypothesis_summary="Test summary",
        )

        # Warm up
        validator.validate(output)

        start = time.perf_counter()
        result = validator.validate(output)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 10, (
            f"Validation took {elapsed_ms:.2f}ms"
        )
        assert result.total_checks == 27


# ═══════════════════════════════════════════════════════════════
#  TESTS: Large Evidence Set
# ═══════════════════════════════════════════════════════════════


class TestLargeEvidenceSet:
    """Test performance with large evidence inputs."""

    def test_50_services_under_100ms(self):
        config = HypothesisAgentConfig(
            features=FeatureFlags(use_llm=False),
        )
        agent = HypothesisAgent(config)
        inp = HypothesisAgentInput(
            log_findings=LogFindings(
                suspicious_services=[
                    {
                        "service": f"svc-{i}",
                        "error_count": 10,
                        "severity_hint": "medium",
                    }
                    for i in range(50)
                ],
            ),
            metric_findings=MetricFindings(
                anomalous_metrics=[
                    {
                        "metric_name": f"metric_{i}",
                        "zscore": 2.5,
                        "severity": "medium",
                        "anomaly_type": "drift",
                    }
                    for i in range(50)
                ],
            ),
        )

        # Warm up
        agent.analyze(inp)

        start = time.perf_counter()
        output = agent.analyze(inp)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, (
            f"Large input took {elapsed_ms:.2f}ms"
        )
        assert output.validation is not None
        assert output.validation.validation_passed is True
