"""
Tests for core/evidence_aggregator.py — Algorithm 1: Evidence Aggregation O(n).

Covers:
  - Log evidence extraction (suspicious services, system signals)
  - Metric evidence extraction (anomalies, correlations, system signals)
  - Dependency evidence extraction (failures, SPOFs, bottlenecks, cycles)
  - Cross-agent correlation detection
  - Evidence capping
  - Severity mapping
  - Dominant severity computation
"""

from __future__ import annotations

import pytest

from agents.hypothesis_agent.config import (
    HypothesisAgentConfig,
    PerformanceConfig,
)
from agents.hypothesis_agent.core.evidence_aggregator import (
    EvidenceAggregator,
)
from agents.hypothesis_agent.schema import (
    AggregatedEvidence,
    DependencyFindings,
    EvidenceSource,
    EvidenceStrength,
    HypothesisAgentInput,
    LogFindings,
    MetricFindings,
    Severity,
)


# ═══════════════════════════════════════════════════════════════
#  FIXTURES
# ═══════════════════════════════════════════════════════════════


def _make_input(**kwargs) -> HypothesisAgentInput:
    """Helper to build HypothesisAgentInput with defaults."""
    return HypothesisAgentInput(**kwargs)


def _make_log_findings(**kwargs) -> LogFindings:
    return LogFindings(**kwargs)


def _make_metric_findings(**kwargs) -> MetricFindings:
    return MetricFindings(**kwargs)


def _make_dependency_findings(**kwargs) -> DependencyFindings:
    return DependencyFindings(**kwargs)


@pytest.fixture
def aggregator() -> EvidenceAggregator:
    return EvidenceAggregator()


@pytest.fixture
def rich_input() -> HypothesisAgentInput:
    """Input with evidence from all 3 agents."""
    return _make_input(
        log_findings=_make_log_findings(
            suspicious_services=[
                {
                    "service": "payment-service",
                    "error_count": 55,
                    "severity_hint": "high",
                    "error_keywords_detected": ["timeout", "connection"],
                },
                {
                    "service": "order-service",
                    "error_count": 12,
                    "severity_hint": "medium",
                    "error_keywords_detected": [],
                },
            ],
            total_error_logs=67,
            dominant_service="payment-service",
            system_wide_spike=True,
            potential_upstream_failure=True,
            database_errors_detected=True,
            error_keywords=["timeout", "connection", "database"],
        ),
        metric_findings=_make_metric_findings(
            anomalous_metrics=[
                {
                    "metric_name": "db_connection_pool_usage",
                    "zscore": 4.5,
                    "severity": "critical",
                    "anomaly_type": "spike",
                },
                {
                    "metric_name": "memory_usage",
                    "zscore": 1.5,
                    "severity": "low",
                    "anomaly_type": "drift",
                },
            ],
            correlations=[
                {
                    "metric_1": "cpu_usage",
                    "metric_2": "latency",
                    "correlation_coefficient": 0.85,
                },
            ],
            total_anomalies=2,
            critical_anomalies=1,
            resource_saturation=True,
            cascading_degradation=False,
        ),
        dependency_findings=_make_dependency_findings(
            failed_service="payment-service",
            blast_radius_count=5,
            is_cascading=True,
            cascade_pattern="breadth_first",
            single_points_of_failure=["payment-service"],
            bottleneck_services=["api-gateway"],
            graph_has_cycles=True,
        ),
    )


# ═══════════════════════════════════════════════════════════════
#  TESTS: Log Evidence Extraction
# ═══════════════════════════════════════════════════════════════


class TestLogEvidenceExtraction:
    """Test evidence extraction from log findings."""

    def test_suspicious_services_create_evidence(
        self, aggregator: EvidenceAggregator
    ):
        inp = _make_input(
            log_findings=_make_log_findings(
                suspicious_services=[
                    {
                        "service": "svc-a",
                        "error_count": 60,
                        "severity_hint": "critical",
                        "error_keywords_detected": ["timeout"],
                    }
                ],
            )
        )
        result = aggregator.aggregate(inp)
        assert result.total_evidence_count >= 1
        assert EvidenceSource.LOG_AGENT in result.sources_represented

        # High error count + critical → STRONG
        log_items = [
            e for e in result.evidence_items
            if e.source == EvidenceSource.LOG_AGENT
        ]
        assert any(e.strength == EvidenceStrength.STRONG for e in log_items)

    def test_system_wide_spike_evidence(
        self, aggregator: EvidenceAggregator
    ):
        inp = _make_input(
            log_findings=_make_log_findings(system_wide_spike=True)
        )
        result = aggregator.aggregate(inp)
        desc = [e.description.lower() for e in result.evidence_items]
        assert any("system-wide" in d for d in desc)

    def test_upstream_failure_evidence(
        self, aggregator: EvidenceAggregator
    ):
        inp = _make_input(
            log_findings=_make_log_findings(
                potential_upstream_failure=True
            )
        )
        result = aggregator.aggregate(inp)
        desc = [e.description.lower() for e in result.evidence_items]
        assert any("upstream" in d for d in desc)

    def test_database_errors_evidence(
        self, aggregator: EvidenceAggregator
    ):
        inp = _make_input(
            log_findings=_make_log_findings(
                database_errors_detected=True
            )
        )
        result = aggregator.aggregate(inp)
        desc = [e.description.lower() for e in result.evidence_items]
        assert any("database" in d for d in desc)


# ═══════════════════════════════════════════════════════════════
#  TESTS: Metric Evidence Extraction
# ═══════════════════════════════════════════════════════════════


class TestMetricEvidenceExtraction:
    """Test evidence extraction from metric findings."""

    def test_anomalous_metrics_strength_by_zscore(
        self, aggregator: EvidenceAggregator
    ):
        inp = _make_input(
            metric_findings=_make_metric_findings(
                anomalous_metrics=[
                    {
                        "metric_name": "cpu",
                        "zscore": 4.0,
                        "severity": "high",
                        "anomaly_type": "spike",
                    }
                ],
            )
        )
        result = aggregator.aggregate(inp)
        metric_items = [
            e for e in result.evidence_items
            if e.source == EvidenceSource.METRICS_AGENT
        ]
        assert len(metric_items) >= 1
        assert metric_items[0].strength == EvidenceStrength.STRONG

    def test_resource_saturation_evidence(
        self, aggregator: EvidenceAggregator
    ):
        inp = _make_input(
            metric_findings=_make_metric_findings(
                resource_saturation=True
            )
        )
        result = aggregator.aggregate(inp)
        desc = [e.description.lower() for e in result.evidence_items]
        assert any("saturation" in d for d in desc)

    def test_cascading_degradation_evidence(
        self, aggregator: EvidenceAggregator
    ):
        inp = _make_input(
            metric_findings=_make_metric_findings(
                cascading_degradation=True
            )
        )
        result = aggregator.aggregate(inp)
        desc = [e.description.lower() for e in result.evidence_items]
        assert any("cascading" in d for d in desc)


# ═══════════════════════════════════════════════════════════════
#  TESTS: Dependency Evidence Extraction
# ═══════════════════════════════════════════════════════════════


class TestDependencyEvidenceExtraction:
    """Test evidence extraction from dependency findings."""

    def test_failed_service_evidence(
        self, aggregator: EvidenceAggregator
    ):
        inp = _make_input(
            dependency_findings=_make_dependency_findings(
                failed_service="db-service"
            )
        )
        result = aggregator.aggregate(inp)
        dep_items = [
            e for e in result.evidence_items
            if e.source == EvidenceSource.DEPENDENCY_AGENT
        ]
        assert len(dep_items) >= 1
        assert dep_items[0].strength == EvidenceStrength.STRONG

    def test_cascading_failure_evidence(
        self, aggregator: EvidenceAggregator
    ):
        inp = _make_input(
            dependency_findings=_make_dependency_findings(
                is_cascading=True,
                cascade_pattern="breadth_first",
                blast_radius_count=10,
            )
        )
        result = aggregator.aggregate(inp)
        desc = [e.description.lower() for e in result.evidence_items]
        assert any("cascading" in d for d in desc)

    def test_spof_evidence(
        self, aggregator: EvidenceAggregator
    ):
        inp = _make_input(
            dependency_findings=_make_dependency_findings(
                single_points_of_failure=["gateway"]
            )
        )
        result = aggregator.aggregate(inp)
        desc = [e.description.lower() for e in result.evidence_items]
        assert any("single point" in d for d in desc)

    def test_graph_cycles_evidence(
        self, aggregator: EvidenceAggregator
    ):
        inp = _make_input(
            dependency_findings=_make_dependency_findings(
                graph_has_cycles=True
            )
        )
        result = aggregator.aggregate(inp)
        desc = [e.description.lower() for e in result.evidence_items]
        assert any("cycle" in d for d in desc)


# ═══════════════════════════════════════════════════════════════
#  TESTS: Cross-Agent Correlations
# ═══════════════════════════════════════════════════════════════


class TestCrossAgentCorrelation:
    """Test cross-agent correlation detection."""

    def test_same_service_in_multiple_agents(
        self, aggregator: EvidenceAggregator, rich_input: HypothesisAgentInput
    ):
        """'payment-service' appears in both log and dep findings."""
        result = aggregator.aggregate(rich_input)
        assert len(result.correlations) >= 1

    def test_db_log_and_metric_correlation(
        self, aggregator: EvidenceAggregator, rich_input: HypothesisAgentInput
    ):
        """DB errors in logs + strong metric anomaly → correlation."""
        result = aggregator.aggregate(rich_input)
        db_corr = [
            c for c in result.correlations
            if "database" in c.description.lower()
        ]
        assert len(db_corr) >= 1

    def test_cascade_and_spike_correlation(
        self, aggregator: EvidenceAggregator, rich_input: HypothesisAgentInput
    ):
        """Cascading failure + system-wide spike → correlation."""
        result = aggregator.aggregate(rich_input)
        cascade_corr = [
            c for c in result.correlations
            if "cascading" in c.description.lower()
        ]
        assert len(cascade_corr) >= 1

    def test_no_correlation_single_source(
        self, aggregator: EvidenceAggregator
    ):
        """No correlations when only one source is present."""
        inp = _make_input(
            log_findings=_make_log_findings(
                suspicious_services=[
                    {"service": "svc", "error_count": 5}
                ],
            )
        )
        result = aggregator.aggregate(inp)
        assert len(result.correlations) == 0


# ═══════════════════════════════════════════════════════════════
#  TESTS: Evidence Capping & Metadata
# ═══════════════════════════════════════════════════════════════


class TestEvidenceCapping:
    """Test evidence count limits and metadata."""

    def test_evidence_cap(self):
        config = HypothesisAgentConfig(
            performance=PerformanceConfig(max_evidence_items=3)
        )
        agg = EvidenceAggregator(config)
        inp = _make_input(
            log_findings=_make_log_findings(
                suspicious_services=[
                    {"service": f"svc-{i}", "error_count": 10}
                    for i in range(10)
                ],
            )
        )
        result = agg.aggregate(inp)
        assert result.total_evidence_count <= 3

    def test_aggregated_evidence_fields(
        self, aggregator: EvidenceAggregator, rich_input: HypothesisAgentInput
    ):
        result = aggregator.aggregate(rich_input)
        assert result.total_evidence_count > 0
        assert result.aggregation_latency_ms >= 0
        assert isinstance(result.sources_represented, list)
        assert isinstance(result.dominant_severity, Severity)

    def test_empty_input_returns_empty_evidence(
        self, aggregator: EvidenceAggregator
    ):
        inp = _make_input()
        result = aggregator.aggregate(inp)
        assert result.total_evidence_count == 0
        assert len(result.evidence_items) == 0
        assert result.dominant_severity == Severity.MEDIUM
