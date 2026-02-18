"""
File: tests/test_metric_agent.py
Purpose: Integration tests for the MetricsAgent pipeline.

Test Cases:
  1. Empty input  → 0 anomalies, confidence < 0.3
  2. Single anomaly → correct detection + severity
  3. Correlated anomalies → correlations detected
  4. 1000-metric benchmark → <100ms deterministic
  5. Division-by-zero edge case → no crash
"""

from __future__ import annotations

import time
import unittest

from agents.metrics_agent.agent import MetricsAgent
from agents.metrics_agent.config import MetricsAgentConfig
from agents.metrics_agent.schema import (
    AnomalyType,
    BaselineStats,
    MetricsAnalysisInput,
    Severity,
    TrendType,
)


class TestMetricsAgentEmpty(unittest.TestCase):
    """Test Case 1: Empty input → 0 anomalies, confidence < 0.3."""

    def setUp(self) -> None:
        self.config = MetricsAgentConfig()
        self.agent = MetricsAgent(config=self.config)

    def test_empty_metrics(self) -> None:
        """Empty metrics dict → 0 anomalies, low confidence."""
        input_data = MetricsAnalysisInput(
            service="test-service",
            time_window="2026-02-13T10:00Z to 2026-02-13T10:15Z",
            metrics={},
            baseline={},
        )
        output = self.agent.analyze(input_data)

        self.assertEqual(output.agent, "metrics_agent")
        self.assertEqual(len(output.anomalous_metrics), 0)
        self.assertLess(output.confidence_score, 0.3)
        self.assertEqual(output.classification_source, "deterministic")
        self.assertIsNotNone(output.metadata)
        self.assertIsNotNone(output.validation)
        self.assertTrue(output.validation.validation_passed)

    def test_stable_metrics_no_anomalies(self) -> None:
        """Stable metrics within baseline → 0 anomalies."""
        input_data = MetricsAnalysisInput(
            service="test-service",
            time_window="2026-02-13T10:00Z to 2026-02-13T10:15Z",
            metrics={
                "cpu_percent": [50.1, 50.2, 50.0, 49.8, 50.3],
            },
            baseline={
                "cpu_percent": BaselineStats(mean=50.0, stddev=10.0),
            },
        )
        output = self.agent.analyze(input_data)

        self.assertEqual(len(output.anomalous_metrics), 0)
        self.assertLess(output.confidence_score, 0.3)


class TestMetricsAgentSingleAnomaly(unittest.TestCase):
    """Test Case 2: Single anomaly → correct detection + severity."""

    def setUp(self) -> None:
        self.config = MetricsAgentConfig()
        self.agent = MetricsAgent(config=self.config)

    def test_cpu_spike_detected(self) -> None:
        """CPU spike from 50 mean to 95.2 → anomalous, high/critical."""
        input_data = MetricsAnalysisInput(
            service="payment-api",
            time_window="2026-02-13T10:00Z to 2026-02-13T10:15Z",
            metrics={
                "cpu_percent": [45.2, 48.1, 52.3, 87.6, 95.2],
            },
            baseline={
                "cpu_percent": BaselineStats(mean=50.0, stddev=10.0),
            },
        )
        output = self.agent.analyze(input_data)

        self.assertEqual(output.agent, "metrics_agent")
        self.assertGreaterEqual(len(output.anomalous_metrics), 1)

        cpu_anomaly = next(
            (m for m in output.anomalous_metrics
             if m.metric_name == "cpu_percent"),
            None,
        )
        self.assertIsNotNone(cpu_anomaly)
        self.assertTrue(cpu_anomaly.is_anomalous)
        self.assertGreater(abs(cpu_anomaly.zscore), 3.0)
        self.assertTrue(cpu_anomaly.threshold_breached)
        self.assertIn(
            cpu_anomaly.severity,
            [Severity.CRITICAL, Severity.HIGH],
        )
        self.assertGreater(output.confidence_score, 0.3)

    def test_latency_spike_detected(self) -> None:
        """P99 latency spike → anomalous."""
        input_data = MetricsAnalysisInput(
            service="payment-api",
            time_window="2026-02-13T10:00Z to 2026-02-13T10:15Z",
            metrics={
                "api_latency_p99_ms": [100.0, 110.0, 120.0, 500.0, 1500.0],
            },
            baseline={
                "api_latency_p99_ms": BaselineStats(mean=120.0, stddev=30.0),
            },
        )
        output = self.agent.analyze(input_data)

        self.assertGreaterEqual(len(output.anomalous_metrics), 1)
        lat = next(
            (m for m in output.anomalous_metrics
             if m.metric_name == "api_latency_p99_ms"),
            None,
        )
        self.assertIsNotNone(lat)
        self.assertTrue(lat.is_anomalous)


class TestMetricsAgentCorrelatedAnomalies(unittest.TestCase):
    """Test Case 3: Correlated anomalies → correlations detected."""

    def setUp(self) -> None:
        self.config = MetricsAgentConfig()
        self.agent = MetricsAgent(config=self.config)

    def test_cpu_latency_correlation(self) -> None:
        """CPU and latency rising together → positive correlation."""
        input_data = MetricsAnalysisInput(
            service="payment-api",
            time_window="2026-02-13T10:00Z to 2026-02-13T10:15Z",
            metrics={
                "cpu_percent": [45.0, 55.0, 70.0, 85.0, 95.0],
                "api_latency_p99_ms": [100.0, 200.0, 400.0, 800.0, 1500.0],
            },
            baseline={
                "cpu_percent": BaselineStats(mean=50.0, stddev=10.0),
                "api_latency_p99_ms": BaselineStats(mean=120.0, stddev=30.0),
            },
        )
        output = self.agent.analyze(input_data)

        # Both should be anomalous
        self.assertGreaterEqual(len(output.anomalous_metrics), 2)

        # Should detect correlation between CPU and latency
        self.assertGreater(len(output.correlations), 0)

        # System summary should flag cascading degradation
        self.assertTrue(output.system_summary.cascading_degradation)
        self.assertGreater(output.confidence_score, 0.5)


class TestMetricsAgentPerformance(unittest.TestCase):
    """Test Case 4: 1000-metric benchmark → <100ms deterministic."""

    def setUp(self) -> None:
        self.config = MetricsAgentConfig()
        self.agent = MetricsAgent(config=self.config)

    def test_1000_metrics_under_100ms(self) -> None:
        """1000 metrics must complete in <100ms deterministic."""
        metrics = {}
        baseline = {}
        for i in range(1000):
            name = f"metric_{i}"
            metrics[name] = [float(j) for j in range(5)]
            baseline[name] = BaselineStats(mean=2.0, stddev=1.0)

        input_data = MetricsAnalysisInput(
            service="benchmark-service",
            time_window="2026-02-13T10:00Z to 2026-02-13T10:15Z",
            metrics=metrics,
            baseline=baseline,
        )

        start = time.perf_counter()
        output = self.agent.analyze(input_data)
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.assertEqual(output.agent, "metrics_agent")
        self.assertLess(elapsed_ms, 100.0, f"Took {elapsed_ms:.2f}ms (limit: 100ms)")
        self.assertEqual(
            output.system_summary.total_metrics_analyzed, 1000
        )


class TestMetricsAgentEdgeCases(unittest.TestCase):
    """Test Case 5: Division-by-zero and edge cases → no crash."""

    def setUp(self) -> None:
        self.config = MetricsAgentConfig()
        self.agent = MetricsAgent(config=self.config)

    def test_zero_stddev_no_crash(self) -> None:
        """Baseline stddev=0 → no division by zero."""
        input_data = MetricsAnalysisInput(
            service="edge-service",
            time_window="2026-02-13T10:00Z to 2026-02-13T10:15Z",
            metrics={
                "cpu_percent": [50.0, 50.0, 50.0, 50.0, 60.0],
            },
            baseline={
                "cpu_percent": BaselineStats(mean=50.0, stddev=0.0),
            },
        )
        output = self.agent.analyze(input_data)
        self.assertEqual(output.agent, "metrics_agent")
        # Should not crash — z-score returns 0.0 when stddev=0

    def test_zero_previous_value_no_crash(self) -> None:
        """Previous value=0 → growth_rate capped at 999.9."""
        input_data = MetricsAnalysisInput(
            service="edge-service",
            time_window="2026-02-13T10:00Z to 2026-02-13T10:15Z",
            metrics={
                "request_rate_per_sec": [0.0, 0.0, 0.0, 0.0, 100.0],
            },
            baseline={
                "request_rate_per_sec": BaselineStats(mean=0.0, stddev=0.0),
            },
        )
        output = self.agent.analyze(input_data)
        self.assertEqual(output.agent, "metrics_agent")
        # Should not crash

    def test_single_value_timeseries(self) -> None:
        """Single-element timeseries → no crash."""
        input_data = MetricsAnalysisInput(
            service="edge-service",
            time_window="2026-02-13T10:00Z to 2026-02-13T10:15Z",
            metrics={
                "cpu_percent": [95.0],
            },
            baseline={
                "cpu_percent": BaselineStats(mean=50.0, stddev=10.0),
            },
        )
        output = self.agent.analyze(input_data)
        self.assertEqual(output.agent, "metrics_agent")

    def test_validation_checks_executed(self) -> None:
        """All 23 validation checks executed."""
        input_data = MetricsAnalysisInput(
            service="test-service",
            time_window="2026-02-13T10:00Z to 2026-02-13T10:15Z",
            metrics={
                "cpu_percent": [45.2, 48.1, 52.3, 87.6, 95.2],
            },
            baseline={
                "cpu_percent": BaselineStats(mean=50.0, stddev=10.0),
            },
        )
        output = self.agent.analyze(input_data)

        self.assertIsNotNone(output.validation)
        self.assertEqual(output.validation.checks_executed, 23)

    def test_health_check(self) -> None:
        """health_check() returns valid structure."""
        health = self.agent.health_check()

        self.assertEqual(health["agent"], "metrics_agent")
        self.assertIn(health["status"], ["healthy", "degraded"])
        self.assertIn("components", health)
        self.assertIn("metrics", health)


class TestMetricsAgentValidation(unittest.TestCase):
    """Validate the 23-check system end-to-end."""

    def setUp(self) -> None:
        self.config = MetricsAgentConfig()
        self.agent = MetricsAgent(config=self.config)

    def test_output_passes_all_23_checks(self) -> None:
        """A valid output with anomalies should pass all 23 checks."""
        input_data = MetricsAnalysisInput(
            service="payment-api",
            time_window="2026-02-13T10:00Z to 2026-02-13T10:15Z",
            metrics={
                "cpu_percent": [45.0, 55.0, 70.0, 85.0, 95.0],
                "api_latency_p99_ms": [100.0, 200.0, 400.0, 800.0, 1500.0],
            },
            baseline={
                "cpu_percent": BaselineStats(mean=50.0, stddev=10.0),
                "api_latency_p99_ms": BaselineStats(mean=120.0, stddev=30.0),
            },
        )
        output = self.agent.analyze(input_data)

        self.assertIsNotNone(output.validation)
        self.assertEqual(output.validation.checks_executed, 23)
        self.assertTrue(
            output.validation.validation_passed,
            f"Validation failed: {[e.model_dump() for e in output.validation.errors]}",
        )

    def test_correlation_id_propagation(self) -> None:
        """Correlation ID is propagated through the pipeline."""
        cid = "test-correlation-123"
        input_data = MetricsAnalysisInput(
            service="test-service",
            time_window="2026-02-13T10:00Z to 2026-02-13T10:15Z",
            metrics={"cpu_percent": [50.0, 60.0, 70.0, 80.0, 90.0]},
            baseline={"cpu_percent": BaselineStats(mean=50.0, stddev=10.0)},
        )
        output = self.agent.analyze(input_data, correlation_id=cid)

        self.assertEqual(output.correlation_id, cid)
        self.assertEqual(output.metadata.correlation_id, cid)


if __name__ == "__main__":
    unittest.main()
