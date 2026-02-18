"""
File: tests/test_performance.py
Purpose: Performance benchmark — 1000 metrics in <100ms.

Asserts the deterministic pipeline processes 1000 metrics within
the performance budget specified in the architecture.
"""

from __future__ import annotations

import time
import unittest

from agents.metrics_agent.agent import MetricsAgent
from agents.metrics_agent.config import MetricsAgentConfig
from agents.metrics_agent.schema import BaselineStats, MetricsAnalysisInput


class TestPerformanceBenchmark(unittest.TestCase):
    """1000-metric performance benchmark."""

    def setUp(self) -> None:
        self.config = MetricsAgentConfig()
        self.agent = MetricsAgent(config=self.config)

    def test_1000_metrics_deterministic_under_100ms(self) -> None:
        """Deterministic pipeline: 1000 metrics < 100ms."""
        metrics = {}
        baseline = {}
        for i in range(1000):
            name = f"metric_{i}"
            # Mix of normal and anomalous values
            if i % 10 == 0:
                metrics[name] = [50.0, 55.0, 60.0, 85.0, 95.0]
            else:
                metrics[name] = [50.0, 51.0, 49.0, 50.5, 50.2]
            baseline[name] = BaselineStats(mean=50.0, stddev=10.0)

        input_data = MetricsAnalysisInput(
            service="benchmark-service",
            time_window="2026-02-13T10:00Z to 2026-02-13T10:15Z",
            metrics=metrics,
            baseline=baseline,
        )

        start = time.perf_counter()
        output = self.agent.analyze(input_data)
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.assertLess(
            elapsed_ms, 100.0,
            f"Deterministic pipeline took {elapsed_ms:.2f}ms (limit: 100ms)",
        )
        self.assertEqual(
            output.system_summary.total_metrics_analyzed, 1000
        )
        self.assertGreater(
            output.system_summary.total_anomalies_detected, 0
        )

    def test_aggregator_latency(self) -> None:
        """MetricAggregator alone: 1000 metrics < 50ms."""
        from agents.metrics_agent.core.metric_aggregator import MetricAggregator

        agg = MetricAggregator(MetricsAgentConfig())
        metrics = {
            f"metric_{i}": [float(j) for j in range(5)]
            for i in range(1000)
        }
        baseline = {
            f"metric_{i}": BaselineStats(mean=2.0, stddev=1.0)
            for i in range(1000)
        }
        input_data = MetricsAnalysisInput(
            service="benchmark-service",
            time_window="2026-02-13T10:00Z to 2026-02-13T10:15Z",
            metrics=metrics,
            baseline=baseline,
        )

        start = time.perf_counter()
        result = agg.aggregate(input_data)
        elapsed_ms = (time.perf_counter() - start) * 1000

        self.assertLess(
            elapsed_ms, 50.0,
            f"Aggregator took {elapsed_ms:.2f}ms (limit: 50ms)",
        )
        self.assertEqual(result.total_metrics_analyzed, 1000)


if __name__ == "__main__":
    unittest.main()
