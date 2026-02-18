"""Tests for reporting.analysis.performance_analyzer."""

from __future__ import annotations

import pytest

from reporting.analysis.performance_analyzer import PerformanceAnalyzer


@pytest.fixture
def analyzer() -> PerformanceAnalyzer:
    return PerformanceAnalyzer(z_threshold=2.0)


class TestPerformanceAnalyzer:
    def test_empty_executions(self, analyzer: PerformanceAnalyzer) -> None:
        result = analyzer.analyze([])
        assert result.avg_latency == 0.0
        assert result.anomalies == []

    def test_single_execution(self, analyzer: PerformanceAnalyzer) -> None:
        execs = [{"agent_name": "a", "duration": 2.0}]
        result = analyzer.analyze(execs, total_duration=2.0)
        assert result.avg_latency == 2.0
        assert result.p50_latency == 2.0

    def test_percentiles(self, analyzer: PerformanceAnalyzer) -> None:
        execs = [{"agent_name": f"a{i}", "duration": float(i)} for i in range(1, 101)]
        result = analyzer.analyze(execs, total_duration=10.0)
        assert result.p50_latency == pytest.approx(50.5, rel=0.05)
        assert result.p95_latency > result.p50_latency

    def test_anomaly_detection(self, analyzer: PerformanceAnalyzer) -> None:
        execs = [
            {"agent_name": "normal1", "duration": 1.0},
            {"agent_name": "normal2", "duration": 1.1},
            {"agent_name": "normal3", "duration": 0.9},
            {"agent_name": "normal4", "duration": 1.0},
            {"agent_name": "normal5", "duration": 1.05},
            {"agent_name": "normal6", "duration": 0.95},
            {"agent_name": "normal7", "duration": 1.02},
            {"agent_name": "normal8", "duration": 0.98},
            {"agent_name": "slow", "duration": 100.0},
        ]
        result = analyzer.analyze(execs)
        assert len(result.anomalies) >= 1
        assert any(a["agent_name"] == "slow" for a in result.anomalies)

    def test_parallel_speedup(self, analyzer: PerformanceAnalyzer) -> None:
        execs = [
            {"agent_name": "a", "duration": 3.0},
            {"agent_name": "b", "duration": 4.0},
        ]
        result = analyzer.analyze(execs, total_duration=4.0)
        assert result.parallel_speedup == 1.75

    def test_agent_latencies_map(self, analyzer: PerformanceAnalyzer) -> None:
        execs = [
            {"agent_name": "x", "duration": 1.5},
            {"agent_name": "y", "duration": 2.5},
        ]
        result = analyzer.analyze(execs)
        assert result.agent_latencies["x"] == 1.5
        assert result.agent_latencies["y"] == 2.5
