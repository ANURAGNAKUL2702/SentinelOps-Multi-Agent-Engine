"""Tests for reporting.analysis.cost_analyzer."""

from __future__ import annotations

import pytest

from reporting.analysis.cost_analyzer import CostAnalyzer, CostSummary


@pytest.fixture
def analyzer() -> CostAnalyzer:
    return CostAnalyzer(outlier_multiplier=2.0)


class TestCostAnalyzer:
    def test_empty_records(self, analyzer: CostAnalyzer) -> None:
        result = analyzer.analyze([])
        assert result.total_cost == 0.0
        assert result.cost_by_agent == {}

    def test_single_record(self, analyzer: CostAnalyzer) -> None:
        records = [{"agent_name": "log_agent", "cost": 0.01, "tokens_input": 100, "tokens_output": 50}]
        result = analyzer.analyze(records)
        assert result.total_cost == 0.01
        assert result.cost_by_agent["log_agent"] == 0.01

    def test_multiple_agents(self, analyzer: CostAnalyzer) -> None:
        records = [
            {"agent_name": "log_agent", "cost": 0.01, "tokens_input": 100, "tokens_output": 50},
            {"agent_name": "rca_agent", "cost": 0.04, "tokens_input": 500, "tokens_output": 200},
            {"agent_name": "log_agent", "cost": 0.005, "tokens_input": 50, "tokens_output": 25},
        ]
        result = analyzer.analyze(records)
        assert result.total_cost == pytest.approx(0.055)
        assert result.cost_by_agent["log_agent"] == pytest.approx(0.015)
        assert result.cost_by_agent["rca_agent"] == pytest.approx(0.04)

    def test_outlier_detection(self, analyzer: CostAnalyzer) -> None:
        records = [
            {"agent_name": "a", "cost": 0.01, "tokens_input": 100, "tokens_output": 0},
            {"agent_name": "b", "cost": 0.01, "tokens_input": 100, "tokens_output": 0},
            {"agent_name": "c", "cost": 0.01, "tokens_input": 100, "tokens_output": 0},
            {"agent_name": "d", "cost": 0.02, "tokens_input": 100, "tokens_output": 0},
            {"agent_name": "e", "cost": 0.01, "tokens_input": 100, "tokens_output": 0},
            {"agent_name": "f", "cost": 0.02, "tokens_input": 100, "tokens_output": 0},
            {"agent_name": "g", "cost": 0.01, "tokens_input": 100, "tokens_output": 0},
            {"agent_name": "h", "cost": 0.01, "tokens_input": 100, "tokens_output": 0},
            {"agent_name": "outlier", "cost": 5.0, "tokens_input": 10000, "tokens_output": 0},
        ]
        result = analyzer.analyze(records)
        assert len(result.outliers) >= 1

    def test_cost_per_token(self, analyzer: CostAnalyzer) -> None:
        records = [{"agent_name": "a", "cost": 0.10, "tokens_input": 1000, "tokens_output": 0}]
        result = analyzer.analyze(records)
        assert result.cost_per_token == pytest.approx(0.0001)

    def test_frozen_summary(self, analyzer: CostAnalyzer) -> None:
        result = analyzer.analyze([])
        with pytest.raises(Exception):
            result.total_cost = 999  # type: ignore[misc]

    def test_compare_runs(self, analyzer: CostAnalyzer) -> None:
        prev = CostSummary(
            total_cost=0.10, avg_cost=0.05, max_cost=0.08, min_cost=0.02,
            std_dev=0.02, cost_by_agent={}, outlier_threshold=0.1,
            outliers=[], cost_per_token=0.0001,
        )
        curr = CostSummary(
            total_cost=0.15, avg_cost=0.075, max_cost=0.10, min_cost=0.03,
            std_dev=0.03, cost_by_agent={}, outlier_threshold=0.15,
            outliers=[], cost_per_token=0.0001,
        )
        delta = analyzer.compare_runs(curr, prev)
        assert delta["total_delta"] == pytest.approx(0.05)
        assert delta["pct_change"] == pytest.approx(50.0)
