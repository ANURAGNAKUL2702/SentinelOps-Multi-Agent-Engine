"""Tests for reporting.visualizations.cost_breakdown_chart."""

from __future__ import annotations

import base64

import pytest

from reporting.visualizations.cost_breakdown_chart import CostBreakdownChart


@pytest.fixture
def chart() -> CostBreakdownChart:
    return CostBreakdownChart()


class TestCostBreakdownChart:
    def test_pie_chart(self, chart: CostBreakdownChart) -> None:
        costs = {"log_agent": 0.01, "rca_agent": 0.04, "validator": 0.005}
        result = chart.generate_pie_chart(costs)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_bar_chart(self, chart: CostBreakdownChart) -> None:
        costs = {"log_agent": 0.01, "rca_agent": 0.04}
        result = chart.generate_bar_chart(costs)
        assert isinstance(result, str)
        assert len(result) > 100

    def test_zero_costs(self, chart: CostBreakdownChart) -> None:
        costs = {"a": 0.0, "b": 0.0}
        result = chart.generate_pie_chart(costs)
        assert isinstance(result, str)

    def test_empty_costs(self, chart: CostBreakdownChart) -> None:
        result = chart.generate_pie_chart({})
        assert isinstance(result, str)
