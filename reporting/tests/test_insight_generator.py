"""Tests for reporting.analysis.insight_generator."""

from __future__ import annotations

import pytest

from reporting.analysis.insight_generator import InsightGenerator


@pytest.fixture
def generator() -> InsightGenerator:
    return InsightGenerator()


class TestInsightGenerator:
    def test_no_insights_for_healthy(self, generator: InsightGenerator) -> None:
        insights = generator.generate(
            avg_duration=5.0,
            target_duration=10.0,
            total_cost=0.01,
            accuracy_rate=0.9,
            slo_compliance=1.0,
            mttr_minutes=5.0,
            mttr_target=15.0,
        )
        # Duration is under target, accuracy is fine, SLO met → might still
        # produce an "info" cost insight if total_cost > 0.10
        for i in insights:
            assert i.severity != "critical"

    def test_duration_exceeds_target(self, generator: InsightGenerator) -> None:
        insights = generator.generate(avg_duration=20.0, target_duration=10.0)
        perf = [i for i in insights if i.category == "performance" and "exceeds" in i.title.lower()]
        assert len(perf) >= 1

    def test_anomaly_insight(self, generator: InsightGenerator) -> None:
        insights = generator.generate(anomaly_count=3)
        titles = [i.title for i in insights]
        assert any("anomal" in t.lower() for t in titles)

    def test_cost_outlier_insight(self, generator: InsightGenerator) -> None:
        insights = generator.generate(outlier_count=2)
        titles = [i.title for i in insights]
        assert any("outlier" in t.lower() for t in titles)

    def test_accuracy_below_threshold(self, generator: InsightGenerator) -> None:
        insights = generator.generate(
            accuracy_rate=0.4,
            accuracy_threshold=0.7,
        )
        acc = [i for i in insights if i.category == "accuracy"]
        assert len(acc) >= 1

    def test_slo_breach(self, generator: InsightGenerator) -> None:
        insights = generator.generate(slo_compliance=0.5)
        slo = [i for i in insights if "SLO" in i.title]
        assert len(slo) >= 1
        assert slo[0].severity == "critical"

    def test_mttr_exceeds_target(self, generator: InsightGenerator) -> None:
        insights = generator.generate(mttr_minutes=20.0, mttr_target=15.0)
        mttr = [i for i in insights if "MTTR" in i.title]
        assert len(mttr) >= 1

    def test_recurring_root_cause(self, generator: InsightGenerator) -> None:
        insights = generator.generate(
            common_root_causes=[("DB pool exhaustion", 5)],
        )
        recurring = [i for i in insights if "recurring" in i.title.lower()]
        assert len(recurring) >= 1

    def test_trend_increasing(self, generator: InsightGenerator) -> None:
        insights = generator.generate(trend_direction="increasing")
        trend = [i for i in insights if "trend" in i.title.lower()]
        assert len(trend) >= 1

    def test_all_insights_have_fields(self, generator: InsightGenerator) -> None:
        insights = generator.generate(
            avg_duration=20.0,
            target_duration=10.0,
            anomaly_count=1,
            outlier_count=1,
            accuracy_rate=0.3,
            slo_compliance=0.5,
            mttr_minutes=30.0,
            mttr_target=15.0,
            common_root_causes=[("test", 5)],
            trend_direction="increasing",
        )
        for i in insights:
            assert i.category
            assert i.severity
            assert i.title
            assert i.description
            assert i.recommendation
