"""Tests for core/prevention_advisor.py — Algorithm 7."""

from __future__ import annotations

from agents.incident_commander_agent.core.prevention_advisor import (
    generate_prevention_recommendations,
)
from agents.incident_commander_agent.schema import PreventionCategory
from agents.root_cause_agent.schema import CausalLink
from agents.validation_agent.schema import (
    Hallucination,
    HallucinationType,
    ValidationReport,
)


class TestPreventionAdvisor:
    def test_base_recommendations_always_present(self):
        recs = generate_prevention_recommendations("some_failure")
        assert len(recs) >= 2

    def test_db_keyword_adds_recommendation(self):
        recs = generate_prevention_recommendations("database_failure")
        titles = [r.title for r in recs]
        assert any("database" in t.lower() for t in titles)

    def test_cascading_adds_circuit_breaker(self):
        recs = generate_prevention_recommendations("cascading_failure")
        titles = [r.title for r in recs]
        assert any("circuit break" in t.lower() for t in titles)

    def test_hallucinations_add_observability(self):
        report = ValidationReport(
            accuracy_score=0.9,
            hallucinations=[
                Hallucination(
                    hallucination_type=HallucinationType.METRIC,
                    description="Made up metric",
                ),
            ],
        )
        recs = generate_prevention_recommendations(
            "some_failure", validation_report=report,
        )
        titles = [r.title for r in recs]
        assert any("observability" in t.lower() for t in titles)

    def test_low_accuracy_adds_tooling_rec(self):
        report = ValidationReport(accuracy_score=0.5)
        recs = generate_prevention_recommendations(
            "some_failure", validation_report=report,
        )
        titles = [r.title for r in recs]
        assert any("tooling" in t.lower() or "analysis" in t.lower() for t in titles)

    def test_deduplication(self):
        recs = generate_prevention_recommendations("cascading_cascade_failure")
        titles = [r.title for r in recs]
        assert len(titles) == len(set(titles))
