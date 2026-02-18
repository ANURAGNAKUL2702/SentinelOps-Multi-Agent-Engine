"""Tests for fallback.py — deterministic fallback pipeline."""

from __future__ import annotations

import time

from agents.incident_commander_agent.fallback import DeterministicFallback
from agents.incident_commander_agent.tests.conftest import (
    make_validation_report,
    make_verdict,
)


class TestDeterministicFallback:
    def test_produces_complete_response(self):
        fb = DeterministicFallback()
        verdict = make_verdict()
        report = make_validation_report()
        response = fb.execute(verdict, report, "corr-001")

        assert response.agent == "incident_commander_agent"
        assert response.root_cause_summary != ""
        assert response.classification_source == "fallback"
        assert len(response.runbook.steps) > 0
        assert len(response.action_items) > 0
        assert len(response.prevention_recommendations) > 0
        assert response.correlation_id == "corr-001"

    def test_fallback_under_100ms(self):
        fb = DeterministicFallback()
        verdict = make_verdict()
        report = make_validation_report()

        start = time.perf_counter()
        response = fb.execute(verdict, report)
        elapsed = (time.perf_counter() - start) * 1000

        assert elapsed < 100, f"Fallback took {elapsed:.1f}ms (>100ms)"

    def test_escalation_for_low_confidence(self):
        fb = DeterministicFallback()
        verdict = make_verdict(confidence=0.2)
        report = make_validation_report()
        response = fb.execute(verdict, report)
        assert response.escalation_decision.should_escalate is True

    def test_unknown_root_cause(self):
        fb = DeterministicFallback()
        verdict = make_verdict(root_cause="totally_unknown_thing")
        report = make_validation_report()
        response = fb.execute(verdict, report)
        assert len(response.runbook.steps) > 0

    def test_metadata_populated(self):
        fb = DeterministicFallback()
        verdict = make_verdict()
        report = make_validation_report()
        response = fb.execute(verdict, report)
        assert response.metadata is not None
        assert response.metadata.used_fallback is True
        assert response.metadata.used_llm is False
        assert response.metadata.total_pipeline_ms > 0
