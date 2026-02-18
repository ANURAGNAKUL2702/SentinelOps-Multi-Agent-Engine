"""Tests for core/escalation_calculator.py — Algorithm 8."""

from __future__ import annotations

from agents.incident_commander_agent.core.escalation_calculator import (
    calculate_escalation,
)
from agents.incident_commander_agent.config import EscalationConfig
from agents.incident_commander_agent.schema import BlastRadius
from agents.incident_commander_agent.tests.conftest import (
    make_validation_report,
    make_verdict,
)
from agents.validation_agent.schema import (
    Hallucination,
    HallucinationType,
)


class TestEscalationCalculator:
    def test_high_confidence_no_escalation(self):
        verdict = make_verdict(confidence=0.9)
        report = make_validation_report(accuracy_score=0.9)
        decision = calculate_escalation(verdict, report)
        assert decision.should_escalate is False
        assert decision.auto_resolve_confidence > 0.5

    def test_low_confidence_triggers_escalation(self):
        verdict = make_verdict(confidence=0.3)
        report = make_validation_report(accuracy_score=0.9)
        decision = calculate_escalation(verdict, report)
        assert decision.should_escalate is True
        assert "confidence" in decision.reason.lower()

    def test_low_accuracy_triggers_escalation(self):
        verdict = make_verdict(confidence=0.9)
        report = make_validation_report(accuracy_score=0.4)
        decision = calculate_escalation(verdict, report)
        assert decision.should_escalate is True
        assert "accuracy" in decision.reason.lower()

    def test_high_user_impact_triggers_escalation(self):
        verdict = make_verdict(confidence=0.9)
        report = make_validation_report(accuracy_score=0.9)
        br = BlastRadius(estimated_users=20_000)
        decision = calculate_escalation(verdict, report, br)
        assert decision.should_escalate is True
        assert "user impact" in decision.reason.lower() or "impact" in decision.reason.lower()

    def test_hallucinations_trigger_escalation(self):
        verdict = make_verdict(confidence=0.9)
        report = make_validation_report(
            accuracy_score=0.9,
            hallucinations=[
                Hallucination(
                    hallucination_type=HallucinationType.SERVICE,
                    description="fake",
                ),
            ],
        )
        decision = calculate_escalation(verdict, report)
        assert decision.should_escalate is True
        assert "hallucination" in decision.reason.lower()

    def test_unknown_root_cause_triggers_escalation(self):
        verdict = make_verdict(root_cause="unknown")
        report = make_validation_report(accuracy_score=0.9)
        decision = calculate_escalation(verdict, report)
        assert decision.should_escalate is True

    def test_auto_resolve_confidence_range(self):
        verdict = make_verdict(confidence=0.9)
        report = make_validation_report(accuracy_score=0.9)
        decision = calculate_escalation(verdict, report)
        assert 0.0 <= decision.auto_resolve_confidence <= 1.0

    def test_vp_escalation_for_large_impact(self):
        verdict = make_verdict(confidence=0.3)
        report = make_validation_report(accuracy_score=0.3)
        br = BlastRadius(estimated_users=100_000)
        decision = calculate_escalation(verdict, report, br)
        assert "VP" in decision.suggested_escalation_path
