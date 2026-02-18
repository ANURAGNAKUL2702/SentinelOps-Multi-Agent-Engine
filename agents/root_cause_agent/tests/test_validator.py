"""
Tests for validator.py — 30 validation checks.
"""

import pytest

from agents.root_cause_agent.schema import (
    Evidence,
    EvidenceSourceAgent,
    EvidenceType,
    ImpactAssessment,
    RootCauseVerdict,
    ValidationSeverity,
    VerdictMetadata,
)
from agents.root_cause_agent.validator import VerdictValidator


def _make_valid_verdict(**kwargs) -> RootCauseVerdict:
    """Build a verdict that passes all 30 checks."""
    defaults = dict(
        root_cause="Database connection pool exhausted in payment-service",
        confidence=0.85,
        evidence_trail=[
            Evidence(
                source=EvidenceSourceAgent.LOG_AGENT,
                evidence_type=EvidenceType.DIRECT,
                description="Error spike in payment-service",
                confidence=0.8,
            ),
            Evidence(
                source=EvidenceSourceAgent.METRICS_AGENT,
                evidence_type=EvidenceType.CORRELATED,
                description="Latency anomaly",
                confidence=0.75,
            ),
        ],
        reasoning=(
            "Multiple agents corroborate database connection pool "
            "exhaustion as the root cause of the incident."
        ),
        affected_services=["payment-service", "order-service"],
        impact=ImpactAssessment(
            affected_services=["payment-service", "order-service"],
            affected_count=2,
            severity_score=0.7,
            blast_radius=3,
        ),
        pipeline_latency_ms=50.0,
        metadata=VerdictMetadata(used_llm=False, used_fallback=True),
    )
    defaults.update(kwargs)
    return RootCauseVerdict(**defaults)


class TestVerdictValidator:
    def test_valid_verdict_passes(self):
        validator = VerdictValidator()
        result = validator.validate(_make_valid_verdict())
        assert result.validation_passed is True

    def test_empty_root_cause_fails(self):
        validator = VerdictValidator()
        result = validator.validate(_make_valid_verdict(root_cause=""))
        assert result.validation_passed is False

    def test_generic_root_cause_fails(self):
        validator = VerdictValidator()
        result = validator.validate(_make_valid_verdict(root_cause="unknown"))
        assert result.validation_passed is False

    def test_no_evidence_fails(self):
        validator = VerdictValidator()
        result = validator.validate(_make_valid_verdict(evidence_trail=[]))
        assert result.validation_passed is False

    def test_short_reasoning_warning(self):
        validator = VerdictValidator()
        result = validator.validate(
            _make_valid_verdict(reasoning="short")
        )
        # Short reasoning is a warning, not critical
        all_issues = result.errors + result.warnings
        warning_names = [e.check_name for e in all_issues]
        assert "reasoning_min_length" in warning_names

    def test_high_confidence_few_evidence_warning(self):
        validator = VerdictValidator()
        result = validator.validate(
            _make_valid_verdict(
                confidence=0.95,
                evidence_trail=[
                    Evidence(
                        source=EvidenceSourceAgent.LOG_AGENT,
                        evidence_type=EvidenceType.DIRECT,
                        description="only one",
                        confidence=0.9,
                    ),
                ],
            )
        )
        all_issues = result.errors + result.warnings
        check_names = [e.check_name for e in all_issues]
        assert "high_confidence_needs_evidence" in check_names

    def test_agent_identifier_check(self):
        validator = VerdictValidator()
        # Default agent should be "root_cause_agent"
        result = validator.validate(_make_valid_verdict())
        critical_names = [e.check_name for e in result.errors]
        assert "agent_identifier" not in critical_names

    def test_impact_count_consistency(self):
        validator = VerdictValidator()
        result = validator.validate(
            _make_valid_verdict(
                impact=ImpactAssessment(
                    affected_services=["a"],
                    affected_count=5,  # inconsistent
                    severity_score=0.5,
                    blast_radius=5,
                ),
            )
        )
        all_issues = result.errors + result.warnings
        check_names = [e.check_name for e in all_issues]
        assert "impact_count_consistency" in check_names

    def test_total_checks_count(self):
        validator = VerdictValidator()
        result = validator.validate(_make_valid_verdict())
        assert result.total_checks == 30
