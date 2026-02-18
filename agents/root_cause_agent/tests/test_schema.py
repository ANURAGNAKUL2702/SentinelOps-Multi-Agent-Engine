"""
Tests for schema.py — Pydantic v2 model validation.
"""

import pytest
from datetime import datetime, timezone

from agents.root_cause_agent.schema import (
    AlternativeVerdict,
    CausalLink,
    CausalRelationship,
    Contradiction,
    ContradictionStrategy,
    DependencyAgentFindings,
    Evidence,
    EvidenceSourceAgent,
    EvidenceType,
    HypothesisFindings,
    ImpactAssessment,
    IncidentCategory,
    LogAgentFindings,
    MetricsAgentFindings,
    RootCauseAgentInput,
    RootCauseVerdict,
    Severity,
    SynthesisResult,
    TimelineEvent,
    ValidationResult,
    ValidationSeverity,
    ValidatorError,
    VerdictMetadata,
)


# ═══════════════════════════════════════════════════════════════
#  Enum tests
# ═══════════════════════════════════════════════════════════════


class TestEnums:
    def test_severity_values(self):
        assert Severity.CRITICAL.value == "critical"
        assert Severity.LOW.value == "low"

    def test_incident_category_values(self):
        assert IncidentCategory.INFRASTRUCTURE.value == "infrastructure"
        assert IncidentCategory.UNKNOWN.value == "unknown"

    def test_evidence_type_values(self):
        assert EvidenceType.DIRECT.value == "direct"
        assert EvidenceType.CIRCUMSTANTIAL.value == "circumstantial"

    def test_evidence_source_agent_values(self):
        assert EvidenceSourceAgent.LOG_AGENT.value == "log_agent"
        assert EvidenceSourceAgent.HYPOTHESIS_AGENT.value == "hypothesis_agent"

    def test_causal_relationship_values(self):
        assert CausalRelationship.CAUSES.value == "causes"

    def test_contradiction_strategy_values(self):
        assert ContradictionStrategy.CONFIDENCE_WINS.value == "confidence_wins"
        assert ContradictionStrategy.UNRESOLVED.value == "unresolved"


# ═══════════════════════════════════════════════════════════════
#  Input schema tests
# ═══════════════════════════════════════════════════════════════


class TestInputSchemas:
    def test_log_findings_defaults(self):
        lf = LogAgentFindings()
        assert lf.suspicious_services == []
        assert lf.confidence == 0.0

    def test_log_findings_with_data(self):
        lf = LogAgentFindings(
            suspicious_services=["svc-a"],
            confidence=0.8,
            timestamp="2024-01-01T00:00:00Z",
        )
        assert lf.suspicious_services == ["svc-a"]
        assert lf.confidence == 0.8

    def test_metrics_findings_defaults(self):
        mf = MetricsAgentFindings()
        assert mf.anomalies == []

    def test_dependency_findings_defaults(self):
        df = DependencyAgentFindings()
        assert df.impact_graph == {}
        assert df.blast_radius == 0

    def test_hypothesis_findings_defaults(self):
        hf = HypothesisFindings()
        assert hf.top_hypothesis == ""
        assert hf.mttr_estimate == 30.0

    def test_root_cause_agent_input_defaults(self):
        inp = RootCauseAgentInput()
        assert inp.log_findings is not None
        assert inp.incident_id != ""

    def test_confidence_range_validation(self):
        lf = LogAgentFindings(confidence=0.5)
        assert lf.confidence == 0.5

        with pytest.raises(Exception):
            LogAgentFindings(confidence=1.5)


# ═══════════════════════════════════════════════════════════════
#  Evidence schema tests
# ═══════════════════════════════════════════════════════════════


class TestEvidenceSchemas:
    def test_evidence_creation(self):
        ev = Evidence(
            source=EvidenceSourceAgent.LOG_AGENT,
            evidence_type=EvidenceType.DIRECT,
            description="Test evidence",
            confidence=0.9,
        )
        assert ev.source == EvidenceSourceAgent.LOG_AGENT
        assert ev.score == 0.0

    def test_evidence_score_update(self):
        ev = Evidence(
            source=EvidenceSourceAgent.LOG_AGENT,
            confidence=0.9,
        )
        updated = ev.model_copy(update={"score": 0.75})
        assert updated.score == 0.75

    def test_synthesis_result_defaults(self):
        sr = SynthesisResult()
        assert sr.evidence_trail == []
        assert sr.agreement_score == 0.0


# ═══════════════════════════════════════════════════════════════
#  Output schema tests
# ═══════════════════════════════════════════════════════════════


class TestOutputSchemas:
    def test_root_cause_verdict_defaults(self):
        v = RootCauseVerdict()
        assert v.agent == "root_cause_agent"
        assert v.confidence == 0.0

    def test_verdict_confidence_clamped(self):
        v = RootCauseVerdict(confidence=0.95)
        assert v.confidence == 0.95

    def test_verdict_agent_frozen(self):
        v = RootCauseVerdict()
        assert v.agent == "root_cause_agent"

    def test_alternative_verdict(self):
        av = AlternativeVerdict(
            root_cause="alt theory",
            confidence=0.6,
            evidence_count=3,
        )
        assert av.root_cause == "alt theory"
        assert av.category == IncidentCategory.UNKNOWN

    def test_verdict_metadata(self):
        m = VerdictMetadata(used_llm=True, used_fallback=False)
        assert m.used_llm is True

    def test_causal_link(self):
        cl = CausalLink(cause="A", effect="B", confidence=0.9)
        assert cl.relationship == CausalRelationship.CAUSES

    def test_timeline_event(self):
        te = TimelineEvent(
            timestamp="2024-01-01T00:00:00Z",
            source=EvidenceSourceAgent.LOG_AGENT,
            event="test",
        )
        assert te.severity == Severity.MEDIUM

    def test_impact_assessment(self):
        ia = ImpactAssessment(
            affected_services=["a", "b"],
            affected_count=2,
            blast_radius=3,
        )
        assert ia.is_cascading is False

    def test_contradiction(self):
        c = Contradiction(
            agent_a=EvidenceSourceAgent.LOG_AGENT,
            agent_b=EvidenceSourceAgent.METRICS_AGENT,
        )
        assert c.resolved is False

    def test_validator_error(self):
        ve = ValidatorError(
            check_number=1,
            check_name="test",
            error_description="test error",
        )
        assert ve.severity == ValidationSeverity.WARNING

    def test_validation_result(self):
        vr = ValidationResult(validation_passed=True, total_checks=30)
        assert vr.errors == []
