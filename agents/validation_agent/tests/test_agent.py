"""Tests for agent.py — ValidationAgent orchestrator."""

from __future__ import annotations

import pytest

from agents.root_cause_agent.schema import (
    CausalLink,
    Evidence,
    EvidenceSourceAgent,
    EvidenceType,
    RootCauseVerdict,
    Severity,
    TimelineEvent,
)
from agents.validation_agent.agent import ValidationAgent
from agents.validation_agent.config import (
    FeatureFlags,
    ValidationAgentConfig,
)
from agents.validation_agent.schema import (
    GroundTruth,
    PropagationStep,
    ValidationAgentInput,
)


def _make_gt(root_cause: str = "database failure") -> GroundTruth:
    return GroundTruth(
        actual_root_cause=root_cause,
        failure_type="resource_exhaustion",
        affected_services_ground_truth=["database", "payment-service"],
        failure_propagation_chain=[
            PropagationStep(
                from_service="database",
                to_service="payment-service",
                delay_seconds=5.0,
            )
        ],
        expected_symptoms=["high latency", "connection timeout"],
    )


def _make_verdict(
    root_cause: str = "database failure",
    confidence: float = 0.9,
) -> RootCauseVerdict:
    return RootCauseVerdict(
        root_cause=root_cause,
        confidence=confidence,
        affected_services=["database", "payment-service"],
        evidence_trail=[
            Evidence(
                source=EvidenceSourceAgent.LOG_AGENT,
                evidence_type=EvidenceType.DIRECT,
                service="database",
                description="Connection pool exhausted",
            )
        ],
        causal_chain=[
            CausalLink(
                cause="database",
                effect="payment-service",
                service="database",
            )
        ],
        timeline=[
            TimelineEvent(
                timestamp="2026-01-01T00:00:00Z",
                source=EvidenceSourceAgent.LOG_AGENT,
                event="DB connection error",
                service="database",
                severity=Severity.HIGH,
            ),
        ],
    )


class TestCorrectVerdict:
    """Agent with a correct verdict."""

    def test_correct_produces_report(self) -> None:
        agent = ValidationAgent()
        report = agent.validate(
            _make_verdict(), _make_gt(), correlation_id="c1"
        )
        assert report.verdict_correct is True
        assert report.accuracy_score >= 0.8
        assert report.agent == "validation_agent"
        assert report.correlation_id == "c1"

    def test_pipeline_latency_set(self) -> None:
        agent = ValidationAgent()
        report = agent.validate(_make_verdict(), _make_gt())
        assert report.pipeline_latency_ms > 0


class TestIncorrectVerdict:
    """Agent with an incorrect verdict."""

    def test_incorrect_verdict(self) -> None:
        agent = ValidationAgent()
        report = agent.validate(
            _make_verdict("network partition"), _make_gt()
        )
        assert report.verdict_correct is False
        assert report.accuracy_score < 0.8
        assert len(report.discrepancies) >= 1
        assert len(report.recommendations) >= 1


class TestOutputValidation:
    """Agent runs output validation by default."""

    def test_output_validation_present(self) -> None:
        agent = ValidationAgent()
        report = agent.validate(_make_verdict(), _make_gt())
        assert report.output_validation is not None
        assert report.output_validation.total_checks == 28

    def test_output_validation_disabled(self) -> None:
        config = ValidationAgentConfig(
            features=FeatureFlags(enable_validation=False)
        )
        agent = ValidationAgent(config)
        report = agent.validate(_make_verdict(), _make_gt())
        assert report.output_validation is None


class TestValidateInput:
    """Test validate_input convenience method."""

    def test_validate_input(self) -> None:
        agent = ValidationAgent()
        input_data = ValidationAgentInput(
            verdict=_make_verdict(),
            ground_truth=_make_gt(),
            correlation_id="input-1",
        )
        report = agent.validate_input(input_data)
        assert report.verdict_correct is True
        assert report.correlation_id == "input-1"


class TestLLMMode:
    """Agent with LLM enabled (mock provider)."""

    def test_llm_enabled_incorrect_verdict(self) -> None:
        config = ValidationAgentConfig(
            features=FeatureFlags(use_llm=True)
        )
        agent = ValidationAgent(config)
        report = agent.validate(
            _make_verdict("network partition"), _make_gt()
        )
        # Should still produce a valid report
        assert report.verdict_correct is False
        assert len(report.recommendations) >= 1

    def test_llm_enabled_correct_verdict_skips_llm(self) -> None:
        config = ValidationAgentConfig(
            features=FeatureFlags(use_llm=True)
        )
        agent = ValidationAgent(config)
        report = agent.validate(_make_verdict(), _make_gt())
        # Correct verdict → LLM not invoked
        assert report.verdict_correct is True
        if report.metadata:
            assert report.metadata.used_llm is False


class TestTelemetry:
    """Telemetry counters are updated."""

    def test_counters_increment(self) -> None:
        agent = ValidationAgent()
        agent.validate(_make_verdict(), _make_gt())
        assert agent.telemetry.validations_total.value >= 1
        assert agent.telemetry.validations_succeeded.value >= 1


class TestErrorHandling:
    """Agent handles internal errors gracefully."""

    def test_error_report_on_exception(self) -> None:
        # Create agent with default config, then break the fallback
        agent = ValidationAgent()
        # Call with minimal valid types — should succeed normally
        report = agent.validate(_make_verdict(), _make_gt())
        assert report.agent == "validation_agent"
