"""Tests for fallback.py — Deterministic validation pipeline."""

from __future__ import annotations

import time

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
from agents.validation_agent.config import ValidationAgentConfig
from agents.validation_agent.fallback import DeterministicFallback
from agents.validation_agent.schema import GroundTruth, PropagationStep
from agents.validation_agent.telemetry import TelemetryCollector


@pytest.fixture()
def fallback() -> DeterministicFallback:
    config = ValidationAgentConfig()
    telemetry = TelemetryCollector()
    return DeterministicFallback(config, telemetry)


def _make_gt(root_cause: str = "database failure") -> GroundTruth:
    return GroundTruth(
        actual_root_cause=root_cause,
        failure_type="resource_exhaustion",
        injected_at="2026-01-01T00:00:00Z",
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
    services: list[str] | None = None,
) -> RootCauseVerdict:
    return RootCauseVerdict(
        root_cause=root_cause,
        confidence=confidence,
        affected_services=services or ["database", "payment-service"],
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
            TimelineEvent(
                timestamp="2026-01-01T00:00:05Z",
                source=EvidenceSourceAgent.METRICS_AGENT,
                event="Payment latency spike",
                service="payment-service",
                severity=Severity.MEDIUM,
            ),
        ],
    )


class TestDeterministicCorrect:
    """Test with correct verdict."""

    def test_correct_verdict_produces_report(
        self, fallback: DeterministicFallback
    ) -> None:
        gt = _make_gt()
        verdict = _make_verdict()
        report = fallback.validate(verdict, gt, correlation_id="test-1")

        assert report.agent == "validation_agent"
        assert report.verdict_correct is True
        assert report.accuracy_score >= 0.8
        assert report.classification_source == "deterministic"
        assert report.correlation_id == "test-1"

    def test_scores_in_range(
        self, fallback: DeterministicFallback
    ) -> None:
        gt = _make_gt()
        verdict = _make_verdict()
        report = fallback.validate(verdict, gt)

        assert 0.0 <= report.accuracy_score <= 1.0
        assert 0.0 <= report.precision <= 1.0
        assert 0.0 <= report.recall <= 1.0
        assert 0.0 <= report.f1_score <= 1.0
        assert 0.0 <= report.confidence_calibration_error <= 1.0
        assert 0.0 <= report.evidence_accuracy <= 1.0
        assert 0.0 <= report.timeline_accuracy <= 1.0
        assert 0.0 <= report.affected_services_accuracy <= 1.0

    def test_metadata_populated(
        self, fallback: DeterministicFallback
    ) -> None:
        gt = _make_gt()
        verdict = _make_verdict()
        report = fallback.validate(verdict, gt, correlation_id="meta-1")

        assert report.metadata is not None
        assert report.metadata.correlation_id == "meta-1"
        assert report.metadata.used_fallback is True
        assert report.metadata.used_llm is False
        assert report.metadata.total_pipeline_ms > 0

    def test_confusion_matrix_present(
        self, fallback: DeterministicFallback
    ) -> None:
        gt = _make_gt()
        verdict = _make_verdict()
        report = fallback.validate(verdict, gt)
        assert report.confusion_matrix is not None
        assert report.confusion_matrix.total >= 1


class TestDeterministicIncorrect:
    """Test with incorrect verdict."""

    def test_wrong_root_cause(
        self, fallback: DeterministicFallback
    ) -> None:
        gt = _make_gt("database failure")
        verdict = _make_verdict("network partition")
        report = fallback.validate(verdict, gt)

        assert report.verdict_correct is False
        assert report.accuracy_score < 0.8
        assert len(report.discrepancies) >= 1
        assert len(report.recommendations) >= 1

    def test_extra_services_detected(
        self, fallback: DeterministicFallback
    ) -> None:
        gt = _make_gt()
        verdict = _make_verdict(
            services=["database", "payment-service", "ghost-service"]
        )
        report = fallback.validate(verdict, gt)
        # Ghost service should produce a hallucination
        assert len(report.hallucinations) >= 1


class TestFallbackPerformance:
    """Deterministic path must complete in <50ms."""

    def test_under_50ms(self, fallback: DeterministicFallback) -> None:
        gt = _make_gt()
        verdict = _make_verdict()
        t0 = time.perf_counter()
        _ = fallback.validate(verdict, gt)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 50, f"Fallback took {elapsed:.1f}ms, budget is 50ms"

    def test_no_llm_calls(self, fallback: DeterministicFallback) -> None:
        gt = _make_gt()
        verdict = _make_verdict()
        report = fallback.validate(verdict, gt)
        assert report.metadata is not None
        assert report.metadata.used_llm is False


class TestHistory:
    """Test with historical data for calibration."""

    def test_history_parsed(self, fallback: DeterministicFallback) -> None:
        gt = _make_gt()
        verdict = _make_verdict()
        history = [
            {"confidence": 0.9, "accuracy": 1.0},
            {"confidence": 0.8, "accuracy": 0.0},
            {"confidence": 0.5, "accuracy": 0.5},
        ]
        report = fallback.validate(verdict, gt, history=history)
        assert 0.0 <= report.confidence_calibration_error <= 1.0
        assert len(report.calibration_curve) >= 1
