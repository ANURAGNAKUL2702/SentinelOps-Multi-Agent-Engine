"""Tests for validation agent schema models."""

from __future__ import annotations

import pytest

from agents.validation_agent.schema import (
    CalibrationBin,
    ConfusionMatrixResult,
    Discrepancy,
    DiscrepancySeverity,
    DiscrepancyType,
    GroundTruth,
    Hallucination,
    HallucinationType,
    PropagationStep,
    ValidationAgentInput,
    ValidationCheckSeverity,
    ValidationMetadata,
    ValidationReport,
    ValidatorError,
    ValidatorResult,
    RootCauseVerdict,
)


class TestGroundTruth:
    """Tests for GroundTruth model."""

    def test_minimal_ground_truth(self) -> None:
        gt = GroundTruth(actual_root_cause="db_failure")
        assert gt.actual_root_cause == "db_failure"
        assert gt.failure_type == ""
        assert gt.affected_services_ground_truth == []

    def test_full_ground_truth(self) -> None:
        gt = GroundTruth(
            actual_root_cause="database_connection_pool_exhaustion",
            failure_type="resource_exhaustion",
            injected_at="2026-01-01T00:00:00Z",
            affected_services_ground_truth=["database", "payment-service"],
            failure_propagation_chain=[
                PropagationStep(
                    from_service="database",
                    to_service="payment-service",
                    delay_seconds=2.5,
                    mechanism="connection_timeout",
                )
            ],
            expected_symptoms=["high_latency", "connection_errors"],
            simulation_metadata={"scenario_id": "test-001"},
        )
        assert gt.failure_type == "resource_exhaustion"
        assert len(gt.failure_propagation_chain) == 1
        assert gt.failure_propagation_chain[0].delay_seconds == 2.5

    def test_propagation_step(self) -> None:
        step = PropagationStep(
            from_service="api-gateway",
            to_service="auth-service",
            delay_seconds=0.5,
        )
        assert step.from_service == "api-gateway"
        assert step.mechanism == ""


class TestDiscrepancy:
    """Tests for Discrepancy model."""

    def test_discrepancy_defaults(self) -> None:
        d = Discrepancy(
            discrepancy_type=DiscrepancyType.ROOT_CAUSE_MISMATCH,
        )
        assert d.severity == DiscrepancySeverity.MEDIUM
        assert d.description == ""

    def test_all_discrepancy_types(self) -> None:
        for dt in DiscrepancyType:
            d = Discrepancy(discrepancy_type=dt, description="test")
            assert d.discrepancy_type == dt


class TestHallucination:
    """Tests for Hallucination model."""

    def test_hallucination_defaults(self) -> None:
        h = Hallucination(hallucination_type=HallucinationType.SERVICE)
        assert h.fabricated_value == ""

    def test_all_hallucination_types(self) -> None:
        for ht in HallucinationType:
            h = Hallucination(
                hallucination_type=ht,
                description="test",
                fabricated_value="fake",
            )
            assert h.hallucination_type == ht


class TestConfusionMatrix:
    """Tests for ConfusionMatrixResult model."""

    def test_total_property(self) -> None:
        cm = ConfusionMatrixResult(tp=5, fp=2, tn=10, fn=3)
        assert cm.total == 20

    def test_empty_matrix(self) -> None:
        cm = ConfusionMatrixResult()
        assert cm.total == 0
        assert cm.matrix == []


class TestCalibrationBin:
    """Tests for CalibrationBin model."""

    def test_basic_bin(self) -> None:
        b = CalibrationBin(
            bin_start=0.0, bin_end=0.1,
            avg_confidence=0.05, avg_accuracy=0.1, count=10,
        )
        assert b.count == 10


class TestValidationReport:
    """Tests for ValidationReport model."""

    def test_default_report(self) -> None:
        r = ValidationReport()
        assert r.agent == "validation_agent"
        assert r.verdict_correct is False
        assert r.accuracy_score == 0.0
        assert r.analysis_timestamp != ""

    def test_score_clamping(self) -> None:
        r = ValidationReport(accuracy_score=0.95, precision=0.8)
        assert r.accuracy_score == 0.95
        assert r.precision == 0.8

    def test_full_report(self) -> None:
        r = ValidationReport(
            verdict_correct=True,
            accuracy_score=0.95,
            precision=1.0,
            recall=1.0,
            f1_score=1.0,
            confidence_calibration_error=0.05,
            evidence_accuracy=0.9,
            timeline_accuracy=0.85,
            affected_services_accuracy=1.0,
            recommendations=["All good"],
            confusion_matrix=ConfusionMatrixResult(tp=1),
        )
        assert r.verdict_correct is True
        assert r.f1_score == 1.0


class TestValidatorResult:
    """Tests for ValidatorResult model."""

    def test_default_result(self) -> None:
        r = ValidatorResult()
        assert r.validation_passed is True
        assert r.total_checks == 0

    def test_with_errors(self) -> None:
        err = ValidatorError(
            check_number=1,
            check_name="test",
            error_description="failed",
            severity=ValidationCheckSeverity.CRITICAL,
        )
        r = ValidatorResult(
            validation_passed=False,
            total_checks=1,
            errors=[err],
        )
        assert not r.validation_passed


class TestValidationAgentInput:
    """Tests for ValidationAgentInput model."""

    def test_input_construction(self) -> None:
        v = RootCauseVerdict(root_cause="test")
        gt = GroundTruth(actual_root_cause="test")
        inp = ValidationAgentInput(verdict=v, ground_truth=gt)
        assert inp.incident_id != ""
        assert inp.history == []
