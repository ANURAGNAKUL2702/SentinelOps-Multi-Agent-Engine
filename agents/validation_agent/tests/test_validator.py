"""Tests for validator.py — 28 output validation checks."""

from __future__ import annotations

import pytest

from agents.validation_agent.schema import (
    ConfusionMatrixResult,
    Discrepancy,
    DiscrepancySeverity,
    DiscrepancyType,
    Hallucination,
    HallucinationType,
    ValidationCheckSeverity,
    ValidationMetadata,
    ValidationReport,
)
from agents.validation_agent.validator import ReportValidator


@pytest.fixture()
def validator() -> ReportValidator:
    return ReportValidator()


def _valid_report(**overrides) -> ValidationReport:
    """Build a minimal valid report."""
    defaults = dict(
        verdict_correct=True,
        accuracy_score=0.95,
        precision=0.9,
        recall=0.9,
        f1_score=0.9,
        confidence_calibration_error=0.1,
        evidence_accuracy=0.8,
        timeline_accuracy=0.8,
        affected_services_accuracy=0.9,
        metadata=ValidationMetadata(correlation_id="test-123"),
    )
    defaults.update(overrides)
    return ValidationReport(**defaults)


class TestInputPresence:
    """Checks 1-3."""

    def test_01_report_present(self, validator: ReportValidator) -> None:
        report = _valid_report()
        result = validator.validate(report)
        # Check 1 should pass (report is not None)
        assert not any(e.check_number == 1 for e in result.errors)

    def test_02_agent_field_correct(self, validator: ReportValidator) -> None:
        report = _valid_report()
        result = validator.validate(report)
        # Agent field is auto-set to 'validation_agent'
        assert not any(e.check_number == 2 for e in result.errors)

    def test_03_timestamp_present(self, validator: ReportValidator) -> None:
        report = _valid_report()
        result = validator.validate(report)
        assert not any(e.check_number == 3 for e in result.errors)
        assert not any(e.check_number == 3 for e in result.warnings)


class TestScoreRanges:
    """Checks 4-8 — scores clamped by schema, so mainly test pass."""

    def test_04_accuracy_range_valid(self, validator: ReportValidator) -> None:
        report = _valid_report(accuracy_score=0.5)
        result = validator.validate(report)
        assert not any(e.check_number == 4 for e in result.errors)

    def test_05_precision_valid(self, validator: ReportValidator) -> None:
        report = _valid_report(precision=1.0)
        result = validator.validate(report)
        assert not any(e.check_number == 5 for e in result.errors)

    def test_06_recall_valid(self, validator: ReportValidator) -> None:
        report = _valid_report(recall=0.0)
        result = validator.validate(report)
        assert not any(e.check_number == 6 for e in result.errors)

    def test_07_f1_valid(self, validator: ReportValidator) -> None:
        report = _valid_report(f1_score=0.5)
        result = validator.validate(report)
        assert not any(e.check_number == 7 for e in result.errors)

    def test_08_calibration_valid(self, validator: ReportValidator) -> None:
        report = _valid_report(confidence_calibration_error=0.0)
        result = validator.validate(report)
        assert not any(e.check_number == 8 for e in result.errors)


class TestPrecisionRecallConsistency:
    """Checks 9-11."""

    def test_09_f1_not_exceeding_max_pr(
        self, validator: ReportValidator
    ) -> None:
        report = _valid_report(precision=0.6, recall=0.6, f1_score=0.6)
        result = validator.validate(report)
        assert not any(e.check_number == 9 for e in result.errors)
        assert not any(e.check_number == 9 for e in result.warnings)

    def test_10_f1_formula_consistent(
        self, validator: ReportValidator
    ) -> None:
        # 2*0.8*0.6 / (0.8+0.6) = 0.6857
        report = _valid_report(precision=0.8, recall=0.6, f1_score=0.69)
        result = validator.validate(report)
        assert not any(e.check_number == 10 for e in result.warnings)

    def test_10_f1_formula_inconsistent(
        self, validator: ReportValidator
    ) -> None:
        # Wrong F1 for given P and R
        report = _valid_report(precision=0.8, recall=0.6, f1_score=0.3)
        result = validator.validate(report)
        warnings_10 = [e for e in result.warnings if e.check_number == 10]
        assert len(warnings_10) >= 1


class TestVerdictCorrectness:
    """Checks 12-14."""

    def test_12_correct_verdict_high_accuracy(
        self, validator: ReportValidator
    ) -> None:
        # Correct verdict with accuracy >= 0.8 — should pass
        report = _valid_report(verdict_correct=True, accuracy_score=0.9)
        result = validator.validate(report)
        assert not any(e.check_number == 12 for e in result.errors)

    def test_12_correct_verdict_low_accuracy_fails(
        self, validator: ReportValidator
    ) -> None:
        report = _valid_report(verdict_correct=True, accuracy_score=0.3)
        result = validator.validate(report)
        errors_12 = [e for e in result.errors if e.check_number == 12]
        assert len(errors_12) >= 1

    def test_13_incorrect_verdict_has_discrepancies(
        self, validator: ReportValidator
    ) -> None:
        report = _valid_report(
            verdict_correct=False,
            accuracy_score=0.3,
            discrepancies=[
                Discrepancy(
                    discrepancy_type=DiscrepancyType.ROOT_CAUSE_MISMATCH,
                    description="Wrong root cause",
                )
            ],
        )
        result = validator.validate(report)
        assert not any(e.check_number == 13 for e in result.warnings)

    def test_13_incorrect_verdict_no_discrepancies_warns(
        self, validator: ReportValidator
    ) -> None:
        report = _valid_report(
            verdict_correct=False,
            accuracy_score=0.3,
            discrepancies=[],
        )
        result = validator.validate(report)
        warnings_13 = [e for e in result.warnings if e.check_number == 13]
        assert len(warnings_13) >= 1


class TestEvidenceTimeline:
    """Checks 15-17."""

    def test_15_evidence_range(self, validator: ReportValidator) -> None:
        report = _valid_report(evidence_accuracy=0.5)
        result = validator.validate(report)
        assert not any(e.check_number == 15 for e in result.errors)

    def test_16_timeline_range(self, validator: ReportValidator) -> None:
        report = _valid_report(timeline_accuracy=1.0)
        result = validator.validate(report)
        assert not any(e.check_number == 16 for e in result.errors)

    def test_17_services_range(self, validator: ReportValidator) -> None:
        report = _valid_report(affected_services_accuracy=0.0)
        result = validator.validate(report)
        assert not any(e.check_number == 17 for e in result.errors)


class TestHallucinationChecks:
    """Checks 18-20."""

    def test_18_valid_hallucination_types(
        self, validator: ReportValidator
    ) -> None:
        report = _valid_report(
            hallucinations=[
                Hallucination(
                    hallucination_type=HallucinationType.SERVICE,
                    description="ghost svc",
                    fabricated_value="ghost",
                )
            ]
        )
        result = validator.validate(report)
        assert not any(e.check_number == 18 for e in result.errors)

    def test_19_no_duplicate_hallucinations(
        self, validator: ReportValidator
    ) -> None:
        h = Hallucination(
            hallucination_type=HallucinationType.SERVICE,
            description="ghost",
            fabricated_value="ghost-svc",
        )
        report = _valid_report(hallucinations=[h, h])
        result = validator.validate(report)
        warnings_19 = [e for e in result.warnings if e.check_number == 19]
        assert len(warnings_19) >= 1

    def test_20_hallucination_descriptions(
        self, validator: ReportValidator
    ) -> None:
        report = _valid_report(
            hallucinations=[
                Hallucination(
                    hallucination_type=HallucinationType.SERVICE,
                    description="valid description",
                    fabricated_value="ghost",
                )
            ]
        )
        result = validator.validate(report)
        assert not any(e.check_number == 20 for e in result.warnings)


class TestRecommendationChecks:
    """Checks 21-22."""

    def test_21_recommendations_present_when_low_accuracy(
        self, validator: ReportValidator
    ) -> None:
        report = _valid_report(
            accuracy_score=0.5,
            verdict_correct=False,
            recommendations=["Fix something"],
            discrepancies=[
                Discrepancy(
                    discrepancy_type=DiscrepancyType.ROOT_CAUSE_MISMATCH,
                    description="wrong",
                )
            ],
        )
        result = validator.validate(report)
        assert not any(e.check_number == 21 for e in result.warnings)

    def test_21_no_recommendations_low_accuracy_warns(
        self, validator: ReportValidator
    ) -> None:
        report = _valid_report(
            accuracy_score=0.5,
            verdict_correct=False,
            recommendations=[],
            discrepancies=[
                Discrepancy(
                    discrepancy_type=DiscrepancyType.ROOT_CAUSE_MISMATCH,
                    description="wrong",
                )
            ],
        )
        result = validator.validate(report)
        warnings_21 = [e for e in result.warnings if e.check_number == 21]
        assert len(warnings_21) >= 1

    def test_22_recommendations_non_empty_strings(
        self, validator: ReportValidator
    ) -> None:
        report = _valid_report(recommendations=["Good recommendation"])
        result = validator.validate(report)
        assert not any(e.check_number == 22 for e in result.warnings)


class TestConfusionMatrix:
    """Checks 23-24."""

    def test_23_matrix_sums(self, validator: ReportValidator) -> None:
        report = _valid_report(
            confusion_matrix=ConfusionMatrixResult(
                tp=1, fp=0, tn=0, fn=0,
                matrix=[[1]], classes=["database_failure"],
            )
        )
        result = validator.validate(report)
        assert not any(e.check_number == 23 for e in result.warnings)

    def test_24_matrix_classes(self, validator: ReportValidator) -> None:
        report = _valid_report(
            confusion_matrix=ConfusionMatrixResult(
                tp=1, fp=0, tn=0, fn=0,
                matrix=[[1]], classes=["database_failure"],
            )
        )
        result = validator.validate(report)
        assert not any(e.check_number == 24 for e in result.warnings)


class TestMetadata:
    """Checks 25-26."""

    def test_25_metadata_correlation_id(
        self, validator: ReportValidator
    ) -> None:
        report = _valid_report(
            metadata=ValidationMetadata(correlation_id="abc-123")
        )
        result = validator.validate(report)
        assert not any(e.check_number == 25 for e in result.warnings)

    def test_26_metadata_timestamps(
        self, validator: ReportValidator
    ) -> None:
        report = _valid_report(
            metadata=ValidationMetadata(
                correlation_id="abc",
                validation_start="2026-01-01T00:00:00Z",
                validation_end="2026-01-01T00:00:01Z",
            )
        )
        result = validator.validate(report)
        assert not any(e.check_number == 26 for e in result.warnings)


class TestCrossField:
    """Checks 27-28."""

    def test_27_discrepancy_types_valid(
        self, validator: ReportValidator
    ) -> None:
        report = _valid_report(
            discrepancies=[
                Discrepancy(
                    discrepancy_type=DiscrepancyType.SERVICE_MISMATCH,
                    description="missing svc",
                )
            ]
        )
        result = validator.validate(report)
        assert not any(e.check_number == 27 for e in result.errors)

    def test_28_no_duplicate_discrepancies(
        self, validator: ReportValidator
    ) -> None:
        d = Discrepancy(
            discrepancy_type=DiscrepancyType.SERVICE_MISMATCH,
            description="missing svc",
            expected="a",
            actual="b",
        )
        report = _valid_report(discrepancies=[d, d])
        result = validator.validate(report)
        warnings_28 = [e for e in result.warnings if e.check_number == 28]
        assert len(warnings_28) >= 1


class TestOverallValidation:
    """End-to-end validation test."""

    def test_valid_report_passes(self, validator: ReportValidator) -> None:
        report = _valid_report()
        result = validator.validate(report)
        assert result.validation_passed
        assert result.total_checks == 28

    def test_results_have_latency(self, validator: ReportValidator) -> None:
        report = _valid_report()
        result = validator.validate(report)
        assert result.validation_latency_ms >= 0.0
