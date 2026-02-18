"""
File: validator.py
Purpose: 25+ validation checks on the ValidationReport output.
Dependencies: schema models.
Performance: <5ms for all checks.

Validates that the ValidationReport is internally consistent,
scores are within bounds, and cross-field constraints hold.
"""

from __future__ import annotations

import time
from typing import List, Optional

from agents.validation_agent.config import ValidationAgentConfig
from agents.validation_agent.schema import (
    ValidationCheckSeverity,
    ValidationReport,
    ValidatorError,
    ValidatorResult,
)
from agents.validation_agent.telemetry import get_logger

logger = get_logger("validation_agent.validator")


class ReportValidator:
    """Validates the output ValidationReport for consistency.

    Runs 25+ checks grouped by category:
      1-3:   Input presence checks
      4-8:   Score range checks
      9-11:  Precision/recall consistency
      12-14: Verdict correctness consistency
      15-17: Evidence & timeline checks
      18-20: Hallucination checks
      21-22: Recommendation checks
      23-24: Confusion matrix checks
      25-26: Metadata checks
      27-28: Cross-field consistency

    Args:
        config: Agent configuration.
    """

    def __init__(
        self, config: Optional[ValidationAgentConfig] = None
    ) -> None:
        self._config = config or ValidationAgentConfig()

    def validate(self, report: ValidationReport) -> ValidatorResult:
        """Run all validation checks on a report.

        Args:
            report: The ValidationReport to validate.

        Returns:
            ValidatorResult with errors and warnings.
        """
        start = time.perf_counter()
        errors: List[ValidatorError] = []
        warnings: List[ValidatorError] = []

        checks = [
            self._check_01_report_present,
            self._check_02_agent_field,
            self._check_03_timestamp_present,
            self._check_04_accuracy_range,
            self._check_05_precision_range,
            self._check_06_recall_range,
            self._check_07_f1_range,
            self._check_08_calibration_range,
            self._check_09_precision_recall_f1_consistent,
            self._check_10_f1_calculation,
            self._check_11_precision_recall_not_both_nan,
            self._check_12_correct_verdict_high_accuracy,
            self._check_13_incorrect_verdict_has_discrepancies,
            self._check_14_accuracy_consistency,
            self._check_15_evidence_accuracy_range,
            self._check_16_timeline_accuracy_range,
            self._check_17_affected_services_range,
            self._check_18_hallucination_types_valid,
            self._check_19_no_duplicate_hallucinations,
            self._check_20_hallucination_descriptions,
            self._check_21_recommendations_when_low_accuracy,
            self._check_22_recommendations_non_empty_strings,
            self._check_23_confusion_matrix_sums,
            self._check_24_confusion_matrix_classes,
            self._check_25_metadata_correlation_id,
            self._check_26_metadata_timestamps,
            self._check_27_discrepancy_types_valid,
            self._check_28_no_duplicate_discrepancies,
        ]

        for check_fn in checks:
            result = check_fn(report)
            if result is not None:
                if result.severity == ValidationCheckSeverity.CRITICAL:
                    errors.append(result)
                else:
                    warnings.append(result)

        elapsed_ms = (time.perf_counter() - start) * 1000

        passed = len(errors) == 0

        return ValidatorResult(
            validation_passed=passed,
            total_checks=len(checks),
            errors=errors,
            warnings=warnings,
            validation_latency_ms=round(elapsed_ms, 3),
        )

    # ── Checks 1–3: Input presence ─────────────────────────────

    def _check_01_report_present(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 1: Report object is present."""
        if report is None:
            return ValidatorError(
                check_number=1,
                check_name="report_present",
                error_description="ValidationReport is None",
                severity=ValidationCheckSeverity.CRITICAL,
            )
        return None

    def _check_02_agent_field(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 2: Agent field is correct."""
        if report.agent != "validation_agent":
            return ValidatorError(
                check_number=2,
                check_name="agent_field",
                error_description="agent field must be 'validation_agent'",
                expected="validation_agent",
                actual=report.agent,
                severity=ValidationCheckSeverity.CRITICAL,
            )
        return None

    def _check_03_timestamp_present(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 3: Analysis timestamp is present."""
        if not report.analysis_timestamp:
            return ValidatorError(
                check_number=3,
                check_name="timestamp_present",
                error_description="analysis_timestamp is empty",
                severity=ValidationCheckSeverity.WARNING,
            )
        return None

    # ── Checks 4–8: Score ranges ────────────────────────────────

    def _check_04_accuracy_range(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 4: accuracy_score ∈ [0, 1]."""
        if not 0.0 <= report.accuracy_score <= 1.0:
            return ValidatorError(
                check_number=4,
                check_name="accuracy_range",
                error_description="accuracy_score must be in [0, 1]",
                expected="[0.0, 1.0]",
                actual=str(report.accuracy_score),
                severity=ValidationCheckSeverity.CRITICAL,
            )
        return None

    def _check_05_precision_range(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 5: precision ∈ [0, 1]."""
        if not 0.0 <= report.precision <= 1.0:
            return ValidatorError(
                check_number=5,
                check_name="precision_range",
                error_description="precision must be in [0, 1]",
                expected="[0.0, 1.0]",
                actual=str(report.precision),
                severity=ValidationCheckSeverity.CRITICAL,
            )
        return None

    def _check_06_recall_range(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 6: recall ∈ [0, 1]."""
        if not 0.0 <= report.recall <= 1.0:
            return ValidatorError(
                check_number=6,
                check_name="recall_range",
                error_description="recall must be in [0, 1]",
                expected="[0.0, 1.0]",
                actual=str(report.recall),
                severity=ValidationCheckSeverity.CRITICAL,
            )
        return None

    def _check_07_f1_range(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 7: f1_score ∈ [0, 1]."""
        if not 0.0 <= report.f1_score <= 1.0:
            return ValidatorError(
                check_number=7,
                check_name="f1_range",
                error_description="f1_score must be in [0, 1]",
                expected="[0.0, 1.0]",
                actual=str(report.f1_score),
                severity=ValidationCheckSeverity.CRITICAL,
            )
        return None

    def _check_08_calibration_range(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 8: confidence_calibration_error ∈ [0, 1]."""
        if not 0.0 <= report.confidence_calibration_error <= 1.0:
            return ValidatorError(
                check_number=8,
                check_name="calibration_range",
                error_description=(
                    "confidence_calibration_error must be in [0, 1]"
                ),
                expected="[0.0, 1.0]",
                actual=str(report.confidence_calibration_error),
                severity=ValidationCheckSeverity.CRITICAL,
            )
        return None

    # ── Checks 9–11: Precision/Recall consistency ───────────────

    def _check_09_precision_recall_f1_consistent(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 9: F1 <= max(precision, recall)."""
        max_pr = max(report.precision, report.recall)
        if report.f1_score > max_pr + 0.01:
            return ValidatorError(
                check_number=9,
                check_name="f1_consistency",
                error_description=(
                    "f1_score cannot exceed max(precision, recall)"
                ),
                expected=f"<= {max_pr}",
                actual=str(report.f1_score),
                severity=ValidationCheckSeverity.WARNING,
            )
        return None

    def _check_10_f1_calculation(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 10: F1 = 2*P*R/(P+R) when P+R > 0."""
        p, r = report.precision, report.recall
        if p + r > 0:
            expected_f1 = 2 * p * r / (p + r)
            if abs(report.f1_score - expected_f1) > 0.02:
                return ValidatorError(
                    check_number=10,
                    check_name="f1_formula",
                    error_description="f1_score doesn't match 2*P*R/(P+R)",
                    expected=f"{expected_f1:.4f}",
                    actual=str(report.f1_score),
                    severity=ValidationCheckSeverity.WARNING,
                )
        return None

    def _check_11_precision_recall_not_both_nan(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 11: Precision and recall are valid floats."""
        import math
        if math.isnan(report.precision) or math.isnan(report.recall):
            return ValidatorError(
                check_number=11,
                check_name="precision_recall_valid",
                error_description="precision or recall is NaN",
                severity=ValidationCheckSeverity.CRITICAL,
            )
        return None

    # ── Checks 12–14: Verdict correctness ───────────────────────

    def _check_12_correct_verdict_high_accuracy(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 12: If verdict_correct=True, accuracy >= 0.8."""
        if report.verdict_correct and report.accuracy_score < 0.8:
            return ValidatorError(
                check_number=12,
                check_name="correct_verdict_accuracy",
                error_description=(
                    "verdict_correct=True but accuracy_score < 0.8"
                ),
                expected=">= 0.8",
                actual=str(report.accuracy_score),
                severity=ValidationCheckSeverity.CRITICAL,
            )
        return None

    def _check_13_incorrect_verdict_has_discrepancies(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 13: If verdict_correct=False, discrepancies non-empty."""
        if not report.verdict_correct and not report.discrepancies:
            return ValidatorError(
                check_number=13,
                check_name="incorrect_has_discrepancies",
                error_description=(
                    "verdict_correct=False but discrepancies list is empty"
                ),
                severity=ValidationCheckSeverity.WARNING,
            )
        return None

    def _check_14_accuracy_consistency(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 14: If accuracy=1.0, verdict should be correct."""
        if report.accuracy_score >= 1.0 and not report.verdict_correct:
            return ValidatorError(
                check_number=14,
                check_name="accuracy_verdict_consistency",
                error_description=(
                    "accuracy_score=1.0 but verdict_correct=False"
                ),
                severity=ValidationCheckSeverity.WARNING,
            )
        return None

    # ── Checks 15–17: Evidence & timeline ───────────────────────

    def _check_15_evidence_accuracy_range(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 15: evidence_accuracy ∈ [0, 1]."""
        if not 0.0 <= report.evidence_accuracy <= 1.0:
            return ValidatorError(
                check_number=15,
                check_name="evidence_accuracy_range",
                error_description="evidence_accuracy must be in [0, 1]",
                expected="[0.0, 1.0]",
                actual=str(report.evidence_accuracy),
                severity=ValidationCheckSeverity.CRITICAL,
            )
        return None

    def _check_16_timeline_accuracy_range(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 16: timeline_accuracy ∈ [0, 1]."""
        if not 0.0 <= report.timeline_accuracy <= 1.0:
            return ValidatorError(
                check_number=16,
                check_name="timeline_accuracy_range",
                error_description="timeline_accuracy must be in [0, 1]",
                expected="[0.0, 1.0]",
                actual=str(report.timeline_accuracy),
                severity=ValidationCheckSeverity.CRITICAL,
            )
        return None

    def _check_17_affected_services_range(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 17: affected_services_accuracy ∈ [0, 1]."""
        if not 0.0 <= report.affected_services_accuracy <= 1.0:
            return ValidatorError(
                check_number=17,
                check_name="services_accuracy_range",
                error_description=(
                    "affected_services_accuracy must be in [0, 1]"
                ),
                expected="[0.0, 1.0]",
                actual=str(report.affected_services_accuracy),
                severity=ValidationCheckSeverity.CRITICAL,
            )
        return None

    # ── Checks 18–20: Hallucination checks ──────────────────────

    def _check_18_hallucination_types_valid(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 18: All hallucination types are valid enum values."""
        from agents.validation_agent.schema import HallucinationType
        valid_types = set(HallucinationType)
        for h in report.hallucinations:
            if h.hallucination_type not in valid_types:
                return ValidatorError(
                    check_number=18,
                    check_name="hallucination_types",
                    error_description=(
                        f"Invalid hallucination type: "
                        f"{h.hallucination_type}"
                    ),
                    severity=ValidationCheckSeverity.WARNING,
                )
        return None

    def _check_19_no_duplicate_hallucinations(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 19: No duplicate hallucinations."""
        seen = set()
        for h in report.hallucinations:
            key = (h.hallucination_type, h.fabricated_value)
            if key in seen:
                return ValidatorError(
                    check_number=19,
                    check_name="no_duplicate_hallucinations",
                    error_description=(
                        f"Duplicate hallucination: {h.fabricated_value}"
                    ),
                    severity=ValidationCheckSeverity.WARNING,
                )
            seen.add(key)
        return None

    def _check_20_hallucination_descriptions(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 20: Hallucinations have descriptions."""
        for h in report.hallucinations:
            if not h.description:
                return ValidatorError(
                    check_number=20,
                    check_name="hallucination_descriptions",
                    error_description="Hallucination missing description",
                    severity=ValidationCheckSeverity.WARNING,
                )
        return None

    # ── Checks 21–22: Recommendation checks ─────────────────────

    def _check_21_recommendations_when_low_accuracy(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 21: At least 1 recommendation if accuracy < 0.9."""
        threshold = self._config.recommendations.accuracy_for_recs
        if report.accuracy_score < threshold and not report.recommendations:
            return ValidatorError(
                check_number=21,
                check_name="recommendations_present",
                error_description=(
                    f"accuracy_score < {threshold} but no recommendations"
                ),
                severity=ValidationCheckSeverity.WARNING,
            )
        return None

    def _check_22_recommendations_non_empty_strings(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 22: Recommendations are non-empty strings."""
        for rec in report.recommendations:
            if not rec or not rec.strip():
                return ValidatorError(
                    check_number=22,
                    check_name="recommendations_non_empty",
                    error_description="Recommendation is empty string",
                    severity=ValidationCheckSeverity.WARNING,
                )
        return None

    # ── Checks 23–24: Confusion matrix ──────────────────────────

    def _check_23_confusion_matrix_sums(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 23: Confusion matrix TP+FP+TN+FN > 0."""
        cm = report.confusion_matrix
        if cm is not None:
            total = cm.tp + cm.fp + cm.tn + cm.fn
            if total == 0:
                return ValidatorError(
                    check_number=23,
                    check_name="confusion_matrix_sum",
                    error_description=(
                        "Confusion matrix TP+FP+TN+FN sums to zero"
                    ),
                    severity=ValidationCheckSeverity.WARNING,
                )
        return None

    def _check_24_confusion_matrix_classes(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 24: Confusion matrix has classes."""
        cm = report.confusion_matrix
        if cm is not None and cm.matrix and not cm.classes:
            return ValidatorError(
                check_number=24,
                check_name="confusion_matrix_classes",
                error_description="Confusion matrix has data but no classes",
                severity=ValidationCheckSeverity.WARNING,
            )
        return None

    # ── Checks 25–26: Metadata ──────────────────────────────────

    def _check_25_metadata_correlation_id(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 25: Metadata has correlation_id."""
        if report.metadata and not report.metadata.correlation_id:
            return ValidatorError(
                check_number=25,
                check_name="metadata_correlation_id",
                error_description="metadata.correlation_id is empty",
                severity=ValidationCheckSeverity.WARNING,
            )
        return None

    def _check_26_metadata_timestamps(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 26: Metadata has validation timestamps."""
        if report.metadata:
            if not report.metadata.validation_start:
                return ValidatorError(
                    check_number=26,
                    check_name="metadata_timestamps",
                    error_description=(
                        "metadata.validation_start is empty"
                    ),
                    severity=ValidationCheckSeverity.WARNING,
                )
        return None

    # ── Checks 27–28: Cross-field ───────────────────────────────

    def _check_27_discrepancy_types_valid(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 27: All discrepancy types are valid."""
        from agents.validation_agent.schema import DiscrepancyType
        valid_types = set(DiscrepancyType)
        for d in report.discrepancies:
            if d.discrepancy_type not in valid_types:
                return ValidatorError(
                    check_number=27,
                    check_name="discrepancy_types",
                    error_description=(
                        f"Invalid discrepancy type: {d.discrepancy_type}"
                    ),
                    severity=ValidationCheckSeverity.WARNING,
                )
        return None

    def _check_28_no_duplicate_discrepancies(
        self, report: ValidationReport
    ) -> Optional[ValidatorError]:
        """Check 28: No duplicate discrepancies."""
        seen = set()
        for d in report.discrepancies:
            key = (d.discrepancy_type, d.description)
            if key in seen:
                return ValidatorError(
                    check_number=28,
                    check_name="no_duplicate_discrepancies",
                    error_description=(
                        f"Duplicate discrepancy: {d.description[:50]}"
                    ),
                    severity=ValidationCheckSeverity.WARNING,
                )
            seen.add(key)
        return None
