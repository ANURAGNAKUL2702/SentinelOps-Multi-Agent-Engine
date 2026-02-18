"""
File: validator.py
Purpose: 27-check validation for HypothesisAgentOutput.
Dependencies: Schema models only.
Performance: <5ms, O(n) where n=fields.

Validation categories:
  Structural     (5 checks): Required fields/values exist
  Type           (6 checks): Correct data types
  Enum           (2 checks): Valid enum values
  Range          (4 checks): Numeric bounds
  Business Rules (6 checks): Semantic correctness
  Logical        (4 checks): Logical consistency
"""

from __future__ import annotations

import time
from typing import List, Optional

from agents.hypothesis_agent.config import HypothesisAgentConfig
from agents.hypothesis_agent.schema import (
    HypothesisAgentOutput,
    HypothesisStatus,
    IncidentCategory,
    Severity,
    ValidationResult,
    ValidationSeverity,
    ValidatorError,
)
from agents.hypothesis_agent.telemetry import get_logger

logger = get_logger("hypothesis_agent.validator")


class OutputValidator:
    """Validates HypothesisAgentOutput against 27 checks.

    Validation failures are informational — they do NOT block
    output generation. Errors are classified as CRITICAL or WARNING.

    Args:
        config: Agent configuration.
    """

    def __init__(
        self, config: Optional[HypothesisAgentConfig] = None
    ) -> None:
        self._config = config or HypothesisAgentConfig()

    def validate(
        self,
        output: HypothesisAgentOutput,
        correlation_id: str = "",
    ) -> ValidationResult:
        """Run all 27 validation checks.

        Args:
            output: The assembled output to validate.
            correlation_id: Request correlation ID.

        Returns:
            ValidationResult with errors and warnings.
        """
        start = time.perf_counter()
        errors: List[ValidatorError] = []
        warnings: List[ValidatorError] = []
        checks = 0

        # ── Structural Checks (1–5) ────────────────────────────

        # 1. agent field is "hypothesis_agent"
        checks += 1
        if output.agent != "hypothesis_agent":
            errors.append(ValidatorError(
                check_number=1,
                check_name="agent_name",
                error_description=(
                    "agent must be 'hypothesis_agent'"
                ),
                expected="hypothesis_agent",
                actual=output.agent,
                severity=ValidationSeverity.CRITICAL,
            ))

        # 2. analysis_timestamp is not empty
        checks += 1
        if not output.analysis_timestamp:
            errors.append(ValidatorError(
                check_number=2,
                check_name="analysis_timestamp_present",
                error_description="analysis_timestamp is empty",
                expected="ISO 8601 timestamp",
                actual="",
                severity=ValidationSeverity.CRITICAL,
            ))

        # 3. hypotheses list is present
        checks += 1
        if output.hypotheses is None:
            errors.append(ValidatorError(
                check_number=3,
                check_name="hypotheses_present",
                error_description="hypotheses is None",
                expected="List[Hypothesis]",
                actual="None",
                severity=ValidationSeverity.CRITICAL,
            ))

        # 4. classification_source is valid
        checks += 1
        valid_sources = {"llm", "fallback", "cached", "deterministic"}
        if output.classification_source not in valid_sources:
            errors.append(ValidatorError(
                check_number=4,
                check_name="classification_source_valid",
                error_description=(
                    f"classification_source "
                    f"'{output.classification_source}' not valid"
                ),
                expected=str(valid_sources),
                actual=output.classification_source,
                severity=ValidationSeverity.CRITICAL,
            ))

        # 5. incident_id is not empty
        checks += 1
        if not output.incident_id:
            warnings.append(ValidatorError(
                check_number=5,
                check_name="incident_id_present",
                error_description="incident_id is empty",
                expected="non-empty string",
                actual="",
                severity=ValidationSeverity.WARNING,
            ))

        # ── Type Checks (6–11) ─────────────────────────────────

        # 6. hypotheses is a list
        checks += 1
        if not isinstance(output.hypotheses, list):
            errors.append(ValidatorError(
                check_number=6,
                check_name="hypotheses_is_list",
                error_description="hypotheses is not a list",
                expected="list",
                actual=type(output.hypotheses).__name__,
                severity=ValidationSeverity.CRITICAL,
            ))

        # 7. confidence_score is float
        checks += 1
        if not isinstance(output.confidence_score, (int, float)):
            errors.append(ValidatorError(
                check_number=7,
                check_name="confidence_score_is_number",
                error_description="confidence_score is not a number",
                expected="float",
                actual=type(output.confidence_score).__name__,
                severity=ValidationSeverity.CRITICAL,
            ))

        # 8. pattern_matches is a list
        checks += 1
        if not isinstance(output.pattern_matches, list):
            errors.append(ValidatorError(
                check_number=8,
                check_name="pattern_matches_is_list",
                error_description="pattern_matches is not a list",
                expected="list",
                actual=type(output.pattern_matches).__name__,
                severity=ValidationSeverity.CRITICAL,
            ))

        # 9. pipeline_latency_ms is numeric
        checks += 1
        if not isinstance(
            output.pipeline_latency_ms, (int, float)
        ):
            errors.append(ValidatorError(
                check_number=9,
                check_name="pipeline_latency_is_number",
                error_description=(
                    "pipeline_latency_ms is not a number"
                ),
                expected="float",
                actual=type(
                    output.pipeline_latency_ms
                ).__name__,
                severity=ValidationSeverity.WARNING,
            ))

        # 10. estimated_mttr_minutes is numeric
        checks += 1
        if not isinstance(
            output.estimated_mttr_minutes, (int, float)
        ):
            errors.append(ValidatorError(
                check_number=10,
                check_name="mttr_is_number",
                error_description=(
                    "estimated_mttr_minutes is not a number"
                ),
                expected="float",
                actual=type(
                    output.estimated_mttr_minutes
                ).__name__,
                severity=ValidationSeverity.WARNING,
            ))

        # 11. hypothesis_summary is string
        checks += 1
        if not isinstance(output.hypothesis_summary, str):
            errors.append(ValidatorError(
                check_number=11,
                check_name="summary_is_string",
                error_description=(
                    "hypothesis_summary is not a string"
                ),
                expected="str",
                actual=type(
                    output.hypothesis_summary
                ).__name__,
                severity=ValidationSeverity.WARNING,
            ))

        # ── Enum Checks (12–13) ────────────────────────────────

        # 12. category is valid IncidentCategory
        checks += 1
        try:
            IncidentCategory(output.category)
        except (ValueError, KeyError):
            errors.append(ValidatorError(
                check_number=12,
                check_name="category_valid_enum",
                error_description=(
                    f"category '{output.category}' not valid"
                ),
                expected=str(
                    [e.value for e in IncidentCategory]
                ),
                actual=str(output.category),
                severity=ValidationSeverity.WARNING,
            ))

        # 13. severity is valid Severity
        checks += 1
        try:
            Severity(output.severity)
        except (ValueError, KeyError):
            errors.append(ValidatorError(
                check_number=13,
                check_name="severity_valid_enum",
                error_description=(
                    f"severity '{output.severity}' not valid"
                ),
                expected=str([e.value for e in Severity]),
                actual=str(output.severity),
                severity=ValidationSeverity.WARNING,
            ))

        # ── Range Checks (14–17) ───────────────────────────────

        # 14. confidence_score in [0.0, 1.0]
        checks += 1
        if isinstance(output.confidence_score, (int, float)):
            if not (0.0 <= output.confidence_score <= 1.0):
                errors.append(ValidatorError(
                    check_number=14,
                    check_name="confidence_score_range",
                    error_description=(
                        f"confidence_score {output.confidence_score} "
                        f"outside [0.0, 1.0]"
                    ),
                    expected="0.0 <= x <= 1.0",
                    actual=str(output.confidence_score),
                    severity=ValidationSeverity.CRITICAL,
                ))

        # 15. pipeline_latency_ms >= 0
        checks += 1
        if isinstance(output.pipeline_latency_ms, (int, float)):
            if output.pipeline_latency_ms < 0:
                errors.append(ValidatorError(
                    check_number=15,
                    check_name="pipeline_latency_non_negative",
                    error_description=(
                        "pipeline_latency_ms is negative"
                    ),
                    expected=">= 0.0",
                    actual=str(output.pipeline_latency_ms),
                    severity=ValidationSeverity.WARNING,
                ))

        # 16. estimated_mttr_minutes >= 0
        checks += 1
        if isinstance(
            output.estimated_mttr_minutes, (int, float)
        ):
            if output.estimated_mttr_minutes < 0:
                errors.append(ValidatorError(
                    check_number=16,
                    check_name="mttr_non_negative",
                    error_description=(
                        "estimated_mttr_minutes is negative"
                    ),
                    expected=">= 0.0",
                    actual=str(output.estimated_mttr_minutes),
                    severity=ValidationSeverity.WARNING,
                ))

        # 17. hypothesis likelihood_scores in [0.0, 1.0]
        checks += 1
        if isinstance(output.hypotheses, list):
            for idx, h in enumerate(output.hypotheses):
                if not (0.0 <= h.likelihood_score <= 1.0):
                    errors.append(ValidatorError(
                        check_number=17,
                        check_name=(
                            "hypothesis_likelihood_range"
                        ),
                        error_description=(
                            f"hypothesis[{idx}].likelihood_score "
                            f"{h.likelihood_score} "
                            f"outside [0.0, 1.0]"
                        ),
                        expected="0.0 <= x <= 1.0",
                        actual=str(h.likelihood_score),
                        severity=ValidationSeverity.CRITICAL,
                    ))
                    break  # one error per check

        # ── Business Rules (18–23) ──────────────────────────────

        # 18. at least min_hypotheses generated
        checks += 1
        min_h = self._config.limits.min_hypotheses
        if isinstance(output.hypotheses, list):
            if len(output.hypotheses) < min_h:
                warnings.append(ValidatorError(
                    check_number=18,
                    check_name="min_hypotheses_count",
                    error_description=(
                        f"Only {len(output.hypotheses)} hypotheses "
                        f"(minimum={min_h})"
                    ),
                    expected=f">= {min_h}",
                    actual=str(len(output.hypotheses)),
                    severity=ValidationSeverity.WARNING,
                ))

        # 19. no more than max_hypotheses
        checks += 1
        max_h = self._config.limits.max_hypotheses
        if isinstance(output.hypotheses, list):
            if len(output.hypotheses) > max_h:
                warnings.append(ValidatorError(
                    check_number=19,
                    check_name="max_hypotheses_count",
                    error_description=(
                        f"{len(output.hypotheses)} hypotheses "
                        f"exceeds max={max_h}"
                    ),
                    expected=f"<= {max_h}",
                    actual=str(len(output.hypotheses)),
                    severity=ValidationSeverity.WARNING,
                ))

        # 20. recommended_hypothesis references a valid ID
        checks += 1
        if (
            output.recommended_hypothesis
            and isinstance(output.hypotheses, list)
        ):
            ids = {h.hypothesis_id for h in output.hypotheses}
            if output.recommended_hypothesis not in ids:
                warnings.append(ValidatorError(
                    check_number=20,
                    check_name="recommended_hypothesis_valid",
                    error_description=(
                        f"recommended_hypothesis "
                        f"'{output.recommended_hypothesis}' "
                        f"not in hypothesis IDs"
                    ),
                    expected=f"one of {ids}",
                    actual=output.recommended_hypothesis,
                    severity=ValidationSeverity.WARNING,
                ))

        # 21. hypothesis_summary is not empty
        checks += 1
        if not output.hypothesis_summary:
            warnings.append(ValidatorError(
                check_number=21,
                check_name="summary_not_empty",
                error_description="hypothesis_summary is empty",
                expected="non-empty summary",
                actual="",
                severity=ValidationSeverity.WARNING,
            ))

        # 22. each hypothesis has a non-empty theory
        checks += 1
        if isinstance(output.hypotheses, list):
            for idx, h in enumerate(output.hypotheses):
                if not h.theory:
                    errors.append(ValidatorError(
                        check_number=22,
                        check_name="hypothesis_theory_present",
                        error_description=(
                            f"hypothesis[{idx}].theory is empty"
                        ),
                        expected="non-empty theory",
                        actual="",
                        severity=ValidationSeverity.CRITICAL,
                    ))
                    break

        # 23. hypotheses sorted by likelihood descending
        checks += 1
        if isinstance(output.hypotheses, list):
            scores = [
                h.likelihood_score for h in output.hypotheses
            ]
            if scores != sorted(scores, reverse=True):
                warnings.append(ValidatorError(
                    check_number=23,
                    check_name="hypotheses_sorted",
                    error_description=(
                        "hypotheses not sorted by "
                        "likelihood_score descending"
                    ),
                    expected="descending order",
                    actual=str(scores),
                    severity=ValidationSeverity.WARNING,
                ))

        # ── Logical Consistency (24–27) ─────────────────────────

        # 24. if confidence > 0.5, should have strong evidence
        checks += 1
        if (
            output.confidence_score > 0.5
            and isinstance(output.hypotheses, list)
            and output.hypotheses
        ):
            top = output.hypotheses[0]
            if not top.evidence_supporting:
                warnings.append(ValidatorError(
                    check_number=24,
                    check_name="high_confidence_has_evidence",
                    error_description=(
                        "High confidence but top hypothesis "
                        "has no supporting evidence"
                    ),
                    expected="non-empty evidence_supporting",
                    actual="[]",
                    severity=ValidationSeverity.WARNING,
                ))

        # 25. severity should reflect top hypothesis
        checks += 1
        if (
            isinstance(output.hypotheses, list)
            and output.hypotheses
        ):
            top_sev = output.hypotheses[0].severity
            severity_order = {
                Severity.CRITICAL: 4,
                Severity.HIGH: 3,
                Severity.MEDIUM: 2,
                Severity.LOW: 1,
            }
            if (
                severity_order.get(output.severity, 0)
                < severity_order.get(top_sev, 0)
            ):
                warnings.append(ValidatorError(
                    check_number=25,
                    check_name="severity_consistency",
                    error_description=(
                        f"Output severity '{output.severity.value}' "
                        f"lower than top hypothesis "
                        f"'{top_sev.value}'"
                    ),
                    expected=top_sev.value,
                    actual=output.severity.value,
                    severity=ValidationSeverity.WARNING,
                ))

        # 26. category should match top hypothesis
        checks += 1
        if (
            isinstance(output.hypotheses, list)
            and output.hypotheses
        ):
            top_cat = output.hypotheses[0].category
            if output.category != top_cat:
                warnings.append(ValidatorError(
                    check_number=26,
                    check_name="category_consistency",
                    error_description=(
                        f"Output category '{output.category.value}' "
                        f"differs from top hypothesis "
                        f"'{top_cat.value}'"
                    ),
                    expected=top_cat.value,
                    actual=output.category.value,
                    severity=ValidationSeverity.WARNING,
                ))

        # 27. MTTR should match recommended hypothesis
        checks += 1
        if (
            isinstance(output.hypotheses, list)
            and output.hypotheses
        ):
            top_mttr = output.hypotheses[0].estimated_mttr_minutes
            if abs(output.estimated_mttr_minutes - top_mttr) > 0.1:
                warnings.append(ValidatorError(
                    check_number=27,
                    check_name="mttr_consistency",
                    error_description=(
                        f"Output MTTR {output.estimated_mttr_minutes} "
                        f"doesn't match top hypothesis "
                        f"MTTR {top_mttr}"
                    ),
                    expected=str(top_mttr),
                    actual=str(output.estimated_mttr_minutes),
                    severity=ValidationSeverity.WARNING,
                ))

        # ── Assemble result ─────────────────────────────────────

        elapsed_ms = (time.perf_counter() - start) * 1000

        has_critical = any(
            e.severity == ValidationSeverity.CRITICAL
            for e in errors
        )
        passed = not has_critical

        result = ValidationResult(
            validation_passed=passed,
            total_checks=checks,
            errors=errors,
            warnings=warnings,
            validation_latency_ms=round(elapsed_ms, 2),
        )

        log_fn = logger.info if passed else logger.warning
        log_fn(
            f"Validation {'passed' if passed else 'FAILED'} — "
            f"{checks} checks, "
            f"{len(errors)} errors, "
            f"{len(warnings)} warnings, "
            f"{elapsed_ms:.2f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "validation",
                "context": {
                    "passed": passed,
                    "checks": checks,
                    "errors": len(errors),
                    "warnings": len(warnings),
                },
            },
        )

        return result
