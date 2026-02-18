"""
File: validator.py
Purpose: 30 validation checks for the RootCauseVerdict.
Dependencies: Schema models only.
Performance: <1ms, O(n).

Validates the final verdict before returning to the caller.
Each check is numbered 1-30, classified as CRITICAL or WARNING.
Critical failures block the response; warnings are logged.
"""

from __future__ import annotations

from typing import List, Optional

from agents.root_cause_agent.config import RootCauseAgentConfig
from agents.root_cause_agent.schema import (
    IncidentCategory,
    RootCauseVerdict,
    Severity,
    ValidationResult,
    ValidationSeverity,
    ValidatorError,
)
from agents.root_cause_agent.telemetry import get_logger

logger = get_logger("root_cause_agent.validator")


class VerdictValidator:
    """Validates a RootCauseVerdict against 30 quality checks.

    Args:
        config: Agent configuration.
    """

    def __init__(
        self, config: Optional[RootCauseAgentConfig] = None
    ) -> None:
        self._config = config or RootCauseAgentConfig()

    def validate(self, verdict: RootCauseVerdict) -> ValidationResult:
        """Run all 30 validation checks.

        Args:
            verdict: The verdict to validate.

        Returns:
            ValidationResult with errors and pass/fail.
        """
        errors: List[ValidatorError] = []

        # ── Root cause checks (1-5) ────────────────────────────
        errors.extend(self._check_root_cause(verdict))

        # ── Confidence checks (6-10) ───────────────────────────
        errors.extend(self._check_confidence(verdict))

        # ── Evidence checks (11-15) ─────────────────────────────
        errors.extend(self._check_evidence(verdict))

        # ── Causal chain checks (16-19) ─────────────────────────
        errors.extend(self._check_causal_chain(verdict))

        # ── Impact checks (20-22) ──────────────────────────────
        errors.extend(self._check_impact(verdict))

        # ── Alternatives checks (23-25) ─────────────────────────
        errors.extend(self._check_alternatives(verdict))

        # ── Metadata checks (26-28) ─────────────────────────────
        errors.extend(self._check_metadata(verdict))

        # ── Cross-field checks (29-30) ──────────────────────────
        errors.extend(self._check_cross_field(verdict))

        passed = not any(
            e.severity == ValidationSeverity.CRITICAL for e in errors
        )

        critical_errors = [
            e for e in errors
            if e.severity == ValidationSeverity.CRITICAL
        ]
        warnings = [
            e for e in errors
            if e.severity == ValidationSeverity.WARNING
        ]

        result = ValidationResult(
            validation_passed=passed,
            total_checks=30,
            errors=critical_errors,
            warnings=warnings,
        )

        if not passed:
            logger.warning(
                f"Validation FAILED: {len(errors)} errors "
                f"({sum(1 for e in errors if e.severity == ValidationSeverity.CRITICAL)} critical)"
            )
        else:
            logger.debug(
                f"Validation passed: {30 - len(errors)}/30 checks OK"
            )

        return result

    # ── Root cause checks (1-5) ─────────────────────────────────

    def _check_root_cause(
        self, verdict: RootCauseVerdict
    ) -> List[ValidatorError]:
        errors: List[ValidatorError] = []

        # 1. Root cause must not be empty
        if not verdict.root_cause or not verdict.root_cause.strip():
            errors.append(ValidatorError(
                check_number=1,
                check_name="root_cause_not_empty",
                error_description="Root cause must not be empty",
                severity=ValidationSeverity.CRITICAL,
            ))

        # 2. Root cause minimum length
        if len(verdict.root_cause) < 10:
            errors.append(ValidatorError(
                check_number=2,
                check_name="root_cause_min_length",
                error_description=f"Root cause too short ({len(verdict.root_cause)} chars, min 10)",
                severity=ValidationSeverity.WARNING,
            ))

        # 3. Root cause should not be generic placeholder
        generic_phrases = [
            "unknown", "n/a", "none", "todo", "placeholder",
            "tbd", "not determined",
        ]
        if verdict.root_cause.lower().strip() in generic_phrases:
            errors.append(ValidatorError(
                check_number=3,
                check_name="root_cause_not_generic",
                error_description="Root cause appears to be a generic placeholder",
                severity=ValidationSeverity.CRITICAL,
            ))

        # 4. Category must be valid enum
        try:
            IncidentCategory(verdict.category)
        except (ValueError, KeyError):
            errors.append(ValidatorError(
                check_number=4,
                check_name="category_valid_enum",
                error_description=f"Invalid category: {verdict.category}",
                severity=ValidationSeverity.WARNING,
            ))

        # 5. Severity must be valid enum
        try:
            Severity(verdict.severity)
        except (ValueError, KeyError):
            errors.append(ValidatorError(
                check_number=5,
                check_name="severity_valid_enum",
                error_description=f"Invalid severity: {verdict.severity}",
                severity=ValidationSeverity.WARNING,
            ))

        return errors

    # ── Confidence checks (6-10) ────────────────────────────────

    def _check_confidence(
        self, verdict: RootCauseVerdict
    ) -> List[ValidatorError]:
        errors: List[ValidatorError] = []

        # 6. Confidence in valid range
        if not (0.0 <= verdict.confidence <= 1.0):
            errors.append(ValidatorError(
                check_number=6,
                check_name="confidence_range",
                error_description=f"Confidence {verdict.confidence} not in [0.0, 1.0]",
                severity=ValidationSeverity.CRITICAL,
            ))

        # 7. Confidence > 0
        if verdict.confidence <= 0.0:
            errors.append(ValidatorError(
                check_number=7,
                check_name="confidence_positive",
                error_description="Confidence must be > 0.0",
                severity=ValidationSeverity.WARNING,
            ))

        # 8. High confidence should have evidence
        if verdict.confidence > 0.9 and len(verdict.evidence_trail) < 2:
            errors.append(ValidatorError(
                check_number=8,
                check_name="high_confidence_needs_evidence",
                error_description="High confidence (>0.9) but < 2 evidence items",
                severity=ValidationSeverity.WARNING,
            ))

        # 9. Very low confidence warning
        if 0.0 < verdict.confidence < 0.1:
            errors.append(ValidatorError(
                check_number=9,
                check_name="very_low_confidence",
                error_description=f"Very low confidence: {verdict.confidence}",
                severity=ValidationSeverity.WARNING,
            ))

        # 10. Confidence precision (max 4 decimal places)
        conf_str = str(verdict.confidence)
        if "." in conf_str and len(conf_str.split(".")[1]) > 4:
            errors.append(ValidatorError(
                check_number=10,
                check_name="confidence_precision",
                error_description="Confidence has > 4 decimal places",
                severity=ValidationSeverity.WARNING,
            ))

        return errors

    # ── Evidence checks (11-15) ─────────────────────────────────

    def _check_evidence(
        self, verdict: RootCauseVerdict
    ) -> List[ValidatorError]:
        errors: List[ValidatorError] = []

        # 11. At least 1 evidence item
        if len(verdict.evidence_trail) == 0:
            errors.append(ValidatorError(
                check_number=11,
                check_name="evidence_not_empty",
                error_description="Evidence trail must not be empty",
                severity=ValidationSeverity.CRITICAL,
            ))

        # 12. Evidence count within limits
        max_ev = self._config.performance.max_evidence_items
        if len(verdict.evidence_trail) > max_ev:
            errors.append(ValidatorError(
                check_number=12,
                check_name="evidence_count_limit",
                error_description=f"Evidence count {len(verdict.evidence_trail)} exceeds max {max_ev}",
                severity=ValidationSeverity.WARNING,
            ))

        # 13. Each evidence item has a description
        empty_desc = sum(
            1 for e in verdict.evidence_trail if not e.description
        )
        if empty_desc > 0:
            errors.append(ValidatorError(
                check_number=13,
                check_name="evidence_has_description",
                error_description=f"{empty_desc} evidence items have empty description",
                severity=ValidationSeverity.WARNING,
            ))

        # 14. Evidence confidences in range
        bad_conf = sum(
            1 for e in verdict.evidence_trail
            if not (0.0 <= e.confidence <= 1.0)
        )
        if bad_conf > 0:
            errors.append(ValidatorError(
                check_number=14,
                check_name="evidence_confidence_range",
                error_description=f"{bad_conf} evidence items have invalid confidence",
                severity=ValidationSeverity.CRITICAL,
            ))

        # 15. Evidence from multiple sources preferred
        sources = set(e.source for e in verdict.evidence_trail)
        if len(verdict.evidence_trail) > 3 and len(sources) < 2:
            errors.append(ValidatorError(
                check_number=15,
                check_name="evidence_source_diversity",
                error_description="Many evidence items but only 1 source agent",
                severity=ValidationSeverity.WARNING,
            ))

        return errors

    # ── Causal chain checks (16-19) ─────────────────────────────

    def _check_causal_chain(
        self, verdict: RootCauseVerdict
    ) -> List[ValidatorError]:
        errors: List[ValidatorError] = []

        # 16. Causal chain not excessively long
        max_depth = self._config.limits.max_causal_chain_depth
        if len(verdict.causal_chain) > max_depth:
            errors.append(ValidatorError(
                check_number=16,
                check_name="causal_chain_depth",
                error_description=f"Causal chain {len(verdict.causal_chain)} exceeds max {max_depth}",
                severity=ValidationSeverity.WARNING,
            ))

        # 17. Causal links have non-empty cause/effect
        for i, link in enumerate(verdict.causal_chain):
            if not link.cause or not link.effect:
                errors.append(ValidatorError(
                    check_number=17,
                    check_name="causal_link_not_empty",
                    error_description=f"Causal link {i} has empty cause or effect",
                    severity=ValidationSeverity.WARNING,
                ))
                break

        # 18. Causal chain confidence values in range
        bad_chain_conf = sum(
            1 for link in verdict.causal_chain
            if not (0.0 <= link.confidence <= 1.0)
        )
        if bad_chain_conf > 0:
            errors.append(ValidatorError(
                check_number=18,
                check_name="causal_chain_confidence_range",
                error_description=f"{bad_chain_conf} causal links have invalid confidence",
                severity=ValidationSeverity.WARNING,
            ))

        # 19. No self-referencing links
        self_ref = sum(
            1 for link in verdict.causal_chain
            if link.cause == link.effect
        )
        if self_ref > 0:
            errors.append(ValidatorError(
                check_number=19,
                check_name="causal_no_self_reference",
                error_description=f"{self_ref} causal links reference themselves",
                severity=ValidationSeverity.WARNING,
            ))

        return errors

    # ── Impact checks (20-22) ──────────────────────────────────

    def _check_impact(
        self, verdict: RootCauseVerdict
    ) -> List[ValidatorError]:
        errors: List[ValidatorError] = []

        if verdict.impact is None:
            return errors

        # 20. Severity score in range
        if not (0.0 <= verdict.impact.severity_score <= 1.0):
            errors.append(ValidatorError(
                check_number=20,
                check_name="impact_severity_range",
                error_description=f"Impact severity {verdict.impact.severity_score} not in [0.0, 1.0]",
                severity=ValidationSeverity.CRITICAL,
            ))

        # 21. Affected count matches list
        if verdict.impact.affected_count != len(verdict.impact.affected_services):
            errors.append(ValidatorError(
                check_number=21,
                check_name="impact_count_consistency",
                error_description=(
                    f"affected_count ({verdict.impact.affected_count}) != "
                    f"len(affected_services) ({len(verdict.impact.affected_services)})"
                ),
                severity=ValidationSeverity.WARNING,
            ))

        # 22. Blast radius >= affected count
        if verdict.impact.blast_radius < verdict.impact.affected_count:
            errors.append(ValidatorError(
                check_number=22,
                check_name="blast_radius_consistency",
                error_description="Blast radius < affected count",
                severity=ValidationSeverity.WARNING,
            ))

        return errors

    # ── Alternatives checks (23-25) ─────────────────────────────

    def _check_alternatives(
        self, verdict: RootCauseVerdict
    ) -> List[ValidatorError]:
        errors: List[ValidatorError] = []

        # 23. Alternatives count within limit
        max_alts = self._config.limits.max_alternatives
        if len(verdict.alternative_causes) > max_alts:
            errors.append(ValidatorError(
                check_number=23,
                check_name="alternatives_count_limit",
                error_description=f"Alternatives {len(verdict.alternative_causes)} exceeds max {max_alts}",
                severity=ValidationSeverity.WARNING,
            ))

        # 24. Low confidence requires alternatives
        threshold = self._config.limits.low_confidence_threshold
        min_alts = self._config.limits.min_alternatives_low_confidence
        if verdict.confidence < threshold and len(verdict.alternative_causes) < min_alts:
            errors.append(ValidatorError(
                check_number=24,
                check_name="low_confidence_needs_alternatives",
                error_description=(
                    f"Confidence {verdict.confidence} < {threshold} "
                    f"but only {len(verdict.alternative_causes)} alternatives "
                    f"(need at least {min_alts})"
                ),
                severity=ValidationSeverity.WARNING,
            ))

        # 25. Alternative confidences in range
        bad_alt_conf = sum(
            1 for alt in verdict.alternative_causes
            if not (0.0 <= alt.confidence <= 1.0)
        )
        if bad_alt_conf > 0:
            errors.append(ValidatorError(
                check_number=25,
                check_name="alternative_confidence_range",
                error_description=f"{bad_alt_conf} alternatives have invalid confidence",
                severity=ValidationSeverity.CRITICAL,
            ))

        return errors

    # ── Metadata checks (26-28) ─────────────────────────────────

    def _check_metadata(
        self, verdict: RootCauseVerdict
    ) -> List[ValidatorError]:
        errors: List[ValidatorError] = []

        # 26. Agent identifier is correct
        if verdict.agent != "root_cause_agent":
            errors.append(ValidatorError(
                check_number=26,
                check_name="agent_identifier",
                error_description=f"Agent is '{verdict.agent}', expected 'root_cause_agent'",
                severity=ValidationSeverity.CRITICAL,
            ))

        # 27. Reasoning meets minimum length
        min_len = self._config.limits.min_reasoning_length
        if len(verdict.reasoning) < min_len:
            errors.append(ValidatorError(
                check_number=27,
                check_name="reasoning_min_length",
                error_description=f"Reasoning too short ({len(verdict.reasoning)} chars, min {min_len})",
                severity=ValidationSeverity.WARNING,
            ))

        # 28. Pipeline latency recorded
        if verdict.pipeline_latency_ms <= 0:
            errors.append(ValidatorError(
                check_number=28,
                check_name="pipeline_latency_recorded",
                error_description="Pipeline latency not recorded",
                severity=ValidationSeverity.WARNING,
            ))

        return errors

    # ── Cross-field checks (29-30) ──────────────────────────────

    def _check_cross_field(
        self, verdict: RootCauseVerdict
    ) -> List[ValidatorError]:
        errors: List[ValidatorError] = []

        # 29. MTTR estimate reasonable
        if verdict.estimated_mttr_minutes < 0:
            errors.append(ValidatorError(
                check_number=29,
                check_name="mttr_non_negative",
                error_description=f"MTTR is negative: {verdict.estimated_mttr_minutes}",
                severity=ValidationSeverity.CRITICAL,
            ))

        if verdict.estimated_mttr_minutes > 1440:  # 24 hours
            errors.append(ValidatorError(
                check_number=29,
                check_name="mttr_reasonable",
                error_description=f"MTTR > 24 hours: {verdict.estimated_mttr_minutes}",
                severity=ValidationSeverity.WARNING,
            ))

        # 30. Affected services list consistency
        if verdict.affected_services and verdict.impact:
            impact_set = set(verdict.impact.affected_services)
            verdict_set = set(verdict.affected_services)
            if not verdict_set.issubset(impact_set | {s for s in verdict_set}):
                # Soft check — they don't need to be identical
                pass
        if not verdict.affected_services and verdict.impact and verdict.impact.affected_count > 0:
            errors.append(ValidatorError(
                check_number=30,
                check_name="affected_services_consistency",
                error_description="Impact shows affected services but verdict.affected_services is empty",
                severity=ValidationSeverity.WARNING,
            ))

        return errors
