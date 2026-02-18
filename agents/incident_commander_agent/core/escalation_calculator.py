"""
File: core/escalation_calculator.py
Purpose: Algorithm 8 – Decide whether to escalate to a human SRE.
Dependencies: Standard library only + schema + config.
Performance: <1ms per call, pure function.
"""

from __future__ import annotations

from typing import List

from agents.incident_commander_agent.config import EscalationConfig
from agents.incident_commander_agent.schema import (
    BlastRadius,
    EscalationDecision,
    RootCauseVerdict,
    ValidationReport,
)


def calculate_escalation(
    verdict: RootCauseVerdict,
    validation_report: ValidationReport,
    blast_radius: BlastRadius | None = None,
    config: EscalationConfig | None = None,
) -> EscalationDecision:
    """Decide whether to escalate the incident to a human.

    Escalation triggers (any one is sufficient):
    1. Confidence below threshold (verdict.confidence).
    2. Accuracy below threshold (validation_report.accuracy_score).
    3. High user impact (blast_radius.estimated_users).
    4. Hallucinations detected in validation report.
    5. Verdict root cause unknown.

    Args:
        verdict: Root cause verdict from upstream.
        validation_report: Validation report from upstream.
        blast_radius: Computed blast radius (defaults to empty).
        config: Escalation config overrides.

    Returns:
        EscalationDecision with reasoning.
    """
    cfg = config or EscalationConfig()
    br = blast_radius or BlastRadius()

    reasons: List[str] = []
    escalate = False

    # Check 1: Low confidence
    if verdict.confidence < cfg.low_confidence_threshold:
        escalate = True
        reasons.append(
            f"Low confidence ({verdict.confidence:.0%} < "
            f"{cfg.low_confidence_threshold:.0%} threshold)"
        )

    # Check 2: Low accuracy
    if validation_report.accuracy_score < cfg.low_accuracy_threshold:
        escalate = True
        reasons.append(
            f"Low validation accuracy ({validation_report.accuracy_score:.0%} < "
            f"{cfg.low_accuracy_threshold:.0%} threshold)"
        )

    # Check 3: High user impact
    if br.estimated_users > cfg.high_impact_user_threshold:
        escalate = True
        reasons.append(
            f"High user impact ({br.estimated_users:,} > "
            f"{cfg.high_impact_user_threshold:,} threshold)"
        )

    # Check 4: Hallucinations detected
    if validation_report.hallucinations:
        escalate = True
        reasons.append(
            f"{len(validation_report.hallucinations)} "
            f"hallucination(s) detected in analysis"
        )

    # Check 5: Unknown root cause
    root_cause = verdict.root_cause.lower() if verdict.root_cause else ""
    if "unknown" in root_cause or not root_cause:
        escalate = True
        reasons.append("Root cause is unknown or empty")

    # Compute auto-resolve confidence
    auto_confidence = _auto_resolve_confidence(
        verdict.confidence,
        validation_report.accuracy_score,
        len(validation_report.hallucinations),
    )

    if escalate:
        reason_text = "; ".join(reasons)
        return EscalationDecision(
            should_escalate=True,
            reason=f"Escalation needed: {reason_text}.",
            suggested_escalation_path=_escalation_path(br, reasons),
            auto_resolve_confidence=auto_confidence,
        )

    return EscalationDecision(
        should_escalate=False,
        reason="All checks passed — safe for automated resolution.",
        suggested_escalation_path="",
        auto_resolve_confidence=auto_confidence,
    )


def _auto_resolve_confidence(
    confidence: float,
    accuracy: float,
    hallucination_count: int,
) -> float:
    """Compute confidence that the issue can be auto-resolved.

    Combines verdict confidence and validation accuracy, then
    penalizes for each hallucination detected.
    """
    base = (confidence + accuracy) / 2.0
    penalty = hallucination_count * 0.1
    return max(0.0, min(1.0, round(base - penalty, 4)))


def _escalation_path(
    blast_radius: BlastRadius,
    reasons: List[str],
) -> str:
    """Suggest who to escalate to based on blast radius."""
    if blast_radius.estimated_users > 50_000:
        return "VP Engineering + On-call SRE Lead"
    if blast_radius.is_customer_facing:
        return "On-call SRE Lead + Customer Success"
    return "On-call SRE"
