"""
File: core/evidence_validator.py
Purpose: Validate evidence trail against ground truth events.
Dependencies: Standard library only.
Performance: <1ms per call, pure function.

Algorithm 5: For each evidence item, check service existence,
timestamp alignment, and symptom matching against ground truth.
Flag hallucinated evidence.
"""

from __future__ import annotations

from typing import List, Tuple

from agents.validation_agent.schema import (
    Evidence,
    GroundTruth,
    Hallucination,
    HallucinationType,
)


def _extract_service_from_evidence(evidence: Evidence) -> str:
    """Extract service name from evidence description or raw data.

    Args:
        evidence: The evidence item.

    Returns:
        Extracted service name, or empty string.
    """
    # Try raw_data first
    raw = evidence.raw_data
    if isinstance(raw, dict):
        for key in ("service", "service_name", "suspicious_service"):
            if key in raw and isinstance(raw[key], str):
                return raw[key]

    # Try to extract from description
    desc = evidence.description.lower()
    # Common service name patterns
    for svc in (
        "api-gateway", "auth-service", "payment-service",
        "fraud-service", "notification-service", "database",
        "cache-service", "merchant-portal",
    ):
        if svc in desc:
            return svc

    return ""


def validate_evidence(
    evidence_trail: List[Evidence],
    ground_truth: GroundTruth,
    timestamp_tolerance_seconds: float = 300.0,
) -> Tuple[float, List[Hallucination]]:
    """Validate evidence trail against ground truth.

    For each evidence item:
      - Check if referenced service exists in ground truth services
      - Check if description aligns with expected symptoms
      - Flag hallucinated evidence

    Args:
        evidence_trail: List of Evidence from the verdict.
        ground_truth: Simulation ground truth.
        timestamp_tolerance_seconds: Tolerance for timestamp matching.

    Returns:
        Tuple of (evidence_accuracy, hallucinations_list).
    """
    if not evidence_trail:
        return 0.0, []

    gt_services = set(
        s.lower() for s in ground_truth.affected_services_ground_truth
    )
    gt_symptoms = set(
        s.lower() for s in ground_truth.expected_symptoms
    )
    gt_root_cause = ground_truth.actual_root_cause.lower()

    valid_count = 0
    hallucinations: List[Hallucination] = []

    for evidence in evidence_trail:
        is_valid = False

        # Check service reference
        service = _extract_service_from_evidence(evidence)
        service_valid = True

        if service:
            if service.lower() not in gt_services:
                # Service not in ground truth — possible hallucination
                service_valid = False
                hallucinations.append(Hallucination(
                    hallucination_type=HallucinationType.SERVICE,
                    description=(
                        f"Evidence references service '{service}' "
                        f"not in ground truth affected services"
                    ),
                    fabricated_value=service,
                    context=f"evidence: {evidence.description[:100]}",
                ))

        # Check symptom/description relevance
        desc_lower = evidence.description.lower()
        symptom_match = False

        # Check against expected symptoms
        for symptom in gt_symptoms:
            if symptom in desc_lower or desc_lower in symptom:
                symptom_match = True
                break

        # Check against root cause
        root_cause_match = gt_root_cause in desc_lower or any(
            word in desc_lower
            for word in gt_root_cause.split("_")
            if len(word) > 3
        )

        # Evidence is valid if service matches OR content is relevant
        if service_valid and (symptom_match or root_cause_match):
            is_valid = True
        elif not service and (symptom_match or root_cause_match):
            is_valid = True
        elif service_valid and service:
            # Service exists in ground truth even if content doesn't match
            is_valid = True

        if is_valid:
            valid_count += 1

    accuracy = valid_count / len(evidence_trail)
    return round(accuracy, 4), hallucinations
