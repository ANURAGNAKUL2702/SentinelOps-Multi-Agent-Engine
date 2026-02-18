"""
File: core/hallucination_detector.py
Purpose: Detect fabricated services, events, and dependencies.
Dependencies: Standard library only.
Performance: <1ms per call, pure function.

Algorithm 7: Cross-reference verdict elements against ground truth
to find fabricated services, phantom dependencies, and fake metrics.
"""

from __future__ import annotations

from typing import List, Set

from agents.validation_agent.schema import (
    CausalLink,
    Evidence,
    GroundTruth,
    Hallucination,
    HallucinationType,
    PropagationStep,
    RootCauseVerdict,
)


def _get_ground_truth_services(ground_truth: GroundTruth) -> Set[str]:
    """Extract all known services from ground truth.

    Args:
        ground_truth: Simulation ground truth.

    Returns:
        Set of lowercase service names.
    """
    services: Set[str] = set()
    for s in ground_truth.affected_services_ground_truth:
        services.add(s.lower())
    for step in ground_truth.failure_propagation_chain:
        services.add(step.from_service.lower())
        services.add(step.to_service.lower())
    return services


def _get_ground_truth_edges(
    chain: List[PropagationStep],
) -> Set[tuple]:
    """Extract directed edges from propagation chain.

    Args:
        chain: Ground truth propagation chain.

    Returns:
        Set of (from_service, to_service) tuples (lowercase).
    """
    return {
        (step.from_service.lower(), step.to_service.lower())
        for step in chain
    }


def _detect_fabricated_services(
    verdict: RootCauseVerdict,
    gt_services: Set[str],
) -> List[Hallucination]:
    """Detect services in verdict not in ground truth.

    Args:
        verdict: The root cause verdict.
        gt_services: Known services from ground truth.

    Returns:
        List of Hallucination objects for fabricated services.
    """
    hallucinations: List[Hallucination] = []

    if not gt_services:
        return hallucinations

    for service in verdict.affected_services:
        if service.lower() not in gt_services:
            hallucinations.append(Hallucination(
                hallucination_type=HallucinationType.SERVICE,
                description=(
                    f"Verdict references service '{service}' "
                    f"not found in ground truth"
                ),
                fabricated_value=service,
                context="affected_services",
            ))

    return hallucinations


def _detect_phantom_dependencies(
    verdict: RootCauseVerdict,
    gt_edges: Set[tuple],
    gt_services: Set[str],
) -> List[Hallucination]:
    """Detect causal chain edges not in ground truth propagation.

    Args:
        verdict: The root cause verdict.
        gt_edges: Known edges from ground truth chain.
        gt_services: Known services from ground truth.

    Returns:
        List of Hallucination objects for phantom dependencies.
    """
    hallucinations: List[Hallucination] = []

    if not gt_edges and not gt_services:
        return hallucinations

    for link in verdict.causal_chain:
        cause_lower = link.cause.lower()
        effect_lower = link.effect.lower()

        # Check if the services in the causal link exist
        for svc_name, field_name in [
            (cause_lower, "cause"), (effect_lower, "effect")
        ]:
            if svc_name and gt_services and svc_name not in gt_services:
                # Check if it looks like a service name
                if any(c in svc_name for c in ("-", "_", "service")):
                    hallucinations.append(Hallucination(
                        hallucination_type=HallucinationType.DEPENDENCY,
                        description=(
                            f"Causal chain references '{svc_name}' "
                            f"({field_name}) not in ground truth services"
                        ),
                        fabricated_value=svc_name,
                        context=f"causal_chain.{field_name}",
                    ))

        # Check if the edge exists in ground truth
        if gt_edges:
            edge = (cause_lower, effect_lower)
            reverse_edge = (effect_lower, cause_lower)
            if (
                edge not in gt_edges
                and reverse_edge not in gt_edges
                and link.service
                and link.service.lower() not in gt_services
            ):
                hallucinations.append(Hallucination(
                    hallucination_type=HallucinationType.DEPENDENCY,
                    description=(
                        f"Causal link '{link.cause}' → '{link.effect}' "
                        f"not found in ground truth propagation chain"
                    ),
                    fabricated_value=(
                        f"{link.cause} -> {link.effect}"
                    ),
                    context="causal_chain",
                ))

    return hallucinations


def _detect_fake_metrics(
    verdict: RootCauseVerdict,
    ground_truth: GroundTruth,
) -> List[Hallucination]:
    """Detect references to metrics/symptoms not in ground truth.

    Args:
        verdict: The root cause verdict.
        ground_truth: Simulation ground truth.

    Returns:
        List of Hallucination objects for fake metrics.
    """
    hallucinations: List[Hallucination] = []

    if not ground_truth.expected_symptoms:
        return hallucinations

    gt_symptoms = set(s.lower() for s in ground_truth.expected_symptoms)

    for evidence in verdict.evidence_trail:
        raw = evidence.raw_data
        if isinstance(raw, dict):
            metric_name = raw.get("metric", "") or raw.get(
                "metric_name", ""
            )
            if metric_name and isinstance(metric_name, str):
                metric_lower = metric_name.lower()
                # Check if this metric relates to any expected symptom
                is_relevant = any(
                    symptom in metric_lower or metric_lower in symptom
                    for symptom in gt_symptoms
                )
                if not is_relevant:
                    # Also check general relevance
                    is_relevant = any(
                        word in metric_lower
                        for word in (
                            "error", "latency", "timeout",
                            "failure", "cpu", "memory",
                        )
                    )
                if not is_relevant:
                    hallucinations.append(Hallucination(
                        hallucination_type=HallucinationType.METRIC,
                        description=(
                            f"Evidence references metric '{metric_name}' "
                            f"not matching expected symptoms"
                        ),
                        fabricated_value=metric_name,
                        context=f"evidence: {evidence.description[:80]}",
                    ))

    return hallucinations


def detect_hallucinations(
    verdict: RootCauseVerdict,
    ground_truth: GroundTruth,
    check_services: bool = True,
    check_dependencies: bool = True,
    check_metrics: bool = True,
) -> List[Hallucination]:
    """Detect all types of hallucinations in the verdict.

    Args:
        verdict: The root cause verdict to check.
        ground_truth: Simulation ground truth.
        check_services: Whether to check for fabricated services.
        check_dependencies: Whether to check for phantom deps.
        check_metrics: Whether to check for fake metrics.

    Returns:
        List of all detected Hallucination objects.
    """
    gt_services = _get_ground_truth_services(ground_truth)
    gt_edges = _get_ground_truth_edges(
        ground_truth.failure_propagation_chain
    )

    hallucinations: List[Hallucination] = []

    if check_services:
        hallucinations.extend(
            _detect_fabricated_services(verdict, gt_services)
        )

    if check_dependencies:
        hallucinations.extend(
            _detect_phantom_dependencies(verdict, gt_edges, gt_services)
        )

    if check_metrics:
        hallucinations.extend(
            _detect_fake_metrics(verdict, ground_truth)
        )

    return hallucinations
