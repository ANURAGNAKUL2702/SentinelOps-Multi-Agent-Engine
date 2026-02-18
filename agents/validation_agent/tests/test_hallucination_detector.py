"""Tests for hallucination_detector.py — Algorithm 7."""

from __future__ import annotations

import pytest

from agents.root_cause_agent.schema import (
    CausalLink,
    Evidence,
    EvidenceSourceAgent,
    EvidenceType,
    RootCauseVerdict,
)
from agents.validation_agent.core.hallucination_detector import (
    detect_hallucinations,
)
from agents.validation_agent.schema import (
    GroundTruth,
    HallucinationType,
    PropagationStep,
)


def _make_gt(
    services: list[str] | None = None,
    chain: list[PropagationStep] | None = None,
    symptoms: list[str] | None = None,
) -> GroundTruth:
    return GroundTruth(
        actual_root_cause="database failure",
        affected_services_ground_truth=services or ["database", "payment-service"],
        failure_propagation_chain=chain
        or [
            PropagationStep(
                from_service="database", to_service="payment-service"
            )
        ],
        expected_symptoms=symptoms or ["high latency", "connection timeout"],
    )


def _make_verdict(
    services: list[str] | None = None,
    causal_chain: list[CausalLink] | None = None,
    evidence: list[Evidence] | None = None,
) -> RootCauseVerdict:
    return RootCauseVerdict(
        root_cause="database failure",
        confidence=0.9,
        affected_services=services or ["database", "payment-service"],
        causal_chain=causal_chain or [],
        evidence_trail=evidence or [],
    )


class TestNoHallucinations:
    def test_matching_services(self) -> None:
        gt = _make_gt()
        verdict = _make_verdict()
        result = detect_hallucinations(verdict, gt)
        # All services exist in ground truth
        service_hals = [
            h for h in result if h.hallucination_type == HallucinationType.SERVICE
        ]
        assert len(service_hals) == 0


class TestFabricatedServices:
    def test_unknown_service(self) -> None:
        gt = _make_gt()
        verdict = _make_verdict(services=["database", "ghost-service"])
        result = detect_hallucinations(verdict, gt)
        service_hals = [
            h for h in result if h.hallucination_type == HallucinationType.SERVICE
        ]
        assert len(service_hals) >= 1
        assert any("ghost-service" in h.fabricated_value for h in service_hals)

    def test_all_fabricated(self) -> None:
        gt = _make_gt()
        verdict = _make_verdict(services=["unknown-a", "unknown-b"])
        result = detect_hallucinations(verdict, gt)
        service_hals = [
            h for h in result if h.hallucination_type == HallucinationType.SERVICE
        ]
        assert len(service_hals) == 2


class TestPhantomDependencies:
    def test_phantom_causal_link(self) -> None:
        gt = _make_gt()
        verdict = _make_verdict(
            causal_chain=[
                CausalLink(
                    cause="ghost-service",
                    effect="phantom-service",
                    service="ghost-service",
                )
            ]
        )
        result = detect_hallucinations(verdict, gt)
        dep_hals = [
            h for h in result if h.hallucination_type == HallucinationType.DEPENDENCY
        ]
        assert len(dep_hals) >= 1


class TestFakeMetrics:
    def test_irrelevant_metric(self) -> None:
        gt = _make_gt(symptoms=["high latency"])
        evidence = [
            Evidence(
                source=EvidenceSourceAgent.METRICS_AGENT,
                evidence_type=EvidenceType.DIRECT,
                service="database",
                description="custom metric",
                raw_data={"metric": "unicorn_rainbow_count"},
            )
        ]
        verdict = _make_verdict(evidence=evidence)
        result = detect_hallucinations(verdict, gt)
        metric_hals = [
            h for h in result if h.hallucination_type == HallucinationType.METRIC
        ]
        assert len(metric_hals) >= 1

    def test_relevant_metric_no_hallucination(self) -> None:
        gt = _make_gt(symptoms=["high latency"])
        evidence = [
            Evidence(
                source=EvidenceSourceAgent.METRICS_AGENT,
                evidence_type=EvidenceType.DIRECT,
                service="database",
                description="latency metric",
                raw_data={"metric": "error_rate"},
            )
        ]
        verdict = _make_verdict(evidence=evidence)
        result = detect_hallucinations(verdict, gt)
        metric_hals = [
            h for h in result if h.hallucination_type == HallucinationType.METRIC
        ]
        assert len(metric_hals) == 0


class TestFeatureFlags:
    def test_disable_service_check(self) -> None:
        gt = _make_gt()
        verdict = _make_verdict(services=["ghost-service"])
        result = detect_hallucinations(
            verdict, gt, check_services=False
        )
        service_hals = [
            h for h in result if h.hallucination_type == HallucinationType.SERVICE
        ]
        assert len(service_hals) == 0

    def test_disable_all_checks(self) -> None:
        gt = _make_gt()
        verdict = _make_verdict(services=["ghost"])
        result = detect_hallucinations(
            verdict, gt,
            check_services=False,
            check_dependencies=False,
            check_metrics=False,
        )
        assert result == []
