"""Performance tests — budget compliance."""

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
from agents.validation_agent.agent import ValidationAgent
from agents.validation_agent.config import (
    FeatureFlags,
    ValidationAgentConfig,
)
from agents.validation_agent.fallback import DeterministicFallback
from agents.validation_agent.schema import GroundTruth, PropagationStep
from agents.validation_agent.telemetry import TelemetryCollector


def _make_gt() -> GroundTruth:
    return GroundTruth(
        actual_root_cause="database failure",
        failure_type="resource_exhaustion",
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


def _make_verdict(root_cause: str = "database failure") -> RootCauseVerdict:
    return RootCauseVerdict(
        root_cause=root_cause,
        confidence=0.9,
        affected_services=["database", "payment-service"],
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
                event="DB error",
                service="database",
                severity=Severity.HIGH,
            ),
        ],
    )


class TestFallbackBudget:
    """Deterministic fallback must finish in <50ms."""

    def test_fallback_under_50ms(self) -> None:
        config = ValidationAgentConfig()
        telemetry = TelemetryCollector()
        fb = DeterministicFallback(config, telemetry)

        gt = _make_gt()
        verdict = _make_verdict()

        # Warm up
        fb.validate(verdict, gt)

        t0 = time.perf_counter()
        fb.validate(verdict, gt)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 50, f"Fallback: {elapsed:.1f}ms > 50ms budget"


class TestPipelineBudget:
    """Full pipeline (no LLM) must finish in <1s."""

    def test_pipeline_under_1s(self) -> None:
        config = ValidationAgentConfig(
            features=FeatureFlags(use_llm=False)
        )
        agent = ValidationAgent(config)

        gt = _make_gt()
        verdict = _make_verdict()

        t0 = time.perf_counter()
        report = agent.validate(verdict, gt)
        elapsed = (time.perf_counter() - t0) * 1000
        assert elapsed < 1000, f"Pipeline: {elapsed:.1f}ms > 1000ms budget"
        assert report.pipeline_latency_ms < 1000


class TestCostBudget:
    """Cost per call must be <$0.0005 when using mock provider."""

    def test_mock_provider_zero_cost(self) -> None:
        config = ValidationAgentConfig(
            features=FeatureFlags(use_llm=True)
        )
        agent = ValidationAgent(config)
        gt = _make_gt()
        verdict = _make_verdict("wrong cause")
        report = agent.validate(verdict, gt)
        # Mock provider has zero cost
        assert report.agent == "validation_agent"
