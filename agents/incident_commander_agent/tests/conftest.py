"""
conftest.py — shared fixtures for Incident Commander Agent tests.
"""

from __future__ import annotations

import pytest

from agents.root_cause_agent.schema import (
    CausalLink,
    Evidence,
    EvidenceSourceAgent,
    EvidenceType,
    IncidentCategory,
    RootCauseVerdict,
    Severity,
    TimelineEvent,
)
from agents.validation_agent.schema import (
    Hallucination,
    HallucinationType,
    ValidationReport,
)
from agents.incident_commander_agent.config import IncidentCommanderConfig
from agents.incident_commander_agent.schema import IncidentCommanderInput


# ─── Helper factories ───────────────────────────────────────


def make_verdict(
    root_cause: str = "database_connection_pool_exhaustion",
    confidence: float = 0.85,
    affected_services: list[str] | None = None,
    causal_chain: list[CausalLink] | None = None,
    timeline: list[TimelineEvent] | None = None,
) -> RootCauseVerdict:
    """Build a RootCauseVerdict with sane defaults."""
    return RootCauseVerdict(
        root_cause=root_cause,
        confidence=confidence,
        affected_services=affected_services or [
            "payment-service", "order-service",
        ],
        causal_chain=causal_chain or [
            CausalLink(
                cause="database_connection_pool_exhaustion",
                effect="payment-service latency spike",
                confidence=0.9,
            ),
        ],
        timeline=timeline or [
            TimelineEvent(
                timestamp="2025-01-15T10:00:00Z",
                event="Connection pool utilization exceeded 90%",
                source=EvidenceSourceAgent.METRICS_AGENT,
            ),
        ],
        evidence_trail=[
            Evidence(
                source=EvidenceSourceAgent.METRICS_AGENT,
                evidence_type=EvidenceType.DIRECT,
                description="High connection pool utilization",
                confidence=0.9,
            ),
        ],
        reasoning="Database connection pool exhaustion detected via metrics.",
        category=IncidentCategory.DATABASE,
        severity=Severity.HIGH,
        correlation_id="test-corr-001",
    )


def make_validation_report(
    accuracy_score: float = 0.9,
    hallucinations: list | None = None,
) -> ValidationReport:
    """Build a ValidationReport with sane defaults."""
    return ValidationReport(
        verdict_correct=True,
        accuracy_score=accuracy_score,
        precision=0.85,
        recall=0.90,
        f1_score=0.87,
        hallucinations=hallucinations or [],
        correlation_id="test-corr-001",
    )


def make_input(
    root_cause: str = "database_connection_pool_exhaustion",
    confidence: float = 0.85,
    affected_services: list[str] | None = None,
    accuracy_score: float = 0.9,
    hallucinations: list | None = None,
    correlation_id: str = "test-corr-001",
) -> IncidentCommanderInput:
    """Build an IncidentCommanderInput with sane defaults."""
    return IncidentCommanderInput(
        verdict=make_verdict(
            root_cause=root_cause,
            confidence=confidence,
            affected_services=affected_services,
        ),
        validation_report=make_validation_report(
            accuracy_score=accuracy_score,
            hallucinations=hallucinations,
        ),
        correlation_id=correlation_id,
    )


# ─── Fixtures ───────────────────────────────────────────────


@pytest.fixture
def verdict() -> RootCauseVerdict:
    return make_verdict()


@pytest.fixture
def validation_report() -> ValidationReport:
    return make_validation_report()


@pytest.fixture
def commander_input() -> IncidentCommanderInput:
    return make_input()


@pytest.fixture
def config() -> IncidentCommanderConfig:
    return IncidentCommanderConfig()
