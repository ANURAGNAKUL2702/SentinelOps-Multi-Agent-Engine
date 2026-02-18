"""Tests for evidence_validator.py — Algorithm 5."""

from __future__ import annotations

import pytest

from agents.root_cause_agent.schema import (
    Evidence,
    EvidenceSourceAgent,
    EvidenceType,
)
from agents.validation_agent.core.evidence_validator import (
    validate_evidence,
)
from agents.validation_agent.schema import GroundTruth


def _make_evidence(
    description: str = "",
    service: str = "",
    raw_data: dict | None = None,
) -> Evidence:
    """Helper to create evidence items."""
    return Evidence(
        source=EvidenceSourceAgent.LOG_AGENT,
        evidence_type=EvidenceType.DIRECT,
        description=description,
        confidence=0.8,
        raw_data=raw_data or {},
    )


def _make_ground_truth(
    services: list | None = None,
    symptoms: list | None = None,
    root_cause: str = "database_connection_pool_exhaustion",
) -> GroundTruth:
    """Helper to create ground truth."""
    return GroundTruth(
        actual_root_cause=root_cause,
        affected_services_ground_truth=services or ["database", "payment-service"],
        expected_symptoms=symptoms or ["high_latency", "connection_errors"],
    )


class TestValidateEvidence:
    """Tests for evidence validation."""

    def test_all_evidence_correct(self) -> None:
        """All evidence matches ground truth → accuracy=1.0."""
        evidence = [
            _make_evidence(
                description="high_latency detected on database",
                raw_data={"service": "database"},
            ),
            _make_evidence(
                description="connection_errors in payment-service",
                raw_data={"service": "payment-service"},
            ),
        ]
        gt = _make_ground_truth()
        accuracy, hallucinations = validate_evidence(evidence, gt)
        assert accuracy == 1.0
        assert hallucinations == []

    def test_fabricated_service(self) -> None:
        """Evidence references non-existent service → hallucination."""
        evidence = [
            _make_evidence(
                description="errors on fake-service",
                raw_data={"service": "fake-service"},
            ),
        ]
        gt = _make_ground_truth()
        accuracy, hallucinations = validate_evidence(evidence, gt)
        assert len(hallucinations) >= 1
        assert any(h.fabricated_value == "fake-service" for h in hallucinations)

    def test_partial_match(self) -> None:
        """Some evidence valid, some not."""
        evidence = [
            _make_evidence(
                description="database connection issues",
                raw_data={"service": "database"},
            ),
            _make_evidence(
                description="unrelated noise",
                raw_data={"service": "nonexistent-service"},
            ),
        ]
        gt = _make_ground_truth()
        accuracy, hallucinations = validate_evidence(evidence, gt)
        assert 0.0 < accuracy < 1.0

    def test_empty_evidence_trail(self) -> None:
        """Empty evidence → accuracy=0.0."""
        gt = _make_ground_truth()
        accuracy, hallucinations = validate_evidence([], gt)
        assert accuracy == 0.0
        assert hallucinations == []

    def test_evidence_with_symptom_match(self) -> None:
        """Evidence description matches expected symptom."""
        evidence = [
            _make_evidence(
                description="high_latency observed across services",
            ),
        ]
        gt = _make_ground_truth()
        accuracy, hallucinations = validate_evidence(evidence, gt)
        assert accuracy > 0.0

    def test_evidence_with_root_cause_keywords(self) -> None:
        """Evidence mentions root cause keywords."""
        evidence = [
            _make_evidence(
                description="database connection pool showing exhaustion patterns",
            ),
        ]
        gt = _make_ground_truth()
        accuracy, hallucinations = validate_evidence(evidence, gt)
        assert accuracy > 0.0

    def test_timestamp_tolerance(self) -> None:
        """Custom timestamp tolerance parameter accepted."""
        evidence = [
            _make_evidence(
                description="database error",
                raw_data={"service": "database"},
            ),
        ]
        gt = _make_ground_truth()
        accuracy, _ = validate_evidence(evidence, gt, timestamp_tolerance_seconds=60.0)
        assert accuracy > 0.0
