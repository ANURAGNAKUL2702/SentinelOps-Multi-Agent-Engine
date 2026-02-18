"""
Tests for core/confidence_calculator.py — Algorithm 2.
"""

import pytest

from agents.root_cause_agent.config import RootCauseAgentConfig
from agents.root_cause_agent.core.confidence_calculator import ConfidenceCalculator
from agents.root_cause_agent.schema import (
    Evidence,
    EvidenceSourceAgent,
    EvidenceType,
    SynthesisResult,
)


def _make_synthesis(**kwargs) -> SynthesisResult:
    defaults = dict(
        evidence_trail=[
            Evidence(
                source=EvidenceSourceAgent.LOG_AGENT,
                evidence_type=EvidenceType.DIRECT,
                confidence=0.8,
            ),
            Evidence(
                source=EvidenceSourceAgent.METRICS_AGENT,
                evidence_type=EvidenceType.CORRELATED,
                confidence=0.7,
            ),
        ],
        sources_present=[
            EvidenceSourceAgent.LOG_AGENT,
            EvidenceSourceAgent.METRICS_AGENT,
        ],
        agreement_score=0.6,
        primary_service="payment-service",
    )
    defaults.update(kwargs)
    return SynthesisResult(**defaults)


class TestConfidenceCalculator:
    def test_calculate_basic(self):
        calc = ConfidenceCalculator()
        conf = calc.calculate(
            _make_synthesis(),
            agent_confidences=[0.8, 0.7, 0.85, 0.88],
        )
        assert 0.0 < conf < 1.0

    def test_calculate_clamped_to_max(self):
        config = RootCauseAgentConfig()
        calc = ConfidenceCalculator(config)
        conf = calc.calculate(
            _make_synthesis(agreement_score=1.0),
            agent_confidences=[1.0, 1.0, 1.0, 1.0],
        )
        assert conf <= config.confidence.max_confidence

    def test_calculate_minimum_bound(self):
        config = RootCauseAgentConfig()
        calc = ConfidenceCalculator(config)
        conf = calc.calculate(
            _make_synthesis(agreement_score=0.0, sources_present=[]),
            agent_confidences=[],
        )
        assert conf >= config.confidence.min_confidence

    def test_more_agents_higher_confidence(self):
        calc = ConfidenceCalculator()
        conf_2 = calc.calculate(
            _make_synthesis(sources_present=[
                EvidenceSourceAgent.LOG_AGENT,
                EvidenceSourceAgent.METRICS_AGENT,
            ]),
            agent_confidences=[0.7, 0.7],
        )
        conf_4 = calc.calculate(
            _make_synthesis(sources_present=[
                EvidenceSourceAgent.LOG_AGENT,
                EvidenceSourceAgent.METRICS_AGENT,
                EvidenceSourceAgent.DEPENDENCY_AGENT,
                EvidenceSourceAgent.HYPOTHESIS_AGENT,
            ]),
            agent_confidences=[0.7, 0.7, 0.7, 0.7],
        )
        assert conf_4 > conf_2

    def test_higher_agreement_higher_confidence(self):
        calc = ConfidenceCalculator()
        conf_low = calc.calculate(
            _make_synthesis(agreement_score=0.2),
            agent_confidences=[0.7, 0.7],
        )
        conf_high = calc.calculate(
            _make_synthesis(agreement_score=0.9),
            agent_confidences=[0.7, 0.7],
        )
        assert conf_high > conf_low

    def test_calculate_from_evidence(self):
        calc = ConfidenceCalculator()
        evidence = [
            Evidence(
                source=EvidenceSourceAgent.LOG_AGENT,
                evidence_type=EvidenceType.DIRECT,
                confidence=0.9,
            ),
        ]
        conf = calc.calculate_from_evidence(evidence)
        assert 0.0 < conf < 1.0

    def test_calculate_from_empty_evidence(self):
        calc = ConfidenceCalculator()
        conf = calc.calculate_from_evidence([])
        config = RootCauseAgentConfig()
        assert conf == config.confidence.min_confidence
