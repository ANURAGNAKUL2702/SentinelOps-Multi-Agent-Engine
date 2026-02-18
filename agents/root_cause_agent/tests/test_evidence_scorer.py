"""
Tests for core/evidence_scorer.py — Algorithm 5.
"""

import pytest
from datetime import datetime, timezone, timedelta

from agents.root_cause_agent.config import RootCauseAgentConfig
from agents.root_cause_agent.core.evidence_scorer import EvidenceScorer
from agents.root_cause_agent.schema import (
    Evidence,
    EvidenceSourceAgent,
    EvidenceType,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hours_ago_iso(hours: float) -> str:
    dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class TestEvidenceScorer:
    def test_score_all_basic(self):
        scorer = EvidenceScorer()
        evidence = [
            Evidence(
                source=EvidenceSourceAgent.HYPOTHESIS_AGENT,
                evidence_type=EvidenceType.DIRECT,
                confidence=0.9,
                timestamp=_now_iso(),
            ),
        ]
        scored = scorer.score_all(evidence)
        assert len(scored) == 1
        assert scored[0].score > 0

    def test_hypothesis_scored_higher_than_log(self):
        scorer = EvidenceScorer()
        ts = _now_iso()
        evidence = [
            Evidence(
                source=EvidenceSourceAgent.HYPOTHESIS_AGENT,
                evidence_type=EvidenceType.DIRECT,
                confidence=0.9,
                timestamp=ts,
            ),
            Evidence(
                source=EvidenceSourceAgent.LOG_AGENT,
                evidence_type=EvidenceType.DIRECT,
                confidence=0.9,
                timestamp=ts,
            ),
        ]
        scored = scorer.score_all(evidence, reference_time=ts)
        hyp_score = scored[0].score
        log_score = scored[1].score
        assert hyp_score > log_score

    def test_direct_scored_higher_than_circumstantial(self):
        scorer = EvidenceScorer()
        ts = _now_iso()
        evidence = [
            Evidence(
                source=EvidenceSourceAgent.LOG_AGENT,
                evidence_type=EvidenceType.DIRECT,
                confidence=0.8,
                timestamp=ts,
            ),
            Evidence(
                source=EvidenceSourceAgent.LOG_AGENT,
                evidence_type=EvidenceType.CIRCUMSTANTIAL,
                confidence=0.8,
                timestamp=ts,
            ),
        ]
        scored = scorer.score_all(evidence, reference_time=ts)
        assert scored[0].score > scored[1].score

    def test_recency_decay(self):
        scorer = EvidenceScorer()
        now = _now_iso()
        old = _hours_ago_iso(12)
        evidence = [
            Evidence(
                source=EvidenceSourceAgent.LOG_AGENT,
                evidence_type=EvidenceType.DIRECT,
                confidence=0.8,
                timestamp=now,
            ),
            Evidence(
                source=EvidenceSourceAgent.LOG_AGENT,
                evidence_type=EvidenceType.DIRECT,
                confidence=0.8,
                timestamp=old,
            ),
        ]
        scored = scorer.score_all(evidence, reference_time=now)
        assert scored[0].score > scored[1].score

    def test_empty_evidence(self):
        scorer = EvidenceScorer()
        scored = scorer.score_all([])
        assert scored == []

    def test_score_non_negative(self):
        scorer = EvidenceScorer()
        evidence = [
            Evidence(
                source=EvidenceSourceAgent.METRICS_AGENT,
                evidence_type=EvidenceType.CORRELATED,
                confidence=0.5,
                timestamp=_now_iso(),
            ),
        ]
        scored = scorer.score_all(evidence)
        assert all(e.score >= 0 for e in scored)

    def test_max_age_evidence(self):
        scorer = EvidenceScorer()
        very_old = _hours_ago_iso(48)
        evidence = [
            Evidence(
                source=EvidenceSourceAgent.LOG_AGENT,
                evidence_type=EvidenceType.DIRECT,
                confidence=0.9,
                timestamp=very_old,
            ),
        ]
        scored = scorer.score_all(evidence, reference_time=_now_iso())
        # Should still have a score (min_weight applied)
        assert scored[0].score > 0
