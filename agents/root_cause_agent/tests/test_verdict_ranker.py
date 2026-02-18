"""
Tests for core/verdict_ranker.py — Algorithm 4.
"""

import pytest

from agents.root_cause_agent.core.verdict_ranker import VerdictRanker
from agents.root_cause_agent.schema import (
    EvidenceSourceAgent,
    SynthesisResult,
)


def _make_synthesis(**kwargs) -> SynthesisResult:
    defaults = dict(
        agreement_score=0.7,
        sources_present=[EvidenceSourceAgent.LOG_AGENT],
    )
    defaults.update(kwargs)
    return SynthesisResult(**defaults)


class TestVerdictRanker:
    def test_rank_basic(self):
        ranker = VerdictRanker()
        hypotheses = [
            {"theory": "DB failure", "confidence": 0.9, "evidence_supporting": ["a", "b"]},
            {"theory": "Network issue", "confidence": 0.6, "evidence_supporting": ["c"]},
        ]
        top, conf, alts = ranker.rank(hypotheses, _make_synthesis())
        assert top == "DB failure"
        assert conf == 0.9
        assert len(alts) == 1

    def test_rank_empty_hypotheses(self):
        ranker = VerdictRanker()
        top, conf, alts = ranker.rank([], _make_synthesis())
        assert top == "Unknown root cause"
        assert conf == 0.0
        assert alts == []

    def test_alternatives_capped(self):
        from agents.root_cause_agent.config import RootCauseAgentConfig, VerdictLimits
        config = RootCauseAgentConfig(limits=VerdictLimits(max_alternatives=2))
        ranker = VerdictRanker(config)
        hypotheses = [
            {"theory": f"Theory {i}", "confidence": 0.9 - i * 0.1,
             "evidence_supporting": []}
            for i in range(5)
        ]
        top, conf, alts = ranker.rank(hypotheses, _make_synthesis())
        assert len(alts) <= 2

    def test_higher_confidence_ranked_first(self):
        ranker = VerdictRanker()
        hypotheses = [
            {"theory": "Low", "confidence": 0.3, "evidence_supporting": []},
            {"theory": "High", "confidence": 0.95, "evidence_supporting": ["a", "b", "c"]},
        ]
        top, conf, alts = ranker.rank(hypotheses, _make_synthesis())
        assert top == "High"

    def test_evidence_count_affects_score(self):
        ranker = VerdictRanker()
        # Same confidence, but different evidence counts
        hypotheses = [
            {"theory": "Less evidence", "confidence": 0.8, "evidence_supporting": []},
            {"theory": "More evidence", "confidence": 0.8, "evidence_supporting": ["a", "b", "c", "d"]},
        ]
        top, conf, alts = ranker.rank(hypotheses, _make_synthesis())
        assert top == "More evidence"

    def test_alternatives_have_correct_fields(self):
        ranker = VerdictRanker()
        hypotheses = [
            {"theory": "Main", "confidence": 0.9, "evidence_supporting": ["a"]},
            {"theory": "Alt", "confidence": 0.5, "evidence_supporting": [],
             "category": "database"},
        ]
        top, conf, alts = ranker.rank(hypotheses, _make_synthesis())
        assert len(alts) == 1
        assert alts[0].root_cause == "Alt"
        assert 0.0 <= alts[0].confidence <= 1.0
