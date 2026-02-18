"""
Tests for core/hypothesis_ranker.py — Algorithms 5, 6, 8, 9, 10.

Covers:
  - Algorithm 5: Likelihood Scoring
  - Algorithm 6: Pruning (threshold + similarity merge)
  - Algorithm 8: Historical Similarity Search
  - Algorithm 9: MTTR Estimation
  - Algorithm 10: Confidence Score Calculation
  - Edge cases (empty inputs, single hypothesis)
"""

from __future__ import annotations

import pytest

from agents.hypothesis_agent.config import (
    HypothesisAgentConfig,
    HypothesisLimits,
    ScoringWeights,
)
from agents.hypothesis_agent.core.hypothesis_ranker import (
    HypothesisRanker,
)
from agents.hypothesis_agent.schema import (
    AggregatedEvidence,
    CausalChain,
    CausalChainLink,
    CausalRelationship,
    CrossAgentCorrelation,
    EvidenceItem,
    EvidenceSource,
    EvidenceStrength,
    HistoricalIncident,
    Hypothesis,
    HypothesisStatus,
    IncidentCategory,
    PatternMatch,
    PatternName,
    Severity,
)


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════


def _make_evidence(
    total: int = 5,
    strong: int = 2,
    sources: list | None = None,
    correlations: list | None = None,
) -> AggregatedEvidence:
    items = [
        EvidenceItem(
            source=EvidenceSource.LOG_AGENT,
            description=f"Evidence item {i}",
            strength=(
                EvidenceStrength.STRONG if i < strong
                else EvidenceStrength.MODERATE
            ),
        )
        for i in range(total)
    ]
    return AggregatedEvidence(
        evidence_items=items,
        total_evidence_count=total,
        strong_evidence_count=strong,
        sources_represented=sources or [EvidenceSource.LOG_AGENT],
        correlations=correlations or [],
    )


def _hyp(
    theory: str,
    category: IncidentCategory = IncidentCategory.DATABASE,
    severity: Severity = Severity.HIGH,
    supporting: list | None = None,
    contradicting: list | None = None,
    pattern_match: PatternMatch | None = None,
    causal_chain: CausalChain | None = None,
    score: float = 0.5,
) -> Hypothesis:
    return Hypothesis(
        theory=theory,
        category=category,
        severity=severity,
        likelihood_score=score,
        evidence_supporting=supporting or [],
        evidence_contradicting=contradicting or [],
        pattern_match=pattern_match,
        causal_chain=causal_chain,
    )


@pytest.fixture
def ranker() -> HypothesisRanker:
    return HypothesisRanker()


# ═══════════════════════════════════════════════════════════════
#  TESTS: Algorithm 5 — Likelihood Scoring
# ═══════════════════════════════════════════════════════════════


class TestLikelihoodScoring:
    """Test hypothesis scoring algorithm."""

    def test_more_supporting_evidence_higher_score(
        self, ranker: HypothesisRanker
    ):
        evidence = _make_evidence(total=10, strong=3)
        h1 = _hyp("Theory A", supporting=["e1", "e2", "e3"])
        h2 = _hyp("Theory B", supporting=["e1"])

        scored1 = ranker._score_hypothesis(h1, evidence, [])
        scored2 = ranker._score_hypothesis(h2, evidence, [])
        assert scored1.likelihood_score >= scored2.likelihood_score

    def test_contradicting_evidence_lowers_score(
        self, ranker: HypothesisRanker
    ):
        evidence = _make_evidence(total=5)
        h_clean = _hyp("Theory A", supporting=["e1"])
        h_contra = _hyp(
            "Theory B",
            supporting=["e1"],
            contradicting=["c1", "c2"],
        )

        s_clean = ranker._score_hypothesis(h_clean, evidence, [])
        s_contra = ranker._score_hypothesis(h_contra, evidence, [])
        assert s_clean.likelihood_score >= s_contra.likelihood_score

    def test_pattern_match_boosts_score(
        self, ranker: HypothesisRanker
    ):
        evidence = _make_evidence(total=5)
        pm = PatternMatch(
            pattern_name=PatternName.DATABASE_CONNECTION_POOL_EXHAUSTION,
            match_score=0.8,
        )
        h_with = _hyp("Theory A", supporting=["e1"], pattern_match=pm)
        h_without = _hyp("Theory B", supporting=["e1"])

        s_with = ranker._score_hypothesis(h_with, evidence, [pm])
        s_without = ranker._score_hypothesis(h_without, evidence, [])
        assert s_with.likelihood_score > s_without.likelihood_score

    def test_correlation_boosts_score(
        self, ranker: HypothesisRanker
    ):
        corr = CrossAgentCorrelation(
            sources=[EvidenceSource.LOG_AGENT, EvidenceSource.METRICS_AGENT],
            description="Service overlap",
            correlation_score=0.8,
        )
        evidence = _make_evidence(total=5, correlations=[corr])
        h = _hyp("Theory A", supporting=["e1"])
        scored = ranker._score_hypothesis(h, evidence, [])
        assert scored.likelihood_score > 0

    def test_score_clamped_to_01(
        self, ranker: HypothesisRanker
    ):
        evidence = _make_evidence(total=1)
        h = _hyp("Theory", supporting=["e1"] * 100)
        scored = ranker._score_hypothesis(h, evidence, [])
        assert 0.0 <= scored.likelihood_score <= 1.0


# ═══════════════════════════════════════════════════════════════
#  TESTS: Algorithm 6 — Pruning
# ═══════════════════════════════════════════════════════════════


class TestPruning:
    """Test hypothesis pruning algorithm."""

    def test_below_threshold_pruned(self):
        config = HypothesisAgentConfig(
            limits=HypothesisLimits(pruning_threshold=0.3)
        )
        ranker = HypothesisRanker(config)
        hypotheses = [
            _hyp("Good", score=0.8),
            _hyp("Bad", score=0.1),
            _hyp("Ok", score=0.5),
        ]
        pruned = ranker._prune_hypotheses(hypotheses)
        assert len(pruned) == 2

    def test_similar_hypotheses_merged(
        self, ranker: HypothesisRanker
    ):
        h1 = _hyp(
            "Database connection pool exhaustion issue",
            category=IncidentCategory.DATABASE,
            supporting=["e1"],
            score=0.8,
        )
        h2 = _hyp(
            "Database connection pool exhaustion problem",
            category=IncidentCategory.DATABASE,
            supporting=["e2"],
            score=0.6,
        )
        pruned = ranker._prune_hypotheses([h1, h2])
        # Should merge into 1
        assert len(pruned) == 1
        # Merged should have combined evidence
        assert len(pruned[0].evidence_supporting) >= 2

    def test_different_categories_not_merged(
        self, ranker: HypothesisRanker
    ):
        h1 = _hyp(
            "Database connection pool issue",
            category=IncidentCategory.DATABASE,
            score=0.8,
        )
        h2 = _hyp(
            "Network partition detected",
            category=IncidentCategory.NETWORK,
            score=0.6,
        )
        pruned = ranker._prune_hypotheses([h1, h2])
        assert len(pruned) == 2

    def test_keeps_at_least_one_even_below_threshold(
        self, ranker: HypothesisRanker
    ):
        hypotheses = [_hyp("Low", score=0.01)]
        pruned = ranker._prune_hypotheses(hypotheses)
        assert len(pruned) >= 1


# ═══════════════════════════════════════════════════════════════
#  TESTS: Algorithm 8 — Historical Similarity
# ═══════════════════════════════════════════════════════════════


class TestHistoricalSimilarity:
    """Test historical similarity search."""

    def test_matching_category_boosts_score(
        self, ranker: HypothesisRanker
    ):
        h = _hyp(
            "Database pool exhaustion causing service failures",
            category=IncidentCategory.DATABASE,
            score=0.5,
        )
        incident = HistoricalIncident(
            incident_id="inc-1",
            title="Database pool exhaustion",
            root_cause="Connection pool was exhausted",
            category=IncidentCategory.DATABASE,
            mttr_minutes=30.0,
        )
        boosted = ranker._apply_historical_boost(
            [h], [incident]
        )
        assert boosted[0].likelihood_score >= h.likelihood_score

    def test_no_boost_for_different_category(
        self, ranker: HypothesisRanker
    ):
        h = _hyp(
            "Network partition issue",
            category=IncidentCategory.NETWORK,
            score=0.5,
        )
        incident = HistoricalIncident(
            title="Memory leak issue",
            root_cause="Memory was leaking",
            category=IncidentCategory.APPLICATION,
        )
        boosted = ranker._apply_historical_boost(
            [h], [incident]
        )
        # Score should be unchanged (no match)
        assert boosted[0].likelihood_score == h.likelihood_score


# ═══════════════════════════════════════════════════════════════
#  TESTS: Algorithm 9 — MTTR Estimation
# ═══════════════════════════════════════════════════════════════


class TestMTTREstimation:
    """Test MTTR estimation sources."""

    def test_mttr_from_historical(
        self, ranker: HypothesisRanker
    ):
        h = _hyp("DB issue", category=IncidentCategory.DATABASE)
        incident = HistoricalIncident(
            category=IncidentCategory.DATABASE,
            mttr_minutes=42.0,
        )
        result = ranker._estimate_mttr(h, [], [incident])
        assert result.estimated_mttr_minutes == 42.0

    def test_mttr_from_pattern(
        self, ranker: HypothesisRanker
    ):
        pm = PatternMatch(
            pattern_name=PatternName.DATABASE_CONNECTION_POOL_EXHAUSTION,
            match_score=0.8,
        )
        h = _hyp("DB pool issue", pattern_match=pm)
        result = ranker._estimate_mttr(h, [pm], [])
        assert result.estimated_mttr_minutes == 45.0

    def test_mttr_from_severity_default(
        self, ranker: HypothesisRanker
    ):
        h = _hyp("Unknown issue", severity=Severity.CRITICAL)
        result = ranker._estimate_mttr(h, [], [])
        assert result.estimated_mttr_minutes == 60.0

    def test_mttr_medium_severity_default(
        self, ranker: HypothesisRanker
    ):
        h = _hyp(
            "Medium issue",
            severity=Severity.MEDIUM,
            category=IncidentCategory.UNKNOWN,
        )
        result = ranker._estimate_mttr(h, [], [])
        assert result.estimated_mttr_minutes == 30.0


# ═══════════════════════════════════════════════════════════════
#  TESTS: Algorithm 10 — Confidence Score
# ═══════════════════════════════════════════════════════════════


class TestConfidenceScore:
    """Test overall confidence score calculation."""

    def test_confidence_with_strong_evidence(
        self, ranker: HypothesisRanker
    ):
        hypotheses = [_hyp("Strong theory", score=0.9)]
        evidence = _make_evidence(
            total=5,
            strong=3,
            sources=[
                EvidenceSource.LOG_AGENT,
                EvidenceSource.METRICS_AGENT,
                EvidenceSource.DEPENDENCY_AGENT,
            ],
        )
        pm = PatternMatch(
            pattern_name=PatternName.DATABASE_CONNECTION_POOL_EXHAUSTION,
            match_score=0.8,
        )
        conf = ranker.compute_overall_confidence(
            hypotheses, evidence, [pm]
        )
        assert conf > 0.5

    def test_confidence_with_no_hypotheses(
        self, ranker: HypothesisRanker
    ):
        conf = ranker.compute_overall_confidence(
            [], _make_evidence(), []
        )
        assert conf == 0.1

    def test_confidence_clamped_to_01(
        self, ranker: HypothesisRanker
    ):
        conf = ranker.compute_overall_confidence(
            [_hyp("x", score=1.0)],
            _make_evidence(total=10, strong=10),
            [PatternMatch(
                pattern_name=PatternName.DATABASE_CONNECTION_POOL_EXHAUSTION,
                match_score=1.0,
            )],
        )
        assert 0.0 <= conf <= 1.0


# ═══════════════════════════════════════════════════════════════
#  TESTS: Full Ranking Pipeline
# ═══════════════════════════════════════════════════════════════


class TestFullRankingPipeline:
    """Test end-to-end ranking pipeline."""

    def test_rank_returns_sorted_hypotheses(
        self, ranker: HypothesisRanker
    ):
        evidence = _make_evidence(total=5, strong=2)
        hypotheses = [
            _hyp("Theory A", supporting=["e1"], score=0.3),
            _hyp("Theory B", supporting=["e1", "e2", "e3"], score=0.8),
            _hyp("Theory C", supporting=["e1", "e2"], score=0.5),
        ]
        ranked = ranker.rank(hypotheses, evidence, [], [])
        scores = [h.likelihood_score for h in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_enforces_max_limit(self):
        config = HypothesisAgentConfig(
            limits=HypothesisLimits(max_hypotheses=2)
        )
        ranker = HypothesisRanker(config)
        evidence = _make_evidence()
        hypotheses = [
            _hyp(f"Theory {i}", score=0.5 + i * 0.1,
                 category=IncidentCategory(
                     ["database", "application", "network", "deployment"][i]
                 ))
            for i in range(4)
        ]
        ranked = ranker.rank(hypotheses, evidence, [], [])
        assert len(ranked) <= 2

    def test_rank_empty_input(self, ranker: HypothesisRanker):
        result = ranker.rank([], _make_evidence(), [], [])
        assert result == []
