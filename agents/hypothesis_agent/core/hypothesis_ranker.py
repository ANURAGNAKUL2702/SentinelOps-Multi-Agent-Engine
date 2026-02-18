"""
File: core/hypothesis_ranker.py
Purpose: Algorithms 5, 6, 8, 9, 10 — Scoring, Pruning, History, MTTR, Confidence.
Dependencies: Schema models only.
Performance: <10ms, O(n²) for pruning dedup.

Algorithms:
  5. Likelihood Scoring O(1) per hypothesis
  6. Hypothesis Pruning O(n²) merge similar
  8. Historical Similarity Search O(h)
  9. MTTR Estimation
  10. Confidence Score Calculation
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set, Tuple

from agents.hypothesis_agent.config import HypothesisAgentConfig
from agents.hypothesis_agent.schema import (
    AggregatedEvidence,
    CrossAgentCorrelation,
    HistoricalIncident,
    Hypothesis,
    HypothesisStatus,
    IncidentCategory,
    PatternMatch,
    Severity,
    ValidationTest,
)
from agents.hypothesis_agent.telemetry import get_logger

logger = get_logger("hypothesis_agent.hypothesis_ranker")


class HypothesisRanker:
    """Scores, prunes, and ranks hypotheses.

    Pipeline::

        hypotheses → score → prune → historical check → MTTR → rank

    Args:
        config: Agent configuration with scoring weights.
    """

    def __init__(
        self, config: Optional[HypothesisAgentConfig] = None
    ) -> None:
        self._config = config or HypothesisAgentConfig()

    def rank(
        self,
        hypotheses: List[Hypothesis],
        evidence: AggregatedEvidence,
        pattern_matches: List[PatternMatch],
        historical: List[HistoricalIncident],
        correlation_id: str = "",
    ) -> List[Hypothesis]:
        """Score, prune, and rank hypotheses.

        Args:
            hypotheses: Raw hypotheses from generation.
            evidence: Aggregated evidence.
            pattern_matches: Matched patterns.
            historical: Historical incidents for similarity.
            correlation_id: Request correlation ID.

        Returns:
            Ranked list of hypotheses (best first).
        """
        start = time.perf_counter()

        if not hypotheses:
            return []

        # ── Algorithm 5: Likelihood Scoring ─────────────────────
        scored = [
            self._score_hypothesis(h, evidence, pattern_matches)
            for h in hypotheses
        ]

        # ── Algorithm 6: Pruning ────────────────────────────────
        pruned = self._prune_hypotheses(scored)

        # ── Algorithm 8: Historical Similarity ──────────────────
        if (
            historical
            and self._config.features.enable_historical_search
        ):
            pruned = self._apply_historical_boost(
                pruned, historical
            )

        # ── Algorithm 9: MTTR Estimation ────────────────────────
        pruned = [
            self._estimate_mttr(h, pattern_matches, historical)
            for h in pruned
        ]

        # ── Sort by likelihood descending ───────────────────────
        pruned.sort(
            key=lambda h: h.likelihood_score, reverse=True
        )

        # ── Enforce limits ──────────────────────────────────────
        max_h = self._config.limits.max_hypotheses
        if len(pruned) > max_h:
            for h in pruned[max_h:]:
                h = h.model_copy(
                    update={"status": HypothesisStatus.PRUNED}
                )
            pruned = pruned[:max_h]

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"Hypothesis ranking complete — "
            f"{len(pruned)} hypotheses ranked, "
            f"{elapsed_ms:.2f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "ranking",
                "context": {
                    "input_count": len(hypotheses),
                    "output_count": len(pruned),
                    "top_score": (
                        pruned[0].likelihood_score if pruned else 0
                    ),
                },
            },
        )

        return pruned

    # ── Algorithm 5: Likelihood Scoring ─────────────────────────

    def _score_hypothesis(
        self,
        hypothesis: Hypothesis,
        evidence: AggregatedEvidence,
        pattern_matches: List[PatternMatch],
    ) -> Hypothesis:
        """Calculate likelihood score for a hypothesis.

        Score = (supporting * w_s) - (contradicting * w_c)
              + (pattern * w_p) + (correlation * w_x)

        Clamped to [0.0, 1.0].

        Args:
            hypothesis: Hypothesis to score.
            evidence: Aggregated evidence.
            pattern_matches: Pattern matches.

        Returns:
            Hypothesis with updated likelihood_score.
        """
        w = self._config.scoring

        # Supporting evidence contribution
        supporting_count = len(hypothesis.evidence_supporting)
        max_supporting = max(
            evidence.total_evidence_count, 1
        )
        supporting_score = min(
            1.0, supporting_count / max_supporting
        ) * w.supporting_evidence

        # Contradicting evidence penalty
        contradicting_count = len(hypothesis.evidence_contradicting)
        contradicting_penalty = min(
            1.0, contradicting_count / max(max_supporting, 1)
        ) * w.contradicting_evidence

        # Pattern match boost
        pattern_score = 0.0
        if hypothesis.pattern_match:
            pattern_score = (
                hypothesis.pattern_match.match_score
                * w.pattern_match
            )

        # Cross-agent correlation boost
        correlation_score = 0.0
        if evidence.correlations:
            avg_corr = sum(
                c.correlation_score for c in evidence.correlations
            ) / len(evidence.correlations)
            correlation_score = avg_corr * w.cross_agent_correlation

        # Causal chain boost
        causal_score = 0.0
        if hypothesis.causal_chain and hypothesis.causal_chain.chain:
            causal_score = (
                hypothesis.causal_chain.chain_confidence
                * w.causal_chain_strength
            )

        # Final score
        raw_score = (
            supporting_score
            - contradicting_penalty
            + pattern_score
            + correlation_score
            + causal_score
        )

        # Clamp
        final_score = max(0.0, min(1.0, raw_score))

        return hypothesis.model_copy(
            update={"likelihood_score": round(final_score, 4)}
        )

    # ── Algorithm 6: Hypothesis Pruning ─────────────────────────

    def _prune_hypotheses(
        self, hypotheses: List[Hypothesis]
    ) -> List[Hypothesis]:
        """Prune low-quality and duplicate hypotheses.

        1. Remove below threshold.
        2. Merge similar hypotheses (O(n²) comparison).

        Args:
            hypotheses: Scored hypotheses.

        Returns:
            Pruned list.
        """
        threshold = self._config.limits.pruning_threshold

        # Remove below threshold
        surviving = [
            h for h in hypotheses
            if h.likelihood_score >= threshold
        ]

        if len(surviving) <= 1:
            # Ensure minimum hypotheses
            if not surviving and hypotheses:
                surviving = hypotheses[:1]
            return surviving

        # Merge similar (O(n²))
        merged: List[Hypothesis] = []
        used: Set[int] = set()

        for i in range(len(surviving)):
            if i in used:
                continue

            current = surviving[i]
            for j in range(i + 1, len(surviving)):
                if j in used:
                    continue

                if self._are_similar(current, surviving[j]):
                    # Merge: keep higher score, combine evidence
                    current = self._merge_hypotheses(
                        current, surviving[j]
                    )
                    used.add(j)

            merged.append(current)

        return merged

    def _are_similar(
        self, h1: Hypothesis, h2: Hypothesis
    ) -> bool:
        """Check if two hypotheses are similar enough to merge.

        Similar if same category and overlapping evidence.
        """
        if h1.category != h2.category:
            return False

        # Check theory text overlap (simple word overlap)
        words1 = set(h1.theory.lower().split())
        words2 = set(h2.theory.lower().split())
        if not words1 or not words2:
            return False

        overlap = len(words1 & words2)
        min_words = min(len(words1), len(words2))
        if min_words == 0:
            return False

        return (overlap / min_words) > 0.6

    def _merge_hypotheses(
        self, primary: Hypothesis, secondary: Hypothesis
    ) -> Hypothesis:
        """Merge two similar hypotheses, keeping the stronger one."""
        combined_supporting = list(set(
            primary.evidence_supporting
            + secondary.evidence_supporting
        ))
        combined_contradicting = list(set(
            primary.evidence_contradicting
            + secondary.evidence_contradicting
        ))

        # Keep the hypothesis with higher score as base
        if primary.likelihood_score >= secondary.likelihood_score:
            base = primary
        else:
            base = secondary

        return base.model_copy(update={
            "evidence_supporting": combined_supporting,
            "evidence_contradicting": combined_contradicting,
            "likelihood_score": max(
                primary.likelihood_score,
                secondary.likelihood_score,
            ),
        })

    # ── Algorithm 8: Historical Similarity Search ───────────────

    def _apply_historical_boost(
        self,
        hypotheses: List[Hypothesis],
        historical: List[HistoricalIncident],
    ) -> List[Hypothesis]:
        """Boost hypothesis scores based on historical similarity.

        Compares hypothesis category and theory against past incidents.

        Args:
            hypotheses: Current hypotheses.
            historical: Past incidents.

        Returns:
            Hypotheses with boosted scores if historically matched.
        """
        boosted: List[Hypothesis] = []

        for h in hypotheses:
            best_sim = 0.0
            best_match: Optional[HistoricalIncident] = None

            for incident in historical:
                sim = self._compute_similarity(h, incident)
                if sim > best_sim:
                    best_sim = sim
                    best_match = incident

            if best_sim > 0.5 and best_match:
                boost = (
                    best_sim
                    * self._config.scoring.historical_similarity
                )
                new_score = min(
                    1.0, h.likelihood_score + boost
                )
                h = h.model_copy(update={
                    "likelihood_score": round(new_score, 4),
                    "reasoning": (
                        h.reasoning
                        + f" [Historical match: '{best_match.title}' "
                        f"(similarity={best_sim:.2f})]"
                    ),
                })

            boosted.append(h)

        return boosted

    def _compute_similarity(
        self,
        hypothesis: Hypothesis,
        incident: HistoricalIncident,
    ) -> float:
        """Compute similarity between hypothesis and historical incident.

        Uses category match + keyword overlap.

        Returns:
            Similarity score 0.0-1.0.
        """
        score = 0.0

        # Category match: 0.4 weight
        if hypothesis.category == incident.category:
            score += 0.4

        # Keyword overlap: 0.6 weight
        h_words = set(hypothesis.theory.lower().split())
        i_words = set(
            incident.root_cause.lower().split()
            + incident.title.lower().split()
        )
        if h_words and i_words:
            overlap = len(h_words & i_words)
            max_possible = min(len(h_words), len(i_words))
            if max_possible > 0:
                score += 0.6 * (overlap / max_possible)

        return min(1.0, score)

    # ── Algorithm 9: MTTR Estimation ────────────────────────────

    def _estimate_mttr(
        self,
        hypothesis: Hypothesis,
        pattern_matches: List[PatternMatch],
        historical: List[HistoricalIncident],
    ) -> Hypothesis:
        """Estimate MTTR for a hypothesis.

        Sources (priority order):
        1. Historical incidents with same category.
        2. Pattern library typical MTTR.
        3. Severity-based default.

        Args:
            hypothesis: Hypothesis to estimate MTTR for.
            pattern_matches: Matched patterns.
            historical: Historical incidents.

        Returns:
            Hypothesis with updated estimated_mttr_minutes.
        """
        mttr = 30.0  # default

        # 1. Historical average
        category_incidents = [
            inc for inc in historical
            if inc.category == hypothesis.category
        ]
        if category_incidents:
            mttr = sum(
                inc.mttr_minutes for inc in category_incidents
            ) / len(category_incidents)
        # 2. Pattern library
        elif hypothesis.pattern_match:
            from agents.hypothesis_agent.core.pattern_matcher import (
                PATTERN_LIBRARY,
            )
            for pattern in PATTERN_LIBRARY:
                if (
                    pattern.pattern_name
                    == hypothesis.pattern_match.pattern_name
                ):
                    mttr = pattern.typical_mttr_minutes
                    break
        # 3. Severity-based default
        else:
            severity_mttr = {
                Severity.CRITICAL: 60.0,
                Severity.HIGH: 45.0,
                Severity.MEDIUM: 30.0,
                Severity.LOW: 15.0,
            }
            mttr = severity_mttr.get(hypothesis.severity, 30.0)

        return hypothesis.model_copy(
            update={"estimated_mttr_minutes": round(mttr, 1)}
        )

    # ── Algorithm 10: Confidence Score Calculation ──────────────

    def compute_overall_confidence(
        self,
        hypotheses: List[Hypothesis],
        evidence: AggregatedEvidence,
        pattern_matches: List[PatternMatch],
    ) -> float:
        """Calculate overall confidence for the analysis.

        Factors:
        - Top hypothesis score (40%)
        - Evidence quality (20%)
        - Pattern match strength (20%)
        - Cross-agent correlation (20%)

        Args:
            hypotheses: Ranked hypotheses.
            evidence: Aggregated evidence.
            pattern_matches: Pattern matches.

        Returns:
            Confidence score 0.0-1.0.
        """
        if not hypotheses:
            return 0.1

        # Factor 1: Top hypothesis score (40%)
        top_score = hypotheses[0].likelihood_score
        top_factor = top_score * 0.4

        # Factor 2: Evidence quality (20%)
        if evidence.total_evidence_count > 0:
            quality = (
                evidence.strong_evidence_count
                / evidence.total_evidence_count
            )
        else:
            quality = 0.0
        evidence_factor = quality * 0.2

        # Factor 3: Pattern match strength (20%)
        pattern_factor = 0.0
        if pattern_matches:
            best_match = max(
                pm.match_score for pm in pattern_matches
            )
            pattern_factor = best_match * 0.2

        # Factor 4: Cross-agent agreement (20%)
        source_count = len(evidence.sources_represented)
        agreement_factor = min(1.0, source_count / 3.0) * 0.2

        confidence = (
            top_factor
            + evidence_factor
            + pattern_factor
            + agreement_factor
        )

        return round(max(0.0, min(1.0, confidence)), 4)
