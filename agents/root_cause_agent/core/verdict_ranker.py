"""
File: core/verdict_ranker.py
Purpose: Algorithm 4 — Rank hypotheses by confidence × evidence_count × agreement.
Dependencies: Schema models only.
Performance: <1ms, O(n log n) where n = hypotheses.

Scores each hypothesis verdict and returns them sorted descending.
Top becomes the root cause, rest become alternatives.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from agents.root_cause_agent.config import RootCauseAgentConfig
from agents.root_cause_agent.schema import (
    AlternativeVerdict,
    Evidence,
    EvidenceSourceAgent,
    IncidentCategory,
    SynthesisResult,
)
from agents.root_cause_agent.telemetry import get_logger

logger = get_logger("root_cause_agent.verdict_ranker")


class VerdictRanker:
    """Ranks root cause candidates by composite score.

    Score formula::

        score = confidence × (1 + log2(evidence_count + 1)) × agreement
                + signature_boost

    The signature_boost rewards hypotheses whose THEORY text matches
    distinctive keywords found in the evidence trail.  This prevents
    generic hypotheses (e.g. "network partition") from winning when
    scenario-specific evidence (e.g. "OutOfMemoryError") is present.

    Args:
        config: Agent configuration.
    """

    # Keyword families that uniquely identify each failure type.
    # If ANY keyword from a family appears in the evidence trail,
    # hypotheses mentioning the corresponding root cause get a boost.
    _SIGNATURE_KEYWORDS: Dict[str, List[str]] = {
        "memory_leak": [
            "outofmemoryerror", "oomkilled", "oom killer",
            "heap", "gc overhead", "memory leak",
            "exit code 137", "java heap space",
        ],
        "cpu_spike": [
            "cpu saturat", "thread pool", "thread starvation",
            "cpu at 100", "cpu-bound", "rejected execution",
            "worker threads blocked", "0 idle workers",
        ],
        "database": [
            "connection pool exhausted", "query timeout",
            "slow query", "hikaricp", "deadlock",
            "sqlexception", "lock wait",
            "query execution timeout",
        ],
        "network": [
            "packet loss", "tcp retransmission",
            "connection reset by peer", "no route to host",
            "dns resolution", "network partition",
            "unreachable", "broken pipe",
        ],
    }

    # Map from keyword family to theory-text snippets that should get boosted.
    _THEORY_MARKERS: Dict[str, List[str]] = {
        "memory_leak": ["memory leak", "oom", "heap", "memory"],
        "cpu_spike": ["cpu", "thread pool", "saturation"],
        "database": ["database", "connection pool", "db", "query"],
        "network": ["network", "partition", "connectivity"],
    }

    def __init__(
        self, config: Optional[RootCauseAgentConfig] = None
    ) -> None:
        self._config = config or RootCauseAgentConfig()

    def rank(
        self,
        hypotheses: List[Dict[str, Any]],
        synthesis: SynthesisResult,
        correlation_id: str = "",
    ) -> Tuple[str, float, List[AlternativeVerdict]]:
        """Rank hypotheses and return top verdict + alternatives.

        Args:
            hypotheses: Ranked hypothesis dicts from hypothesis agent.
            synthesis: Evidence synthesis result.
            correlation_id: Request correlation ID.

        Returns:
            Tuple of (top_verdict_str, top_confidence, alternatives).
        """
        if not hypotheses:
            return ("Unknown root cause", 0.0, [])

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for hyp in hypotheses:
            score = self._compute_score(hyp, synthesis)
            scored.append((score, hyp))

        # Sort descending by score, then by confidence tie-breaking
        scored.sort(
            key=lambda x: (
                x[0],
                x[1].get("confidence", x[1].get("likelihood_score", 0.0)),
            ),
            reverse=True,
        )

        # Top verdict
        top_score, top_hyp = scored[0]
        top_verdict = top_hyp.get("theory", top_hyp.get("root_cause", "Unknown"))
        top_confidence = float(
            top_hyp.get("confidence", top_hyp.get("likelihood_score", 0.0))
        )

        # Alternatives
        alternatives: List[AlternativeVerdict] = []
        for score, hyp in scored[1:]:
            theory = hyp.get("theory", hyp.get("root_cause", ""))
            conf = float(
                hyp.get("confidence", hyp.get("likelihood_score", 0.0))
            )
            ev_count = len(hyp.get("evidence_supporting", []))
            cat_str = hyp.get("category", "unknown")
            try:
                cat = IncidentCategory(cat_str)
            except (ValueError, KeyError):
                cat = IncidentCategory.UNKNOWN

            alternatives.append(AlternativeVerdict(
                root_cause=theory,
                confidence=round(min(1.0, max(0.0, conf)), 4),
                evidence_count=ev_count,
                category=cat,
            ))

        # Cap alternatives
        max_alts = self._config.limits.max_alternatives
        alternatives = alternatives[:max_alts]

        logger.debug(
            f"Verdict ranked: top='{top_verdict[:50]}' "
            f"conf={top_confidence:.2f}, "
            f"alts={len(alternatives)}",
            extra={
                "correlation_id": correlation_id,
                "layer": "verdict_ranking",
            },
        )

        return (top_verdict, top_confidence, alternatives)

    def _compute_score(
        self, hyp: Dict[str, Any], synthesis: SynthesisResult
    ) -> float:
        """Compute composite score for a hypothesis.

        Combines base confidence with evidence count, agreement, and
        a **signature boost** that rewards hypotheses matching distinctive
        keywords found in the evidence trail.

        Args:
            hyp: Hypothesis dictionary.
            synthesis: Synthesis result.

        Returns:
            Composite score.
        """
        import math

        confidence = float(
            hyp.get("confidence", hyp.get("likelihood_score", 0.0))
        )
        evidence_count = len(hyp.get("evidence_supporting", []))
        agreement = synthesis.agreement_score

        # Base score = confidence × (1 + log2(evidence + 1)) × (0.5 + agreement × 0.5)
        evidence_factor = 1.0 + math.log2(evidence_count + 1)
        agreement_factor = 0.5 + agreement * 0.5
        base_score = confidence * evidence_factor * agreement_factor

        # ── Signature boost: match evidence keywords ↔ hypothesis theory ──
        # Collect all evidence text for keyword matching
        evidence_text = " ".join(
            e.description.lower() for e in synthesis.evidence_trail
        )
        theory = hyp.get("theory", hyp.get("root_cause", "")).lower()

        boost = 0.0
        for family, keywords in self._SIGNATURE_KEYWORDS.items():
            # Check if distinctive keywords from this family exist in evidence
            evidence_hit = any(kw in evidence_text for kw in keywords)
            if not evidence_hit:
                continue

            # Check if the hypothesis theory matches this family
            markers = self._THEORY_MARKERS.get(family, [])
            theory_hit = any(m in theory for m in markers)

            if theory_hit:
                # Hypothesis matches the evidence → BOOST
                boost += 0.35
            else:
                # Evidence suggests a different root cause → PENALISE
                boost -= 0.15

        score = base_score + boost
        return round(max(0.0, score), 6)
