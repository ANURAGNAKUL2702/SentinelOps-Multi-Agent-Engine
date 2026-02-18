"""
File: core/evidence_scorer.py
Purpose: Algorithm 5 — Score evidence by source reliability, timestamp recency, evidence type.
Dependencies: Schema models only.
Performance: <1ms, O(n) where n = evidence items.

score = base_weight[source] × recency_decay(hours_old) × type_multiplier[evidence_type]
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import List, Optional

from agents.root_cause_agent.config import RootCauseAgentConfig
from agents.root_cause_agent.schema import (
    Evidence,
    EvidenceSourceAgent,
    EvidenceType,
)
from agents.root_cause_agent.telemetry import get_logger

logger = get_logger("root_cause_agent.evidence_scorer")


class EvidenceScorer:
    """Scores evidence items by source weight, recency, and type.

    Formula::

        score = base_weight[source]
                × recency_decay(hours_old)
                × type_multiplier[evidence_type]
                × confidence

    Where recency_decay = exp(-ln(2) × hours_old / half_life)

    Args:
        config: Agent configuration.
    """

    def __init__(
        self, config: Optional[RootCauseAgentConfig] = None
    ) -> None:
        self._config = config or RootCauseAgentConfig()

    def score_all(
        self,
        evidence: List[Evidence],
        reference_time: Optional[str] = None,
        correlation_id: str = "",
    ) -> List[Evidence]:
        """Score all evidence items and return with updated scores.

        Args:
            evidence: List of evidence items to score.
            reference_time: ISO-8601 reference timestamp for recency.
                Defaults to now.
            correlation_id: Request correlation ID.

        Returns:
            Same evidence list with score fields populated.
        """
        ref_dt = self._parse_timestamp(reference_time)
        scored: List[Evidence] = []

        for ev in evidence:
            score = self._compute_score(ev, ref_dt)
            scored.append(ev.model_copy(update={"score": round(score, 6)}))

        logger.debug(
            f"Scored {len(scored)} evidence items",
            extra={
                "correlation_id": correlation_id,
                "layer": "evidence_scoring",
            },
        )

        return scored

    def _compute_score(
        self, ev: Evidence, ref_dt: datetime
    ) -> float:
        """Compute score for a single evidence item.

        Args:
            ev: Evidence item.
            ref_dt: Reference datetime for recency.

        Returns:
            Computed score.
        """
        base_weight = self._source_weight(ev.source)
        recency = self._recency_decay(ev.timestamp, ref_dt)
        type_mult = self._type_multiplier(ev.evidence_type)

        return base_weight * recency * type_mult * ev.confidence

    def _source_weight(self, source: EvidenceSourceAgent) -> float:
        """Get base weight for a source agent.

        Args:
            source: The source agent.

        Returns:
            Weight value.
        """
        sw = self._config.source_weights
        weights = {
            EvidenceSourceAgent.HYPOTHESIS_AGENT: sw.hypothesis_agent,
            EvidenceSourceAgent.DEPENDENCY_AGENT: sw.dependency_agent,
            EvidenceSourceAgent.METRICS_AGENT: sw.metrics_agent,
            EvidenceSourceAgent.LOG_AGENT: sw.log_agent,
        }
        return weights.get(source, 0.5)

    def _type_multiplier(self, ev_type: EvidenceType) -> float:
        """Get multiplier for evidence type.

        Args:
            ev_type: The evidence type.

        Returns:
            Multiplier value.
        """
        et = self._config.evidence_types
        multipliers = {
            EvidenceType.DIRECT: et.direct,
            EvidenceType.CORRELATED: et.correlated,
            EvidenceType.CIRCUMSTANTIAL: et.circumstantial,
        }
        return multipliers.get(ev_type, 0.5)

    def _recency_decay(
        self, timestamp_str: str, ref_dt: datetime
    ) -> float:
        """Compute exponential recency decay.

        decay = exp(-ln(2) × hours_old / half_life)

        Args:
            timestamp_str: ISO-8601 timestamp.
            ref_dt: Reference datetime.

        Returns:
            Decay factor between min_weight and 1.0.
        """
        if not timestamp_str:
            return 1.0  # no timestamp = assume current

        try:
            ev_dt = datetime.fromisoformat(
                timestamp_str.replace("Z", "+00:00")
            )
            hours_old = (ref_dt - ev_dt).total_seconds() / 3600.0
            if hours_old < 0:
                hours_old = 0.0
        except (ValueError, TypeError):
            return 1.0

        half_life = self._config.recency.half_life_hours
        max_age = self._config.recency.max_age_hours
        min_weight = self._config.recency.min_weight

        if hours_old >= max_age:
            return min_weight

        decay = math.exp(-math.log(2) * hours_old / half_life)
        return max(min_weight, decay)

    def _parse_timestamp(
        self, ts: Optional[str]
    ) -> datetime:
        """Parse an ISO-8601 timestamp or return now.

        Args:
            ts: Optional timestamp string.

        Returns:
            Parsed datetime.
        """
        if ts:
            try:
                return datetime.fromisoformat(
                    ts.replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass
        return datetime.now(timezone.utc)
