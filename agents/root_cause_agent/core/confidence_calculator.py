"""
File: core/confidence_calculator.py
Purpose: Algorithm 2 — Bayesian P(root_cause|evidence) scoring.
Dependencies: Schema models only.
Performance: <1ms, O(n) where n = evidence items.

Computes posterior confidence using:
  P(root_cause | evidence) ≈ prior × agent_factor × agreement_factor × strength_factor
Clamped to [min_confidence, max_confidence].
"""

from __future__ import annotations

import math
from typing import List, Optional

from agents.root_cause_agent.config import RootCauseAgentConfig
from agents.root_cause_agent.schema import (
    Evidence,
    EvidenceSourceAgent,
    EvidenceType,
    SynthesisResult,
)
from agents.root_cause_agent.telemetry import get_logger

logger = get_logger("root_cause_agent.confidence_calculator")


class ConfidenceCalculator:
    """Bayesian confidence scoring for root cause verdicts.

    Formula::

        posterior = prior
                    × (1 + agent_count_weight × (n_agents / 4))
                    × (1 + agreement_weight × agreement_score)
                    × (1 + evidence_strength_weight × avg_evidence_conf)

    Clamped to [min_confidence, max_confidence].

    Args:
        config: Agent configuration.
    """

    def __init__(
        self, config: Optional[RootCauseAgentConfig] = None
    ) -> None:
        self._config = config or RootCauseAgentConfig()

    def calculate(
        self,
        synthesis: SynthesisResult,
        agent_confidences: List[float],
        correlation_id: str = "",
    ) -> float:
        """Compute Bayesian posterior confidence.

        Args:
            synthesis: Synthesis result with evidence and agreement.
            agent_confidences: Individual agent confidence values.
            correlation_id: Request correlation ID.

        Returns:
            Posterior confidence clamped to [min, max].
        """
        cfg = self._config.confidence

        # Prior
        posterior = cfg.prior

        # Agent count factor
        n_agents = len(synthesis.sources_present)
        agent_factor = 1.0 + cfg.agent_count_weight * (n_agents / 4.0)
        posterior *= agent_factor

        # Agreement factor
        agreement_factor = 1.0 + cfg.agreement_weight * synthesis.agreement_score
        posterior *= agreement_factor

        # Evidence strength factor
        avg_conf = 0.0
        if agent_confidences:
            avg_conf = sum(agent_confidences) / len(agent_confidences)
        strength_factor = 1.0 + cfg.evidence_strength_weight * avg_conf
        posterior *= strength_factor

        # Direct evidence bonus
        direct_count = sum(
            1 for e in synthesis.evidence_trail
            if e.evidence_type == EvidenceType.DIRECT
        )
        if direct_count > 0:
            posterior *= (1.0 + min(direct_count, 5) * 0.02)

        # Clamp
        result = max(cfg.min_confidence, min(cfg.max_confidence, posterior))
        result = round(result, 4)

        logger.debug(
            f"Confidence calculated: {result} "
            f"(prior={cfg.prior}, agents={n_agents}, "
            f"agreement={synthesis.agreement_score:.2f}, "
            f"avg_conf={avg_conf:.2f})",
            extra={
                "correlation_id": correlation_id,
                "layer": "confidence_calculation",
            },
        )

        return result

    def calculate_from_evidence(
        self,
        evidence_trail: List[Evidence],
        correlation_id: str = "",
    ) -> float:
        """Simplified confidence from evidence trail alone.

        Args:
            evidence_trail: List of evidence items.
            correlation_id: Request correlation ID.

        Returns:
            Confidence score 0.0-1.0.
        """
        if not evidence_trail:
            return self._config.confidence.min_confidence

        avg_conf = sum(e.confidence for e in evidence_trail) / len(evidence_trail)

        sources = set(e.source for e in evidence_trail)
        source_bonus = len(sources) / 4.0 * 0.2

        direct_ratio = sum(
            1 for e in evidence_trail
            if e.evidence_type == EvidenceType.DIRECT
        ) / max(len(evidence_trail), 1)

        result = avg_conf * 0.6 + source_bonus + direct_ratio * 0.2
        result = max(
            self._config.confidence.min_confidence,
            min(self._config.confidence.max_confidence, result),
        )

        return round(result, 4)
