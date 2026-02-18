"""
File: llm/explanation_builder.py
Purpose: Generate human-readable explanation from verdict and evidence.
Dependencies: Schema models only.
Performance: <1ms, O(n).

Builds a structured, human-readable explanation for the root cause
verdict, combining evidence summaries, causal chain narratives,
and impact descriptions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents.root_cause_agent.config import RootCauseAgentConfig
from agents.root_cause_agent.schema import (
    CausalLink,
    Contradiction,
    Evidence,
    ImpactAssessment,
    TimelineEvent,
)
from agents.root_cause_agent.telemetry import get_logger

logger = get_logger("root_cause_agent.llm.explanation_builder")


class ExplanationBuilder:
    """Builds structured explanations for root cause verdicts.

    Produces a multi-section narrative:
    1. Summary sentence.
    2. Evidence overview.
    3. Causal chain narrative.
    4. Impact statement.
    5. Contradictions (if any).

    Args:
        config: Agent configuration.
    """

    def __init__(
        self, config: Optional[RootCauseAgentConfig] = None
    ) -> None:
        self._config = config or RootCauseAgentConfig()

    def build(
        self,
        root_cause: str,
        confidence: float,
        evidence_trail: List[Evidence],
        causal_chain: List[CausalLink],
        impact: Optional[ImpactAssessment] = None,
        contradictions: Optional[List[Contradiction]] = None,
        timeline: Optional[List[TimelineEvent]] = None,
        correlation_id: str = "",
    ) -> str:
        """Build a human-readable explanation.

        Args:
            root_cause: Identified root cause.
            confidence: Overall confidence.
            evidence_trail: Evidence supporting the verdict.
            causal_chain: Causal chain links.
            impact: Impact assessment.
            contradictions: Any contradictions found.
            timeline: Incident timeline.
            correlation_id: Request correlation ID.

        Returns:
            Multi-section explanation string.
        """
        sections: List[str] = []

        # ── 1. Summary ──────────────────────────────────────────
        conf_label = self._confidence_label(confidence)
        sections.append(
            f"Root Cause Analysis: {root_cause} "
            f"({conf_label} confidence: {confidence:.0%})"
        )

        # ── 2. Evidence overview ────────────────────────────────
        if evidence_trail:
            sections.append(self._evidence_section(evidence_trail))

        # ── 3. Causal chain ─────────────────────────────────────
        if causal_chain:
            sections.append(self._causal_section(causal_chain))

        # ── 4. Impact ───────────────────────────────────────────
        if impact:
            sections.append(self._impact_section(impact))

        # ── 5. Contradictions ───────────────────────────────────
        if contradictions:
            sections.append(self._contradiction_section(contradictions))

        # ── 6. Timeline summary ─────────────────────────────────
        if timeline:
            sections.append(self._timeline_section(timeline))

        explanation = "\n\n".join(sections)

        logger.debug(
            f"Explanation built: {len(explanation)} chars",
            extra={
                "correlation_id": correlation_id,
                "layer": "explanation_building",
            },
        )

        return explanation

    def _confidence_label(self, confidence: float) -> str:
        """Map confidence to human-readable label."""
        if confidence >= 0.9:
            return "very high"
        elif confidence >= 0.7:
            return "high"
        elif confidence >= 0.5:
            return "moderate"
        elif confidence >= 0.3:
            return "low"
        return "very low"

    def _evidence_section(self, evidence: List[Evidence]) -> str:
        """Build evidence overview section."""
        lines = [f"Evidence Summary ({len(evidence)} items):"]
        # Group by source
        from collections import Counter
        source_counts = Counter(e.source.value for e in evidence)
        for source, count in source_counts.most_common():
            lines.append(f"  - {source}: {count} item(s)")

        # Top direct evidence
        direct = [e for e in evidence if e.evidence_type.value == "direct"]
        if direct:
            lines.append("Key direct evidence:")
            for e in direct[:3]:
                lines.append(f"  * {e.description} (conf: {e.confidence:.0%})")

        return "\n".join(lines)

    def _causal_section(self, chain: List[CausalLink]) -> str:
        """Build causal chain narrative."""
        lines = ["Causal Chain:"]
        for i, link in enumerate(chain, 1):
            svc = f" [{link.service}]" if link.service else ""
            lines.append(
                f"  {i}. {link.cause} → {link.effect} "
                f"({link.relationship.value}{svc}, "
                f"conf: {link.confidence:.0%})"
            )
        return "\n".join(lines)

    def _impact_section(self, impact: ImpactAssessment) -> str:
        """Build impact statement."""
        lines = [
            f"Impact Assessment:",
            f"  - Affected services: {impact.affected_count}",
            f"  - Blast radius: {impact.blast_radius}",
            f"  - Severity score: {impact.severity_score:.2f}",
            f"  - Cascading failure: {'Yes' if impact.is_cascading else 'No'}",
        ]
        if impact.affected_services:
            lines.append(
                f"  - Services: {', '.join(impact.affected_services[:10])}"
            )
        return "\n".join(lines)

    def _contradiction_section(
        self, contradictions: List[Contradiction]
    ) -> str:
        """Build contradictions section."""
        resolved = [c for c in contradictions if c.resolved]
        unresolved = [c for c in contradictions if not c.resolved]
        lines = [f"Contradictions ({len(contradictions)} detected):"]
        for c in resolved:
            lines.append(
                f"  - {c.agent_a.value} vs {c.agent_b.value}: "
                f"resolved by {c.resolution_strategy.value} → {c.winner}"
            )
        for c in unresolved:
            lines.append(
                f"  - UNRESOLVED: {c.agent_a.value} ({c.claim_a}) vs "
                f"{c.agent_b.value} ({c.claim_b})"
            )
        return "\n".join(lines)

    def _timeline_section(self, timeline: List[TimelineEvent]) -> str:
        """Build timeline summary."""
        lines = [f"Timeline ({len(timeline)} events):"]
        for ev in timeline[:10]:
            lines.append(
                f"  [{ev.timestamp}] {ev.source.value}: {ev.event}"
            )
        if len(timeline) > 10:
            lines.append(f"  ... and {len(timeline) - 10} more events")
        return "\n".join(lines)
