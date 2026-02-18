"""
File: core/priority_ranker.py
Purpose: Algorithm 4 – Rank actions by urgency × impact × confidence.
Dependencies: Standard library only + schema.
Performance: <1ms per call, pure function.
"""

from __future__ import annotations

from typing import Dict, List

from agents.incident_commander_agent.schema import (
    ActionItem,
    ActionPriority,
    BlastRadius,
)


_PRIORITY_WEIGHT: Dict[ActionPriority, float] = {
    ActionPriority.P0: 4.0,
    ActionPriority.P1: 3.0,
    ActionPriority.P2: 2.0,
    ActionPriority.P3: 1.0,
}


def _score_action(
    action: ActionItem,
    blast_radius: BlastRadius,
    confidence: float,
) -> float:
    """Compute a composite score for a single action.

    Score = priority_weight × (1 + availability_impact) × confidence
    Automated actions get a 1.2x boost (faster to execute).

    Args:
        action: The action item.
        blast_radius: Current blast radius.
        confidence: Overall confidence in root cause (0.0–1.0).

    Returns:
        Composite priority score (higher = more urgent).
    """
    weight = _PRIORITY_WEIGHT.get(action.priority, 2.0)
    impact_factor = 1.0 + blast_radius.availability_impact
    auto_boost = 1.2 if action.is_automated else 1.0
    return weight * impact_factor * confidence * auto_boost


def rank_actions(
    actions: List[ActionItem],
    blast_radius: BlastRadius | None = None,
    confidence: float = 0.8,
) -> List[ActionItem]:
    """Rank action items by computed urgency score.

    Actions are sorted descending by composite score (most urgent first).
    Ties are broken by the existing action_id for stability.

    Args:
        actions: Unranked action items.
        blast_radius: Blast radius context (defaults to empty).
        confidence: Root cause confidence 0.0–1.0.

    Returns:
        New list of ActionItem sorted most-urgent first.
    """
    br = blast_radius or BlastRadius()

    scored = [
        (a, _score_action(a, br, confidence)) for a in actions
    ]

    scored.sort(key=lambda pair: (-pair[1], pair[0].action_id))

    return [a for a, _ in scored]
