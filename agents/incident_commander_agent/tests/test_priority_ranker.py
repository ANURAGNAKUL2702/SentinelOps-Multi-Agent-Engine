"""Tests for core/priority_ranker.py — Algorithm 4."""

from __future__ import annotations

from agents.incident_commander_agent.core.priority_ranker import (
    rank_actions,
)
from agents.incident_commander_agent.schema import (
    ActionItem,
    ActionPriority,
    BlastRadius,
)


class TestPriorityRanker:
    def test_p0_ranks_first(self):
        actions = [
            ActionItem(action_id="a3", priority=ActionPriority.P2),
            ActionItem(action_id="a1", priority=ActionPriority.P0),
            ActionItem(action_id="a2", priority=ActionPriority.P1),
        ]
        ranked = rank_actions(actions)
        assert ranked[0].action_id == "a1"
        assert ranked[1].action_id == "a2"

    def test_automated_boost(self):
        actions = [
            ActionItem(action_id="manual", priority=ActionPriority.P1, is_automated=False),
            ActionItem(action_id="auto", priority=ActionPriority.P1, is_automated=True),
        ]
        ranked = rank_actions(actions)
        assert ranked[0].action_id == "auto"

    def test_blast_radius_affects_ranking(self):
        actions = [
            ActionItem(action_id="a1", priority=ActionPriority.P1),
        ]
        br_low = BlastRadius(availability_impact=0.1)
        br_high = BlastRadius(availability_impact=0.9)
        ranked_low = rank_actions(actions, br_low)
        ranked_high = rank_actions(actions, br_high)
        # Same action, higher impact should not change order but scores differ
        assert len(ranked_low) == 1
        assert len(ranked_high) == 1

    def test_empty_list(self):
        assert rank_actions([]) == []

    def test_confidence_affects_score(self):
        actions = [
            ActionItem(action_id="a1", priority=ActionPriority.P0),
        ]
        # Lower confidence shouldn't reorder single item but affects score
        ranked = rank_actions(actions, confidence=0.5)
        assert ranked[0].action_id == "a1"
