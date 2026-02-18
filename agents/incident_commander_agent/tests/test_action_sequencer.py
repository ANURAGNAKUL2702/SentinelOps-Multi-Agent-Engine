"""Tests for core/action_sequencer.py — Algorithm 5."""

from __future__ import annotations

import pytest

from agents.incident_commander_agent.core.action_sequencer import (
    CycleError,
    sequence_actions,
)
from agents.incident_commander_agent.schema import ActionItem


class TestActionSequencer:
    def test_no_dependencies(self):
        actions = [
            ActionItem(action_id="a1"),
            ActionItem(action_id="a2"),
        ]
        ordered = sequence_actions(actions)
        ids = [a.action_id for a in ordered]
        assert set(ids) == {"a1", "a2"}

    def test_linear_dependency(self):
        actions = [
            ActionItem(action_id="a3", dependencies=["a2"]),
            ActionItem(action_id="a1"),
            ActionItem(action_id="a2", dependencies=["a1"]),
        ]
        ordered = sequence_actions(actions)
        ids = [a.action_id for a in ordered]
        assert ids.index("a1") < ids.index("a2")
        assert ids.index("a2") < ids.index("a3")

    def test_cycle_raises(self):
        actions = [
            ActionItem(action_id="a1", dependencies=["a2"]),
            ActionItem(action_id="a2", dependencies=["a1"]),
        ]
        with pytest.raises(CycleError):
            sequence_actions(actions)

    def test_empty_list(self):
        assert sequence_actions([]) == []

    def test_unknown_dependency_ignored(self):
        actions = [
            ActionItem(action_id="a1", dependencies=["nonexistent"]),
        ]
        ordered = sequence_actions(actions)
        assert len(ordered) == 1
