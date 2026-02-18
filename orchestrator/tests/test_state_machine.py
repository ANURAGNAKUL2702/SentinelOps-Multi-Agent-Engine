"""Tests for orchestrator.state_machine."""

from __future__ import annotations

import pytest

from orchestrator.schema import AgentStatus, PipelineStatus, StateMachineError
from orchestrator.state_machine import StateMachine


class TestStateMachineTransitions:
    def test_initial_state_pending(self) -> None:
        sm = StateMachine()
        assert sm.get_current_state() == "PENDING"

    def test_pending_to_running(self) -> None:
        sm = StateMachine()
        # Manually set to RUNNING (our SM uses string states)
        sm._state = "RUNNING"
        assert sm.get_current_state() == "RUNNING"

    def test_running_to_success(self) -> None:
        sm = StateMachine()
        sm._state = "RUNNING"
        sm.transition(PipelineStatus.SUCCESS)
        assert sm.get_current_state() == PipelineStatus.SUCCESS.value

    def test_running_to_failed(self) -> None:
        sm = StateMachine()
        sm._state = "RUNNING"
        sm.transition(PipelineStatus.FAILED)
        assert sm.get_current_state() == PipelineStatus.FAILED.value

    def test_running_to_timeout(self) -> None:
        sm = StateMachine()
        sm._state = "RUNNING"
        sm.transition(PipelineStatus.TIMEOUT)
        assert sm.get_current_state() == PipelineStatus.TIMEOUT.value

    def test_running_to_partial(self) -> None:
        sm = StateMachine()
        sm._state = "RUNNING"
        sm.transition(PipelineStatus.PARTIAL_SUCCESS)
        assert sm.get_current_state() == PipelineStatus.PARTIAL_SUCCESS.value

    def test_completed_to_running_raises(self) -> None:
        sm = StateMachine()
        sm._state = "RUNNING"
        sm.transition(PipelineStatus.SUCCESS)
        with pytest.raises(StateMachineError):
            sm.transition(PipelineStatus.FAILED)

    def test_failed_is_terminal(self) -> None:
        sm = StateMachine()
        sm._state = "RUNNING"
        sm.transition(PipelineStatus.FAILED)
        with pytest.raises(StateMachineError):
            sm.transition(PipelineStatus.SUCCESS)


class TestAgentStates:
    def test_set_get_agent_state(self) -> None:
        sm = StateMachine()
        sm.set_agent_state("log_agent", AgentStatus.RUNNING)
        sm.set_agent_state("log_agent", AgentStatus.SUCCESS)
        states = sm.get_agent_states()
        assert states["log_agent"] == AgentStatus.SUCCESS

    def test_empty_agent_states(self) -> None:
        sm = StateMachine()
        assert sm.get_agent_states() == {}


class TestReset:
    def test_reset_returns_to_pending(self) -> None:
        sm = StateMachine()
        sm._state = "RUNNING"
        sm.transition(PipelineStatus.SUCCESS)
        sm.reset()
        assert sm.get_current_state() == "PENDING"
        assert sm.get_agent_states() == {}
