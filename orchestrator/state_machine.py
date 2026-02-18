"""Pipeline and per-agent state machine."""

from __future__ import annotations

import threading
from typing import Dict

from .schema import AgentStatus, PipelineStatus, StateMachineError


# Valid transitions for the pipeline state machine.
_VALID_TRANSITIONS: Dict[PipelineStatus, set[PipelineStatus]] = {
    PipelineStatus.SUCCESS: set(),          # terminal
    PipelineStatus.PARTIAL_SUCCESS: set(),  # terminal
    PipelineStatus.FAILED: set(),           # terminal
    PipelineStatus.TIMEOUT: set(),          # terminal
}

# PENDING is treated as a "virtual" state that only transitions to RUNNING.
# RUNNING can transition to any terminal state.
_PENDING = "PENDING"
_RUNNING = "RUNNING"


class StateMachine:
    """Track pipeline execution state through its lifecycle.

    States: PENDING → RUNNING → COMPLETED / FAILED / PARTIAL_SUCCESS / TIMEOUT.
    No transitions are allowed *from* terminal states.
    """

    def __init__(self) -> None:
        self._state: str = _PENDING
        self._agent_states: Dict[str, AgentStatus] = {}
        self._lock = threading.Lock()

    def transition(self, to_state: PipelineStatus) -> bool:
        """Attempt to transition the pipeline to *to_state*.

        Args:
            to_state: Target state.

        Returns:
            ``True`` if the transition succeeded.

        Raises:
            StateMachineError: If the transition is invalid.
        """
        with self._lock:
            if self._state == _PENDING:
                # PENDING → RUNNING is the only valid first step
                if to_state.value.upper() == _RUNNING:
                    self._state = _RUNNING
                    return True
                raise StateMachineError(_PENDING, to_state.value)

            if self._state == _RUNNING:
                # RUNNING → any terminal
                if to_state in (
                    PipelineStatus.SUCCESS,
                    PipelineStatus.PARTIAL_SUCCESS,
                    PipelineStatus.FAILED,
                    PipelineStatus.TIMEOUT,
                ):
                    self._state = to_state.value
                    return True
                raise StateMachineError(_RUNNING, to_state.value)

            # Already in a terminal state — no transitions allowed
            raise StateMachineError(self._state, to_state.value)

    def get_current_state(self) -> str:
        """Return the current pipeline state label."""
        with self._lock:
            return self._state

    def set_agent_state(self, agent_name: str, state: AgentStatus) -> None:
        """Set the execution state for *agent_name*.

        Args:
            agent_name: Agent identifier.
            state: New agent state.
        """
        with self._lock:
            self._agent_states[agent_name] = state

    def get_agent_states(self) -> Dict[str, AgentStatus]:
        """Return a copy of all agent states."""
        with self._lock:
            return dict(self._agent_states)

    def reset(self) -> None:
        """Reset to PENDING (for reuse across pipeline runs)."""
        with self._lock:
            self._state = _PENDING
            self._agent_states.clear()
