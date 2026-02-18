"""Graceful degradation and error categorisation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import ValidationError

from .config import OrchestratorConfig
from .schema import AgentError, CircuitBreakerOpenError
from .telemetry import get_logger

_logger = get_logger(__name__)

# Agents whose failure aborts downstream stages.
_CRITICAL_AGENTS = frozenset({
    "hypothesis_agent",
    "root_cause_agent",
})


class ErrorHandler:
    """Categorise agent exceptions and decide whether to abort.

    Strategies (configured via :class:`OrchestratorConfig`):
        * **fail_fast** — abort on the first error regardless of agent.
        * **continue** — skip failed agents when ``allow_partial_results``
          is ``True`` and the agent is not critical.
        * **abort_if_critical** — abort only when a critical agent
          (hypothesis, root_cause) fails.
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        self._errors: List[AgentError] = []

    # ---- public -----------------------------------------------------------

    def categorize_error(self, error: Exception) -> str:
        """Return a canonical error-type label for *error*.

        Args:
            error: The exception.

        Returns:
            One of ``TIMEOUT``, ``VALIDATION_ERROR``, ``LLM_ERROR``,
            ``CIRCUIT_OPEN``, ``UNKNOWN``.
        """
        if isinstance(error, asyncio.TimeoutError):
            return "TIMEOUT"
        if isinstance(error, ValidationError):
            return "VALIDATION_ERROR"
        if isinstance(error, CircuitBreakerOpenError):
            return "CIRCUIT_OPEN"
        msg = str(error).lower()
        if "rate_limit" in msg or "llm" in msg or "groq" in msg or "api" in msg:
            return "LLM_ERROR"
        return "UNKNOWN"

    def handle_agent_error(
        self,
        agent_name: str,
        error: Exception,
        stage: str = "",
        retries: int = 0,
    ) -> None:
        """Record an agent error.

        Args:
            agent_name: Agent that failed.
            error: The exception caught.
            stage: Pipeline stage label.
            retries: Number of retries attempted before failure.
        """
        error_type = self.categorize_error(error)
        agent_error = AgentError(
            agent_name=agent_name,
            error_type=error_type,
            error_message=str(error),
            timestamp=datetime.now(timezone.utc),
            retries_attempted=retries,
        )
        self._errors.append(agent_error)
        _logger.error(
            f"Agent {agent_name} failed ({error_type}): {error}",
            extra={"agent_name": agent_name, "stage": stage},
        )

    def should_abort(self, agent_name: str, stage: str = "") -> bool:
        """Decide whether the pipeline should abort.

        Args:
            agent_name: Agent that failed.
            stage: Pipeline stage label.

        Returns:
            ``True`` if the pipeline should be aborted.
        """
        if self.config.fail_fast:
            return True
        if self.is_critical_agent(agent_name):
            return True
        return False

    @staticmethod
    def is_critical_agent(agent_name: str) -> bool:
        """Return ``True`` if *agent_name* is critical (failure aborts pipeline).

        Args:
            agent_name: Agent identifier.

        Returns:
            Whether the agent is critical.
        """
        return agent_name in _CRITICAL_AGENTS

    def get_errors(self) -> List[AgentError]:
        """Return all recorded errors."""
        return list(self._errors)

    def reset(self) -> None:
        """Clear error list."""
        self._errors.clear()
