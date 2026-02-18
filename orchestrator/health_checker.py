"""Agent health monitoring and readiness probes."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict

from .config import OrchestratorConfig
from .schema import HealthStatus, HealthStatusValue
from .telemetry import get_logger

_logger = get_logger(__name__)


class HealthChecker:
    """Lightweight health/readiness probes for agents.

    If an agent exposes a ``health_check()`` method it is called;
    otherwise the agent is assumed healthy.
    """

    def __init__(self, config: OrchestratorConfig) -> None:
        self.config = config
        self._last_status: Dict[str, HealthStatus] = {}

    async def check_agent_health(
        self,
        agent_name: str,
        agent_instance: Any,
    ) -> HealthStatus:
        """Probe a single agent.

        Args:
            agent_name: Agent identifier.
            agent_instance: The agent object.

        Returns:
            :class:`HealthStatus` indicating health.
        """
        now = datetime.now(timezone.utc)
        checker = getattr(agent_instance, "health_check", None)

        if checker is None:
            # No health endpoint → assume healthy
            status = HealthStatus(
                agent_name=agent_name,
                is_healthy=True,
                status=HealthStatusValue.HEALTHY,
                last_checked=now,
            )
            self._last_status[agent_name] = status
            return status

        try:
            result = checker()
            if asyncio.iscoroutine(result):
                result = await asyncio.wait_for(
                    result, timeout=self.config.health_check_timeout
                )
            is_healthy = bool(result)
            status = HealthStatus(
                agent_name=agent_name,
                is_healthy=is_healthy,
                status=HealthStatusValue.HEALTHY if is_healthy else HealthStatusValue.UNHEALTHY,
                last_checked=now,
            )
        except asyncio.TimeoutError:
            status = HealthStatus(
                agent_name=agent_name,
                is_healthy=False,
                status=HealthStatusValue.UNHEALTHY,
                last_checked=now,
                error_message="Health check timed out",
            )
        except Exception as exc:
            status = HealthStatus(
                agent_name=agent_name,
                is_healthy=False,
                status=HealthStatusValue.UNHEALTHY,
                last_checked=now,
                error_message=str(exc),
            )

        self._last_status[agent_name] = status
        return status

    async def check_all_agents(
        self,
        agents: Dict[str, Any],
    ) -> Dict[str, HealthStatus]:
        """Probe all agents concurrently.

        Args:
            agents: Mapping of agent_name → agent_instance.

        Returns:
            Mapping of agent_name → :class:`HealthStatus`.
        """
        tasks = {
            name: self.check_agent_health(name, inst)
            for name, inst in agents.items()
        }
        results: Dict[str, HealthStatus] = {}
        for name, coro in tasks.items():
            results[name] = await coro
        return results

    def is_agent_ready(self, agent_name: str) -> bool:
        """Return whether the last probe for *agent_name* was healthy.

        Args:
            agent_name: Agent identifier.

        Returns:
            ``True`` if healthy (or never checked).
        """
        status = self._last_status.get(agent_name)
        if status is None:
            return True  # never checked → assume ready
        return status.is_healthy
