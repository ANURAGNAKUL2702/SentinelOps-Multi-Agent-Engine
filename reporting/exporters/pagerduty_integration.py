"""PagerDuty integration — trigger / resolve incidents."""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any, Dict, Optional

from ..telemetry import get_logger

_logger = get_logger(__name__)

_PD_EVENTS_URL = "https://events.pagerduty.com/v2/enqueue"


class PagerDutyIntegration:
    """Send PagerDuty events via the Events API v2.

    Uses only ``urllib`` — no extra dependencies.

    Args:
        routing_key: PagerDuty integration / routing key.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        routing_key: str,
        timeout: int = 10,
    ) -> None:
        self._routing_key = routing_key
        self._timeout = timeout

    def trigger(
        self,
        *,
        correlation_id: str,
        summary: str,
        severity: str = "warning",
        source: str = "warroom-simulator",
        custom_details: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Trigger a new PagerDuty incident.

        Args:
            correlation_id: Dedup key.
            summary: Human-readable summary (max 1024 chars).
            severity: PagerDuty severity (critical/error/warning/info).
            source: Source identifier.
            custom_details: Optional extra details dict.

        Returns:
            The ``dedup_key`` on success, or ``None``.
        """
        payload: Dict[str, Any] = {
            "routing_key": self._routing_key,
            "dedup_key": correlation_id,
            "event_action": "trigger",
            "payload": {
                "summary": summary[:1024],
                "severity": self._map_severity(severity),
                "source": source,
                "custom_details": custom_details or {},
            },
        }
        return self._send(payload)

    def resolve(
        self,
        *,
        correlation_id: str,
    ) -> Optional[str]:
        """Resolve a PagerDuty incident.

        Args:
            correlation_id: Dedup key of the incident to resolve.

        Returns:
            The ``dedup_key`` on success, or ``None``.
        """
        payload: Dict[str, Any] = {
            "routing_key": self._routing_key,
            "dedup_key": correlation_id,
            "event_action": "resolve",
        }
        return self._send(payload)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _map_severity(severity: str) -> str:
        mapping = {
            "P0_CRITICAL": "critical",
            "P1_HIGH": "error",
            "P2_MEDIUM": "warning",
            "P3_LOW": "info",
        }
        return mapping.get(severity, severity)

    def _send(self, payload: Dict[str, Any]) -> Optional[str]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            _PD_EVENTS_URL,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                dedup = body.get("dedup_key", payload.get("dedup_key"))
                _logger.info("PagerDuty event sent: %s", dedup)
                return dedup
        except (urllib.error.URLError, OSError):
            _logger.exception("Failed to send PagerDuty event")
            return None
