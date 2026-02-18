"""Slack notifier — send incident summaries to a Slack channel."""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional

from ..telemetry import get_logger

_logger = get_logger(__name__)


class SlackNotifier:
    """Post incident notifications to Slack via incoming webhooks.

    Uses only ``urllib`` so there are no extra dependencies.

    Args:
        webhook_url: Slack incoming-webhook URL.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        webhook_url: str,
        timeout: int = 10,
    ) -> None:
        self._url = webhook_url
        self._timeout = timeout

    def notify(
        self,
        *,
        correlation_id: str,
        root_cause: str = "",
        severity: str = "P2_MEDIUM",
        confidence: float = 0.0,
        duration: float = 0.0,
        total_cost: float = 0.0,
        status: str = "resolved",
    ) -> bool:
        """Send a Slack notification for an incident.

        Args:
            correlation_id: Unique incident ID.
            root_cause: Identified root cause.
            severity: Severity label.
            confidence: Root-cause confidence.
            duration: Pipeline duration in seconds.
            total_cost: Total LLM cost.
            status: Incident status.

        Returns:
            ``True`` if the message was accepted by Slack, else ``False``.
        """
        color = self._severity_color(severity)
        blocks = self._build_payload(
            correlation_id=correlation_id,
            root_cause=root_cause,
            severity=severity,
            confidence=confidence,
            duration=duration,
            total_cost=total_cost,
            status=status,
            color=color,
        )
        return self._post(blocks)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_payload(self, *, color: str, **fields: Any) -> Dict[str, Any]:
        attachment = {
            "color": color,
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"🚨 Incident Report — {fields['severity']}",
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Correlation ID:*\n{fields['correlation_id']}"},
                        {"type": "mrkdwn", "text": f"*Status:*\n{fields['status']}"},
                        {"type": "mrkdwn", "text": f"*Root Cause:*\n{fields['root_cause']}"},
                        {"type": "mrkdwn", "text": f"*Confidence:*\n{fields['confidence']:.0%}"},
                        {"type": "mrkdwn", "text": f"*Duration:*\n{fields['duration']:.1f}s"},
                        {"type": "mrkdwn", "text": f"*Cost:*\n${fields['total_cost']:.4f}"},
                    ],
                },
            ],
        }
        return {"attachments": [attachment]}

    @staticmethod
    def _severity_color(severity: str) -> str:
        mapping = {
            "P0_CRITICAL": "#dc3545",
            "P1_HIGH": "#fd7e14",
            "P2_MEDIUM": "#ffc107",
            "P3_LOW": "#28a745",
        }
        return mapping.get(severity, "#6c757d")

    def _post(self, payload: Dict[str, Any]) -> bool:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                ok = resp.status == 200
                if ok:
                    _logger.info("Slack notification sent")
                else:
                    _logger.warning("Slack returned status %d", resp.status)
                return ok
        except (urllib.error.URLError, OSError):
            _logger.exception("Failed to send Slack notification")
            return False
