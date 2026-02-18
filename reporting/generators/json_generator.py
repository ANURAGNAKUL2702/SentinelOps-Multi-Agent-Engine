"""Generate machine-readable JSON reports from IncidentReport models."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Optional

from ..config import ReportingConfig
from ..schema import IncidentReport
from ..telemetry import get_logger

_logger = get_logger(__name__)


def _default_serializer(obj: Any) -> Any:
    """JSON fallback serializer for datetime and enums."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "value"):
        return obj.value
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class JSONGenerator:
    """Serialize :class:`IncidentReport` to JSON.

    Args:
        config: Reporting configuration.
    """

    def __init__(self, config: Optional[ReportingConfig] = None) -> None:
        self.config = config or ReportingConfig()

    def generate(self, report: IncidentReport) -> str:
        """Convert *report* to a pretty-printed JSON string.

        Args:
            report: The incident report model.

        Returns:
            JSON string representation.
        """
        data = report.model_dump(mode="python")
        return json.dumps(data, indent=2, default=_default_serializer)

    def save(self, report: IncidentReport, path: str) -> str:
        """Serialize and save the JSON report to *path*.

        Args:
            report: The incident report model.
            path: Destination file path.

        Returns:
            Absolute path of saved file.
        """
        content = self.generate(report)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        _logger.info("JSON report saved", extra={"path": path})
        return os.path.abspath(path)
