"""Generate rich HTML incident reports with embedded CSS and charts."""

from __future__ import annotations

import os
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..config import ReportingConfig
from ..schema import IncidentReport
from ..telemetry import get_logger

_logger = get_logger(__name__)


class HTMLGenerator:
    """Render :class:`IncidentReport` as HTML via Jinja2 with autoescaping.

    Args:
        config: Reporting configuration.
    """

    def __init__(self, config: Optional[ReportingConfig] = None) -> None:
        self.config = config or ReportingConfig()
        self._env = Environment(
            loader=FileSystemLoader(self.config.templates_dir),
            autoescape=select_autoescape(["html", "j2"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def generate(self, report: IncidentReport) -> str:
        """Render *report* as an HTML string.

        Args:
            report: The incident report model.

        Returns:
            Rendered HTML text.

        Raises:
            FileNotFoundError: If the template file is missing.
        """
        template_path = "incident_report.html.j2"
        full = os.path.join(self.config.templates_dir, template_path)
        if not os.path.isfile(full):
            raise FileNotFoundError(f"Template not found: {full}")
        template = self._env.get_template(template_path)
        return template.render(report=report)

    def save(self, report: IncidentReport, path: str) -> str:
        """Render and save the HTML report to *path*.

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
        _logger.info("HTML report saved", extra={"path": path})
        return os.path.abspath(path)
