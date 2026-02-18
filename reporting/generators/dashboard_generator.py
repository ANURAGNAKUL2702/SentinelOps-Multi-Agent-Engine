"""Generate executive dashboards with KPIs, trend charts, and incident tables."""

from __future__ import annotations

import os
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..config import ReportingConfig
from ..schema import DashboardData
from ..telemetry import get_logger

_logger = get_logger(__name__)


class DashboardGenerator:
    """Render :class:`DashboardData` as an HTML executive dashboard.

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

    def generate(self, dashboard: DashboardData) -> str:
        """Render the dashboard as HTML.

        Args:
            dashboard: Dashboard data model.

        Returns:
            HTML string.

        Raises:
            FileNotFoundError: If the template is missing.
        """
        template_path = "executive_summary.html.j2"
        full = os.path.join(self.config.templates_dir, template_path)
        if not os.path.isfile(full):
            raise FileNotFoundError(f"Template not found: {full}")
        template = self._env.get_template(template_path)
        return template.render(dashboard=dashboard)

    def save(self, dashboard: DashboardData, path: str) -> str:
        """Render and save the dashboard HTML to *path*.

        Args:
            dashboard: Dashboard data model.
            path: Destination file path.

        Returns:
            Absolute path of saved file.
        """
        content = self.generate(dashboard)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        _logger.info("Dashboard saved", extra={"path": path})
        return os.path.abspath(path)
