"""Dependency graph visualisation using matplotlib (networkx optional)."""

from __future__ import annotations

import base64
import io
from typing import Dict, List, Optional, Set

from ..config import ReportingConfig

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False


class DependencyGraphViz:
    """Render a service dependency graph.

    Uses matplotlib only so there is no hard dependency on graphviz.

    Args:
        config: Reporting configuration.
    """

    def __init__(self, config: Optional[ReportingConfig] = None) -> None:
        self.config = config or ReportingConfig()

    def generate(
        self,
        services: List[str],
        edges: List[tuple[str, str]],
        affected: Optional[Set[str]] = None,
        *,
        fmt: str = "png",
    ) -> str:
        """Create a dependency graph chart.

        Args:
            services: List of service names (nodes).
            edges: Directed edges ``(source, target)``.
            affected: Set of service names to highlight in red.
            fmt: Image format.

        Returns:
            Base64-encoded image.
        """
        if not _HAS_MPL:
            raise ImportError("matplotlib is required for dependency graphs")

        affected = affected or set()
        fig, ax = plt.subplots(
            figsize=(
                self.config.chart_width / self.config.chart_dpi,
                self.config.chart_height / self.config.chart_dpi,
            ),
            dpi=self.config.chart_dpi,
        )

        if not services:
            ax.text(0.5, 0.5, "No services", ha="center", va="center")
            ax.axis("off")
            result = self._fig_to_base64(fig, fmt)
            plt.close(fig)
            return result

        # Simple circular layout
        import math

        n = len(services)
        positions: Dict[str, tuple[float, float]] = {}
        for i, svc in enumerate(services):
            angle = 2 * math.pi * i / max(n, 1)
            positions[svc] = (math.cos(angle), math.sin(angle))

        # Draw edges
        for src, tgt in edges:
            if src in positions and tgt in positions:
                x0, y0 = positions[src]
                x1, y1 = positions[tgt]
                ax.annotate(
                    "",
                    xy=(x1, y1),
                    xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#888", lw=1.2),
                )

        # Draw nodes
        for svc, (x, y) in positions.items():
            colour = "#dc3545" if svc in affected else "#4361ee"
            ax.scatter(x, y, s=600, c=colour, zorder=5, edgecolors="white", linewidth=2)
            ax.text(x, y - 0.15, svc, ha="center", fontsize=7, fontweight="bold")

        ax.set_title("Service Dependency Graph", fontweight="bold")
        ax.axis("off")
        fig.tight_layout()
        result = self._fig_to_base64(fig, fmt)
        plt.close(fig)
        return result

    @staticmethod
    def _fig_to_base64(fig: object, fmt: str) -> str:
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, bbox_inches="tight")  # type: ignore[union-attr]
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("ascii")
