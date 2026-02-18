"""Cost breakdown charts — pie and bar charts for LLM cost analysis."""

from __future__ import annotations

import base64
import io
from typing import Dict, Optional

from ..config import ReportingConfig

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False


class CostBreakdownChart:
    """Generate pie / bar charts for cost breakdown.

    Args:
        config: Reporting configuration.
    """

    def __init__(self, config: Optional[ReportingConfig] = None) -> None:
        self.config = config or ReportingConfig()

    # ------------------------------------------------------------------
    # Pie chart
    # ------------------------------------------------------------------

    def generate_pie_chart(
        self,
        cost_by_agent: Dict[str, float],
        *,
        fmt: str = "png",
    ) -> str:
        """Create a pie chart of costs by agent.

        Args:
            cost_by_agent: Mapping ``{agent_name: cost}``.
            fmt: Image format.

        Returns:
            Base64-encoded image.
        """
        if not _HAS_MPL:
            raise ImportError("matplotlib is required")

        fig, ax = plt.subplots(
            figsize=(
                self.config.chart_width / self.config.chart_dpi,
                self.config.chart_height / self.config.chart_dpi,
            ),
            dpi=self.config.chart_dpi,
        )

        # Filter out zero-cost agents
        filtered = {k: v for k, v in cost_by_agent.items() if v > 0}

        if not filtered:
            ax.text(0.5, 0.5, "$0.00 (all deterministic)", ha="center", va="center", fontsize=11)
            ax.axis("off")
        else:
            labels = list(filtered.keys())
            values = list(filtered.values())
            ax.pie(
                values,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
                textprops={"fontsize": 8},
            )
            ax.set_title("Cost by Agent", fontweight="bold")

        fig.tight_layout()
        result = self._fig_to_base64(fig, fmt)
        plt.close(fig)
        return result

    # ------------------------------------------------------------------
    # Bar chart
    # ------------------------------------------------------------------

    def generate_bar_chart(
        self,
        tokens_by_agent: Dict[str, int],
        *,
        fmt: str = "png",
    ) -> str:
        """Create a bar chart of token usage by agent.

        Args:
            tokens_by_agent: Mapping ``{agent_name: token_count}``.
            fmt: Image format.

        Returns:
            Base64-encoded image.
        """
        if not _HAS_MPL:
            raise ImportError("matplotlib is required")

        fig, ax = plt.subplots(
            figsize=(
                self.config.chart_width / self.config.chart_dpi,
                self.config.chart_height / self.config.chart_dpi,
            ),
            dpi=self.config.chart_dpi,
        )

        if not tokens_by_agent or all(v == 0 for v in tokens_by_agent.values()):
            ax.text(0.5, 0.5, "No token data", ha="center", va="center", fontsize=11)
            ax.axis("off")
        else:
            agents = list(tokens_by_agent.keys())
            tokens = list(tokens_by_agent.values())
            ax.barh(agents, tokens, color="#4361ee", edgecolor="white")
            ax.set_xlabel("Tokens")
            ax.set_title("Token Usage by Agent", fontweight="bold")

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
