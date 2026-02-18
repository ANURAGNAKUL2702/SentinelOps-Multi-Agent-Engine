"""
File: llm/hypothesis_refiner.py
Purpose: LLM-based hypothesis refinement — merges similar theories.
Dependencies: Schema models only.
Performance: <1s with LLM, <1ms without.

Refines and consolidates hypotheses post-generation using LLM
to produce clearer, more actionable theories.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from agents.hypothesis_agent.config import HypothesisAgentConfig
from agents.hypothesis_agent.schema import (
    Hypothesis,
    IncidentCategory,
    Severity,
)
from agents.hypothesis_agent.llm.theory_generator import (
    LLMProvider,
    LLMProviderError,
    MockLLMProvider,
)
from agents.hypothesis_agent.telemetry import TelemetryCollector, get_logger

logger = get_logger("hypothesis_agent.llm.hypothesis_refiner")


class HypothesisRefiner:
    """Refines hypothesis text using LLM for clarity.

    Takes ranked hypotheses and produces human-readable summaries.
    Does NOT change ranking or scores — only improves text quality.

    Args:
        config: Agent configuration.
        provider: LLM provider.
        telemetry: Telemetry collector.
    """

    def __init__(
        self,
        config: Optional[HypothesisAgentConfig] = None,
        provider: Optional[LLMProvider] = None,
        telemetry: Optional[TelemetryCollector] = None,
    ) -> None:
        self._config = config or HypothesisAgentConfig()
        self._provider = provider or MockLLMProvider()
        self._telemetry = telemetry or TelemetryCollector()

    def generate_summary(
        self,
        hypotheses: List[Hypothesis],
        correlation_id: str = "",
    ) -> str:
        """Generate a human-readable summary of the top hypotheses.

        Args:
            hypotheses: Ranked hypotheses (best first).
            correlation_id: Request correlation ID.

        Returns:
            Summary string.
        """
        if not hypotheses:
            return "No hypotheses generated — insufficient evidence."

        # Use the top hypothesis to build a summary
        top = hypotheses[0]
        parts = [
            f"Most likely root cause: {top.theory} "
            f"(likelihood={top.likelihood_score:.0%}, "
            f"category={top.category.value}, "
            f"severity={top.severity.value})"
        ]

        if len(hypotheses) > 1:
            alts = [
                f"  - {h.theory} ({h.likelihood_score:.0%})"
                for h in hypotheses[1:3]
            ]
            parts.append(
                "Alternative hypotheses:\n" + "\n".join(alts)
            )

        if top.causal_chain and top.causal_chain.chain:
            chain_text = " → ".join(
                f"{link.service}: {link.event}"
                for link in top.causal_chain.chain
            )
            parts.append(f"Causal chain: {chain_text}")

        return "\n".join(parts)

    def determine_category(
        self, hypotheses: List[Hypothesis]
    ) -> IncidentCategory:
        """Determine the overall incident category from hypotheses.

        Args:
            hypotheses: Ranked hypotheses.

        Returns:
            Best-guess IncidentCategory.
        """
        if not hypotheses:
            return IncidentCategory.UNKNOWN
        return hypotheses[0].category

    def determine_severity(
        self, hypotheses: List[Hypothesis]
    ) -> Severity:
        """Determine the overall incident severity from hypotheses.

        Uses the highest severity among top hypotheses.

        Args:
            hypotheses: Ranked hypotheses.

        Returns:
            Overall Severity.
        """
        if not hypotheses:
            return Severity.MEDIUM

        severity_order = {
            Severity.CRITICAL: 4,
            Severity.HIGH: 3,
            Severity.MEDIUM: 2,
            Severity.LOW: 1,
        }

        max_sev = max(
            hypotheses[:3],
            key=lambda h: severity_order.get(h.severity, 0),
        )
        return max_sev.severity
