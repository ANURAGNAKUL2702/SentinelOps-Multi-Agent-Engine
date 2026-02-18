"""
File: llm/causal_reasoner.py
Purpose: Algorithm 4 — Causal Chain Construction (LLM-enhanced).
Dependencies: Schema models only.
Performance: <2s with LLM, <5ms fallback.

Enhances hypothesis causal chains using LLM reasoning.
Uses the theory_generator's LLM provider and circuit breaker
for consistency.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from agents.hypothesis_agent.config import HypothesisAgentConfig
from agents.hypothesis_agent.schema import (
    AggregatedEvidence,
    CausalChain,
    CausalChainLink,
    CausalRelationship,
    Hypothesis,
)
from agents.hypothesis_agent.llm.theory_generator import (
    LLMProvider,
    LLMProviderError,
    MockLLMProvider,
)
from agents.hypothesis_agent.telemetry import TelemetryCollector, get_logger

logger = get_logger("hypothesis_agent.llm.causal_reasoner")


class CausalReasoner:
    """Enhances hypothesis causal chains using LLM reasoning.

    For hypotheses that lack or have weak causal chains,
    this module asks the LLM to construct detailed chains
    explaining how the root cause propagated through the system.

    Args:
        config: Agent configuration.
        provider: LLM provider (shared with theory_generator).
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

    def enhance_chains(
        self,
        hypotheses: List[Hypothesis],
        evidence: AggregatedEvidence,
        correlation_id: str = "",
    ) -> List[Hypothesis]:
        """Enhance causal chains for hypotheses lacking them.

        Only calls LLM for hypotheses without causal chains
        or with low-confidence chains.

        Args:
            hypotheses: Hypotheses to enhance.
            evidence: Aggregated evidence for context.
            correlation_id: Request correlation ID.

        Returns:
            Hypotheses with enhanced causal chains.
        """
        if not self._config.features.enable_causal_reasoning:
            return hypotheses

        start = time.perf_counter()
        enhanced: List[Hypothesis] = []

        for h in hypotheses:
            if self._needs_enhancement(h):
                try:
                    chain = self._build_chain_via_llm(
                        h, evidence, correlation_id
                    )
                    if chain:
                        h = h.model_copy(
                            update={"causal_chain": chain}
                        )
                except LLMProviderError:
                    # Keep existing chain if LLM fails
                    pass

            enhanced.append(h)

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"Causal reasoning complete — "
            f"{len(enhanced)} hypotheses processed, "
            f"{elapsed_ms:.2f}ms",
            extra={
                "correlation_id": correlation_id,
                "layer": "causal_reasoning",
            },
        )

        return enhanced

    def _needs_enhancement(self, h: Hypothesis) -> bool:
        """Check if a hypothesis needs causal chain enhancement."""
        if h.causal_chain is None:
            return True
        if not h.causal_chain.chain:
            return True
        if h.causal_chain.chain_confidence < 0.4:
            return True
        return False

    def _build_chain_via_llm(
        self,
        hypothesis: Hypothesis,
        evidence: AggregatedEvidence,
        correlation_id: str = "",
    ) -> Optional[CausalChain]:
        """Ask LLM to construct a causal chain for a hypothesis.

        Args:
            hypothesis: The hypothesis to build chain for.
            evidence: Evidence context.
            correlation_id: Request correlation ID.

        Returns:
            CausalChain or None on failure.
        """
        system_prompt = (
            "You are an expert at tracing incident causality. "
            "Given a root cause hypothesis and supporting evidence, "
            "construct a causal chain showing how the root cause "
            "propagated through the system step by step.\n\n"
            "Respond with ONLY valid JSON:\n"
            "{\n"
            '  "chain": [\n'
            "    {\n"
            '      "step": 1,\n'
            '      "service": "<service name>",\n'
            '      "event": "<what happened>",\n'
            '      "relationship": "causes|contributes_to|correlates_with"\n'
            "    }\n"
            "  ],\n"
            '  "root_cause_service": "<first service>",\n'
            '  "terminal_effect": "<final symptom>",\n'
            '  "confidence": <float 0.0-1.0>\n'
            "}"
        )

        evidence_summary = [
            {
                "description": e.description,
                "source": e.source.value,
                "severity": e.severity.value,
            }
            for e in evidence.evidence_items[:10]
        ]

        user_prompt = (
            f"Hypothesis: {hypothesis.theory}\n"
            f"Category: {hypothesis.category.value}\n"
            f"Supporting evidence: "
            f"{json.dumps(hypothesis.evidence_supporting)}\n"
            f"Evidence summary: "
            f"{json.dumps(evidence_summary, indent=2)}\n\n"
            "Build the causal chain."
        )

        try:
            response = self._provider.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                timeout_seconds=self._config.llm.timeout_seconds,
            )

            input_tokens = self._provider.count_tokens(
                system_prompt + user_prompt
            )
            output_tokens = self._provider.count_tokens(
                json.dumps(response, default=str)
            )
            self._telemetry.record_llm_call(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_per_1k_input=(
                    self._config.llm.cost_per_1k_input_tokens
                ),
                cost_per_1k_output=(
                    self._config.llm.cost_per_1k_output_tokens
                ),
                success=True,
                correlation_id=correlation_id,
            )

            return self._parse_chain(response)

        except LLMProviderError:
            self._telemetry.record_llm_call(
                success=False, correlation_id=correlation_id
            )
            raise

    def _parse_chain(
        self, response: Dict[str, Any]
    ) -> Optional[CausalChain]:
        """Parse LLM response into a CausalChain."""
        try:
            chain_raw = response.get("chain", [])
            if not isinstance(chain_raw, list) or not chain_raw:
                return None

            links: List[CausalChainLink] = []
            for raw in chain_raw:
                if not isinstance(raw, dict):
                    continue
                rel_str = raw.get("relationship", "causes")
                try:
                    rel = CausalRelationship(rel_str)
                except ValueError:
                    rel = CausalRelationship.CAUSES

                links.append(CausalChainLink(
                    step=int(raw.get("step", len(links) + 1)),
                    service=str(raw.get("service", "unknown")),
                    event=str(raw.get("event", "unknown")),
                    relationship=rel,
                ))

            if not links:
                return None

            confidence = float(response.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))

            return CausalChain(
                chain=links,
                root_cause_service=str(
                    response.get(
                        "root_cause_service", links[0].service
                    )
                ),
                terminal_effect=str(
                    response.get(
                        "terminal_effect", links[-1].event
                    )
                ),
                chain_confidence=confidence,
            )

        except (KeyError, TypeError, ValueError) as e:
            logger.warning(
                f"Failed to parse causal chain: {e}",
                extra={"layer": "causal_reasoning"},
            )
            return None
