"""
File: core/causal_chain_builder.py
Purpose: Algorithm 3 — Merge hypothesis causal chains with dep graph into final chain.
Dependencies: Schema models only.
Performance: <2ms, O(n) where n = chain links.

Merges causal chains from the hypothesis agent with dependency
graph edges, deduplicates, validates no cycles, and returns
the final List[CausalLink].
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from agents.root_cause_agent.config import RootCauseAgentConfig
from agents.root_cause_agent.schema import (
    CausalLink,
    CausalRelationship,
    DependencyAgentFindings,
    HypothesisFindings,
    RootCauseAgentInput,
)
from agents.root_cause_agent.telemetry import get_logger

logger = get_logger("root_cause_agent.causal_chain_builder")


class CausalChainBuilder:
    """Builds the final causal chain from hypothesis chains + dep graph.

    Pipeline::

        HypothesisChains ──┐
                           ├──  merge → dedup → validate_no_cycles → trim
        DepGraph edges   ──┘

    Args:
        config: Agent configuration.
    """

    def __init__(
        self, config: Optional[RootCauseAgentConfig] = None
    ) -> None:
        self._config = config or RootCauseAgentConfig()

    def build(
        self,
        input_data: RootCauseAgentInput,
        primary_service: str = "",
        correlation_id: str = "",
    ) -> List[CausalLink]:
        """Build the final causal chain.

        Args:
            input_data: Root cause agent input.
            primary_service: The most-blamed service.
            correlation_id: Request correlation ID.

        Returns:
            Ordered list of CausalLink objects (root_cause → symptoms).
        """
        links: List[CausalLink] = []

        # ── Extract from hypothesis causal chains ───────────────
        hyp_links = self._extract_hypothesis_links(
            input_data.hypothesis_findings
        )
        links.extend(hyp_links)

        # ── Extract from dependency graph ───────────────────────
        dep_links = self._extract_dependency_links(
            input_data.dependency_findings, primary_service
        )
        links.extend(dep_links)

        # ── Deduplicate ─────────────────────────────────────────
        links = self._deduplicate(links)

        # ── Validate no cycles ──────────────────────────────────
        if self._has_cycle(links):
            links = self._remove_cycle(links)

        # ── Trim to max depth ──────────────────────────────────
        max_depth = self._config.limits.max_causal_chain_depth
        if len(links) > max_depth:
            links = links[:max_depth]

        logger.debug(
            f"Causal chain built: {len(links)} links",
            extra={
                "correlation_id": correlation_id,
                "layer": "causal_chain",
            },
        )

        return links

    def _extract_hypothesis_links(
        self, findings: HypothesisFindings
    ) -> List[CausalLink]:
        """Extract causal links from hypothesis agent chains."""
        links: List[CausalLink] = []

        for chain_data in findings.causal_chains:
            chain_links = chain_data.get("chain", [])
            if isinstance(chain_links, list):
                for i in range(len(chain_links) - 1):
                    current = chain_links[i]
                    next_item = chain_links[i + 1]

                    cause = (
                        current.get("event", str(current))
                        if isinstance(current, dict)
                        else str(current)
                    )
                    effect = (
                        next_item.get("event", str(next_item))
                        if isinstance(next_item, dict)
                        else str(next_item)
                    )
                    service = (
                        current.get("service", "")
                        if isinstance(current, dict)
                        else ""
                    )

                    links.append(CausalLink(
                        cause=cause,
                        effect=effect,
                        relationship=CausalRelationship.CAUSES,
                        confidence=findings.top_confidence,
                        service=service,
                    ))

        # If no chains but we have a top hypothesis, create a simple link
        if not links and findings.top_hypothesis:
            links.append(CausalLink(
                cause=findings.top_hypothesis,
                effect="Service degradation observed",
                relationship=CausalRelationship.CAUSES,
                confidence=findings.top_confidence,
                service="",
            ))

        return links

    def _extract_dependency_links(
        self,
        findings: DependencyAgentFindings,
        primary_service: str,
    ) -> List[CausalLink]:
        """Extract causal links from dependency graph edges."""
        links: List[CausalLink] = []

        # Use critical paths first
        for path in findings.critical_paths:
            for i in range(len(path) - 1):
                links.append(CausalLink(
                    cause=f"Failure in {path[i]}",
                    effect=f"Impact on {path[i + 1]}",
                    relationship=CausalRelationship.CAUSES,
                    confidence=findings.confidence,
                    service=path[i],
                ))

        # Add edges from impact graph if relevant to primary service
        if primary_service and primary_service in findings.impact_graph:
            downstream = findings.impact_graph[primary_service]
            for target in downstream[:5]:  # limit
                links.append(CausalLink(
                    cause=f"Failure in {primary_service}",
                    effect=f"Impact on {target}",
                    relationship=CausalRelationship.CONTRIBUTES_TO,
                    confidence=findings.confidence * 0.7,
                    service=primary_service,
                ))

        return links

    def _deduplicate(self, links: List[CausalLink]) -> List[CausalLink]:
        """Remove duplicate causal links, keeping highest confidence."""
        seen: Dict[Tuple[str, str], CausalLink] = {}
        for link in links:
            key = (link.cause, link.effect)
            existing = seen.get(key)
            if existing is None or link.confidence > existing.confidence:
                seen[key] = link
        return list(seen.values())

    def _has_cycle(self, links: List[CausalLink]) -> bool:
        """Detect cycles in the causal chain using DFS."""
        graph: Dict[str, List[str]] = {}
        for link in links:
            graph.setdefault(link.cause, []).append(link.effect)

        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.discard(node)
            return False

        for node in graph:
            if node not in visited:
                if dfs(node):
                    return True
        return False

    def _remove_cycle(self, links: List[CausalLink]) -> List[CausalLink]:
        """Remove the last-added link that creates a cycle."""
        result: List[CausalLink] = []
        for link in links:
            result.append(link)
            if self._has_cycle(result):
                result.pop()
        return result
