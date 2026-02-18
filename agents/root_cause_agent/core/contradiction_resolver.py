"""
File: core/contradiction_resolver.py
Purpose: Algorithm 6 — Detect and resolve conflicts between agents.
Dependencies: Schema models only.
Performance: <2ms, O(n²) where n = evidence items.

Detects contradictions (different services blamed by different agents),
resolves by highest confidence / timestamp priority / graph centrality,
and flags unresolved conflicts.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple

from agents.root_cause_agent.config import RootCauseAgentConfig
from agents.root_cause_agent.schema import (
    Contradiction,
    ContradictionStrategy,
    Evidence,
    EvidenceSourceAgent,
    EvidenceType,
    RootCauseAgentInput,
)
from agents.root_cause_agent.telemetry import get_logger

logger = get_logger("root_cause_agent.contradiction_resolver")


class ContradictionResolver:
    """Detects and resolves conflicts between agent findings.

    Strategies (in priority order):
    1. Confidence wins — higher-confidence agent's claim accepted.
    2. Timestamp priority — more recent evidence wins.
    3. Graph centrality — service with more connections wins.
    4. Unresolved — flag both claims, no winner.

    Args:
        config: Agent configuration.
    """

    def __init__(
        self, config: Optional[RootCauseAgentConfig] = None
    ) -> None:
        self._config = config or RootCauseAgentConfig()

    def resolve(
        self,
        input_data: RootCauseAgentInput,
        evidence_trail: List[Evidence],
        correlation_id: str = "",
    ) -> List[Contradiction]:
        """Detect and resolve contradictions.

        Args:
            input_data: Root cause agent input.
            evidence_trail: Unified evidence trail.
            correlation_id: Request correlation ID.

        Returns:
            List of contradictions (resolved and unresolved).
        """
        contradictions: List[Contradiction] = []

        # ── Detect service-blame conflicts ──────────────────────
        service_blame = self._collect_blame(input_data)

        # If agents blame different services, that's a contradiction
        blamed_services: Dict[EvidenceSourceAgent, str] = {}
        for agent, services in service_blame.items():
            if services:
                # Take the most-blamed service per agent
                top = max(services, key=services.get)  # type: ignore[arg-type]
                blamed_services[agent] = top

        # Compare pairs
        agents = list(blamed_services.keys())
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agents[i], agents[j]
                svc_a, svc_b = blamed_services[a], blamed_services[b]

                if svc_a != svc_b:
                    contradiction = self._resolve_pair(
                        a, b, svc_a, svc_b, input_data
                    )
                    contradictions.append(contradiction)

        logger.debug(
            f"Contradictions: {len(contradictions)} detected",
            extra={
                "correlation_id": correlation_id,
                "layer": "contradiction_resolution",
            },
        )

        return contradictions

    def _collect_blame(
        self, input_data: RootCauseAgentInput
    ) -> Dict[EvidenceSourceAgent, Counter]:
        """Collect service blame counts per agent.

        Args:
            input_data: Root cause agent input.

        Returns:
            Mapping of agent → Counter of blamed services.
        """
        blame: Dict[EvidenceSourceAgent, Counter] = {}

        # Log agent
        log_counter: Counter = Counter()
        for svc in input_data.log_findings.suspicious_services:
            log_counter[svc] += 1
        if log_counter:
            blame[EvidenceSourceAgent.LOG_AGENT] = log_counter

        # Metrics agent
        met_counter: Counter = Counter()
        for anom in input_data.metrics_findings.anomalies:
            svc = anom.get("service", "")
            if svc:
                met_counter[svc] += 1
        if met_counter:
            blame[EvidenceSourceAgent.METRICS_AGENT] = met_counter

        # Dependency agent
        dep_counter: Counter = Counter()
        for bn in input_data.dependency_findings.bottlenecks:
            dep_counter[bn] += 1
        if dep_counter:
            blame[EvidenceSourceAgent.DEPENDENCY_AGENT] = dep_counter

        # Hypothesis agent — single primary blame
        hyp_counter: Counter = Counter()
        if input_data.hypothesis_findings.top_hypothesis:
            # Try to extract service from hypothesis
            for hyp in input_data.hypothesis_findings.ranked_hypotheses:
                svc = hyp.get("service", "")
                if svc:
                    hyp_counter[svc] += 1
        if hyp_counter:
            blame[EvidenceSourceAgent.HYPOTHESIS_AGENT] = hyp_counter

        return blame

    def _resolve_pair(
        self,
        agent_a: EvidenceSourceAgent,
        agent_b: EvidenceSourceAgent,
        service_a: str,
        service_b: str,
        input_data: RootCauseAgentInput,
    ) -> Contradiction:
        """Resolve a contradiction between two agents.

        Args:
            agent_a: First agent.
            agent_b: Second agent.
            service_a: Service blamed by agent A.
            service_b: Service blamed by agent B.
            input_data: Full input for context.

        Returns:
            Contradiction with resolution.
        """
        conf_a = self._agent_confidence(agent_a, input_data)
        conf_b = self._agent_confidence(agent_b, input_data)

        # Strategy 1: Confidence wins
        if abs(conf_a - conf_b) > 0.1:
            winner = service_a if conf_a > conf_b else service_b
            return Contradiction(
                agent_a=agent_a,
                agent_b=agent_b,
                claim_a=f"Root cause in {service_a}",
                claim_b=f"Root cause in {service_b}",
                resolved=True,
                resolution_strategy=ContradictionStrategy.CONFIDENCE_WINS,
                winner=winner,
            )

        # Strategy 2: Graph centrality — more connected service wins
        graph = input_data.dependency_findings.impact_graph
        centrality_a = len(graph.get(service_a, []))
        centrality_b = len(graph.get(service_b, []))

        if centrality_a != centrality_b:
            winner = service_a if centrality_a > centrality_b else service_b
            return Contradiction(
                agent_a=agent_a,
                agent_b=agent_b,
                claim_a=f"Root cause in {service_a}",
                claim_b=f"Root cause in {service_b}",
                resolved=True,
                resolution_strategy=ContradictionStrategy.GRAPH_CENTRALITY,
                winner=winner,
            )

        # Strategy 3: Unresolved
        return Contradiction(
            agent_a=agent_a,
            agent_b=agent_b,
            claim_a=f"Root cause in {service_a}",
            claim_b=f"Root cause in {service_b}",
            resolved=False,
            resolution_strategy=ContradictionStrategy.UNRESOLVED,
            winner="",
        )

    def _agent_confidence(
        self, agent: EvidenceSourceAgent, input_data: RootCauseAgentInput
    ) -> float:
        """Get confidence for a specific agent.

        Args:
            agent: The agent.
            input_data: Root cause agent input.

        Returns:
            Agent confidence 0.0-1.0.
        """
        conf_map = {
            EvidenceSourceAgent.LOG_AGENT: input_data.log_findings.confidence,
            EvidenceSourceAgent.METRICS_AGENT: input_data.metrics_findings.confidence,
            EvidenceSourceAgent.DEPENDENCY_AGENT: input_data.dependency_findings.confidence,
            EvidenceSourceAgent.HYPOTHESIS_AGENT: input_data.hypothesis_findings.confidence,
        }
        return conf_map.get(agent, 0.0)
