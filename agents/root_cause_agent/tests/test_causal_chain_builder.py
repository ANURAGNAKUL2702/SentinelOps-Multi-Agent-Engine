"""
Tests for core/causal_chain_builder.py — Algorithm 3.
"""

import pytest

from agents.root_cause_agent.core.causal_chain_builder import CausalChainBuilder
from agents.root_cause_agent.schema import (
    CausalRelationship,
    DependencyAgentFindings,
    HypothesisFindings,
    RootCauseAgentInput,
)


def _make_input(**kwargs) -> RootCauseAgentInput:
    defaults = dict(
        hypothesis_findings=HypothesisFindings(
            top_hypothesis="DB overload",
            top_confidence=0.85,
            causal_chains=[
                {"chain": [
                    {"event": "DB saturation", "service": "db"},
                    {"event": "Timeout in API", "service": "api"},
                    {"event": "User errors", "service": "frontend"},
                ]}
            ],
            confidence=0.85,
            timestamp="2024-01-01T00:00:00Z",
        ),
        dependency_findings=DependencyAgentFindings(
            impact_graph={"db": ["api", "cache"], "api": ["frontend"]},
            critical_paths=[["db", "api", "frontend"]],
            confidence=0.8,
            timestamp="2024-01-01T00:00:00Z",
        ),
    )
    defaults.update(kwargs)
    return RootCauseAgentInput(**defaults)


class TestCausalChainBuilder:
    def test_build_from_hypothesis_chains(self):
        builder = CausalChainBuilder()
        links = builder.build(_make_input(), primary_service="db")
        assert len(links) > 0

    def test_links_have_cause_and_effect(self):
        builder = CausalChainBuilder()
        links = builder.build(_make_input(), primary_service="db")
        for link in links:
            assert link.cause != ""
            assert link.effect != ""

    def test_deduplication(self):
        builder = CausalChainBuilder()
        links = builder.build(_make_input(), primary_service="db")
        pairs = [(l.cause, l.effect) for l in links]
        assert len(pairs) == len(set(pairs))

    def test_no_cycles(self):
        builder = CausalChainBuilder()
        links = builder.build(_make_input(), primary_service="db")
        # Verify no cycles
        assert not builder._has_cycle(links)

    def test_max_depth_respected(self):
        from agents.root_cause_agent.config import RootCauseAgentConfig, VerdictLimits
        config = RootCauseAgentConfig(limits=VerdictLimits(max_causal_chain_depth=2))
        builder = CausalChainBuilder(config)
        links = builder.build(_make_input(), primary_service="db")
        assert len(links) <= 2

    def test_empty_hypothesis_chains(self):
        inp = RootCauseAgentInput(
            hypothesis_findings=HypothesisFindings(
                top_hypothesis="Something failed",
                top_confidence=0.5,
                confidence=0.5,
            ),
        )
        builder = CausalChainBuilder()
        links = builder.build(inp, primary_service="")
        # Should still produce at least one link from top_hypothesis
        assert len(links) >= 1

    def test_relationship_types(self):
        builder = CausalChainBuilder()
        links = builder.build(_make_input(), primary_service="db")
        relationships = {l.relationship for l in links}
        assert CausalRelationship.CAUSES in relationships
