"""Hypothesis Agent — THE BRAIN of the war-room system.

Receives findings from Log, Metrics, and Dependency agents,
generates root cause hypotheses using hybrid deterministic + LLM pipeline.

Architecture: 60% deterministic + 40% LLM
Pipeline: Evidence Aggregation → Pattern Matching → Hypothesis Generation → Ranking → Validation
"""

from agents.hypothesis_agent.agent import HypothesisAgent
from agents.hypothesis_agent.config import HypothesisAgentConfig
from agents.hypothesis_agent.schema import (
    HypothesisAgentInput,
    HypothesisAgentOutput,
)

__all__ = [
    "HypothesisAgent",
    "HypothesisAgentConfig",
    "HypothesisAgentInput",
    "HypothesisAgentOutput",
]
