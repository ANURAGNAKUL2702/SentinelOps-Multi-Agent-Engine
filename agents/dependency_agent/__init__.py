"""Dependency Agent — analyzes service dependency graphs and distributed traces."""

from agents.dependency_agent.agent import DependencyAgent
from agents.dependency_agent.config import DependencyAgentConfig
from agents.dependency_agent.schema import (
    DependencyAgentOutput,
    DependencyAnalysisInput,
)

__all__ = [
    "DependencyAgent",
    "DependencyAgentConfig",
    "DependencyAgentOutput",
    "DependencyAnalysisInput",
]
