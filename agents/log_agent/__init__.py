"""
Log Agent — Hybrid architecture (80% deterministic + 20% LLM).

Public API::

    from agents.log_agent import LogAgent, LogAgentConfig, LogAnalysisInput

    agent = LogAgent(LogAgentConfig())
    output = agent.analyze(LogAnalysisInput(...))

Modules:
    agent       — Hybrid orchestrator (3-phase pipeline)
    config      — Feature flags, thresholds, environment config
    schema      — Pydantic schemas for all data contracts
    fallback    — Deterministic rule-based classification
    validator   — 23-check output validation
    telemetry   — Structured logging + Prometheus metrics
    core/       — Signal extraction + anomaly detection
    llm/        — LLM classification + synthesis
"""

from agents.log_agent.agent import LogAgent
from agents.log_agent.config import LogAgentConfig
from agents.log_agent.schema import LogAgentOutput, LogAnalysisInput

__all__ = [
    "LogAgent",
    "LogAgentConfig",
    "LogAgentOutput",
    "LogAnalysisInput",
]
