"""
File: llm/synthesizer.py
Purpose: Output assembly — builds DependencyAgentOutput from analysis results.
Dependencies: Schema models only
Performance: <1ms, O(1) complexity

Assembles the final output from classification results, graph data,
impact analysis, and pipeline metadata.  No LLM calls — purely
deterministic assembly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from agents.dependency_agent.config import DependencyAgentConfig
from agents.dependency_agent.schema import (
    BottleneckDetectionResult,
    ClassificationResult,
    DependencyAgentOutput,
    DependencyAnalysisInput,
    GraphBuildResult,
    ImpactAnalysisResult,
    PipelineMetadata,
    TraceAnalysisResult,
)
from agents.dependency_agent.telemetry import get_logger

logger = get_logger("dependency_agent.synthesizer")


class Synthesizer:
    """Output assembly — builds the final DependencyAgentOutput.

    Combines all analysis results with pipeline metadata into
    the canonical output format.

    Args:
        config: Agent configuration.

    Example::

        synthesizer = Synthesizer(DependencyAgentConfig())
        output = synthesizer.assemble_output(
            classification=result,
            input_data=input_data,
            pipeline_latency_ms=45.2,
        )
    """

    def __init__(
        self, config: Optional[DependencyAgentConfig] = None
    ) -> None:
        self._config = config or DependencyAgentConfig()

    def assemble_output(
        self,
        classification: ClassificationResult,
        input_data: DependencyAnalysisInput,
        pipeline_latency_ms: float = 0.0,
        correlation_id: str = "",
        graph_build_time_ms: float = 0.0,
        trace_analysis_time_ms: float = 0.0,
        impact_calculation_time_ms: float = 0.0,
        classification_time_ms: float = 0.0,
        validation_time_ms: float = 0.0,
    ) -> DependencyAgentOutput:
        """Assemble the final DependencyAgentOutput.

        Args:
            classification: Classification result from LLM or fallback.
            input_data: Original input.
            pipeline_latency_ms: Total pipeline time.
            correlation_id: Request correlation ID.
            graph_build_time_ms: Graph build phase time.
            trace_analysis_time_ms: Trace analysis phase time.
            impact_calculation_time_ms: Impact calculation phase time.
            classification_time_ms: Classification phase time.
            validation_time_ms: Validation phase time.

        Returns:
            Complete DependencyAgentOutput.
        """
        used_llm = classification.classification_source == "llm"
        used_fallback = classification.classification_source in (
            "deterministic", "fallback"
        )
        cache_hit = classification.classification_source == "cached"

        metadata = PipelineMetadata(
            graph_build_time_ms=round(graph_build_time_ms, 2),
            trace_analysis_time_ms=round(trace_analysis_time_ms, 2),
            impact_calculation_time_ms=round(
                impact_calculation_time_ms, 2
            ),
            classification_time_ms=round(classification_time_ms, 2),
            validation_time_ms=round(validation_time_ms, 2),
            total_time_ms=round(pipeline_latency_ms, 2),
            used_llm=used_llm,
            used_fallback=used_fallback,
            cache_hit=cache_hit,
            correlation_id=correlation_id,
        )

        return DependencyAgentOutput(
            agent="dependency_agent",
            analysis_timestamp=datetime.now(
                timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            time_window=input_data.time_window,
            failed_service=classification.failed_service,
            dependency_analysis=classification.dependency_analysis,
            impact_analysis=classification.impact_analysis,
            critical_path=classification.critical_path,
            bottlenecks=classification.bottlenecks,
            cascading_failure_risk=classification.cascading_failure_risk,
            single_points_of_failure=classification.single_points_of_failure,
            confidence_score=classification.confidence_score,
            confidence_reasoning=classification.confidence_reasoning,
            correlation_id=correlation_id,
            classification_source=classification.classification_source,
            pipeline_latency_ms=round(pipeline_latency_ms, 2),
            metadata=metadata,
        )
