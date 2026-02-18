"""
File: llm/synthesizer.py
Purpose: Output assembly — builds MetricsAgentOutput from classification results.
Dependencies: Schema models only
Performance: <1ms, O(1) complexity

Assembles the final output from classification results, aggregation data,
and pipeline metadata.  No LLM calls — purely deterministic assembly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from agents.metrics_agent.config import MetricsAgentConfig
from agents.metrics_agent.schema import (
    ClassificationResult,
    AggregationResult,
    MetricsAgentOutput,
    MetricsAnalysisInput,
    PipelineMetadata,
)
from agents.metrics_agent.telemetry import get_logger

logger = get_logger("metrics_agent.synthesizer")


class Synthesizer:
    """Output assembly — builds the final MetricsAgentOutput.

    Combines classification results with pipeline metadata into
    the canonical output format.

    Args:
        config: Agent configuration.

    Example::

        synthesizer = Synthesizer(MetricsAgentConfig())
        output = synthesizer.assemble_output(
            classification=result,
            aggregation=agg,
            input_data=input_data,
            pipeline_latency_ms=45.2,
            correlation_id="abc-123",
        )
    """

    def __init__(self, config: Optional[MetricsAgentConfig] = None) -> None:
        self._config = config or MetricsAgentConfig()

    def assemble_output(
        self,
        classification: ClassificationResult,
        aggregation: AggregationResult,
        input_data: MetricsAnalysisInput,
        pipeline_latency_ms: float = 0.0,
        correlation_id: str = "",
        extraction_time_ms: float = 0.0,
        validation_time_ms: float = 0.0,
    ) -> MetricsAgentOutput:
        """Assemble the final MetricsAgentOutput.

        Args:
            classification: Classification result from LLM or fallback.
            aggregation: Aggregation result from metric aggregator.
            input_data: Original input.
            pipeline_latency_ms: Total pipeline time.
            correlation_id: Request correlation ID.
            extraction_time_ms: Extraction phase time.
            validation_time_ms: Validation phase time.

        Returns:
            Complete MetricsAgentOutput.
        """
        used_llm = classification.classification_source == "llm"
        used_fallback = classification.classification_source in (
            "deterministic", "fallback"
        )
        cache_hit = classification.classification_source == "cached"

        metadata = PipelineMetadata(
            extraction_time_ms=round(extraction_time_ms, 2),
            classification_time_ms=round(
                classification.classification_latency_ms, 2
            ),
            validation_time_ms=round(validation_time_ms, 2),
            total_time_ms=round(pipeline_latency_ms, 2),
            used_llm=used_llm,
            used_fallback=used_fallback,
            cache_hit=cache_hit,
            correlation_id=correlation_id,
        )

        return MetricsAgentOutput(
            agent="metrics_agent",
            analysis_timestamp=datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            time_window=input_data.time_window,
            service=input_data.service,
            anomalous_metrics=classification.anomalous_metrics,
            correlations=classification.correlations,
            system_summary=classification.system_summary,
            confidence_score=classification.confidence_score,
            confidence_reasoning=classification.confidence_reasoning,
            correlation_id=correlation_id,
            classification_source=classification.classification_source,
            pipeline_latency_ms=round(pipeline_latency_ms, 2),
            metadata=metadata,
        )
