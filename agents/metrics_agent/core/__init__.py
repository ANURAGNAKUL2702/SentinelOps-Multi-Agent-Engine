"""Core deterministic analysis modules.

Exports:
    MetricAggregator: Z-score, growth rate, trend, threshold
    AnomalyDetector: Anomaly type classification + severity
    CorrelationDetector: Pearson correlation detection
"""

from agents.metrics_agent.core.anomaly_detector import AnomalyDetector
from agents.metrics_agent.core.correlation_detector import CorrelationDetector
from agents.metrics_agent.core.metric_aggregator import MetricAggregator

__all__ = [
    "MetricAggregator",
    "AnomalyDetector",
    "CorrelationDetector",
]
