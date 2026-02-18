"""
File: fallback.py
Purpose: Deterministic validation without LLM — <50ms budget.
Dependencies: All 7 core algorithms.
Performance: <50ms total, no network calls.

Runs the entire validation pipeline using deterministic algorithms
only, generating rule-based recommendations instead of LLM analysis.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple

from agents.validation_agent.config import ValidationAgentConfig
from agents.validation_agent.core.accuracy_calculator import (
    calculate_accuracy,
)
from agents.validation_agent.core.confidence_calibrator import (
    calculate_calibration,
)
from agents.validation_agent.core.confusion_matrix import (
    build_confusion_matrix,
)
from agents.validation_agent.core.evidence_validator import (
    validate_evidence,
)
from agents.validation_agent.core.hallucination_detector import (
    detect_hallucinations,
)
from agents.validation_agent.core.precision_recall import (
    calculate_precision_recall,
)
from agents.validation_agent.core.timeline_validator import (
    validate_timeline,
)
from agents.validation_agent.schema import (
    Discrepancy,
    DiscrepancySeverity,
    DiscrepancyType,
    GroundTruth,
    Hallucination,
    RootCauseVerdict,
    ValidationMetadata,
    ValidationReport,
)
from agents.validation_agent.telemetry import TelemetryCollector, get_logger

logger = get_logger("validation_agent.fallback")


class DeterministicFallback:
    """Deterministic validation pipeline without LLM.

    10-phase pipeline:
      1. Accuracy calculation
      2. Precision/recall/F1
      3. Confidence calibration
      4. Confusion matrix
      5. Evidence validation
      6. Timeline validation
      7. Hallucination detection
      8. Affected services accuracy
      9. Discrepancy identification
      10. Recommendation generation + assembly

    Args:
        config: Agent configuration.
        telemetry: Telemetry collector.
    """

    def __init__(
        self,
        config: ValidationAgentConfig,
        telemetry: TelemetryCollector,
    ) -> None:
        self._config = config
        self._telemetry = telemetry

    def validate(
        self,
        verdict: RootCauseVerdict,
        ground_truth: GroundTruth,
        correlation_id: str = "",
        history: List[Dict[str, Any]] | None = None,
    ) -> ValidationReport:
        """Run the deterministic validation pipeline.

        Args:
            verdict: Root cause verdict to validate.
            ground_truth: Simulation ground truth.
            correlation_id: Request correlation ID.
            history: Historical prediction data for calibration.

        Returns:
            Complete ValidationReport.
        """
        pipeline_start = time.perf_counter()
        start_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Phase 1: Accuracy calculation
        t0 = time.perf_counter()
        verdict_correct, accuracy_score = calculate_accuracy(
            verdict.root_cause,
            ground_truth.actual_root_cause,
            fuzzy_threshold=self._config.accuracy.fuzzy_threshold,
        )
        accuracy_ms = (time.perf_counter() - t0) * 1000
        self._telemetry.measure_value("accuracy_calculation", accuracy_ms)

        # Phase 2: Precision/recall/F1
        t0 = time.perf_counter()
        precision, recall, f1 = calculate_precision_recall(
            verdict_correct,
            verdict.confidence,
            threshold=self._config.precision_recall.confidence_threshold,
        )
        pr_ms = (time.perf_counter() - t0) * 1000
        self._telemetry.measure_value("precision_recall", pr_ms)

        # Phase 3: Confidence calibration
        t0 = time.perf_counter()
        cal_history = self._parse_history(history)
        calibration_error, calibration_curve = calculate_calibration(
            verdict.confidence,
            accuracy_score,
            cal_history,
            num_bins=self._config.calibration.num_bins,
        )
        cal_ms = (time.perf_counter() - t0) * 1000
        self._telemetry.measure_value("calibration", cal_ms)

        # Phase 4: Confusion matrix
        t0 = time.perf_counter()
        confusion = build_confusion_matrix(
            verdict.root_cause,
            ground_truth.actual_root_cause,
            self._config.all_failure_types,
        )
        cm_ms = (time.perf_counter() - t0) * 1000
        self._telemetry.measure_value("confusion_matrix", cm_ms)

        # Phase 5: Evidence validation
        t0 = time.perf_counter()
        evidence_accuracy, evidence_hallucinations = validate_evidence(
            verdict.evidence_trail,
            ground_truth,
            self._config.evidence.timestamp_tolerance_seconds,
        )
        ev_ms = (time.perf_counter() - t0) * 1000
        self._telemetry.measure_value("evidence_validation", ev_ms)

        # Phase 6: Timeline validation
        t0 = time.perf_counter()
        timeline_accuracy = validate_timeline(
            verdict.timeline,
            ground_truth.failure_propagation_chain,
            self._config.timeline.timing_tolerance_seconds,
            self._config.timeline.order_weight,
            self._config.timeline.timing_weight,
        )
        tl_ms = (time.perf_counter() - t0) * 1000
        self._telemetry.measure_value("timeline_validation", tl_ms)

        # Phase 7: Hallucination detection
        t0 = time.perf_counter()
        hallucinations = detect_hallucinations(
            verdict,
            ground_truth,
            self._config.hallucination.check_services,
            self._config.hallucination.check_dependencies,
            self._config.hallucination.check_metrics,
        )
        # Merge evidence hallucinations (dedup)
        all_hallucinations = self._merge_hallucinations(
            hallucinations, evidence_hallucinations
        )
        hall_ms = (time.perf_counter() - t0) * 1000
        self._telemetry.measure_value("hallucination_detection", hall_ms)

        if all_hallucinations:
            self._telemetry.hallucinations_detected.inc(
                len(all_hallucinations)
            )

        # Phase 8: Affected services accuracy (Jaccard similarity)
        affected_services_accuracy = self._jaccard_similarity(
            verdict.affected_services,
            ground_truth.affected_services_ground_truth,
        )

        # Phase 9: Discrepancy identification
        discrepancies = self._identify_discrepancies(
            verdict,
            ground_truth,
            verdict_correct,
            accuracy_score,
            all_hallucinations,
        )

        # Phase 10: Recommendations + assembly
        recommendations = self._generate_recommendations(
            accuracy_score,
            evidence_accuracy,
            timeline_accuracy,
            calibration_error,
            precision,
            recall,
            all_hallucinations,
        )

        end_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        pipeline_ms = (time.perf_counter() - pipeline_start) * 1000

        metadata = ValidationMetadata(
            correlation_id=correlation_id,
            validation_start=start_ts,
            validation_end=end_ts,
            accuracy_calculation_ms=round(accuracy_ms, 3),
            precision_recall_ms=round(pr_ms, 3),
            evidence_validation_ms=round(ev_ms, 3),
            timeline_validation_ms=round(tl_ms, 3),
            hallucination_detection_ms=round(hall_ms, 3),
            confusion_matrix_ms=round(cm_ms, 3),
            calibration_ms=round(cal_ms, 3),
            total_pipeline_ms=round(pipeline_ms, 3),
            used_llm=False,
            used_fallback=True,
            cache_hit=False,
        )

        return ValidationReport(
            verdict_correct=verdict_correct,
            accuracy_score=accuracy_score,
            precision=precision,
            recall=recall,
            f1_score=f1,
            confidence_calibration_error=calibration_error,
            evidence_accuracy=evidence_accuracy,
            timeline_accuracy=timeline_accuracy,
            affected_services_accuracy=affected_services_accuracy,
            discrepancies=discrepancies,
            hallucinations=all_hallucinations,
            recommendations=recommendations,
            confusion_matrix=confusion,
            calibration_curve=calibration_curve,
            metadata=metadata,
            correlation_id=correlation_id,
            classification_source="deterministic",
            pipeline_latency_ms=round(pipeline_ms, 3),
        )

    def _parse_history(
        self, history: List[Dict[str, Any]] | None
    ) -> List[Tuple[float, float]]:
        """Parse history dicts into (confidence, accuracy) tuples.

        Args:
            history: List of dicts with 'confidence' and 'accuracy'.

        Returns:
            List of (confidence, accuracy) tuples.
        """
        if not history:
            return []
        result: List[Tuple[float, float]] = []
        for item in history:
            conf = item.get("confidence", 0.0)
            acc = item.get("accuracy", 0.0)
            if isinstance(conf, (int, float)) and isinstance(
                acc, (int, float)
            ):
                result.append((float(conf), float(acc)))
        return result

    def _jaccard_similarity(
        self,
        set_a: List[str],
        set_b: List[str],
    ) -> float:
        """Calculate Jaccard similarity between two service lists.

        Args:
            set_a: First service list.
            set_b: Second service list.

        Returns:
            Jaccard similarity 0.0–1.0.
        """
        a = set(s.lower() for s in set_a)
        b = set(s.lower() for s in set_b)
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return round(intersection / union, 4) if union > 0 else 0.0

    def _merge_hallucinations(
        self,
        primary: List[Hallucination],
        secondary: List[Hallucination],
    ) -> List[Hallucination]:
        """Merge and deduplicate hallucination lists.

        Args:
            primary: Primary hallucination list.
            secondary: Secondary hallucination list.

        Returns:
            Deduplicated list.
        """
        seen = set()
        result: List[Hallucination] = []
        for h in primary + secondary:
            key = (h.hallucination_type, h.fabricated_value)
            if key not in seen:
                seen.add(key)
                result.append(h)
        return result

    def _identify_discrepancies(
        self,
        verdict: RootCauseVerdict,
        ground_truth: GroundTruth,
        verdict_correct: bool,
        accuracy_score: float,
        hallucinations: List[Hallucination],
    ) -> List[Discrepancy]:
        """Identify specific discrepancies between verdict and ground truth.

        Args:
            verdict: The root cause verdict.
            ground_truth: Simulation ground truth.
            verdict_correct: Whether prediction is correct.
            accuracy_score: Fuzzy match score.
            hallucinations: Detected hallucinations.

        Returns:
            List of Discrepancy objects.
        """
        discrepancies: List[Discrepancy] = []

        # Root cause mismatch
        if not verdict_correct:
            discrepancies.append(Discrepancy(
                discrepancy_type=DiscrepancyType.ROOT_CAUSE_MISMATCH,
                description=(
                    f"Predicted '{verdict.root_cause}' but actual is "
                    f"'{ground_truth.actual_root_cause}'"
                ),
                expected=ground_truth.actual_root_cause,
                actual=verdict.root_cause,
                severity=DiscrepancySeverity.CRITICAL,
            ))

        # Service mismatches
        predicted_services = set(
            s.lower() for s in verdict.affected_services
        )
        actual_services = set(
            s.lower() for s in ground_truth.affected_services_ground_truth
        )
        missing_services = actual_services - predicted_services
        extra_services = predicted_services - actual_services

        if missing_services:
            discrepancies.append(Discrepancy(
                discrepancy_type=DiscrepancyType.SERVICE_MISMATCH,
                description=(
                    f"Missing services: {', '.join(sorted(missing_services))}"
                ),
                expected=", ".join(sorted(actual_services)),
                actual=", ".join(sorted(predicted_services)),
                severity=DiscrepancySeverity.HIGH,
            ))

        if extra_services:
            discrepancies.append(Discrepancy(
                discrepancy_type=DiscrepancyType.SERVICE_MISMATCH,
                description=(
                    f"Extra services not in ground truth: "
                    f"{', '.join(sorted(extra_services))}"
                ),
                expected=", ".join(sorted(actual_services)),
                actual=", ".join(sorted(predicted_services)),
                severity=DiscrepancySeverity.MEDIUM,
            ))

        # Confidence miscalibration
        if verdict.confidence >= 0.9 and not verdict_correct:
            discrepancies.append(Discrepancy(
                discrepancy_type=DiscrepancyType.CONFIDENCE_MISCALIBRATION,
                description=(
                    f"High confidence ({verdict.confidence:.2f}) "
                    f"but incorrect verdict"
                ),
                expected="Low confidence for incorrect prediction",
                actual=f"confidence={verdict.confidence:.2f}",
                severity=DiscrepancySeverity.HIGH,
            ))
        elif verdict.confidence <= 0.3 and verdict_correct:
            discrepancies.append(Discrepancy(
                discrepancy_type=DiscrepancyType.CONFIDENCE_MISCALIBRATION,
                description=(
                    f"Low confidence ({verdict.confidence:.2f}) "
                    f"but correct verdict"
                ),
                expected="Higher confidence for correct prediction",
                actual=f"confidence={verdict.confidence:.2f}",
                severity=DiscrepancySeverity.LOW,
            ))

        # Hallucination-based discrepancies
        for h in hallucinations:
            discrepancies.append(Discrepancy(
                discrepancy_type=DiscrepancyType.EVIDENCE_FABRICATION,
                description=h.description,
                expected="Element exists in ground truth",
                actual=h.fabricated_value,
                severity=DiscrepancySeverity.HIGH,
            ))

        return discrepancies

    def _generate_recommendations(
        self,
        accuracy: float,
        evidence_accuracy: float,
        timeline_accuracy: float,
        calibration_error: float,
        precision: float,
        recall: float,
        hallucinations: List[Hallucination],
    ) -> List[str]:
        """Generate rule-based recommendations.

        Args:
            accuracy: Overall accuracy score.
            evidence_accuracy: Evidence accuracy score.
            timeline_accuracy: Timeline accuracy score.
            calibration_error: Calibration error.
            precision: Precision score.
            recall: Recall score.
            hallucinations: Detected hallucinations.

        Returns:
            List of recommendation strings.
        """
        thresholds = self._config.recommendations
        recs: List[str] = []

        if accuracy < thresholds.critical_accuracy:
            recs.append(
                "Critical: Root cause detection failed. Review "
                "Log/Metrics/Dependency agent outputs for missing signals."
            )

        if evidence_accuracy < thresholds.evidence_accuracy:
            recs.append(
                "Evidence quality issues detected. Check evidence "
                "aggregation in Hypothesis Agent."
            )

        if timeline_accuracy < thresholds.timeline_accuracy:
            recs.append(
                "Timeline reconstruction inaccurate. Verify "
                "timestamp handling across agents."
            )

        if hallucinations:
            recs.append(
                "Hallucinations detected: agents referenced "
                "non-existent services/metrics. Strengthen "
                "validation in upstream agents."
            )

        if calibration_error > thresholds.calibration_error:
            recs.append(
                "Confidence scores poorly calibrated. Consider "
                "recalibrating confidence calculations."
            )

        if precision < 0.5 and recall >= 0.7:
            recs.append(
                "Over-predicting positive cases. Increase "
                "confidence thresholds."
            )

        if recall < 0.5 and precision >= 0.7:
            recs.append(
                "Missing true positives. Agents may be "
                "too conservative."
            )

        # Always provide at least one recommendation if accuracy < 0.9
        if not recs and accuracy < thresholds.accuracy_for_recs:
            recs.append(
                "Consider improving cross-agent correlation "
                "to strengthen root cause identification."
            )

        return recs
