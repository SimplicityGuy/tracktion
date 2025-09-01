"""Metrics tracking and reporting for feedback learning loop."""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from services.file_rename_service.app.feedback.models import (
    Feedback,
    FeedbackAction,
)
from services.file_rename_service.app.feedback.storage import FeedbackStorage

logger = logging.getLogger(__name__)


class MetricsTracker:
    """Track and report metrics for feedback and learning."""

    def __init__(
        self,
        storage: FeedbackStorage,
        window_size: int = 100,
        trend_periods: int = 10,
    ):
        """Initialize metrics tracker.

        Args:
            storage: Feedback storage backend
            window_size: Window size for moving averages
            trend_periods: Number of periods for trend analysis
        """
        self.storage = storage
        self.window_size = window_size
        self.trend_periods = trend_periods

    async def calculate_metrics(
        self,
        model_version: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, Any]:
        """Calculate comprehensive metrics.

        Args:
            model_version: Filter by model version
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            Dictionary of calculated metrics
        """
        # Get feedbacks
        feedbacks = await self.storage.get_feedbacks(
            model_version=model_version,
            start_date=start_date,
            end_date=end_date,
            limit=10000,
        )

        if not feedbacks:
            return self._empty_metrics()

        # Calculate basic metrics
        basic_metrics = self._calculate_basic_metrics(feedbacks)

        # Calculate trend metrics
        trend_metrics = self._calculate_trend_metrics(feedbacks)

        # Calculate performance metrics
        perf_metrics = self._calculate_performance_metrics(feedbacks)

        # Calculate confidence analysis
        confidence_metrics = self._calculate_confidence_metrics(feedbacks)

        # Combine all metrics
        return {
            **basic_metrics,
            "trends": trend_metrics,
            "performance": perf_metrics,
            "confidence_analysis": confidence_metrics,
            "period": {
                "start": start_date.isoformat() if start_date else None,
                "end": end_date.isoformat() if end_date else None,
                "total_feedbacks": len(feedbacks),
            },
        }

    def _calculate_basic_metrics(self, feedbacks: list[Feedback]) -> dict[str, Any]:
        """Calculate basic metrics from feedbacks.

        Args:
            feedbacks: List of feedback

        Returns:
            Basic metrics dictionary
        """
        total = len(feedbacks)

        # Count by action
        approved = sum(1 for f in feedbacks if f.user_action == FeedbackAction.APPROVED)
        rejected = sum(1 for f in feedbacks if f.user_action == FeedbackAction.REJECTED)
        modified = sum(1 for f in feedbacks if f.user_action == FeedbackAction.MODIFIED)

        # Calculate rates
        approval_rate = approved / total
        rejection_rate = rejected / total
        modification_rate = modified / total

        # Model version distribution
        model_versions = {}
        for feedback in feedbacks:
            if feedback.model_version not in model_versions:
                model_versions[feedback.model_version] = 0
            model_versions[feedback.model_version] += 1

        return {
            "total_feedbacks": total,
            "approved": approved,
            "rejected": rejected,
            "modified": modified,
            "approval_rate": approval_rate,
            "rejection_rate": rejection_rate,
            "modification_rate": modification_rate,
            "model_versions": model_versions,
        }

    def _calculate_trend_metrics(self, feedbacks: list[Feedback]) -> dict[str, Any]:
        """Calculate trend metrics over time.

        Args:
            feedbacks: List of feedback sorted by time

        Returns:
            Trend metrics dictionary
        """
        if len(feedbacks) < self.window_size:
            return {"insufficient_data": True}

        # Sort by timestamp
        sorted_feedbacks = sorted(feedbacks, key=lambda f: f.timestamp)

        # Calculate moving averages
        approval_trend = []
        confidence_trend = []

        for i in range(len(sorted_feedbacks) - self.window_size + 1):
            window = sorted_feedbacks[i : i + self.window_size]

            # Approval rate in window
            window_approved = sum(1 for f in window if f.user_action == FeedbackAction.APPROVED)
            approval_trend.append(window_approved / len(window))

            # Average confidence in window
            avg_confidence = np.mean([f.confidence_score for f in window])
            confidence_trend.append(avg_confidence)

        # Calculate trend direction
        if len(approval_trend) >= 2:
            recent_trend = approval_trend[-self.trend_periods :]
            trend_direction = "improving" if recent_trend[-1] > recent_trend[0] else "declining"
            trend_strength = abs(recent_trend[-1] - recent_trend[0])
        else:
            trend_direction = "stable"
            trend_strength = 0.0

        return {
            "approval_trend": approval_trend,
            "confidence_trend": confidence_trend,
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "moving_average_window": self.window_size,
        }

    def _calculate_performance_metrics(self, feedbacks: list[Feedback]) -> dict[str, Any]:
        """Calculate performance metrics.

        Args:
            feedbacks: List of feedback

        Returns:
            Performance metrics dictionary
        """
        if not feedbacks:
            return {}

        # Processing time analysis
        processing_times = [f.processing_time_ms for f in feedbacks if f.processing_time_ms]

        if processing_times:
            perf_stats = {
                "avg_processing_time_ms": np.mean(processing_times),
                "p50_processing_time_ms": np.percentile(processing_times, 50),
                "p95_processing_time_ms": np.percentile(processing_times, 95),
                "p99_processing_time_ms": np.percentile(processing_times, 99),
                "max_processing_time_ms": max(processing_times),
                "min_processing_time_ms": min(processing_times),
            }
        else:
            perf_stats = {"no_timing_data": True}

        # Throughput calculation
        if len(feedbacks) >= 2:
            time_span = (feedbacks[-1].timestamp - feedbacks[0].timestamp).total_seconds()
            if time_span > 0:
                throughput = len(feedbacks) / (time_span / 3600)  # per hour
                perf_stats["throughput_per_hour"] = throughput

        return perf_stats

    def _calculate_confidence_metrics(self, feedbacks: list[Feedback]) -> dict[str, Any]:
        """Analyze confidence score relationships.

        Args:
            feedbacks: List of feedback

        Returns:
            Confidence analysis dictionary
        """
        # Group by action
        approved = [f for f in feedbacks if f.user_action == FeedbackAction.APPROVED]
        rejected = [f for f in feedbacks if f.user_action == FeedbackAction.REJECTED]
        modified = [f for f in feedbacks if f.user_action == FeedbackAction.MODIFIED]

        metrics = {}

        # Confidence by action
        if approved:
            metrics["approved_avg_confidence"] = np.mean([f.confidence_score for f in approved])
            metrics["approved_std_confidence"] = np.std([f.confidence_score for f in approved])

        if rejected:
            metrics["rejected_avg_confidence"] = np.mean([f.confidence_score for f in rejected])
            metrics["rejected_std_confidence"] = np.std([f.confidence_score for f in rejected])

        if modified:
            metrics["modified_avg_confidence"] = np.mean([f.confidence_score for f in modified])
            metrics["modified_std_confidence"] = np.std([f.confidence_score for f in modified])

        # Confidence distribution
        all_confidences = [f.confidence_score for f in feedbacks]
        metrics["confidence_distribution"] = {
            "0.0-0.2": sum(1 for c in all_confidences if 0.0 <= c < 0.2) / len(feedbacks),
            "0.2-0.4": sum(1 for c in all_confidences if 0.2 <= c < 0.4) / len(feedbacks),
            "0.4-0.6": sum(1 for c in all_confidences if 0.4 <= c < 0.6) / len(feedbacks),
            "0.6-0.8": sum(1 for c in all_confidences if 0.6 <= c < 0.8) / len(feedbacks),
            "0.8-1.0": sum(1 for c in all_confidences if 0.8 <= c <= 1.0) / len(feedbacks),
        }

        # Calibration analysis
        calibration = self._calculate_calibration(feedbacks)
        metrics["calibration"] = calibration

        return metrics

    def _calculate_calibration(self, feedbacks: list[Feedback]) -> dict[str, float]:
        """Calculate model calibration metrics.

        Args:
            feedbacks: List of feedback

        Returns:
            Calibration metrics
        """
        # Group feedbacks by confidence buckets
        buckets: dict[float, list[Feedback]] = {
            0.1: [],
            0.3: [],
            0.5: [],
            0.7: [],
            0.9: [],
        }

        for feedback in feedbacks:
            # Find appropriate bucket
            for threshold in sorted(buckets.keys()):
                if feedback.confidence_score <= threshold:
                    buckets[threshold].append(feedback)
                    break
            else:
                buckets[0.9].append(feedback)

        calibration_error = 0.0
        bucket_count = 0

        for threshold, bucket_feedbacks in buckets.items():
            if bucket_feedbacks:
                # Expected approval rate based on confidence
                expected = threshold

                # Actual approval rate
                actual = sum(1 for f in bucket_feedbacks if f.user_action == FeedbackAction.APPROVED) / len(
                    bucket_feedbacks
                )

                # Calibration error for this bucket
                calibration_error += abs(expected - actual)
                bucket_count += 1

        # Average calibration error
        avg_calibration_error = calibration_error / bucket_count if bucket_count > 0 else 0.0

        return {
            "average_calibration_error": avg_calibration_error,
            "is_well_calibrated": avg_calibration_error < 0.1,
        }

    async def get_improvement_metrics(
        self,
        baseline_version: str,
        current_version: str,
    ) -> dict[str, Any]:
        """Calculate improvement metrics between model versions.

        Args:
            baseline_version: Baseline model version
            current_version: Current model version

        Returns:
            Improvement metrics
        """
        # Get feedbacks for both versions
        baseline_feedbacks = await self.storage.get_feedbacks(model_version=baseline_version, limit=5000)
        current_feedbacks = await self.storage.get_feedbacks(model_version=current_version, limit=5000)

        if not baseline_feedbacks or not current_feedbacks:
            return {"error": "Insufficient data for comparison"}

        # Calculate metrics for both
        baseline_metrics = self._calculate_basic_metrics(baseline_feedbacks)
        current_metrics = self._calculate_basic_metrics(current_feedbacks)

        # Calculate improvements
        approval_improvement = current_metrics["approval_rate"] - baseline_metrics["approval_rate"]
        rejection_improvement = baseline_metrics["rejection_rate"] - current_metrics["rejection_rate"]

        # Confidence improvements
        baseline_conf = np.mean([f.confidence_score for f in baseline_feedbacks])
        current_conf = np.mean([f.confidence_score for f in current_feedbacks])
        confidence_improvement = current_conf - baseline_conf

        return {
            "baseline_version": baseline_version,
            "current_version": current_version,
            "approval_rate_improvement": approval_improvement,
            "rejection_rate_improvement": rejection_improvement,
            "confidence_improvement": confidence_improvement,
            "relative_improvement": (
                approval_improvement / baseline_metrics["approval_rate"] if baseline_metrics["approval_rate"] > 0 else 0
            ),
            "baseline_metrics": baseline_metrics,
            "current_metrics": current_metrics,
        }

    async def generate_dashboard_data(self) -> dict[str, Any]:
        """Generate data for metrics dashboard.

        Returns:
            Dashboard data dictionary
        """
        # Get recent metrics
        now = datetime.now(UTC)
        last_24h = now - timedelta(hours=24)
        last_7d = now - timedelta(days=7)

        # Calculate metrics for different periods
        metrics_24h = await self.calculate_metrics(start_date=last_24h, end_date=now)
        metrics_7d = await self.calculate_metrics(start_date=last_7d, end_date=now)

        # Get learning metrics
        learning_metrics = await self.storage.get_learning_metrics()

        # Get active experiments (would need experiment manager instance)
        # For now, return placeholder
        active_experiments: list[dict[str, Any]] = []

        return {
            "timestamp": now.isoformat(),
            "metrics_24h": metrics_24h,
            "metrics_7d": metrics_7d,
            "learning_status": {
                "model_version": learning_metrics.model_version if learning_metrics else None,
                "total_feedback": learning_metrics.total_feedback if learning_metrics else 0,
                "last_retrained": (
                    learning_metrics.last_retrained.isoformat()
                    if learning_metrics and learning_metrics.last_retrained
                    else None
                ),
                "next_retrain_at": (
                    learning_metrics.next_retrain_at.isoformat()
                    if learning_metrics and learning_metrics.next_retrain_at
                    else None
                ),
            },
            "active_experiments": active_experiments,
            "health_status": await self._calculate_health_status(),
        }

    async def _calculate_health_status(self) -> dict[str, Any]:
        """Calculate system health status.

        Returns:
            Health status dictionary
        """
        # Get recent feedbacks
        recent = await self.storage.get_feedbacks(limit=100)

        if not recent:
            return {"status": "no_data", "healthy": False}

        # Check processing times
        recent_times = [f.processing_time_ms for f in recent if f.processing_time_ms]
        if recent_times:
            avg_time = np.mean(recent_times)
            p95_time = np.percentile(recent_times, 95)
        else:
            avg_time = 0
            p95_time = 0

        # Health checks
        checks = {
            "processing_time_ok": p95_time < 500,  # <500ms at p95
            "throughput_ok": len(recent) > 10,  # At least some activity
            "error_rate_ok": True,  # Would check error logs
        }

        all_healthy = all(checks.values())

        return {
            "status": "healthy" if all_healthy else "degraded",
            "healthy": all_healthy,
            "checks": checks,
            "metrics": {
                "avg_processing_time_ms": avg_time,
                "p95_processing_time_ms": p95_time,
                "recent_feedback_count": len(recent),
            },
        }

    def _empty_metrics(self) -> dict[str, Any]:
        """Return empty metrics structure.

        Returns:
            Empty metrics dictionary
        """
        return {
            "total_feedbacks": 0,
            "approved": 0,
            "rejected": 0,
            "modified": 0,
            "approval_rate": 0.0,
            "rejection_rate": 0.0,
            "modification_rate": 0.0,
            "model_versions": {},
            "trends": {"insufficient_data": True},
            "performance": {"no_timing_data": True},
            "confidence_analysis": {},
        }
