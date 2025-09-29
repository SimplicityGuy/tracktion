"""Alert management system for structure monitoring."""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, cast

from redis.asyncio import Redis
from services.notification_service.src.channels.discord import DiscordNotificationService
from services.notification_service.src.core.base import (
    AlertType,
    NotificationMessage,
)
from services.tracklist_service.src.monitoring.structure_monitor import ChangeReport, StructureMonitor

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


def severity_to_alert_type(severity: AlertSeverity) -> AlertType:
    """Map AlertSeverity to AlertType."""
    mapping = {
        AlertSeverity.INFO: AlertType.GENERAL,
        AlertSeverity.WARNING: AlertType.MONITORING,
        AlertSeverity.ERROR: AlertType.ERROR,
        AlertSeverity.CRITICAL: AlertType.CRITICAL,
    }
    return mapping.get(severity, AlertType.GENERAL)


@dataclass
class Alert:
    """Alert data structure."""

    severity: AlertSeverity
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    page_type: str | None = None
    change_report: ChangeReport | None = None
    channels: list[str] = field(default_factory=list)  # Legacy field - now just strings
    metadata: dict[str, Any] = field(default_factory=dict)
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "page_type": self.page_type,
            "change_report": (self.change_report.to_dict() if self.change_report else None),
            "channels": self.channels,
            "metadata": self.metadata,
            "resolved": self.resolved,
        }


@dataclass
class HealthStatus:
    """Health status for the scraper."""

    healthy: bool
    success_rate: float
    last_check: datetime
    failed_extractions: list[str] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def requires_alert(self) -> bool:
        """Check if health status requires an alert."""
        return not self.healthy or self.success_rate < 0.95 or len(self.anomalies) > 0


class AlertManager:
    """Manages alerts and notifications for the scraping system."""

    def __init__(
        self,
        redis_client: Redis | None = None,
        notification_service: DiscordNotificationService | None = None,
    ):
        """Initialize alert manager.

        Args:
            redis_client: Redis client for storing alerts
            notification_service: Discord notification service for sending alerts
        """
        self.redis_client = redis_client
        self.notification_service = notification_service or DiscordNotificationService(redis_client)
        self._alert_history: list[Alert] = []
        self._health_checks: dict[str, HealthStatus] = {}
        self._anomaly_detectors: dict[str, Callable[..., Any]] = {}
        self._monitoring_tasks: list[asyncio.Task[Any]] = []

    async def check_parser_health(self, page_type: str | None = None) -> HealthStatus:
        """Check health status of parser(s).

        Args:
            page_type: Specific page type to check, or None for all

        Returns:
            HealthStatus object
        """
        if self.redis_client:
            # Get recent extraction metrics from Redis
            key = f"metrics:extraction:{page_type}" if page_type else "metrics:extraction:*"

            # Get success/failure counts
            success_key = f"{key}:success"
            failure_key = f"{key}:failure"

            try:
                if page_type:
                    success_count = int(await self.redis_client.get(success_key) or 0)
                    failure_count = int(await self.redis_client.get(failure_key) or 0)
                else:
                    # Aggregate across all page types
                    success_keys = await self.redis_client.keys("metrics:extraction:*:success")
                    failure_keys = await self.redis_client.keys("metrics:extraction:*:failure")

                    success_values = [int(await self.redis_client.get(k) or 0) for k in success_keys]
                    failure_values = [int(await self.redis_client.get(k) or 0) for k in failure_keys]
                    success_count = sum(success_values)
                    failure_count = sum(failure_values)

                total = success_count + failure_count
                success_rate = success_count / total if total > 0 else 1.0

                # Check for anomalies
                anomalies = await self._detect_anomalies(page_type)

                # Get failed extraction details
                failed_key = f"failed:extractions:{page_type or '*'}"
                failed_extractions = []
                if page_type:
                    failed_data = await cast("Awaitable[list[Any]]", self.redis_client.lrange(failed_key, 0, 9))
                    failed_extractions = [json.loads(f) for f in failed_data if f]

                status = HealthStatus(
                    healthy=success_rate >= 0.95 and len(anomalies) == 0,
                    success_rate=success_rate,
                    last_check=datetime.now(UTC),
                    failed_extractions=failed_extractions,
                    anomalies=anomalies,
                    metrics={
                        "success_count": float(success_count),
                        "failure_count": float(failure_count),
                        "total_count": float(total),
                    },
                )
            except Exception as e:
                logger.error(f"Error checking parser health: {e}")
                status = HealthStatus(
                    healthy=False,
                    success_rate=0.0,
                    last_check=datetime.now(UTC),
                    anomalies=[f"Health check failed: {e!s}"],
                )
        else:
            # No Redis, return default healthy status
            status = HealthStatus(healthy=True, success_rate=1.0, last_check=datetime.now(UTC))

        # Cache health status
        if page_type:
            self._health_checks[page_type] = status

        return status

    async def send_alert(self, severity: str, message: str, channels: list[str] | None = None) -> None:
        """Send alert using Discord notification service.

        Args:
            severity: Alert severity level
            message: Alert message
            channels: Legacy parameter - ignored (Discord channels determined by severity)
        """
        severity_enum = AlertSeverity(severity)
        alert_type = severity_to_alert_type(severity_enum)

        notification = NotificationMessage(
            alert_type=alert_type,
            title=f"{severity_enum.value.title()} Alert",
            message=message,
        )

        result = await self.notification_service.send(notification)

        # Store in local history for compatibility
        alert = Alert(
            severity=severity_enum,
            message=message,
            channels=[],  # Legacy channels not used with Discord
        )
        self._alert_history.append(alert)
        if len(self._alert_history) > 1000:
            self._alert_history = self._alert_history[-1000:]

        if not result.success:
            logger.error(f"Failed to send Discord notification: {result.error}")

    async def send_change_alert(self, change_report: ChangeReport) -> None:
        """Send alert for structural changes.

        Args:
            change_report: Change report from structure monitor
        """
        # Determine severity based on change report
        if change_report.has_breaking_changes:
            severity = AlertSeverity.CRITICAL
        elif change_report.severity == "high":
            severity = AlertSeverity.ERROR
        elif change_report.severity == "medium":
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO

        alert_type = severity_to_alert_type(severity)

        # Build Discord message fields
        fields: list[dict[str, Any]] = [
            {"name": "Page Type", "value": change_report.page_type, "inline": True},
            {"name": "Changes", "value": str(len(change_report.changes)), "inline": True},
            {"name": "Severity", "value": change_report.severity, "inline": True},
            {"name": "Match %", "value": f"{change_report.fingerprint_match_percentage:.1f}%", "inline": True},
            {
                "name": "Review Required",
                "value": "Yes" if change_report.requires_manual_review else "No",
                "inline": True,
            },
        ]

        notification = NotificationMessage(
            alert_type=alert_type,
            title="🔄 Structural Changes Detected",
            message=f"Structural changes detected for {change_report.page_type}",
            fields=fields,
        )

        result = await self.notification_service.send(notification)

        # Store in local history for compatibility
        alert = Alert(
            severity=severity,
            message=f"Structural changes detected for {change_report.page_type}",
            details={
                "page_type": change_report.page_type,
                "change_count": len(change_report.changes),
                "severity": change_report.severity,
                "match_percentage": change_report.fingerprint_match_percentage,
                "requires_review": change_report.requires_manual_review,
            },
            page_type=change_report.page_type,
            change_report=change_report,
            channels=[],
        )
        self._alert_history.append(alert)

        if not result.success:
            logger.error(f"Failed to send change alert: {result.error}")

    async def send_health_alert(self, health_status: HealthStatus, page_type: str | None = None) -> None:
        """Send alert for health status issues.

        Args:
            health_status: Health status to alert on
            page_type: Page type for the health status
        """
        if not health_status.requires_alert:
            return

        # Determine severity
        if health_status.success_rate < 0.5:
            severity = AlertSeverity.CRITICAL
        elif health_status.success_rate < 0.8:
            severity = AlertSeverity.ERROR
        elif health_status.success_rate < 0.95 or health_status.anomalies:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO

        alert_type = severity_to_alert_type(severity)

        # Build Discord message fields
        fields: list[dict[str, Any]] = [
            {"name": "Success Rate", "value": f"{health_status.success_rate:.1%}", "inline": True},
            {"name": "Anomalies", "value": str(len(health_status.anomalies)), "inline": True},
            {"name": "Failed Extractions", "value": str(len(health_status.failed_extractions)), "inline": True},
        ]

        # Add metrics
        for key, value in health_status.metrics.items():
            fields.append({"name": key.title(), "value": str(value), "inline": True})

        # Add anomalies if present (limited)
        if health_status.anomalies:
            anomaly_text = "\n".join(health_status.anomalies[:3])
            if len(health_status.anomalies) > 3:
                anomaly_text += f"\n... and {len(health_status.anomalies) - 3} more"
            fields.append({"name": "Anomalies", "value": anomaly_text, "inline": False})

        notification = NotificationMessage(
            alert_type=alert_type,
            title=f"🏥 Health Alert{f': {page_type}' if page_type else ''}",
            message=f"Parser health degraded{f' for {page_type}' if page_type else ''}",
            fields=fields,
        )

        result = await self.notification_service.send(notification)

        # Store in local history for compatibility
        alert = Alert(
            severity=severity,
            message=f"Parser health degraded{f' for {page_type}' if page_type else ''}",
            details={
                "success_rate": health_status.success_rate,
                "anomaly_count": len(health_status.anomalies),
                "failed_count": len(health_status.failed_extractions),
                "metrics": health_status.metrics,
            },
            page_type=page_type,
            channels=[],
            metadata={
                "anomalies": health_status.anomalies,
                "failed_extractions": health_status.failed_extractions[:5],
            },
        )
        self._alert_history.append(alert)

        if not result.success:
            logger.error(f"Failed to send health alert: {result.error}")

    # Legacy methods removed - using DiscordNotificationService instead

    async def _detect_anomalies(self, page_type: str | None = None) -> list[str]:
        """Detect anomalies in extraction patterns.

        Args:
            page_type: Page type to check

        Returns:
            List of detected anomalies
        """
        anomalies = []

        if self.redis_client:
            # Check for sudden drop in extraction rate
            rate_key = f"metrics:rate:{page_type or '*'}"
            current_rate = float(await self.redis_client.get(f"{rate_key}:current") or 0)
            avg_rate = float(await self.redis_client.get(f"{rate_key}:average") or 0)

            if avg_rate > 0 and current_rate < avg_rate * 0.5:
                anomalies.append(f"Extraction rate dropped by {(1 - current_rate / avg_rate) * 100:.1f}%")

            # Check for repeated failures
            failure_pattern_key = f"failures:pattern:{page_type or '*'}"
            failure_pattern = await self.redis_client.get(failure_pattern_key)
            if failure_pattern:
                try:
                    pattern_data = json.loads(failure_pattern)
                    if isinstance(pattern_data, dict) and pattern_data.get("consecutive_failures", 0) > 5:
                        anomalies.append(f"Consecutive failures: {pattern_data['consecutive_failures']}")
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Invalid failure pattern data: {failure_pattern}")

        # Run custom anomaly detectors
        for name, detector in self._anomaly_detectors.items():
            try:
                result = await detector(page_type)
                if result:
                    anomalies.append(f"{name}: {result}")
            except Exception as e:
                logger.error(f"Anomaly detector {name} failed: {e}")

        return anomalies

    def register_anomaly_detector(self, name: str, detector: Callable[..., Any]) -> None:
        """Register a custom anomaly detector.

        Args:
            name: Name of the detector
            detector: Async callable that returns anomaly string or None
        """
        self._anomaly_detectors[name] = detector

    async def start_monitoring(self, structure_monitor: StructureMonitor, interval: int = 3600) -> None:
        """Start periodic monitoring tasks.

        Args:
            structure_monitor: Structure monitor instance
            interval: Check interval in seconds (default 1 hour)
        """

        async def monitor_loop() -> None:
            while True:
                try:
                    # Check health for each page type
                    for page_type in ["search", "tracklist", "dj"]:
                        health = await self.check_parser_health(page_type)
                        if health.requires_alert:
                            await self.send_health_alert(health, page_type)

                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Monitoring loop error: {e}")
                    await asyncio.sleep(60)  # Wait a minute before retrying

        task = asyncio.create_task(monitor_loop())
        self._monitoring_tasks.append(task)

    async def stop_monitoring(self) -> None:
        """Stop all monitoring tasks."""
        for task in self._monitoring_tasks:
            task.cancel()
        await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        self._monitoring_tasks.clear()

    def get_recent_alerts(self, limit: int = 10, severity: AlertSeverity | None = None) -> list[Alert]:
        """Get recent alerts from history.

        Args:
            limit: Maximum number of alerts to return
            severity: Filter by severity level

        Returns:
            List of recent alerts
        """
        alerts = self._alert_history[-limit:]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts

    async def get_active_alerts(self) -> list[Alert]:
        """Get currently active alerts.

        Returns:
            List of active alerts
        """
        # Return alerts from last hour that are still relevant
        current_time = datetime.now(UTC)
        one_hour_ago = current_time - timedelta(hours=1)

        return [alert for alert in self._alert_history if alert.timestamp > one_hour_ago and not alert.resolved]
