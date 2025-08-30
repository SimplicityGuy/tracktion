"""Alert management system for structure monitoring."""

import asyncio
import json
import logging
from datetime import datetime, timedelta, UTC
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import aiohttp
from redis.asyncio import Redis

from services.tracklist_service.src.monitoring.structure_monitor import ChangeReport, StructureMonitor

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Available alert notification channels."""

    LOG = "log"
    SLACK = "slack"
    EMAIL = "email"
    WEBHOOK = "webhook"
    DASHBOARD = "dashboard"


@dataclass
class Alert:
    """Alert data structure."""

    severity: AlertSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    page_type: Optional[str] = None
    change_report: Optional[ChangeReport] = None
    channels: List[AlertChannel] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False

    def to_dict(self) -> dict:
        """Convert alert to dictionary."""
        return {
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "page_type": self.page_type,
            "change_report": self.change_report.to_dict() if self.change_report else None,
            "channels": [c.value for c in self.channels],
            "metadata": self.metadata,
            "resolved": self.resolved,
        }


@dataclass
class HealthStatus:
    """Health status for the scraper."""

    healthy: bool
    success_rate: float
    last_check: datetime
    failed_extractions: List[str] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def requires_alert(self) -> bool:
        """Check if health status requires an alert."""
        return not self.healthy or self.success_rate < 0.95 or len(self.anomalies) > 0


class AlertManager:
    """Manages alerts and notifications for the scraping system."""

    def __init__(
        self,
        redis_client: Optional[Redis] = None,
        slack_webhook_url: Optional[str] = None,
        email_config: Optional[Dict[str, str]] = None,
        dashboard_url: Optional[str] = None,
    ):
        """Initialize alert manager.

        Args:
            redis_client: Redis client for storing alerts
            slack_webhook_url: Slack webhook URL for notifications
            email_config: Email configuration for notifications
            dashboard_url: Dashboard URL for posting alerts
        """
        self.redis_client = redis_client
        self.slack_webhook_url = slack_webhook_url
        self.email_config = email_config
        self.dashboard_url = dashboard_url
        self._alert_history: List[Alert] = []
        self._health_checks: Dict[str, HealthStatus] = {}
        self._anomaly_detectors: Dict[str, Callable] = {}
        self._monitoring_tasks: List[asyncio.Task] = []

    async def check_parser_health(self, page_type: Optional[str] = None) -> HealthStatus:
        """Check health status of parser(s).

        Args:
            page_type: Specific page type to check, or None for all

        Returns:
            HealthStatus object
        """
        if self.redis_client:
            # Get recent extraction metrics from Redis
            if page_type:
                key = f"metrics:extraction:{page_type}"
            else:
                key = "metrics:extraction:*"

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
                    failed_result = self.redis_client.lrange(failed_key, 0, 9)
                    if hasattr(failed_result, "__await__"):
                        failed_data = await failed_result  # type: ignore[misc]
                    else:
                        failed_data = failed_result
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
                    anomalies=[f"Health check failed: {str(e)}"],
                )
        else:
            # No Redis, return default healthy status
            status = HealthStatus(healthy=True, success_rate=1.0, last_check=datetime.now(UTC))

        # Cache health status
        if page_type:
            self._health_checks[page_type] = status

        return status

    async def send_alert(self, severity: str, message: str, channels: List[str]) -> None:
        """Send alert to specified channels.

        Args:
            severity: Alert severity level
            message: Alert message
            channels: List of channels to send to
        """
        alert = Alert(
            severity=AlertSeverity(severity),
            message=message,
            channels=[AlertChannel(c) for c in channels],
        )

        await self._dispatch_alert(alert)

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

        # Create alert
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
            channels=self._get_channels_for_severity(severity),
        )

        await self._dispatch_alert(alert)

    async def send_health_alert(self, health_status: HealthStatus, page_type: Optional[str] = None) -> None:
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

        # Create alert
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
            channels=self._get_channels_for_severity(severity),
            metadata={
                "anomalies": health_status.anomalies,
                "failed_extractions": health_status.failed_extractions[:5],  # Limit to 5
            },
        )

        await self._dispatch_alert(alert)

    async def _dispatch_alert(self, alert: Alert) -> None:
        """Dispatch alert to configured channels.

        Args:
            alert: Alert to dispatch
        """
        # Store in history
        self._alert_history.append(alert)
        if len(self._alert_history) > 1000:
            self._alert_history = self._alert_history[-1000:]  # Keep last 1000

        # Store in Redis if available
        if self.redis_client:
            await self._store_alert_in_redis(alert)

        # Send to each channel
        for channel in alert.channels:
            try:
                if channel == AlertChannel.LOG:
                    await self._send_to_log(alert)
                elif channel == AlertChannel.SLACK:
                    await self._send_to_slack(alert)
                elif channel == AlertChannel.EMAIL:
                    await self._send_to_email(alert)
                elif channel == AlertChannel.WEBHOOK:
                    await self._send_to_webhook(alert)
                elif channel == AlertChannel.DASHBOARD:
                    await self._send_to_dashboard(alert)
            except Exception as e:
                logger.error(f"Failed to send alert to {channel.value}: {e}")

    async def _send_to_log(self, alert: Alert) -> None:
        """Send alert to log."""
        log_method = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.ERROR: logger.error,
            AlertSeverity.CRITICAL: logger.critical,
        }.get(alert.severity, logger.info)

        log_method(f"[ALERT] {alert.message}", extra={"alert": alert.to_dict()})

    async def _send_to_slack(self, alert: Alert) -> None:
        """Send alert to Slack."""
        if not self.slack_webhook_url:
            return

        # Format message for Slack
        color = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ff9900",
            AlertSeverity.ERROR: "#ff0000",
            AlertSeverity.CRITICAL: "#990000",
        }.get(alert.severity, "#808080")

        attachment = {
            "color": color,
            "title": f"{alert.severity.value.upper()}: {alert.message}",
            "fields": [{"title": k, "value": str(v), "short": True} for k, v in alert.details.items()],
            "footer": "Tracklist Service Alert",
            "ts": int(alert.timestamp.timestamp()),
        }

        payload = {"attachments": [attachment]}

        async with aiohttp.ClientSession() as session:
            async with session.post(self.slack_webhook_url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Failed to send Slack alert: {response.status}")

    async def _send_to_email(self, alert: Alert) -> None:
        """Send alert via email."""
        # TODO: Implement email sending
        logger.info(f"Email alert not implemented: {alert.message}")

    async def _send_to_webhook(self, alert: Alert) -> None:
        """Send alert to webhook."""
        # TODO: Implement generic webhook
        logger.info(f"Webhook alert not implemented: {alert.message}")

    async def _send_to_dashboard(self, alert: Alert) -> None:
        """Send alert to dashboard."""
        if not self.dashboard_url:
            return

        async with aiohttp.ClientSession() as session:
            async with session.post(self.dashboard_url, json=alert.to_dict()) as response:
                if response.status != 200:
                    logger.error(f"Failed to send dashboard alert: {response.status}")

    async def _store_alert_in_redis(self, alert: Alert) -> None:
        """Store alert in Redis for history."""
        if not self.redis_client:
            return

        key = f"alerts:history:{alert.page_type or 'global'}"
        # Handle potential sync/async Redis methods
        result1 = self.redis_client.lpush(key, json.dumps(alert.to_dict()))
        result2 = self.redis_client.ltrim(key, 0, 99)  # Keep last 100
        result3 = self.redis_client.expire(key, 86400 * 7)  # 7 days

        if hasattr(result1, "__await__"):
            await result1
        if hasattr(result2, "__await__"):
            await result2
        if hasattr(result3, "__await__"):
            await result3

    def _get_channels_for_severity(self, severity: AlertSeverity) -> List[AlertChannel]:
        """Get appropriate channels based on severity."""
        channels = [AlertChannel.LOG]

        # Always add dashboard for visibility
        channels.append(AlertChannel.DASHBOARD)

        if severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            channels.append(AlertChannel.SLACK)
            if severity == AlertSeverity.CRITICAL:
                channels.append(AlertChannel.EMAIL)

        return channels

    async def _detect_anomalies(self, page_type: Optional[str] = None) -> List[str]:
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

    def register_anomaly_detector(self, name: str, detector: Callable) -> None:
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

    def get_recent_alerts(self, limit: int = 10, severity: Optional[AlertSeverity] = None) -> List[Alert]:
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

    async def get_active_alerts(self) -> List[Alert]:
        """Get currently active alerts.

        Returns:
            List of active alerts
        """
        # Return alerts from last hour that are still relevant
        current_time = datetime.now(UTC)
        one_hour_ago = current_time - timedelta(hours=1)

        active_alerts = [
            alert for alert in self._alert_history if alert.timestamp > one_hour_ago and not alert.resolved
        ]

        return active_alerts
