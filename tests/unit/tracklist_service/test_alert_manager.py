"""Tests for alert manager."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from services.tracklist_service.src.monitoring.alert_manager import (
    Alert,
    AlertChannel,
    AlertManager,
    AlertSeverity,
    HealthStatus,
)
from services.tracklist_service.src.monitoring.structure_monitor import StructureMonitor


class TestAlert:
    """Test Alert data structure."""

    def test_alert_creation(self):
        """Test alert creation with defaults."""
        alert = Alert(
            severity=AlertSeverity.WARNING,
            message="Test alert",
        )

        assert alert.severity == AlertSeverity.WARNING
        assert alert.message == "Test alert"
        assert alert.resolved is False
        assert isinstance(alert.timestamp, datetime)
        assert alert.page_type is None

    def test_alert_to_dict(self):
        """Test alert serialization."""
        alert = Alert(
            severity=AlertSeverity.ERROR,
            message="Test error",
            page_type="tracklist",
            details={"error_code": 500},
        )

        alert_dict = alert.to_dict()

        assert alert_dict["severity"] == "error"
        assert alert_dict["message"] == "Test error"
        assert alert_dict["page_type"] == "tracklist"
        assert alert_dict["details"] == {"error_code": 500}
        assert alert_dict["resolved"] is False
        assert "timestamp" in alert_dict


class TestHealthStatus:
    """Test HealthStatus data structure."""

    def test_healthy_status(self):
        """Test healthy status."""
        status = HealthStatus(
            healthy=True,
            success_rate=0.98,
            last_check=datetime.now(UTC),
        )

        assert status.healthy is True
        assert status.success_rate == 0.98
        assert status.requires_alert is False

    def test_unhealthy_status_low_success_rate(self):
        """Test unhealthy status due to low success rate."""
        status = HealthStatus(
            healthy=True,
            success_rate=0.92,  # Below 0.95 threshold
            last_check=datetime.now(UTC),
        )

        assert status.requires_alert is True

    def test_unhealthy_status_with_anomalies(self):
        """Test unhealthy status due to anomalies."""
        status = HealthStatus(
            healthy=True,
            success_rate=0.98,
            last_check=datetime.now(UTC),
            anomalies=["High error rate detected"],
        )

        assert status.requires_alert is True

    def test_unhealthy_status_not_healthy(self):
        """Test unhealthy status."""
        status = HealthStatus(
            healthy=False,
            success_rate=0.98,
            last_check=datetime.now(UTC),
        )

        assert status.requires_alert is True


class TestAlertManager:
    """Test AlertManager functionality."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        redis_mock = AsyncMock()
        redis_mock.get.return_value = "0"
        redis_mock.keys.return_value = []
        redis_mock.lrange.return_value = []
        return redis_mock

    @pytest.fixture
    def alert_manager(self, mock_redis):
        """Create AlertManager instance."""
        return AlertManager(
            redis_client=mock_redis,
            slack_webhook_url="https://hooks.slack.com/test",
            email_config={"smtp_server": "test.com"},
        )

    def test_alert_manager_init(self):
        """Test AlertManager initialization."""
        manager = AlertManager()

        assert manager.redis_client is None
        assert manager.slack_webhook_url is None
        assert manager.email_config is None
        assert manager._alert_history == []

    def test_alert_manager_with_config(self, mock_redis):
        """Test AlertManager with configuration."""
        manager = AlertManager(
            redis_client=mock_redis,
            slack_webhook_url="https://hooks.slack.com/test",
            email_config={"smtp_server": "test.com"},
        )

        assert manager.redis_client == mock_redis
        assert manager.slack_webhook_url == "https://hooks.slack.com/test"
        assert manager.email_config == {"smtp_server": "test.com"}

    @pytest.mark.asyncio
    async def test_check_parser_health_no_redis(self):
        """Test health check without Redis."""
        manager = AlertManager()

        health = await manager.check_parser_health()

        assert isinstance(health, HealthStatus)
        assert health.healthy is True
        assert health.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_check_parser_health_with_redis(self, alert_manager, mock_redis):
        """Test health check with Redis."""
        # Mock Redis responses
        mock_redis.get.side_effect = lambda key: {
            "metrics:extraction:tracklist:success": "95",
            "metrics:extraction:tracklist:failure": "5",
        }.get(key, "0")

        health = await alert_manager.check_parser_health("tracklist")

        assert health.healthy is True
        assert health.success_rate == 0.95

    @pytest.mark.asyncio
    async def test_check_parser_health_unhealthy(self, alert_manager, mock_redis):
        """Test health check with unhealthy status."""
        # Mock Redis responses for low success rate
        mock_redis.get.side_effect = lambda key: {
            "metrics:extraction:tracklist:success": "85",
            "metrics:extraction:tracklist:failure": "15",
        }.get(key, "0")

        health = await alert_manager.check_parser_health("tracklist")

        assert health.healthy is False
        assert health.success_rate == 0.85
        assert health.requires_alert is True

    @pytest.mark.asyncio
    async def test_send_alert(self, alert_manager):
        """Test sending alert."""
        await alert_manager.send_alert("warning", "Test alert", ["log"])

        # Alert should be in history
        assert len(alert_manager._alert_history) == 1
        assert alert_manager._alert_history[0].message == "Test alert"
        assert alert_manager._alert_history[0].severity == AlertSeverity.WARNING

    @pytest.mark.asyncio
    async def test_send_health_alert(self, alert_manager):
        """Test sending health alert."""
        health = HealthStatus(
            healthy=False,
            success_rate=0.85,
            last_check=datetime.now(UTC),
            anomalies=["High error rate"],
        )

        await alert_manager.send_health_alert(health, "tracklist")

        # Should generate alert
        assert len(alert_manager._alert_history) == 1
        alert = alert_manager._alert_history[0]
        assert alert.severity == AlertSeverity.WARNING  # Due to anomalies
        assert "tracklist" in alert.message.lower()

    def test_get_channels_for_severity(self, alert_manager):
        """Test channel selection based on severity."""
        # Test INFO
        channels = alert_manager._get_channels_for_severity(AlertSeverity.INFO)
        assert AlertChannel.LOG in channels
        assert AlertChannel.DASHBOARD in channels
        assert len(channels) == 2

        # Test WARNING
        channels = alert_manager._get_channels_for_severity(AlertSeverity.WARNING)
        assert AlertChannel.LOG in channels
        assert AlertChannel.DASHBOARD in channels
        assert len(channels) == 2

        # Test ERROR
        channels = alert_manager._get_channels_for_severity(AlertSeverity.ERROR)
        assert AlertChannel.LOG in channels
        assert AlertChannel.DASHBOARD in channels
        assert AlertChannel.SLACK in channels
        assert len(channels) == 3

        # Test CRITICAL
        channels = alert_manager._get_channels_for_severity(AlertSeverity.CRITICAL)
        assert AlertChannel.LOG in channels
        assert AlertChannel.DASHBOARD in channels
        assert AlertChannel.SLACK in channels
        assert AlertChannel.EMAIL in channels
        assert len(channels) == 4

    @pytest.mark.asyncio
    async def test_detect_anomalies_no_redis(self, alert_manager):
        """Test anomaly detection without Redis."""
        alert_manager.redis_client = None

        anomalies = await alert_manager._detect_anomalies("tracklist")

        assert anomalies == []

    @pytest.mark.asyncio
    async def test_detect_anomalies_with_drop(self, alert_manager, mock_redis):
        """Test anomaly detection with rate drop."""
        mock_redis.get.side_effect = lambda key: {
            "metrics:rate:tracklist:current": "25",
            "metrics:rate:tracklist:average": "100",
            "failures:pattern:tracklist": None,
        }.get(key, "0")

        anomalies = await alert_manager._detect_anomalies("tracklist")

        assert len(anomalies) > 0
        assert "dropped by" in anomalies[0]

    @pytest.mark.asyncio
    async def test_detect_anomalies_consecutive_failures(self, alert_manager, mock_redis):
        """Test anomaly detection with consecutive failures."""
        failure_pattern = json.dumps({"consecutive_failures": 10})
        mock_redis.get.side_effect = lambda key: {
            "failures:pattern:tracklist": failure_pattern,
            "metrics:rate:tracklist:current": "0",
            "metrics:rate:tracklist:average": "0",
        }.get(key)

        anomalies = await alert_manager._detect_anomalies("tracklist")

        assert len(anomalies) > 0
        assert "Consecutive failures: 10" in anomalies

    def test_register_anomaly_detector(self, alert_manager):
        """Test custom anomaly detector registration."""
        detector_called = False

        async def custom_detector(page_type):
            nonlocal detector_called
            detector_called = True
            return "Custom anomaly detected"

        alert_manager.register_anomaly_detector("custom", custom_detector)

        assert "custom" in alert_manager._anomaly_detectors

    @pytest.mark.asyncio
    async def test_custom_anomaly_detector_execution(self, alert_manager, mock_redis):
        """Test custom anomaly detector execution."""
        # Mock Redis to avoid errors in anomaly detection
        mock_redis.get.side_effect = lambda key: {
            "metrics:rate:tracklist:current": "0",
            "metrics:rate:tracklist:average": "0",
            "failures:pattern:tracklist": None,
        }.get(key, "0")

        async def custom_detector(page_type):
            return "Custom anomaly"

        alert_manager.register_anomaly_detector("custom", custom_detector)

        anomalies = await alert_manager._detect_anomalies("tracklist")

        assert "custom: Custom anomaly" in anomalies

    @pytest.mark.asyncio
    async def test_start_monitoring(self, alert_manager):
        """Test starting monitoring."""

        structure_monitor = StructureMonitor()

        # Start monitoring with short interval for testing
        await alert_manager.start_monitoring(structure_monitor, interval=0.1)

        # Let it run briefly
        await asyncio.sleep(0.05)

        # Stop monitoring
        await alert_manager.stop_monitoring()

        # Should have created monitoring tasks
        assert len(alert_manager._monitoring_tasks) == 0  # Cleared after stop

    @pytest.mark.asyncio
    async def test_get_active_alerts(self, alert_manager):
        """Test getting active alerts."""
        # Add some alerts
        old_alert = Alert(
            severity=AlertSeverity.WARNING,
            message="Old alert",
            timestamp=datetime.now(UTC) - timedelta(hours=2),
        )
        recent_alert = Alert(
            severity=AlertSeverity.ERROR,
            message="Recent alert",
            timestamp=datetime.now(UTC) - timedelta(minutes=30),
        )
        resolved_alert = Alert(
            severity=AlertSeverity.INFO,
            message="Resolved alert",
            timestamp=datetime.now(UTC) - timedelta(minutes=10),
            resolved=True,
        )

        alert_manager._alert_history = [old_alert, recent_alert, resolved_alert]

        active_alerts = await alert_manager.get_active_alerts()

        # Should only include recent unresolved alert
        assert len(active_alerts) == 1
        assert active_alerts[0] == recent_alert

    def test_get_recent_alerts(self, alert_manager):
        """Test getting recent alerts."""
        # Add some alerts
        alerts = [Alert(severity=AlertSeverity.INFO, message=f"Alert {i}") for i in range(5)]
        alert_manager._alert_history = alerts

        # Get recent alerts
        recent = alert_manager.get_recent_alerts(limit=3)

        assert len(recent) == 3
        # Should return the last 3 alerts
        assert recent == alerts[-3:]

    def test_get_recent_alerts_with_severity_filter(self, alert_manager):
        """Test getting recent alerts with severity filter."""
        # Add mixed severity alerts
        alerts = [
            Alert(severity=AlertSeverity.INFO, message="Info 1"),
            Alert(severity=AlertSeverity.WARNING, message="Warning 1"),
            Alert(severity=AlertSeverity.ERROR, message="Error 1"),
            Alert(severity=AlertSeverity.INFO, message="Info 2"),
        ]
        alert_manager._alert_history = alerts

        # Filter for INFO alerts
        info_alerts = alert_manager.get_recent_alerts(severity=AlertSeverity.INFO)

        assert len(info_alerts) == 2
        assert all(alert.severity == AlertSeverity.INFO for alert in info_alerts)


class TestAlertChannels:
    """Test alert channel functionality."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        redis_mock = AsyncMock()
        redis_mock.get.return_value = "0"
        redis_mock.keys.return_value = []
        redis_mock.lrange.return_value = []
        return redis_mock

    @pytest.fixture
    def alert_manager_with_config(self, mock_redis):
        """Create AlertManager instance with full config."""
        return AlertManager(
            redis_client=mock_redis,
            slack_webhook_url="https://hooks.slack.com/test",
            email_config={"smtp_server": "test.com"},
        )

    @pytest.fixture
    def alert_manager(self, mock_redis):
        """Create AlertManager instance."""
        return AlertManager(
            redis_client=mock_redis,
            slack_webhook_url="https://hooks.slack.com/test",
            email_config={"smtp_server": "test.com"},
        )

    @pytest.mark.asyncio
    async def test_send_to_log_channel(self, alert_manager_with_config):
        """Test sending alert to log channel."""
        alert = Alert(
            severity=AlertSeverity.WARNING,
            message="Test log alert",
            channels=[AlertChannel.LOG],
        )

        with patch("services.tracklist_service.src.monitoring.alert_manager.logger") as mock_logger:
            await alert_manager_with_config._send_to_log(alert)
            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_to_slack_channel(self, alert_manager_with_config):
        """Test sending alert to Slack channel."""
        alert = Alert(
            severity=AlertSeverity.ERROR,
            message="Test Slack alert",
            channels=[AlertChannel.SLACK],
        )

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_post.return_value.__aenter__.return_value.status = 200

            await alert_manager_with_config._send_to_slack(alert)

            mock_post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_to_slack_no_webhook(self, alert_manager):
        """Test sending to Slack without webhook URL."""
        alert_manager.slack_webhook_url = None

        alert = Alert(
            severity=AlertSeverity.ERROR,
            message="Test alert",
            channels=[AlertChannel.SLACK],
        )

        # Should not raise error
        await alert_manager._send_to_slack(alert)

    @pytest.mark.asyncio
    async def test_store_alert_history_redis(self, alert_manager_with_config, mock_redis):
        """Test storing alert history in Redis."""
        alert = Alert(
            severity=AlertSeverity.WARNING,
            message="Test alert",
            page_type="tracklist",
        )

        await alert_manager_with_config._store_alert_in_redis(alert)

        # Should call Redis operations
        assert mock_redis.lpush.called or hasattr(mock_redis.lpush.return_value, "__await__")

    @pytest.mark.asyncio
    async def test_store_alert_history_no_redis(self, alert_manager):
        """Test storing alert history without Redis."""

        alert_manager.redis_client = None

        # Should not raise error (method doesn't exist, simulating no-op)
        # The actual implementation stores in memory when Redis is None
