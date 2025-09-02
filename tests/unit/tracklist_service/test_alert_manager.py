"""Tests for alert manager."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from services.notification_service.src.core.base import AlertType
from services.tracklist_service.src.monitoring.alert_manager import (
    Alert,
    AlertManager,
    AlertSeverity,
    HealthStatus,
    severity_to_alert_type,
)
from services.tracklist_service.src.monitoring.structure_monitor import ChangeReport


class TestSeverityMapping:
    """Test severity to alert type mapping."""

    def test_severity_to_alert_type_mapping(self):
        """Test that severities map to correct alert types."""
        assert severity_to_alert_type(AlertSeverity.INFO) == AlertType.GENERAL
        assert severity_to_alert_type(AlertSeverity.WARNING) == AlertType.MONITORING
        assert severity_to_alert_type(AlertSeverity.ERROR) == AlertType.ERROR
        assert severity_to_alert_type(AlertSeverity.CRITICAL) == AlertType.CRITICAL


class TestAlert:
    """Test Alert data structure."""

    def test_alert_creation(self):
        """Test alert creation with defaults."""
        alert = Alert(
            severity=AlertSeverity.WARNING,
            message="Test alert",
            details={"key": "value"},
        )

        assert alert.severity == AlertSeverity.WARNING
        assert alert.message == "Test alert"
        assert alert.details == {"key": "value"}
        assert alert.resolved is False
        assert alert.channels == []

    def test_alert_to_dict(self):
        """Test alert serialization to dictionary."""
        alert = Alert(
            severity=AlertSeverity.ERROR,
            message="Test error",
            page_type="search",
        )

        alert_dict = alert.to_dict()

        assert alert_dict["severity"] == "error"
        assert alert_dict["message"] == "Test error"
        assert alert_dict["page_type"] == "search"
        assert "timestamp" in alert_dict


class TestHealthStatus:
    """Test HealthStatus data structure."""

    def test_healthy_status(self):
        """Test healthy status creation."""
        status = HealthStatus(
            healthy=True,
            success_rate=0.98,
            last_check=datetime.now(UTC),
        )

        assert status.requires_alert is False

    def test_unhealthy_status_low_success_rate(self):
        """Test unhealthy status with low success rate."""
        status = HealthStatus(
            healthy=True,
            success_rate=0.9,  # Below 0.95 threshold
            last_check=datetime.now(UTC),
        )

        assert status.requires_alert is True

    def test_unhealthy_status_with_anomalies(self):
        """Test unhealthy status with anomalies."""
        status = HealthStatus(
            healthy=True,
            success_rate=0.98,
            last_check=datetime.now(UTC),
            anomalies=["Something wrong"],
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
    def mock_notification_service(self):
        """Mock Discord notification service."""
        return AsyncMock()

    @pytest.fixture
    def alert_manager(self, mock_redis, mock_notification_service):
        """Create AlertManager instance."""
        return AlertManager(
            redis_client=mock_redis,
            notification_service=mock_notification_service,
        )

    def test_alert_manager_init(self):
        """Test AlertManager initialization."""
        manager = AlertManager()
        assert manager._alert_history == []
        assert manager._health_checks == {}

    @pytest.mark.asyncio
    async def test_send_alert(self, alert_manager, mock_notification_service):
        """Test sending alert through Discord notification service."""
        await alert_manager.send_alert("error", "Test alert message")

        # Verify notification service was called
        mock_notification_service.send.assert_called_once()
        call_args = mock_notification_service.send.call_args[0][0]

        assert call_args.alert_type == AlertType.ERROR
        assert call_args.title == "Error Alert"
        assert call_args.message == "Test alert message"

    @pytest.mark.asyncio
    async def test_send_change_alert(self, alert_manager, mock_notification_service):
        """Test sending change alert."""
        change_report = ChangeReport(
            page_type="search",
            timestamp=datetime.now(UTC),
            changes=[],  # Changes should be list of StructuralChange objects, not strings
            severity="high",
            fingerprint_match_percentage=80.0,
            requires_manual_review=True,
        )

        await alert_manager.send_change_alert(change_report)

        # Verify notification service was called with correct parameters
        mock_notification_service.send.assert_called_once()
        call_args = mock_notification_service.send.call_args[0][0]

        assert call_args.alert_type == AlertType.ERROR  # high severity maps to ERROR
        assert "🔄 Structural Changes Detected" in call_args.title
        assert "search" in call_args.message

    @pytest.mark.asyncio
    async def test_send_health_alert(self, alert_manager, mock_notification_service):
        """Test sending health alert."""
        health_status = HealthStatus(
            healthy=False,
            success_rate=0.7,
            last_check=datetime.now(UTC),
            anomalies=["Low success rate"],
        )

        await alert_manager.send_health_alert(health_status, "search")

        # Verify notification service was called
        mock_notification_service.send.assert_called_once()
        call_args = mock_notification_service.send.call_args[0][0]

        assert call_args.alert_type == AlertType.ERROR  # 0.7 success rate maps to ERROR
        assert "🏥 Health Alert: search" in call_args.title

    @pytest.mark.asyncio
    async def test_check_parser_health_no_redis(self):
        """Test parser health check without Redis."""
        manager = AlertManager()

        status = await manager.check_parser_health()

        assert status.healthy is True
        assert status.success_rate == 1.0

    @pytest.mark.asyncio
    async def test_check_parser_health_with_redis(self, alert_manager, mock_redis):
        """Test parser health check with Redis data."""
        # Mock Redis responses for success/failure metrics
        mock_redis.get.side_effect = lambda key: {
            "metrics:extraction:search:success": "90",
            "metrics:extraction:search:failure": "10",
        }.get(key, "0")

        status = await alert_manager.check_parser_health("search")

        assert status.success_rate == 0.9
        assert status.healthy is False  # Below 0.95 threshold

    def test_get_recent_alerts(self, alert_manager):
        """Test getting recent alerts."""
        # Add some alerts to history
        alert1 = Alert(AlertSeverity.INFO, "Alert 1")
        alert2 = Alert(AlertSeverity.ERROR, "Alert 2")
        alert_manager._alert_history = [alert1, alert2]

        recent = alert_manager.get_recent_alerts(limit=5)

        assert len(recent) == 2
        assert recent[-1] == alert2  # Most recent

    def test_get_recent_alerts_with_severity_filter(self, alert_manager):
        """Test getting recent alerts with severity filter."""
        alert1 = Alert(AlertSeverity.INFO, "Info alert")
        alert2 = Alert(AlertSeverity.ERROR, "Error alert")
        alert_manager._alert_history = [alert1, alert2]

        error_alerts = alert_manager.get_recent_alerts(severity=AlertSeverity.ERROR)

        assert len(error_alerts) == 1
        assert error_alerts[0] == alert2

    @pytest.mark.asyncio
    async def test_get_active_alerts(self, alert_manager):
        """Test getting active alerts."""
        # Add recent alert
        recent_alert = Alert(AlertSeverity.WARNING, "Recent alert")
        recent_alert.timestamp = datetime.now(UTC)

        # Add old alert
        old_alert = Alert(AlertSeverity.INFO, "Old alert")
        old_alert.timestamp = datetime.now(UTC) - timedelta(hours=2)

        alert_manager._alert_history = [old_alert, recent_alert]

        active = await alert_manager.get_active_alerts()

        assert len(active) == 1
        assert active[0] == recent_alert

    def test_register_anomaly_detector(self, alert_manager):
        """Test registering custom anomaly detector."""

        async def custom_detector(page_type):
            return "Custom anomaly detected"

        alert_manager.register_anomaly_detector("custom", custom_detector)

        assert "custom" in alert_manager._anomaly_detectors
