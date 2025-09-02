"""Tests for base notification abstractions."""

from datetime import UTC, datetime

from services.notification_service.src.core.base import (
    AlertType,
    NotificationMessage,
    NotificationResult,
    NotificationStatus,
)


class TestAlertType:
    """Test AlertType enum."""

    def test_alert_type_values(self) -> None:
        """Test that all alert types have expected values."""
        assert AlertType.GENERAL.value == "general"
        assert AlertType.ERROR.value == "error"
        assert AlertType.CRITICAL.value == "critical"
        assert AlertType.TRACKLIST.value == "tracklist"
        assert AlertType.MONITORING.value == "monitoring"
        assert AlertType.SECURITY.value == "security"

    def test_alert_type_count(self) -> None:
        """Test that we have the expected number of alert types."""
        assert len(AlertType) == 6


class TestNotificationStatus:
    """Test NotificationStatus enum."""

    def test_status_values(self) -> None:
        """Test that all statuses have expected values."""
        assert NotificationStatus.PENDING.value == "pending"
        assert NotificationStatus.SENT.value == "sent"
        assert NotificationStatus.FAILED.value == "failed"
        assert NotificationStatus.RETRYING.value == "retrying"
        assert NotificationStatus.QUEUED.value == "queued"
        assert NotificationStatus.RATE_LIMITED.value == "rate_limited"

    def test_status_count(self) -> None:
        """Test that we have the expected number of statuses."""
        assert len(NotificationStatus) == 6


class TestNotificationResult:
    """Test NotificationResult dataclass."""

    def test_successful_result(self) -> None:
        """Test creating a successful notification result."""
        result = NotificationResult(
            success=True,
            status=NotificationStatus.SENT,
            message_id="msg_123",
        )

        assert result.success is True
        assert result.status == NotificationStatus.SENT
        assert result.message_id == "msg_123"
        assert result.error is None
        assert result.retry_count == 0
        assert isinstance(result.timestamp, datetime)
        assert result.metadata == {}

    def test_failed_result(self) -> None:
        """Test creating a failed notification result."""
        result = NotificationResult(
            success=False,
            status=NotificationStatus.FAILED,
            error="Connection timeout",
            retry_count=3,
        )

        assert result.success is False
        assert result.status == NotificationStatus.FAILED
        assert result.error == "Connection timeout"
        assert result.retry_count == 3
        assert result.message_id is None

    def test_result_to_dict(self) -> None:
        """Test converting result to dictionary."""
        timestamp = datetime.now(UTC)
        result = NotificationResult(
            success=True,
            status=NotificationStatus.SENT,
            message_id="msg_456",
            timestamp=timestamp,
            metadata={"rate_limit": {"remaining": 25}},
        )

        result_dict = result.to_dict()

        assert result_dict["success"] is True
        assert result_dict["status"] == "sent"
        assert result_dict["message_id"] == "msg_456"
        assert result_dict["error"] is None
        assert result_dict["timestamp"] == timestamp.isoformat()
        assert result_dict["retry_count"] == 0
        assert result_dict["metadata"] == {"rate_limit": {"remaining": 25}}


class TestNotificationMessage:
    """Test NotificationMessage dataclass."""

    def test_basic_message(self) -> None:
        """Test creating a basic notification message."""
        message = NotificationMessage(
            alert_type=AlertType.GENERAL,
            title="Test Title",
            message="Test message content",
        )

        assert message.alert_type == AlertType.GENERAL
        assert message.title == "Test Title"
        assert message.message == "Test message content"
        assert message.color is None
        assert message.fields is None
        assert message.metadata == {}
        assert isinstance(message.timestamp, datetime)

    def test_message_with_fields(self) -> None:
        """Test creating a message with fields."""
        fields = [
            {"name": "Field 1", "value": "Value 1", "inline": True},
            {"name": "Field 2", "value": "Value 2", "inline": False},
        ]

        message = NotificationMessage(
            alert_type=AlertType.ERROR,
            title="Error Alert",
            message="An error occurred",
            color=0xFF0000,
            fields=fields,
        )

        assert message.alert_type == AlertType.ERROR
        assert message.color == 0xFF0000
        assert message.fields == fields
        assert len(message.fields) == 2

    def test_message_to_dict(self) -> None:
        """Test converting message to dictionary."""
        timestamp = datetime.now(UTC)
        message = NotificationMessage(
            alert_type=AlertType.CRITICAL,
            title="Critical Alert",
            message="System failure",
            color=0xE74C3C,
            fields=[{"name": "Component", "value": "Database"}],
            metadata={"severity": "high"},
            timestamp=timestamp,
        )

        message_dict = message.to_dict()

        assert message_dict["alert_type"] == "critical"
        assert message_dict["title"] == "Critical Alert"
        assert message_dict["message"] == "System failure"
        assert message_dict["color"] == 0xE74C3C
        assert message_dict["fields"] == [{"name": "Component", "value": "Database"}]
        assert message_dict["metadata"] == {"severity": "high"}
        assert message_dict["timestamp"] == timestamp.isoformat()
