"""Base notification abstractions and types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class AlertType(Enum):
    """Types of alerts with dedicated Discord channels."""

    GENERAL = "general"
    ERROR = "error"
    CRITICAL = "critical"
    TRACKLIST = "tracklist"
    MONITORING = "monitoring"
    SECURITY = "security"


class NotificationStatus(Enum):
    """Status of a notification delivery attempt."""

    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    RETRYING = "retrying"
    QUEUED = "queued"
    RATE_LIMITED = "rate_limited"


@dataclass
class NotificationResult:
    """Result of a notification send attempt."""

    success: bool
    status: NotificationStatus
    message_id: str | None = None
    error: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    retry_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "status": self.status.value,
            "message_id": self.message_id,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
            "retry_count": self.retry_count,
            "metadata": self.metadata,
        }


@dataclass
class NotificationMessage:
    """Standard notification message format."""

    alert_type: AlertType
    title: str
    message: str
    color: int | None = None
    fields: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert message to dictionary."""
        return {
            "alert_type": self.alert_type.value,
            "title": self.title,
            "message": self.message,
            "color": self.color,
            "fields": self.fields,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


class NotificationChannel(ABC):
    """Abstract base class for notification channels."""

    @abstractmethod
    async def send(self, message: NotificationMessage) -> NotificationResult:
        """Send a notification through this channel.

        Args:
            message: The notification message to send

        Returns:
            Result of the send operation
        """

    @abstractmethod
    async def validate_configuration(self) -> bool:
        """Validate channel configuration.

        Returns:
            True if configuration is valid, False otherwise
        """

    @abstractmethod
    async def get_rate_limit_status(self) -> dict[str, Any]:
        """Get current rate limit status for this channel.

        Returns:
            Dictionary containing rate limit information
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the channel is healthy and available.

        Returns:
            True if channel is healthy, False otherwise
        """
