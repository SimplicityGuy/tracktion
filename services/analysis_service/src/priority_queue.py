"""Priority queue configuration and management for the analysis service."""

import enum
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class MessagePriority(enum.IntEnum):
    """Priority levels for analysis messages."""

    CRITICAL = 10  # System-critical files or retries
    HIGH = 7  # User-requested or small files
    NORMAL = 5  # Default priority
    LOW = 3  # Large files or batch operations
    BACKGROUND = 1  # Maintenance or cleanup tasks


@dataclass
class PriorityConfig:
    """Configuration for priority queue behavior."""

    enable_priority: bool = True
    max_priority: int = 10
    default_priority: int = MessagePriority.NORMAL

    # File size thresholds (in MB)
    small_file_threshold: float = 10.0  # Files < 10MB get higher priority
    large_file_threshold: float = 100.0  # Files > 100MB get lower priority

    # Priority adjustments
    small_file_boost: int = 2
    large_file_penalty: int = 2
    retry_boost: int = 3
    user_request_boost: int = 3

    # File format priorities
    format_priorities: dict[str, int] | None = None

    def __post_init__(self) -> None:
        """Initialize default format priorities if not provided."""
        if self.format_priorities is None:
            self.format_priorities = {
                "mp3": MessagePriority.NORMAL,
                "flac": MessagePriority.NORMAL,
                "wav": MessagePriority.LOW,  # Large files
                "m4a": MessagePriority.NORMAL,
                "ogg": MessagePriority.NORMAL,
                "aiff": MessagePriority.LOW,  # Large files
            }


class PriorityCalculator:
    """Calculate message priority based on various attributes."""

    def __init__(self, config: PriorityConfig | None = None) -> None:
        """Initialize the priority calculator.

        Args:
            config: Priority configuration settings
        """
        self.config = config or PriorityConfig()

    def calculate_priority(
        self,
        file_path: str,
        file_size_mb: float | None = None,
        is_retry: bool = False,
        is_user_request: bool = False,
        custom_priority: int | None = None,
        correlation_id: str | None = None,
    ) -> int:
        """Calculate the priority for a message.

        Args:
            file_path: Path to the audio file
            file_size_mb: File size in megabytes
            is_retry: Whether this is a retry attempt
            is_user_request: Whether this was directly requested by a user
            custom_priority: Custom priority override
            correlation_id: Message correlation ID for logging

        Returns:
            Calculated priority value (higher = more important)
        """
        if not self.config.enable_priority:
            return self.config.default_priority

        # Use custom priority if provided
        if custom_priority is not None:
            priority = max(1, min(custom_priority, self.config.max_priority))
            logger.debug(
                f"Using custom priority: {priority}",
                extra={"correlation_id": correlation_id},
            )
            return priority

        # Start with default priority
        priority = self.config.default_priority

        # Adjust for file format
        file_ext = file_path.lower().split(".")[-1] if "." in file_path else ""
        if self.config.format_priorities and file_ext in self.config.format_priorities:
            priority = self.config.format_priorities[file_ext]
            logger.debug(
                f"Format {file_ext} base priority: {priority}",
                extra={"correlation_id": correlation_id},
            )

        # Adjust for file size
        if file_size_mb is not None:
            if file_size_mb < self.config.small_file_threshold:
                priority += self.config.small_file_boost
                logger.debug(
                    f"Small file boost applied: +{self.config.small_file_boost}",
                    extra={"correlation_id": correlation_id},
                )
            elif file_size_mb > self.config.large_file_threshold:
                priority -= self.config.large_file_penalty
                logger.debug(
                    f"Large file penalty applied: -{self.config.large_file_penalty}",
                    extra={"correlation_id": correlation_id},
                )

        # Boost priority for retries
        if is_retry:
            priority += self.config.retry_boost
            logger.debug(
                f"Retry boost applied: +{self.config.retry_boost}",
                extra={"correlation_id": correlation_id},
            )

        # Boost priority for user requests
        if is_user_request:
            priority += self.config.user_request_boost
            logger.debug(
                f"User request boost applied: +{self.config.user_request_boost}",
                extra={"correlation_id": correlation_id},
            )

        # Ensure priority is within valid range
        final_priority = max(1, min(priority, self.config.max_priority))

        logger.info(
            f"Calculated priority for {file_path}: {final_priority}",
            extra={
                "correlation_id": correlation_id,
                "file_size_mb": file_size_mb,
                "is_retry": is_retry,
                "is_user_request": is_user_request,
            },
        )

        return final_priority


def setup_priority_queue(channel: Any, queue_name: str, max_priority: int = 10) -> None:
    """Set up a priority queue in RabbitMQ.

    Args:
        channel: RabbitMQ channel
        queue_name: Name of the queue
        max_priority: Maximum priority value
    """
    # Declare queue with priority support
    channel.queue_declare(
        queue=queue_name,
        durable=True,
        arguments={
            "x-max-priority": max_priority,
            "x-message-ttl": 3600000,  # 1 hour TTL for messages
        },
    )
    logger.info(f"Priority queue '{queue_name}' configured with max priority {max_priority}")


def add_priority_to_message(
    message: dict[str, Any],
    priority_calculator: PriorityCalculator,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    """Add priority field to a message.

    Args:
        message: Message dictionary
        priority_calculator: Priority calculator instance
        correlation_id: Message correlation ID

    Returns:
        Message with priority field added
    """
    # Extract attributes for priority calculation
    file_path = message.get("file_path", "")
    file_size_mb = message.get("file_size_mb")
    is_retry = message.get("retry_count", 0) > 0
    is_user_request = message.get("user_request", False)
    custom_priority = message.get("priority")

    # Calculate priority
    priority = priority_calculator.calculate_priority(
        file_path=file_path,
        file_size_mb=file_size_mb,
        is_retry=is_retry,
        is_user_request=is_user_request,
        custom_priority=custom_priority,
        correlation_id=correlation_id,
    )

    # Add priority to message
    message["priority"] = priority
    return message
