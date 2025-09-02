"""Notification history logging and tracking."""

import asyncio
import json
import logging
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from redis.asyncio import Redis

from services.notification_service.src.core.base import (
    AlertType,
    NotificationMessage,
    NotificationResult,
    NotificationStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class NotificationHistoryEntry:
    """Entry in notification history."""

    message: NotificationMessage
    result: NotificationResult
    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message": self.message.to_dict(),
            "result": self.result.to_dict(),
            "timestamp": self.timestamp.isoformat() if self.timestamp else datetime.now(UTC).isoformat(),
        }


class NotificationHistoryLogger:
    """Manages notification history logging."""

    def __init__(
        self,
        redis_client: Redis | None = None,
        max_memory_entries: int = 1000,
        retention_days: int = 7,
    ):
        """Initialize history logger.

        Args:
            redis_client: Optional Redis client for persistent storage
            max_memory_entries: Maximum entries to keep in memory
            retention_days: Days to retain history in Redis
        """
        self.redis_client = redis_client
        self.max_memory_entries = max_memory_entries
        self.retention_days = retention_days
        self.memory_history: deque[NotificationHistoryEntry] = deque(maxlen=max_memory_entries)
        self.lock = asyncio.Lock()

    async def log(
        self,
        message: NotificationMessage,
        result: NotificationResult,
    ) -> None:
        """Log a notification attempt.

        Args:
            message: The notification message
            result: The result of the send attempt
        """
        entry = NotificationHistoryEntry(message=message, result=result)

        # Add to memory history
        async with self.lock:
            self.memory_history.append(entry)

        # Persist to Redis if available
        if self.redis_client:
            await self._persist_to_redis(entry)

    async def _persist_to_redis(self, entry: NotificationHistoryEntry) -> None:
        """Persist entry to Redis.

        Args:
            entry: History entry to persist
        """
        if not self.redis_client:
            return

        try:
            # Create Redis key based on alert type and date
            timestamp = entry.timestamp or datetime.now(UTC)
            date_str = timestamp.strftime("%Y%m%d")
            key = f"notifications:history:{entry.message.alert_type.value}:{date_str}"

            # Store as JSON
            entry_json = json.dumps(entry.to_dict())

            # Add to list and set expiration
            await self.redis_client.lpush(key, entry_json)
            await self.redis_client.expire(key, 86400 * self.retention_days)

            # Also maintain a global history key with limited entries
            global_key = "notifications:history:global"
            await self.redis_client.lpush(global_key, entry_json)
            await self.redis_client.ltrim(global_key, 0, 999)  # Keep last 1000
            await self.redis_client.expire(global_key, 86400 * self.retention_days)

        except Exception as e:
            logger.error(f"Failed to persist notification to Redis: {e}")

    async def get_recent(
        self,
        limit: int = 10,
        alert_type: AlertType | None = None,
        since: datetime | None = None,
    ) -> list[NotificationHistoryEntry]:
        """Get recent notification history.

        Args:
            limit: Maximum number of entries to return
            alert_type: Filter by alert type
            since: Only return entries since this time

        Returns:
            List of history entries
        """
        entries = []

        # First check memory history
        for entry in reversed(self.memory_history):
            if alert_type and entry.message.alert_type != alert_type:
                continue
            if since and entry.timestamp and entry.timestamp < since:
                continue
            entries.append(entry)
            if len(entries) >= limit:
                break

        # If we need more and have Redis, check there
        if len(entries) < limit and self.redis_client:
            additional = await self._get_from_redis(limit - len(entries), alert_type, since)
            entries.extend(additional)

        return entries[:limit]

    async def _get_from_redis(
        self,
        limit: int,
        alert_type: AlertType | None = None,
        since: datetime | None = None,
    ) -> list[NotificationHistoryEntry]:
        """Get history entries from Redis.

        Args:
            limit: Maximum number of entries to return
            alert_type: Filter by alert type
            since: Only return entries since this time

        Returns:
            List of history entries
        """
        entries: list[NotificationHistoryEntry] = []

        try:
            if alert_type:
                # Get from specific alert type key
                date_str = datetime.now(UTC).strftime("%Y%m%d")
                key = f"notifications:history:{alert_type.value}:{date_str}"
            else:
                # Get from global key
                key = "notifications:history:global"

            # Fetch entries from Redis
            if self.redis_client:
                raw_entries = await self.redis_client.lrange(key, 0, limit * 2)
            else:
                return entries

            for raw_entry in raw_entries:
                try:
                    entry_dict = json.loads(raw_entry)
                    # Reconstruct objects from dictionaries
                    message = NotificationMessage(
                        alert_type=AlertType(entry_dict["message"]["alert_type"]),
                        title=entry_dict["message"]["title"],
                        message=entry_dict["message"]["message"],
                        color=entry_dict["message"].get("color"),
                        fields=entry_dict["message"].get("fields"),
                        metadata=entry_dict["message"].get("metadata", {}),
                        timestamp=datetime.fromisoformat(entry_dict["message"]["timestamp"]),
                    )
                    result = NotificationResult(
                        success=entry_dict["result"]["success"],
                        status=NotificationStatus(entry_dict["result"]["status"]),
                        message_id=entry_dict["result"].get("message_id"),
                        error=entry_dict["result"].get("error"),
                        timestamp=datetime.fromisoformat(entry_dict["result"]["timestamp"]),
                        retry_count=entry_dict["result"].get("retry_count", 0),
                        metadata=entry_dict["result"].get("metadata", {}),
                    )
                    entry = NotificationHistoryEntry(
                        message=message,
                        result=result,
                        timestamp=datetime.fromisoformat(entry_dict["timestamp"]),
                    )

                    if since and entry.timestamp and entry.timestamp < since:
                        continue

                    entries.append(entry)
                    if len(entries) >= limit:
                        break

                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Failed to parse history entry: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to get history from Redis: {e}")

        return entries

    async def get_statistics(
        self,
        alert_type: AlertType | None = None,
        since: datetime | None = None,
    ) -> dict[str, Any]:
        """Get notification statistics.

        Args:
            alert_type: Filter by alert type
            since: Calculate statistics since this time

        Returns:
            Dictionary with statistics
        """
        if since is None:
            since = datetime.now(UTC) - timedelta(hours=24)

        entries = await self.get_recent(limit=1000, alert_type=alert_type, since=since)

        total = len(entries)
        successful = sum(1 for e in entries if e.result.success)
        failed = total - successful
        retried = sum(1 for e in entries if e.result.retry_count > 0)

        # Calculate average retry count for failed messages
        retry_counts = [e.result.retry_count for e in entries if not e.result.success]
        avg_retries = sum(retry_counts) / len(retry_counts) if retry_counts else 0

        # Group by alert type
        by_type: dict[str, int] = {}
        for entry in entries:
            type_name = entry.message.alert_type.value
            by_type[type_name] = by_type.get(type_name, 0) + 1

        return {
            "total": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0,
            "retried": retried,
            "average_retries": avg_retries,
            "by_alert_type": by_type,
            "time_range": {
                "since": since.isoformat(),
                "until": datetime.now(UTC).isoformat(),
            },
        }

    async def cleanup_old_entries(self) -> int:
        """Clean up old entries from Redis.

        Returns:
            Number of entries cleaned up
        """
        if not self.redis_client:
            return 0

        cleaned = 0
        cutoff_date = datetime.now(UTC) - timedelta(days=self.retention_days)

        try:
            # Find old keys
            pattern = "notifications:history:*"
            keys = await self.redis_client.keys(pattern)

            for key in keys:
                # Parse date from key if it has date format
                key_str = key.decode() if isinstance(key, bytes) else key
                parts = key_str.split(":")
                if len(parts) >= 4 and parts[-1].isdigit() and len(parts[-1]) == 8:
                    date_str = parts[-1]
                    try:
                        key_date = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=UTC)
                        if key_date < cutoff_date:
                            await self.redis_client.delete(key)
                            cleaned += 1
                    except ValueError:
                        continue

        except Exception as e:
            logger.error(f"Failed to cleanup old entries: {e}")

        return cleaned
