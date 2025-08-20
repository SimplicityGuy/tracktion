"""Progress tracking system for analysis pipeline using Redis."""

import json
import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4

import redis
import redis.exceptions

logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Status of file processing."""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class FileProgress:
    """Progress information for a single file."""

    file_path: str
    recording_id: str
    correlation_id: str
    status: ProcessingStatus
    queued_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retry_count: int = 0
    error_message: Optional[str] = None
    progress_percentage: float = 0.0
    current_step: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileProgress":
        """Create from dictionary retrieved from Redis."""
        data["status"] = ProcessingStatus(data["status"])
        return cls(**data)


class ProgressTracker:
    """Track progress of audio file analysis using Redis."""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        key_prefix: str = "analysis:progress",
        ttl_seconds: int = 86400,  # 24 hours default TTL
    ) -> None:
        """Initialize the progress tracker.

        Args:
            redis_host: Redis server hostname
            redis_port: Redis server port
            redis_db: Redis database number
            key_prefix: Prefix for Redis keys
            ttl_seconds: Time-to-live for progress entries in seconds
        """
        self.key_prefix = key_prefix
        self.ttl_seconds = ttl_seconds

        # Initialize Redis connection
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

        # Keys for different tracking purposes
        self.queue_key = f"{key_prefix}:queue"
        self.active_key = f"{key_prefix}:active"
        self.completed_key = f"{key_prefix}:completed"
        self.failed_key = f"{key_prefix}:failed"
        self.stats_key = f"{key_prefix}:stats"

    def track_file_queued(
        self,
        file_path: str,
        recording_id: str,
        correlation_id: Optional[str] = None,
    ) -> str:
        """Track that a file has been queued for processing.

        Args:
            file_path: Path to the audio file
            recording_id: Recording ID for database storage
            correlation_id: Optional correlation ID for tracking

        Returns:
            Correlation ID used for tracking
        """
        if not correlation_id:
            correlation_id = str(uuid4())

        progress = FileProgress(
            file_path=file_path,
            recording_id=recording_id,
            correlation_id=correlation_id,
            status=ProcessingStatus.QUEUED,
            queued_at=time.time(),
        )

        # Store progress data
        progress_key = f"{self.key_prefix}:file:{correlation_id}"
        self.redis_client.setex(
            progress_key,
            self.ttl_seconds,
            json.dumps(progress.to_dict()),
        )

        # Add to queue set
        self.redis_client.sadd(self.queue_key, correlation_id)

        # Update queue depth stat
        self._increment_stat("total_queued")
        queue_depth = self.redis_client.scard(self.queue_key)
        self._update_stat("current_queue_depth", queue_depth)

        logger.debug(f"Tracked file queued: {file_path}", extra={"correlation_id": correlation_id})
        return correlation_id

    def track_file_started(self, correlation_id: str, current_step: Optional[str] = None) -> None:
        """Track that file processing has started.

        Args:
            correlation_id: Correlation ID for the file
            current_step: Optional description of current processing step
        """
        progress_key = f"{self.key_prefix}:file:{correlation_id}"
        progress_data = self.redis_client.get(progress_key)

        if not progress_data:
            logger.warning(f"No progress data found for correlation_id: {correlation_id}")
            return

        progress = FileProgress.from_dict(json.loads(progress_data))
        progress.status = ProcessingStatus.IN_PROGRESS
        progress.started_at = time.time()
        progress.current_step = current_step

        # Update progress data
        self.redis_client.setex(
            progress_key,
            self.ttl_seconds,
            json.dumps(progress.to_dict()),
        )

        # Move from queue to active set
        self.redis_client.srem(self.queue_key, correlation_id)
        self.redis_client.sadd(self.active_key, correlation_id)

        # Update stats
        self._increment_stat("total_started")
        active_count = self.redis_client.scard(self.active_key)
        self._update_stat("current_active", active_count)

        logger.debug(f"Tracked file started: {progress.file_path}", extra={"correlation_id": correlation_id})

    def update_progress(
        self,
        correlation_id: str,
        progress_percentage: float,
        current_step: Optional[str] = None,
    ) -> None:
        """Update progress percentage for a file being processed.

        Args:
            correlation_id: Correlation ID for the file
            progress_percentage: Progress percentage (0-100)
            current_step: Optional description of current processing step
        """
        progress_key = f"{self.key_prefix}:file:{correlation_id}"
        progress_data = self.redis_client.get(progress_key)

        if not progress_data:
            logger.warning(f"No progress data found for correlation_id: {correlation_id}")
            return

        progress = FileProgress.from_dict(json.loads(progress_data))
        progress.progress_percentage = min(100.0, max(0.0, progress_percentage))
        if current_step:
            progress.current_step = current_step

        # Update progress data
        self.redis_client.setex(
            progress_key,
            self.ttl_seconds,
            json.dumps(progress.to_dict()),
        )

        logger.debug(
            f"Updated progress: {progress.file_path} - {progress_percentage:.1f}%",
            extra={"correlation_id": correlation_id},
        )

    def track_file_completed(
        self,
        correlation_id: str,
        success: bool = True,
        error_message: Optional[str] = None,
    ) -> None:
        """Track that file processing has completed.

        Args:
            correlation_id: Correlation ID for the file
            success: Whether processing was successful
            error_message: Optional error message if failed
        """
        progress_key = f"{self.key_prefix}:file:{correlation_id}"
        progress_data = self.redis_client.get(progress_key)

        if not progress_data:
            logger.warning(f"No progress data found for correlation_id: {correlation_id}")
            return

        progress = FileProgress.from_dict(json.loads(progress_data))
        progress.completed_at = time.time()
        progress.progress_percentage = 100.0 if success else progress.progress_percentage

        if success:
            progress.status = ProcessingStatus.COMPLETED
            progress.current_step = "Completed"
        else:
            progress.status = ProcessingStatus.FAILED
            progress.error_message = error_message

        # Update progress data
        self.redis_client.setex(
            progress_key,
            self.ttl_seconds,
            json.dumps(progress.to_dict()),
        )

        # Move from active to completed/failed set
        self.redis_client.srem(self.active_key, correlation_id)
        if success:
            self.redis_client.sadd(self.completed_key, correlation_id)
            self._increment_stat("total_completed")
        else:
            self.redis_client.sadd(self.failed_key, correlation_id)
            self._increment_stat("total_failed")

        # Calculate processing time
        if progress.started_at:
            processing_time = progress.completed_at - progress.started_at
            self._update_stat("total_processing_time", processing_time, increment=True)
            self._increment_stat("processed_file_count")

        logger.info(
            f"File processing {'completed' if success else 'failed'}: {progress.file_path}",
            extra={"correlation_id": correlation_id, "success": success},
        )

    def track_file_retry(self, correlation_id: str) -> None:
        """Track that a file is being retried.

        Args:
            correlation_id: Correlation ID for the file
        """
        progress_key = f"{self.key_prefix}:file:{correlation_id}"
        progress_data = self.redis_client.get(progress_key)

        if not progress_data:
            logger.warning(f"No progress data found for correlation_id: {correlation_id}")
            return

        progress = FileProgress.from_dict(json.loads(progress_data))
        progress.status = ProcessingStatus.RETRYING
        progress.retry_count += 1

        # Update progress data
        self.redis_client.setex(
            progress_key,
            self.ttl_seconds,
            json.dumps(progress.to_dict()),
        )

        # Move back to queue
        self.redis_client.srem(self.failed_key, correlation_id)
        self.redis_client.sadd(self.queue_key, correlation_id)

        self._increment_stat("total_retries")

        logger.info(
            f"File queued for retry (attempt {progress.retry_count}): {progress.file_path}",
            extra={"correlation_id": correlation_id},
        )

    def get_progress(self, correlation_id: str) -> Optional[FileProgress]:
        """Get progress information for a specific file.

        Args:
            correlation_id: Correlation ID for the file

        Returns:
            FileProgress object or None if not found
        """
        progress_key = f"{self.key_prefix}:file:{correlation_id}"
        progress_data = self.redis_client.get(progress_key)

        if not progress_data:
            return None

        return FileProgress.from_dict(json.loads(progress_data))

    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status and statistics.

        Returns:
            Dictionary with queue statistics
        """
        stats = self.get_statistics()

        # Get sample of items from each queue
        queued_ids = list(self.redis_client.smembers(self.queue_key))[:10]
        active_ids = list(self.redis_client.smembers(self.active_key))[:10]
        completed_ids = list(self.redis_client.smembers(self.completed_key))[:10]
        failed_ids = list(self.redis_client.smembers(self.failed_key))[:10]

        return {
            "statistics": stats,
            "queue_depth": self.redis_client.scard(self.queue_key),
            "active_count": self.redis_client.scard(self.active_key),
            "completed_count": self.redis_client.scard(self.completed_key),
            "failed_count": self.redis_client.scard(self.failed_key),
            "recent_queued": [self.get_progress(cid) for cid in queued_ids if self.get_progress(cid)],
            "recent_active": [self.get_progress(cid) for cid in active_ids if self.get_progress(cid)],
            "recent_completed": [self.get_progress(cid) for cid in completed_ids if self.get_progress(cid)],
            "recent_failed": [self.get_progress(cid) for cid in failed_ids if self.get_progress(cid)],
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics.

        Returns:
            Dictionary with various statistics
        """
        stats_data = self.redis_client.hgetall(self.stats_key)

        # Convert string values to appropriate types
        stats = {}
        for key, value in stats_data.items():
            try:
                # Try to convert to float first (handles both int and float)
                stats[key] = float(value)
                # Convert to int if it's a whole number
                if stats[key].is_integer():
                    stats[key] = int(stats[key])
            except (ValueError, AttributeError):
                stats[key] = value

        # Calculate average processing time
        if stats.get("processed_file_count", 0) > 0:
            total_time = stats.get("total_processing_time", 0)
            file_count = stats["processed_file_count"]
            stats["average_processing_time"] = total_time / file_count

        # Calculate success rate
        completed = stats.get("total_completed", 0)
        failed = stats.get("total_failed", 0)
        total_processed = completed + failed
        if total_processed > 0:
            stats["success_rate"] = (completed / total_processed) * 100

        return stats

    def clear_old_entries(self, older_than_hours: int = 24) -> int:
        """Clear progress entries older than specified hours.

        Args:
            older_than_hours: Clear entries older than this many hours

        Returns:
            Number of entries cleared
        """
        cutoff_time = time.time() - (older_than_hours * 3600)
        cleared_count = 0

        # Get all correlation IDs from all sets
        all_ids = set()
        all_ids.update(self.redis_client.smembers(self.queue_key))
        all_ids.update(self.redis_client.smembers(self.active_key))
        all_ids.update(self.redis_client.smembers(self.completed_key))
        all_ids.update(self.redis_client.smembers(self.failed_key))

        for correlation_id in all_ids:
            progress_key = f"{self.key_prefix}:file:{correlation_id}"
            progress_data = self.redis_client.get(progress_key)

            if progress_data:
                progress = FileProgress.from_dict(json.loads(progress_data))
                # Check if entry is old
                check_time = progress.completed_at or progress.started_at or progress.queued_at
                if check_time < cutoff_time:
                    # Remove from all sets
                    self.redis_client.srem(self.queue_key, correlation_id)
                    self.redis_client.srem(self.active_key, correlation_id)
                    self.redis_client.srem(self.completed_key, correlation_id)
                    self.redis_client.srem(self.failed_key, correlation_id)
                    # Delete the progress data
                    self.redis_client.delete(progress_key)
                    cleared_count += 1

        logger.info(f"Cleared {cleared_count} old progress entries")
        return cleared_count

    def _increment_stat(self, stat_name: str, amount: int = 1) -> None:
        """Increment a statistic counter.

        Args:
            stat_name: Name of the statistic
            amount: Amount to increment by
        """
        self.redis_client.hincrby(self.stats_key, stat_name, amount)

    def _update_stat(self, stat_name: str, value: Any, increment: bool = False) -> None:
        """Update a statistic value.

        Args:
            stat_name: Name of the statistic
            value: New value for the statistic
            increment: If True, add to existing value instead of replacing
        """
        if increment:
            self.redis_client.hincrbyfloat(self.stats_key, stat_name, value)
        else:
            self.redis_client.hset(self.stats_key, stat_name, value)

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self.redis_client.delete(self.stats_key)
        logger.info("Statistics reset")
