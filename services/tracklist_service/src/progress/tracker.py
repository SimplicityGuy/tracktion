"""Real-time progress tracking for batch operations."""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, UTC, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable
from collections import defaultdict
import statistics

from redis import Redis
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass
class JobProgress:
    """Progress information for a single job."""

    job_id: str
    batch_id: str
    url: str
    status: JobStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time: Optional[float] = None
    retry_count: int = 0
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        if self.started_at:
            data["started_at"] = self.started_at.isoformat()
        if self.completed_at:
            data["completed_at"] = self.completed_at.isoformat()
        return data


@dataclass
class BatchProgress:
    """Progress information for a batch."""

    batch_id: str
    total_jobs: int
    pending: int = 0
    processing: int = 0
    completed: int = 0
    failed: int = 0
    retrying: int = 0
    cancelled: int = 0
    start_time: Optional[datetime] = None
    last_update: Optional[datetime] = None
    estimated_completion: Optional[datetime] = None

    @property
    def progress_percentage(self) -> float:
        """Calculate completion percentage."""
        if self.total_jobs == 0:
            return 0.0
        finished = self.completed + self.failed + self.cancelled
        return (finished / self.total_jobs) * 100

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        finished = self.completed + self.failed
        if finished == 0:
            return 0.0
        return (self.completed / finished) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["progress_percentage"] = self.progress_percentage
        data["success_rate"] = self.success_rate
        if self.start_time:
            data["start_time"] = self.start_time.isoformat()
        if self.last_update:
            data["last_update"] = self.last_update.isoformat()
        if self.estimated_completion:
            data["estimated_completion"] = self.estimated_completion.isoformat()
        return data


class ProgressTracker:
    """Tracks and broadcasts batch processing progress."""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        persistence_ttl: int = 86400,  # 24 hours
    ):
        """Initialize progress tracker.

        Args:
            redis_host: Redis host address
            redis_port: Redis port number
            persistence_ttl: TTL for progress data in seconds
        """
        self.redis = Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.persistence_ttl = persistence_ttl

        # In-memory tracking for performance
        self.job_progress: Dict[str, JobProgress] = {}
        self.batch_progress: Dict[str, BatchProgress] = {}
        self.batch_jobs: Dict[str, Set[str]] = defaultdict(set)

        # WebSocket connections for real-time updates
        self.websocket_connections: Dict[str, List[WebSocket]] = defaultdict(list)

        # Processing time statistics for ETA calculation
        self.processing_times: List[float] = []
        self.max_processing_samples = 100

        # Notification callbacks
        self.completion_callbacks: Dict[str, List[Callable]] = defaultdict(list)

    async def update_progress(
        self,
        batch_id: str,
        job_id: str,
        status: str,
        url: Optional[str] = None,
        error: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update job progress.

        Args:
            batch_id: Batch identifier
            job_id: Job identifier
            status: New job status
            url: Job URL
            error: Error message if failed
            result: Job result if completed
        """
        job_status = JobStatus(status)
        now = datetime.now(UTC)

        # Get or create job progress
        if job_id not in self.job_progress:
            self.job_progress[job_id] = JobProgress(job_id=job_id, batch_id=batch_id, url=url or "", status=job_status)
            self.batch_jobs[batch_id].add(job_id)

        job = self.job_progress[job_id]
        old_status = job.status
        job.status = job_status

        # Update timestamps
        if job_status == JobStatus.PROCESSING and not job.started_at:
            job.started_at = now
        elif job_status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            job.completed_at = now
            if job.started_at:
                job.processing_time = (now - job.started_at).total_seconds()
                self._record_processing_time(job.processing_time)
        elif job_status == JobStatus.RETRYING:
            job.retry_count += 1

        # Store error or result
        if error:
            job.error = error
        if result:
            job.result = result

        # Update batch progress
        await self._update_batch_progress(batch_id, old_status, job_status)

        # Persist to Redis
        await self._persist_job_progress(job)

        # Broadcast update
        await self.broadcast_update(
            batch_id,
            {
                "type": "job_update",
                "job_id": job_id,
                "status": status,
                "batch_progress": self.batch_progress.get(batch_id, BatchProgress(batch_id, 0)).to_dict(),
            },
        )

        # Check for batch completion
        await self._check_batch_completion(batch_id)

    async def _update_batch_progress(self, batch_id: str, old_status: JobStatus, new_status: JobStatus) -> None:
        """Update batch-level progress.

        Args:
            batch_id: Batch identifier
            old_status: Previous job status
            new_status: New job status
        """
        if batch_id not in self.batch_progress:
            total_jobs = len(self.batch_jobs.get(batch_id, []))
            self.batch_progress[batch_id] = BatchProgress(
                batch_id=batch_id, total_jobs=total_jobs, pending=total_jobs, start_time=datetime.now(UTC)
            )

        batch = self.batch_progress[batch_id]

        # Decrement old status counter
        if old_status == JobStatus.PENDING:
            batch.pending = max(0, batch.pending - 1)
        elif old_status == JobStatus.PROCESSING:
            batch.processing = max(0, batch.processing - 1)
        elif old_status == JobStatus.RETRYING:
            batch.retrying = max(0, batch.retrying - 1)

        # Increment new status counter
        if new_status == JobStatus.PENDING:
            batch.pending += 1
        elif new_status == JobStatus.PROCESSING:
            batch.processing += 1
        elif new_status == JobStatus.COMPLETED:
            batch.completed += 1
        elif new_status == JobStatus.FAILED:
            batch.failed += 1
        elif new_status == JobStatus.RETRYING:
            batch.retrying += 1
        elif new_status == JobStatus.CANCELLED:
            batch.cancelled += 1

        batch.last_update = datetime.now(UTC)

        # Update ETA
        batch.estimated_completion = await self.calculate_eta(batch_id)

        # Persist to Redis
        await self._persist_batch_progress(batch)

    async def calculate_eta(self, batch_id: str) -> Optional[datetime]:
        """Calculate estimated time of completion.

        Args:
            batch_id: Batch identifier

        Returns:
            Estimated completion time
        """
        if batch_id not in self.batch_progress:
            return None

        batch = self.batch_progress[batch_id]
        remaining = batch.pending + batch.processing + batch.retrying

        if remaining == 0:
            return datetime.now(UTC)

        # Calculate average processing time
        if not self.processing_times:
            # No data yet, estimate based on job count
            estimated_seconds = float(remaining * 6)  # Default 6 seconds per job
        else:
            avg_time = statistics.mean(self.processing_times)
            estimated_seconds = remaining * avg_time

        return datetime.now(UTC) + timedelta(seconds=estimated_seconds)

    async def broadcast_update(self, batch_id: str, update: Dict[str, Any]) -> None:
        """Broadcast progress update to WebSocket connections.

        Args:
            batch_id: Batch identifier
            update: Update data to broadcast
        """
        connections = self.websocket_connections.get(batch_id, [])
        if not connections:
            return

        update["timestamp"] = datetime.now(UTC).isoformat()
        message = json.dumps(update)

        # Send to all connected clients
        disconnected = []
        for websocket in connections:
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send update to websocket: {e}")
                disconnected.append(websocket)

        # Remove disconnected clients
        for websocket in disconnected:
            self.websocket_connections[batch_id].remove(websocket)

    def add_websocket(self, batch_id: str, websocket: WebSocket) -> None:
        """Add WebSocket connection for batch updates.

        Args:
            batch_id: Batch identifier
            websocket: WebSocket connection
        """
        self.websocket_connections[batch_id].append(websocket)
        logger.info(f"WebSocket connected for batch {batch_id}")

    def remove_websocket(self, batch_id: str, websocket: WebSocket) -> None:
        """Remove WebSocket connection.

        Args:
            batch_id: Batch identifier
            websocket: WebSocket connection
        """
        if batch_id in self.websocket_connections:
            if websocket in self.websocket_connections[batch_id]:
                self.websocket_connections[batch_id].remove(websocket)
                logger.info(f"WebSocket disconnected for batch {batch_id}")

    def register_completion_callback(self, batch_id: str, callback: Callable[[str, BatchProgress], None]) -> None:
        """Register callback for batch completion.

        Args:
            batch_id: Batch identifier
            callback: Callback function
        """
        self.completion_callbacks[batch_id].append(callback)

    async def _check_batch_completion(self, batch_id: str) -> None:
        """Check if batch is complete and trigger callbacks.

        Args:
            batch_id: Batch identifier
        """
        if batch_id not in self.batch_progress:
            return

        batch = self.batch_progress[batch_id]

        # Check if all jobs are finished
        finished = batch.completed + batch.failed + batch.cancelled
        if finished >= batch.total_jobs:
            # Batch is complete
            logger.info(
                f"Batch {batch_id} complete: "
                f"{batch.completed} succeeded, "
                f"{batch.failed} failed, "
                f"{batch.cancelled} cancelled"
            )

            # Trigger callbacks
            for callback in self.completion_callbacks.get(batch_id, []):
                try:
                    await callback(batch_id, batch)
                except Exception as e:
                    logger.error(f"Completion callback failed: {e}")

            # Send completion notification
            await self.broadcast_update(
                batch_id, {"type": "batch_complete", "batch_id": batch_id, "batch_progress": batch.to_dict()}
            )

    def _record_processing_time(self, processing_time: float) -> None:
        """Record processing time for ETA calculation.

        Args:
            processing_time: Job processing time in seconds
        """
        self.processing_times.append(processing_time)

        # Keep only recent samples
        if len(self.processing_times) > self.max_processing_samples:
            self.processing_times = self.processing_times[-self.max_processing_samples :]

    async def _persist_job_progress(self, job: JobProgress) -> None:
        """Persist job progress to Redis.

        Args:
            job: Job progress data
        """
        key = f"job_progress:{job.job_id}"
        self.redis.hset(key, mapping=job.to_dict())
        self.redis.expire(key, self.persistence_ttl)

    async def _persist_batch_progress(self, batch: BatchProgress) -> None:
        """Persist batch progress to Redis.

        Args:
            batch: Batch progress data
        """
        key = f"batch_progress:{batch.batch_id}"
        self.redis.hset(key, mapping=batch.to_dict())
        self.redis.expire(key, self.persistence_ttl)

    def get_job_progress(self, job_id: str) -> Optional[JobProgress]:
        """Get job progress.

        Args:
            job_id: Job identifier

        Returns:
            Job progress or None
        """
        # Check memory first
        if job_id in self.job_progress:
            return self.job_progress[job_id]

        # Check Redis
        key = f"job_progress:{job_id}"
        data = self.redis.hgetall(key)  # type: ignore[misc]
        if data:
            # Convert back to JobProgress
            job = JobProgress(
                job_id=data["job_id"],  # type: ignore[index]
                batch_id=data["batch_id"],  # type: ignore[index]
                url=data["url"],  # type: ignore[index]
                status=JobStatus(data["status"]),  # type: ignore[index]
                retry_count=int(data.get("retry_count", 0)),  # type: ignore[union-attr]
            )
            if data.get("started_at"):  # type: ignore[union-attr]
                job.started_at = datetime.fromisoformat(data["started_at"])  # type: ignore[index]
            if data.get("completed_at"):  # type: ignore[union-attr]
                job.completed_at = datetime.fromisoformat(data["completed_at"])  # type: ignore[index]
            if data.get("processing_time"):  # type: ignore[union-attr]
                job.processing_time = float(data["processing_time"])  # type: ignore[index]
            if data.get("error"):  # type: ignore[union-attr]
                job.error = data["error"]  # type: ignore[index]
            if data.get("result"):  # type: ignore[union-attr]
                job.result = json.loads(data["result"])  # type: ignore[index]

            # Cache in memory
            self.job_progress[job_id] = job
            return job

        return None

    def get_batch_progress(self, batch_id: str) -> Optional[BatchProgress]:
        """Get batch progress.

        Args:
            batch_id: Batch identifier

        Returns:
            Batch progress or None
        """
        # Check memory first
        if batch_id in self.batch_progress:
            return self.batch_progress[batch_id]

        # Check Redis
        key = f"batch_progress:{batch_id}"
        data = self.redis.hgetall(key)  # type: ignore[misc]
        if data:
            # Convert back to BatchProgress
            batch = BatchProgress(
                batch_id=data["batch_id"],  # type: ignore[index]
                total_jobs=int(data["total_jobs"]),  # type: ignore[index]
                pending=int(data.get("pending", 0)),  # type: ignore[union-attr]
                processing=int(data.get("processing", 0)),  # type: ignore[union-attr]
                completed=int(data.get("completed", 0)),  # type: ignore[union-attr]
                failed=int(data.get("failed", 0)),  # type: ignore[union-attr]
                retrying=int(data.get("retrying", 0)),  # type: ignore[union-attr]
                cancelled=int(data.get("cancelled", 0)),  # type: ignore[union-attr]
            )
            if data.get("start_time"):  # type: ignore[union-attr]
                batch.start_time = datetime.fromisoformat(data["start_time"])  # type: ignore[index]
            if data.get("last_update"):  # type: ignore[union-attr]
                batch.last_update = datetime.fromisoformat(data["last_update"])  # type: ignore[index]
            if data.get("estimated_completion"):  # type: ignore[union-attr]
                batch.estimated_completion = datetime.fromisoformat(data["estimated_completion"])  # type: ignore[index]

            # Cache in memory
            self.batch_progress[batch_id] = batch
            return batch

        return None

    def get_batch_jobs(self, batch_id: str) -> List[JobProgress]:
        """Get all jobs in a batch.

        Args:
            batch_id: Batch identifier

        Returns:
            List of job progress objects
        """
        jobs = []

        # Get job IDs for batch
        job_ids = self.batch_jobs.get(batch_id, set())

        # If not in memory, check Redis
        if not job_ids:
            pattern = "job_progress:*"
            for key in self.redis.scan_iter(match=pattern):
                data = self.redis.hgetall(key)  # type: ignore[misc]
                if data and data.get("batch_id") == batch_id:  # type: ignore[union-attr]
                    job_id = data["job_id"]  # type: ignore[index]
                    job_ids.add(job_id)
                    self.batch_jobs[batch_id].add(job_id)

        # Get job progress for each ID
        for job_id in job_ids:
            job = self.get_job_progress(job_id)
            if job:
                jobs.append(job)

        return jobs

    def clear_batch(self, batch_id: str) -> None:
        """Clear batch data from memory and Redis.

        Args:
            batch_id: Batch identifier
        """
        # Clear from memory
        if batch_id in self.batch_progress:
            del self.batch_progress[batch_id]

        job_ids = self.batch_jobs.get(batch_id, set())
        for job_id in job_ids:
            if job_id in self.job_progress:
                del self.job_progress[job_id]

        if batch_id in self.batch_jobs:
            del self.batch_jobs[batch_id]

        # Clear from Redis
        self.redis.delete(f"batch_progress:{batch_id}")
        for job_id in job_ids:
            self.redis.delete(f"job_progress:{job_id}")

        # Clear callbacks
        if batch_id in self.completion_callbacks:
            del self.completion_callbacks[batch_id]

        logger.info(f"Cleared batch {batch_id} data")
