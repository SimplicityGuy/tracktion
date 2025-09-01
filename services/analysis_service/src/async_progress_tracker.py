"""
Async progress tracking and event emission for audio analysis.

This module provides real-time progress tracking with WebSocket support
for streaming updates to clients.
"""

import asyncio
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

try:
    # Try newer redis with async support
    import redis.asyncio as aioredis
except ImportError:
    try:
        # Fallback to aioredis if available
        import aioredis
    except ImportError:
        # No redis support available
        aioredis = None
from aiohttp import web

logger = logging.getLogger(__name__)


class ProgressEventType(Enum):
    """Types of progress events."""

    STARTED = "started"
    PROGRESS = "progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    STAGE_CHANGE = "stage_change"


@dataclass
class ProgressEvent:
    """Progress event data structure."""

    task_id: str
    event_type: ProgressEventType
    timestamp: float
    progress_percent: float = 0.0
    stage: str | None = None
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskProgress:
    """Track progress for a single task."""

    task_id: str
    total_stages: int
    current_stage: int = 0
    stage_name: str = ""
    stage_progress: float = 0.0
    overall_progress: float = 0.0
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class AsyncProgressTracker:
    """
    Async progress tracker with event emission and WebSocket support.
    """

    def __init__(
        self,
        redis_url: str | None = None,
        enable_websocket: bool = True,
        update_interval_seconds: float = 1.0,
        batch_aggregation: bool = True,
    ):
        """
        Initialize the progress tracker.

        Args:
            redis_url: Redis URL for state storage
            enable_websocket: Enable WebSocket support
            update_interval_seconds: Minimum interval between updates
            batch_aggregation: Enable batch progress aggregation
        """
        self.redis_url = redis_url
        self.enable_websocket = enable_websocket
        self.update_interval = update_interval_seconds
        self.batch_aggregation = batch_aggregation

        # Progress tracking
        self.active_tasks: dict[str, TaskProgress] = {}
        self.completed_tasks: set[str] = set()
        self.failed_tasks: set[str] = set()

        # Event listeners
        self.listeners: list[Callable[[ProgressEvent], None]] = []
        self.websocket_connections: set[web.WebSocketResponse] = set()

        # Redis connection
        self.redis: aioredis.Redis | None = None

        # Update throttling
        self.last_update_times: dict[str, float] = {}

        logger.info("AsyncProgressTracker initialized")

    async def initialize(self) -> None:
        """Initialize Redis connection and other async resources."""
        if self.redis_url and aioredis:
            try:
                self.redis = await aioredis.from_url(self.redis_url, encoding="utf-8", decode_responses=True)
                logger.info("Connected to Redis for progress state storage")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e!s}")
                self.redis = None
        elif self.redis_url and not aioredis:
            logger.warning("Redis URL provided but redis library not available")
            self.redis = None

    async def start_task(
        self,
        task_id: str,
        total_stages: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Start tracking a new task.

        Args:
            task_id: Unique task identifier
            total_stages: Total number of stages in the task
            metadata: Optional task metadata
        """
        progress = TaskProgress(
            task_id=task_id,
            total_stages=total_stages,
            metadata=metadata or {},
        )
        self.active_tasks[task_id] = progress

        # Emit start event
        await self._emit_event(
            ProgressEvent(
                task_id=task_id,
                event_type=ProgressEventType.STARTED,
                timestamp=time.time(),
                progress_percent=0.0,
                metadata=metadata or {},
            )
        )

        # Store in Redis if available
        if self.redis:
            await self._store_progress_redis(task_id, progress)

        logger.debug(f"Started tracking task {task_id} with {total_stages} stages")

    async def update_progress(
        self,
        task_id: str,
        stage: str | None = None,
        stage_progress: float | None = None,
        message: str | None = None,
    ) -> None:
        """
        Update progress for a task.

        Args:
            task_id: Task identifier
            stage: Current stage name
            stage_progress: Progress within current stage (0-100)
            message: Optional progress message
        """
        if task_id not in self.active_tasks:
            logger.warning(f"Attempted to update unknown task: {task_id}")
            return

        # Check throttling
        if not self._should_update(task_id):
            return

        progress = self.active_tasks[task_id]

        # Update stage if changed
        if stage and stage != progress.stage_name:
            progress.stage_name = stage
            progress.current_stage += 1
            progress.stage_progress = 0.0

            await self._emit_event(
                ProgressEvent(
                    task_id=task_id,
                    event_type=ProgressEventType.STAGE_CHANGE,
                    timestamp=time.time(),
                    progress_percent=self._calculate_overall_progress(progress),
                    stage=stage,
                    message=message,
                )
            )

        # Update stage progress
        if stage_progress is not None:
            progress.stage_progress = max(0.0, min(100.0, stage_progress))

        # Calculate overall progress
        progress.overall_progress = self._calculate_overall_progress(progress)
        progress.last_update = time.time()

        # Emit progress event
        await self._emit_event(
            ProgressEvent(
                task_id=task_id,
                event_type=ProgressEventType.PROGRESS,
                timestamp=time.time(),
                progress_percent=progress.overall_progress,
                stage=progress.stage_name,
                message=message,
            )
        )

        # Update Redis
        if self.redis:
            await self._store_progress_redis(task_id, progress)

    async def complete_task(self, task_id: str, message: str | None = None) -> None:
        """
        Mark a task as completed.

        Args:
            task_id: Task identifier
            message: Optional completion message
        """
        if task_id not in self.active_tasks:
            logger.warning(f"Attempted to complete unknown task: {task_id}")
            return

        progress = self.active_tasks.pop(task_id)
        self.completed_tasks.add(task_id)

        # Calculate duration
        duration = time.time() - progress.start_time

        # Emit completion event
        await self._emit_event(
            ProgressEvent(
                task_id=task_id,
                event_type=ProgressEventType.COMPLETED,
                timestamp=time.time(),
                progress_percent=100.0,
                message=message,
                metadata={"duration_seconds": duration},
            )
        )

        # Clean up Redis
        if self.redis:
            await self.redis.delete(f"progress:{task_id}")

        logger.debug(f"Task {task_id} completed in {duration:.2f} seconds")

    async def fail_task(self, task_id: str, error: str, message: str | None = None) -> None:
        """
        Mark a task as failed.

        Args:
            task_id: Task identifier
            error: Error description
            message: Optional failure message
        """
        if task_id in self.active_tasks:
            self.active_tasks.pop(task_id)

        self.failed_tasks.add(task_id)

        # Emit failure event
        await self._emit_event(
            ProgressEvent(
                task_id=task_id,
                event_type=ProgressEventType.FAILED,
                timestamp=time.time(),
                message=message or error,
                metadata={"error": error},
            )
        )

        # Clean up Redis
        if self.redis:
            await self.redis.delete(f"progress:{task_id}")

        logger.error(f"Task {task_id} failed: {error}")

    def _calculate_overall_progress(self, progress: TaskProgress) -> float:
        """
        Calculate overall progress percentage.

        Args:
            progress: TaskProgress object

        Returns:
            Overall progress percentage (0-100)
        """
        if progress.total_stages == 0:
            return 0.0

        # Calculate based on completed stages + current stage progress
        completed_stages = max(0, progress.current_stage - 1)
        stage_contribution = 100.0 / progress.total_stages

        overall = (completed_stages * stage_contribution) + ((progress.stage_progress / 100.0) * stage_contribution)

        return min(100.0, max(0.0, overall))

    def _should_update(self, task_id: str) -> bool:
        """
        Check if update should be sent based on throttling.

        Args:
            task_id: Task identifier

        Returns:
            True if update should be sent
        """
        now = time.time()
        last_update = self.last_update_times.get(task_id, 0)

        if now - last_update >= self.update_interval:
            self.last_update_times[task_id] = now
            return True

        return False

    async def _emit_event(self, event: ProgressEvent) -> None:
        """
        Emit a progress event to all listeners.

        Args:
            event: Progress event to emit
        """
        # Call registered listeners
        for listener in self.listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(event)
                else:
                    listener(event)
            except Exception as e:
                logger.error(f"Error in progress listener: {e!s}")

        # Send to WebSocket connections
        if self.enable_websocket:
            await self._broadcast_websocket(event)

    async def _broadcast_websocket(self, event: ProgressEvent) -> None:
        """
        Broadcast event to all WebSocket connections.

        Args:
            event: Event to broadcast
        """
        if not self.websocket_connections:
            return

        # Serialize event
        event_data = {
            "task_id": event.task_id,
            "type": event.event_type.value,
            "timestamp": event.timestamp,
            "progress": event.progress_percent,
            "stage": event.stage,
            "message": event.message,
            "metadata": event.metadata,
        }

        message = json.dumps(event_data)

        # Send to all connections
        dead_connections = set()
        for ws in self.websocket_connections:
            try:
                await ws.send_str(message)
            except ConnectionResetError:
                dead_connections.add(ws)
            except Exception as e:
                logger.error(f"Error broadcasting to WebSocket: {e!s}")
                dead_connections.add(ws)

        # Remove dead connections
        self.websocket_connections -= dead_connections

    async def _store_progress_redis(self, task_id: str, progress: TaskProgress) -> None:
        """
        Store progress in Redis.

        Args:
            task_id: Task identifier
            progress: Progress data
        """
        if not self.redis:
            return

        try:
            key = f"progress:{task_id}"
            data = {
                "task_id": str(progress.task_id),
                "total_stages": int(progress.total_stages),
                "current_stage": int(progress.current_stage),
                "stage_name": str(progress.stage_name),
                "stage_progress": float(progress.stage_progress),
                "overall_progress": float(progress.overall_progress),
                "start_time": float(progress.start_time),
                "last_update": float(progress.last_update),
                "metadata": json.dumps(progress.metadata),
            }

            # Handle async redis clients
            hset_result = self.redis.hset(key, mapping=data)
            if hasattr(hset_result, "__await__"):
                await hset_result

            # Handle both async and sync redis clients for expire
            expire_result = self.redis.expire(key, 3600)
            if hasattr(expire_result, "__await__"):
                await expire_result
            # Note: expire() can return int or bool depending on redis client version

        except Exception as e:
            logger.error(f"Failed to store progress in Redis: {e!s}")

    def add_listener(self, listener: Callable[[ProgressEvent], None]) -> None:
        """
        Add a progress event listener.

        Args:
            listener: Callback function for events
        """
        self.listeners.append(listener)

    def remove_listener(self, listener: Callable[[ProgressEvent], None]) -> None:
        """
        Remove a progress event listener.

        Args:
            listener: Callback function to remove
        """
        if listener in self.listeners:
            self.listeners.remove(listener)

    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """
        Handle WebSocket connection for progress updates.

        Args:
            request: aiohttp request

        Returns:
            WebSocket response
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        # Add to connections
        self.websocket_connections.add(ws)

        try:
            # Send current state
            for task_id, progress in self.active_tasks.items():
                await ws.send_str(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "type": "state",
                            "progress": progress.overall_progress,
                            "stage": progress.stage_name,
                        }
                    )
                )

            # Keep connection alive
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    # Handle client messages if needed
                    pass
                elif msg.type == web.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")

        finally:
            # Remove from connections
            self.websocket_connections.discard(ws)

        return ws


class BatchProgressAggregator:
    """
    Aggregate progress for batch operations.
    """

    def __init__(self, tracker: AsyncProgressTracker):
        """
        Initialize the aggregator.

        Args:
            tracker: AsyncProgressTracker instance
        """
        self.tracker = tracker
        self.batch_tasks: dict[str, list[str]] = {}
        self.batch_progress: dict[str, dict[str, float]] = {}

    async def start_batch(self, batch_id: str, task_ids: list[str], metadata: dict | None = None) -> None:
        """
        Start tracking a batch of tasks.

        Args:
            batch_id: Batch identifier
            task_ids: List of task IDs in the batch
            metadata: Optional batch metadata
        """
        self.batch_tasks[batch_id] = task_ids
        self.batch_progress[batch_id] = dict.fromkeys(task_ids, 0.0)

        # Start batch tracking
        await self.tracker.start_task(batch_id, total_stages=len(task_ids), metadata=metadata)

        logger.debug(f"Started batch {batch_id} with {len(task_ids)} tasks")

    async def update_batch_task(self, batch_id: str, task_id: str, progress: float) -> None:
        """
        Update progress for a task in a batch.

        Args:
            batch_id: Batch identifier
            task_id: Task identifier
            progress: Task progress (0-100)
        """
        if batch_id not in self.batch_tasks:
            return

        if task_id not in self.batch_progress[batch_id]:
            return

        # Update individual task progress
        self.batch_progress[batch_id][task_id] = progress

        # Calculate overall batch progress
        task_count = len(self.batch_tasks[batch_id])
        total_progress = sum(self.batch_progress[batch_id].values())
        overall_progress = total_progress / task_count if task_count > 0 else 0

        # Update batch progress
        await self.tracker.update_progress(
            batch_id,
            stage=f"Processing {task_id}",
            stage_progress=overall_progress,
            message=f"Task {task_id}: {progress:.1f}%",
        )

    async def complete_batch(self, batch_id: str) -> None:
        """
        Mark a batch as completed.

        Args:
            batch_id: Batch identifier
        """
        if batch_id in self.batch_tasks:
            await self.tracker.complete_task(
                batch_id,
                message=f"Batch completed: {len(self.batch_tasks[batch_id])} tasks",
            )
            del self.batch_tasks[batch_id]
            del self.batch_progress[batch_id]
