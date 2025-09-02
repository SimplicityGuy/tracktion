"""Batch synchronization service for processing multiple tracklists."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

import psutil
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from services.tracklist_service.src.messaging.sync_event_publisher import SyncEventPublisher
from services.tracklist_service.src.models.synchronization import SyncConfiguration, SyncEvent
from services.tracklist_service.src.services.sync_service import SynchronizationService, SyncSource, SyncStatus

logger = logging.getLogger(__name__)


class BatchStrategy(Enum):
    """Batch processing strategies."""

    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    ADAPTIVE = "adaptive"
    PRIORITY_BASED = "priority_based"


class BatchStatus(Enum):
    """Batch operation status."""

    PENDING = "pending"
    PROCESSING = "processing"
    PARTIAL_SUCCESS = "partial_success"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchProgress:
    """Track batch operation progress."""

    total: int = 0
    completed: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    in_progress: set[UUID] = field(default_factory=set)
    errors: list[dict[str, Any]] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None

    @property
    def progress_percentage(self) -> int:
        """Calculate progress percentage."""
        if self.total == 0:
            return 0
        return int((self.completed / self.total) * 100)

    @property
    def duration(self) -> timedelta | None:
        """Calculate operation duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        if self.start_time:
            return datetime.now(UTC) - self.start_time
        return None


@dataclass
class BatchResult:
    """Result of a batch operation."""

    batch_id: UUID
    status: BatchStatus
    progress: BatchProgress
    results: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


class BatchSyncService:
    """Service for batch synchronization operations."""

    def __init__(
        self,
        session: AsyncSession,
        sync_service: SynchronizationService | None = None,
        event_publisher: SyncEventPublisher | None = None,
    ):
        """Initialize batch sync service.

        Args:
            session: Database session
            sync_service: Synchronization service
            event_publisher: Event publisher for notifications
        """
        self.session = session
        self.sync_service = sync_service or SynchronizationService(session)
        self.event_publisher = event_publisher

        # Track active batch operations
        self.active_batches: dict[UUID, BatchProgress] = {}

        # Concurrency control
        self.max_parallel = 10
        self.adaptive_threshold = 0.8  # CPU/memory threshold for adaptive strategy

    async def batch_sync_tracklists(
        self,
        tracklist_ids: list[UUID],
        source: SyncSource = SyncSource.ALL,
        strategy: BatchStrategy = BatchStrategy.ADAPTIVE,
        max_parallel: int | None = None,
        continue_on_error: bool = True,
        priority_order: list[UUID] | None = None,
        actor: str = "system",
    ) -> BatchResult:
        """Perform batch synchronization of multiple tracklists.

        Args:
            tracklist_ids: List of tracklist IDs to sync
            source: Sync source
            strategy: Batch processing strategy
            max_parallel: Maximum parallel operations
            continue_on_error: Continue if some operations fail
            priority_order: Priority order for processing
            actor: Who triggered the batch sync

        Returns:
            Batch operation result
        """
        batch_id = uuid4()
        progress = BatchProgress(
            total=len(tracklist_ids),
            start_time=datetime.now(UTC),
        )
        self.active_batches[batch_id] = progress

        try:
            # Publish batch start event
            if self.event_publisher:
                await self.event_publisher.publish_batch_sync(
                    tracklist_ids=tracklist_ids,
                    source=source.value,
                    operation="sync",
                    actor=actor,
                    metadata={"strategy": strategy.value},
                )

            # Apply priority ordering if specified
            if priority_order:
                tracklist_ids = self._apply_priority_order(tracklist_ids, priority_order)

            # Execute based on strategy
            if strategy == BatchStrategy.PARALLEL:
                results = await self._process_parallel(
                    tracklist_ids,
                    source,
                    max_parallel or self.max_parallel,
                    continue_on_error,
                    progress,
                    actor,
                )
            elif strategy == BatchStrategy.SEQUENTIAL:
                results = await self._process_sequential(
                    tracklist_ids,
                    source,
                    continue_on_error,
                    progress,
                    actor,
                )
            elif strategy == BatchStrategy.ADAPTIVE:
                results = await self._process_adaptive(
                    tracklist_ids,
                    source,
                    continue_on_error,
                    progress,
                    actor,
                )
            elif strategy == BatchStrategy.PRIORITY_BASED:
                results = await self._process_priority_based(
                    tracklist_ids,
                    source,
                    continue_on_error,
                    progress,
                    actor,
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # Update final progress
            progress.end_time = datetime.now(UTC)

            # Determine final status
            if progress.failed == 0:
                status = BatchStatus.COMPLETED
            elif progress.successful > 0:
                status = BatchStatus.PARTIAL_SUCCESS
            else:
                status = BatchStatus.FAILED

            # Create batch result
            batch_result = BatchResult(
                batch_id=batch_id,
                status=status,
                progress=progress,
                results=results,
                metadata={
                    "strategy": strategy.value,
                    "source": source.value,
                    "actor": actor,
                },
            )

            # Log summary
            logger.info(
                f"Batch sync completed: {progress.successful}/{progress.total} successful, "
                f"{progress.failed} failed, {progress.skipped} skipped"
            )

            return batch_result

        except Exception as e:
            logger.error(f"Batch sync failed: {e}")
            progress.end_time = datetime.now(UTC)

            return BatchResult(
                batch_id=batch_id,
                status=BatchStatus.FAILED,
                progress=progress,
                results=[],
                metadata={"error": str(e)},
            )
        finally:
            # Clean up
            del self.active_batches[batch_id]

    async def _process_parallel(
        self,
        tracklist_ids: list[UUID],
        source: SyncSource,
        max_parallel: int,
        continue_on_error: bool,
        progress: BatchProgress,
        actor: str,
    ) -> list[dict[str, Any]]:
        """Process tracklists in parallel.

        Args:
            tracklist_ids: List of tracklist IDs
            source: Sync source
            max_parallel: Maximum parallel operations
            continue_on_error: Continue on error
            progress: Progress tracker
            actor: Who triggered the sync

        Returns:
            List of results
        """
        results = []
        semaphore = asyncio.Semaphore(max_parallel)

        async def sync_with_semaphore(tracklist_id: UUID) -> dict[str, Any]:
            async with semaphore:
                progress.in_progress.add(tracklist_id)
                try:
                    result = await self.sync_service.trigger_manual_sync(
                        tracklist_id=tracklist_id,
                        source=source,
                        force=False,
                        actor=actor,
                    )

                    if result.get("status") == SyncStatus.COMPLETED.value:
                        progress.successful += 1
                    else:
                        progress.failed += 1

                    return result

                except Exception as e:
                    progress.failed += 1
                    error_result = {
                        "tracklist_id": str(tracklist_id),
                        "status": "error",
                        "error": str(e),
                    }
                    progress.errors.append(error_result)

                    if not continue_on_error:
                        raise

                    return error_result
                finally:
                    progress.in_progress.discard(tracklist_id)
                    progress.completed += 1

        # Create tasks
        tasks = [sync_with_semaphore(tid) for tid in tracklist_ids]

        # Execute all tasks
        if continue_on_error:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in task_results:
                if isinstance(result, Exception):
                    results.append(
                        {
                            "status": "error",
                            "error": str(result),
                        }
                    )
                else:
                    results.append(result)  # type: ignore[arg-type]
        else:
            results = await asyncio.gather(*tasks)

        return results

    async def _process_sequential(
        self,
        tracklist_ids: list[UUID],
        source: SyncSource,
        continue_on_error: bool,
        progress: BatchProgress,
        actor: str,
    ) -> list[dict[str, Any]]:
        """Process tracklists sequentially.

        Args:
            tracklist_ids: List of tracklist IDs
            source: Sync source
            continue_on_error: Continue on error
            progress: Progress tracker
            actor: Who triggered the sync

        Returns:
            List of results
        """
        results = []

        for tracklist_id in tracklist_ids:
            progress.in_progress.add(tracklist_id)

            try:
                result = await self.sync_service.trigger_manual_sync(
                    tracklist_id=tracklist_id,
                    source=source,
                    force=False,
                    actor=actor,
                )

                if result.get("status") == SyncStatus.COMPLETED.value:
                    progress.successful += 1
                else:
                    progress.failed += 1

                results.append(result)

            except Exception as e:
                progress.failed += 1
                error_result = {
                    "tracklist_id": str(tracklist_id),
                    "status": "error",
                    "error": str(e),
                }
                progress.errors.append(error_result)
                results.append(error_result)

                if not continue_on_error:
                    break
            finally:
                progress.in_progress.discard(tracklist_id)
                progress.completed += 1

        return results

    async def _process_adaptive(
        self,
        tracklist_ids: list[UUID],
        source: SyncSource,
        continue_on_error: bool,
        progress: BatchProgress,
        actor: str,
    ) -> list[dict[str, Any]]:
        """Process tracklists with adaptive strategy.

        Dynamically adjusts parallelism based on system load.

        Args:
            tracklist_ids: List of tracklist IDs
            source: Sync source
            continue_on_error: Continue on error
            progress: Progress tracker
            actor: Who triggered the sync

        Returns:
            List of results
        """
        results = []
        current_parallel = min(5, self.max_parallel)  # Start conservative

        # Process in chunks with adaptive parallelism
        chunk_size = current_parallel
        for i in range(0, len(tracklist_ids), chunk_size):
            chunk = tracklist_ids[i : i + chunk_size]

            # Check system load and adjust parallelism
            load = await self._get_system_load()
            if load < 0.5:
                current_parallel = min(current_parallel + 2, self.max_parallel)
            elif load > self.adaptive_threshold:
                current_parallel = max(current_parallel - 2, 1)

            # Process chunk
            chunk_results = await self._process_parallel(
                chunk,
                source,
                current_parallel,
                continue_on_error,
                progress,
                actor,
            )

            results.extend(chunk_results)

            # Update chunk size for next iteration
            chunk_size = current_parallel

        return results

    async def _process_priority_based(
        self,
        tracklist_ids: list[UUID],
        source: SyncSource,
        continue_on_error: bool,
        progress: BatchProgress,
        actor: str,
    ) -> list[dict[str, Any]]:
        """Process tracklists based on priority.

        High-priority tracklists are processed first with more resources.

        Args:
            tracklist_ids: List of tracklist IDs
            source: Sync source
            continue_on_error: Continue on error
            progress: Progress tracker
            actor: Who triggered the sync

        Returns:
            List of results
        """
        # Categorize by priority
        high_priority, normal_priority, low_priority = await self._categorize_by_priority(tracklist_ids)

        results = []

        # Process high priority with more parallelism
        if high_priority:
            high_results = await self._process_parallel(
                high_priority,
                source,
                self.max_parallel,
                continue_on_error,
                progress,
                actor,
            )
            results.extend(high_results)

        # Process normal priority with moderate parallelism
        if normal_priority:
            normal_results = await self._process_parallel(
                normal_priority,
                source,
                max(self.max_parallel // 2, 3),
                continue_on_error,
                progress,
                actor,
            )
            results.extend(normal_results)

        # Process low priority sequentially or with minimal parallelism
        if low_priority:
            low_results = await self._process_parallel(
                low_priority,
                source,
                2,
                continue_on_error,
                progress,
                actor,
            )
            results.extend(low_results)

        return results

    async def _categorize_by_priority(
        self,
        tracklist_ids: list[UUID],
    ) -> tuple[list[UUID], list[UUID], list[UUID]]:
        """Categorize tracklists by priority.

        Args:
            tracklist_ids: List of tracklist IDs

        Returns:
            Tuple of (high_priority, normal_priority, low_priority) lists
        """
        high_priority = []
        normal_priority = []
        low_priority = []

        # Get sync configurations to determine priority
        for tracklist_id in tracklist_ids:
            config = await self._get_sync_config(tracklist_id)

            if config and config.sync_frequency == "realtime":
                high_priority.append(tracklist_id)
            elif config and config.sync_frequency in ["hourly", "daily"]:
                normal_priority.append(tracklist_id)
            else:
                low_priority.append(tracklist_id)

        return high_priority, normal_priority, low_priority

    async def _get_sync_config(self, tracklist_id: UUID) -> SyncConfiguration | None:
        """Get sync configuration for a tracklist.

        Args:
            tracklist_id: ID of the tracklist

        Returns:
            Sync configuration or None
        """
        query = select(SyncConfiguration).where(SyncConfiguration.tracklist_id == tracklist_id)
        result = await self.session.execute(query)
        config: SyncConfiguration | None = result.scalar_one_or_none()
        return config

    async def _get_system_load(self) -> float:
        """Get current system load.

        Returns:
            System load between 0 and 1
        """
        # Simplified load calculation
        # In production, this would check CPU, memory, and I/O

        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent

            # Weighted average
            load = (cpu_percent * 0.7 + memory_percent * 0.3) / 100
            return float(min(max(load, 0.0), 1.0))
        except Exception:
            # Default to moderate load if unable to determine
            return 0.5

    def _apply_priority_order(
        self,
        tracklist_ids: list[UUID],
        priority_order: list[UUID],
    ) -> list[UUID]:
        """Apply priority ordering to tracklist IDs.

        Args:
            tracklist_ids: Original list of tracklist IDs
            priority_order: Priority order

        Returns:
            Reordered list
        """
        # Create a set for quick lookup
        ids_set = set(tracklist_ids)

        # Start with priority items that are in the list
        ordered = [tid for tid in priority_order if tid in ids_set]

        # Add remaining items
        remaining = [tid for tid in tracklist_ids if tid not in set(ordered)]
        ordered.extend(remaining)

        return ordered

    async def get_batch_status(self, batch_id: UUID) -> dict[str, Any] | None:
        """Get status of a batch operation.

        Args:
            batch_id: Batch operation ID

        Returns:
            Status information or None
        """
        if batch_id in self.active_batches:
            progress = self.active_batches[batch_id]
            return {
                "batch_id": str(batch_id),
                "status": "processing",
                "progress": progress.progress_percentage,
                "total": progress.total,
                "completed": progress.completed,
                "successful": progress.successful,
                "failed": progress.failed,
                "skipped": progress.skipped,
                "duration": (progress.duration.total_seconds() if progress.duration else None),
            }

        return None

    async def cancel_batch(self, batch_id: UUID) -> bool:
        """Cancel a batch operation.

        Args:
            batch_id: Batch operation ID

        Returns:
            True if cancelled successfully
        """
        if batch_id in self.active_batches:
            # In a real implementation, we would set a cancellation flag
            # that the processing loops check
            logger.info(f"Batch operation {batch_id} cancellation requested")
            return True

        return False

    async def aggregate_batch_conflicts(
        self,
        tracklist_ids: list[UUID],
    ) -> dict[str, Any]:
        """Aggregate conflicts across multiple tracklists.

        Args:
            tracklist_ids: List of tracklist IDs

        Returns:
            Aggregated conflict information
        """

        total_conflicts = 0
        conflict_types: dict[str, int] = {}
        affected_tracklists = []

        for tracklist_id in tracklist_ids:
            # Get pending sync events with conflicts
            query = (
                select(SyncEvent)
                .where(
                    and_(
                        SyncEvent.tracklist_id == tracklist_id,
                        SyncEvent.status == "conflict",
                    )
                )
                .order_by(SyncEvent.created_at.desc())
                .limit(1)
            )

            result = await self.session.execute(query)
            event = result.scalar_one_or_none()

            if event and event.conflict_data:
                conflicts = event.conflict_data.get("conflicts", [])
                total_conflicts += len(conflicts)
                affected_tracklists.append(str(tracklist_id))

                # Count conflict types
                for conflict in conflicts:
                    conflict_type = conflict.get("type", "unknown")
                    conflict_types[conflict_type] = conflict_types.get(conflict_type, 0) + 1

        return {
            "total_conflicts": total_conflicts,
            "affected_tracklists": affected_tracklists,
            "conflict_types": conflict_types,
            "summary": f"{len(affected_tracklists)} tracklists have {total_conflicts} conflicts",
        }
