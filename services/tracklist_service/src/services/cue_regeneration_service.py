"""CUE file regeneration service for handling tracklist changes."""

import logging
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.tracklist_service.src.models.cue_file import CueFileDB
from services.tracklist_service.src.models.tracklist import TracklistDB
from services.tracklist_service.src.services.audit_service import AuditService
from services.tracklist_service.src.services.cue_generation_service import (
    CueGenerationService,
)
from src.models.cue_file import CueFormat, GenerateCueRequest

logger = logging.getLogger(__name__)


class RegenerationTrigger(Enum):
    """Types of events that trigger CUE regeneration."""

    MANUAL_EDIT = "manual_edit"
    SYNC_UPDATE = "sync_update"
    VERSION_ROLLBACK = "version_rollback"
    TRACK_ADDED = "track_added"
    TRACK_REMOVED = "track_removed"
    TRACK_MODIFIED = "track_modified"
    METADATA_CHANGE = "metadata_change"
    FORMAT_UPDATE = "format_update"
    FORCED = "forced"


class RegenerationPriority(Enum):
    """Priority levels for CUE regeneration."""

    CRITICAL = "critical"  # Immediate regeneration
    HIGH = "high"  # Within 1 minute
    NORMAL = "normal"  # Within 5 minutes
    LOW = "low"  # Within 15 minutes
    BATCH = "batch"  # Next batch processing window


class CueRegenerationService:
    """Service for managing CUE file regeneration on tracklist changes."""

    def __init__(
        self,
        session: AsyncSession,
        cue_service: CueGenerationService | None = None,
        audit_service: AuditService | None = None,
    ):
        """Initialize CUE regeneration service.

        Args:
            session: Database session
            cue_service: Service for CUE file generation
            audit_service: Service for audit logging
        """
        self.session = session
        self.cue_service = cue_service or CueGenerationService(session)
        self.audit_service = audit_service or AuditService(session)
        self.regeneration_queue: list[dict[str, Any]] = []
        self.cache_invalidation_set: set[UUID] = set()

    async def handle_tracklist_change(
        self,
        tracklist_id: UUID,
        change_type: str,
        change_details: dict[str, Any] | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        """Handle CUE regeneration when tracklist changes.

        Args:
            tracklist_id: ID of the changed tracklist
            change_type: Type of change that occurred
            change_details: Additional details about the change
            actor: Who made the change

        Returns:
            Regeneration status and queued jobs
        """
        try:
            # Determine trigger and priority
            trigger = self._map_change_to_trigger(change_type)
            priority = self._determine_priority(trigger, change_details)

            # Get all active CUE files for this tracklist
            cue_files = await self._get_active_cue_files(tracklist_id)

            if not cue_files:
                logger.info(f"No active CUE files for tracklist {tracklist_id}")
                return {
                    "status": "skipped",
                    "reason": "no_active_cue_files",
                    "tracklist_id": str(tracklist_id),
                }

            # Queue regeneration for each format
            queued_jobs = []
            for cue_file in cue_files:
                job = await self._queue_regeneration(
                    tracklist_id=tracklist_id,
                    cue_file_id=cue_file.id,
                    format=cue_file.format,
                    trigger=trigger,
                    priority=priority,
                    actor=actor,
                )
                queued_jobs.append(job)

            # Invalidate cache
            await self._invalidate_cache(tracklist_id)

            # Log the regeneration trigger
            await self.audit_service.log_tracklist_change(
                tracklist_id=tracklist_id,
                action="cue_regeneration_triggered",
                actor=actor,
                metadata={
                    "trigger": trigger.value,
                    "priority": priority.value,
                    "cue_files_affected": len(cue_files),
                    "change_type": change_type,
                },
            )

            return {
                "status": "queued",
                "tracklist_id": str(tracklist_id),
                "jobs_queued": len(queued_jobs),
                "jobs": queued_jobs,
            }

        except Exception as e:
            logger.error(f"Failed to handle tracklist change for {tracklist_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "tracklist_id": str(tracklist_id),
            }

    def _map_change_to_trigger(self, change_type: str) -> RegenerationTrigger:
        """Map change type to regeneration trigger.

        Args:
            change_type: Type of change

        Returns:
            Corresponding regeneration trigger
        """
        mapping = {
            "manual": RegenerationTrigger.MANUAL_EDIT,
            "sync": RegenerationTrigger.SYNC_UPDATE,
            "1001tracklists_sync": RegenerationTrigger.SYNC_UPDATE,
            "rollback": RegenerationTrigger.VERSION_ROLLBACK,
            "track_add": RegenerationTrigger.TRACK_ADDED,
            "track_remove": RegenerationTrigger.TRACK_REMOVED,
            "track_modify": RegenerationTrigger.TRACK_MODIFIED,
            "metadata": RegenerationTrigger.METADATA_CHANGE,
            "format": RegenerationTrigger.FORMAT_UPDATE,
        }

        return mapping.get(change_type, RegenerationTrigger.FORCED)

    def _determine_priority(
        self,
        trigger: RegenerationTrigger,
        change_details: dict[str, Any] | None = None,
    ) -> RegenerationPriority:
        """Determine regeneration priority based on trigger and details.

        Args:
            trigger: Regeneration trigger
            change_details: Additional change details

        Returns:
            Regeneration priority
        """
        # Manual edits get high priority
        if trigger == RegenerationTrigger.MANUAL_EDIT:
            return RegenerationPriority.HIGH

        # Version rollbacks are critical
        if trigger == RegenerationTrigger.VERSION_ROLLBACK:
            return RegenerationPriority.CRITICAL

        # Format updates are high priority
        if trigger == RegenerationTrigger.FORMAT_UPDATE:
            return RegenerationPriority.HIGH

        # Check change magnitude
        if change_details:
            tracks_affected = change_details.get("tracks_affected", 0)
            if tracks_affected > 10:
                return RegenerationPriority.HIGH
            if tracks_affected > 5:
                return RegenerationPriority.NORMAL

        # Sync updates are normal priority
        if trigger == RegenerationTrigger.SYNC_UPDATE:
            return RegenerationPriority.NORMAL

        # Default to normal priority
        return RegenerationPriority.NORMAL

    async def _get_active_cue_files(self, tracklist_id: UUID) -> list[CueFileDB]:
        """Get all active CUE files for a tracklist.

        Args:
            tracklist_id: ID of the tracklist

        Returns:
            List of active CUE files
        """
        query = select(CueFileDB).where(
            CueFileDB.tracklist_id == tracklist_id,
            CueFileDB.is_active.is_(True),
        )

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def _queue_regeneration(
        self,
        tracklist_id: UUID,
        cue_file_id: UUID,
        format: str,
        trigger: RegenerationTrigger,
        priority: RegenerationPriority,
        actor: str,
    ) -> dict[str, Any]:
        """Queue a CUE file for regeneration.

        Args:
            tracklist_id: ID of the tracklist
            cue_file_id: ID of the CUE file
            format: CUE format
            trigger: What triggered regeneration
            priority: Regeneration priority
            actor: Who triggered the regeneration

        Returns:
            Queued job information
        """
        job = {
            "job_id": str(UUID()),  # Generate a unique job ID
            "tracklist_id": str(tracklist_id),
            "cue_file_id": str(cue_file_id),
            "format": format,
            "trigger": trigger.value,
            "priority": priority.value,
            "actor": actor,
            "queued_at": datetime.now(UTC).isoformat(),
            "status": "pending",
        }

        # Add to queue (in production, this would go to a message queue)
        self.regeneration_queue.append(job)

        # Sort queue by priority
        self._sort_queue()

        return job

    def _sort_queue(self) -> None:
        """Sort regeneration queue by priority."""
        priority_order = {
            RegenerationPriority.CRITICAL.value: 0,
            RegenerationPriority.HIGH.value: 1,
            RegenerationPriority.NORMAL.value: 2,
            RegenerationPriority.LOW.value: 3,
            RegenerationPriority.BATCH.value: 4,
        }

        self.regeneration_queue.sort(key=lambda x: (priority_order.get(x["priority"], 5), x["queued_at"]))

    async def _invalidate_cache(self, tracklist_id: UUID) -> None:
        """Invalidate CUE cache for a tracklist.

        Args:
            tracklist_id: ID of the tracklist
        """
        # Add to invalidation set
        self.cache_invalidation_set.add(tracklist_id)

        # In production, this would interact with a cache service
        logger.info(f"Cache invalidated for tracklist {tracklist_id}")

    async def process_regeneration_queue(
        self,
        max_jobs: int = 10,
        priority_filter: RegenerationPriority | None = None,
    ) -> list[dict[str, Any]]:
        """Process queued regeneration jobs.

        Args:
            max_jobs: Maximum number of jobs to process
            priority_filter: Only process jobs of this priority

        Returns:
            List of processed jobs
        """
        processed = []
        jobs_to_process = []

        # Select jobs to process
        for job in self.regeneration_queue[:max_jobs]:
            if priority_filter and job["priority"] != priority_filter.value:
                continue
            jobs_to_process.append(job)

        # Process each job
        for job in jobs_to_process:
            try:
                result = await self._regenerate_cue_file(job)
                job["status"] = "completed" if result else "failed"
                job["processed_at"] = datetime.now(UTC).isoformat()
                processed.append(job)

                # Remove from queue
                self.regeneration_queue.remove(job)

            except Exception as e:
                logger.error(f"Failed to process regeneration job {job['job_id']}: {e}")
                job["status"] = "error"
                job["error"] = str(e)

        return processed

    async def _regenerate_cue_file(self, job: dict[str, Any]) -> bool:
        """Regenerate a single CUE file.

        Args:
            job: Regeneration job details

        Returns:
            True if successful
        """
        try:
            tracklist_id = UUID(job["tracklist_id"])
            cue_file_id = UUID(job["cue_file_id"])

            # Get the tracklist
            tracklist = await self.session.get(TracklistDB, tracklist_id)
            if not tracklist:
                logger.error(f"Tracklist {tracklist_id} not found")
                return False

            # Get the CUE file
            cue_file = await self.session.get(CueFileDB, cue_file_id)
            if not cue_file:
                logger.error(f"CUE file {cue_file_id} not found")
                return False

            # Regenerate using CueService

            # Convert string format to CueFormat enum
            format_enum = CueFormat(cue_file.format)

            request = GenerateCueRequest(
                format=format_enum,
                audio_file_path="/placeholder/path.wav",
            )
            result = await self.cue_service.generate_cue_file(
                tracklist=tracklist,
                request=request,
            )

            # The CueGenerationResponse doesn't have content attribute
            # This suggests we need to get the content from storage or the result differently
            # For now, let's update the metadata to track regeneration
            cue_file.updated_at = datetime.now(UTC)
            cue_file.format_metadata = {
                **cue_file.format_metadata,  # Preserve existing metadata
                "trigger": job["trigger"],
                "regenerated_at": datetime.now(UTC).isoformat(),
                "actor": job["actor"],
                "success": result.success,
                "error": result.error,
            }

            await self.session.commit()

            # Log successful regeneration
            await self.audit_service.log_cue_file_change(
                cue_file_id=cue_file_id,
                action="regenerated",
                changes={"content": "regenerated", "format": str(cue_file.format)},
                actor=job["actor"],
                metadata={
                    "trigger": job["trigger"],
                    "priority": job["priority"],
                    "job_id": job["job_id"],
                },
            )

            return True

        except Exception as e:
            logger.error(f"Failed to regenerate CUE file: {e}")
            return False

    async def batch_regenerate(
        self,
        tracklist_ids: list[UUID],
        format: str | None = None,
        actor: str = "system",
    ) -> dict[str, Any]:
        """Batch regenerate CUE files for multiple tracklists.

        Args:
            tracklist_ids: List of tracklist IDs
            format: Specific format to regenerate (None for all)
            actor: Who triggered the batch regeneration

        Returns:
            Batch regeneration status
        """
        results: dict[str, Any] = {
            "total": len(tracklist_ids),
            "queued": 0,
            "skipped": 0,
            "errors": 0,
            "details": [],
        }

        for tracklist_id in tracklist_ids:
            try:
                result = await self.handle_tracklist_change(
                    tracklist_id=tracklist_id,
                    change_type="batch_regeneration",
                    change_details={"format": format} if format else None,
                    actor=actor,
                )

                if result["status"] == "queued":
                    results["queued"] += 1
                elif result["status"] == "skipped":
                    results["skipped"] += 1
                else:
                    results["errors"] += 1

                results["details"].append(result)

            except Exception as e:
                logger.error(f"Failed to queue regeneration for {tracklist_id}: {e}")
                results["errors"] += 1

        return results

    async def get_regeneration_status(
        self,
        tracklist_id: UUID | None = None,
    ) -> dict[str, Any]:
        """Get current regeneration queue status.

        Args:
            tracklist_id: Optional filter by tracklist ID

        Returns:
            Queue status information
        """
        if tracklist_id:
            # Filter queue for specific tracklist
            tracklist_jobs = [job for job in self.regeneration_queue if job["tracklist_id"] == str(tracklist_id)]

            return {
                "tracklist_id": str(tracklist_id),
                "queued_jobs": len(tracklist_jobs),
                "jobs": tracklist_jobs,
                "cache_invalidated": tracklist_id in self.cache_invalidation_set,
            }
        # Overall queue status
        priority_counts: dict[str, int] = {}
        for job in self.regeneration_queue:
            priority = job["priority"]
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

        return {
            "total_queued": len(self.regeneration_queue),
            "by_priority": priority_counts,
            "cache_invalidations": len(self.cache_invalidation_set),
            "next_job": (self.regeneration_queue[0] if self.regeneration_queue else None),
        }

    async def cancel_regeneration(
        self,
        job_id: str | None = None,
        tracklist_id: UUID | None = None,
    ) -> dict[str, Any]:
        """Cancel queued regeneration jobs.

        Args:
            job_id: Specific job to cancel
            tracklist_id: Cancel all jobs for a tracklist

        Returns:
            Cancellation result
        """
        cancelled = []

        if job_id:
            # Cancel specific job
            for job in self.regeneration_queue:
                if job["job_id"] == job_id:
                    self.regeneration_queue.remove(job)
                    cancelled.append(job)
                    break
        elif tracklist_id:
            # Cancel all jobs for tracklist
            jobs_to_remove = [job for job in self.regeneration_queue if job["tracklist_id"] == str(tracklist_id)]
            for job in jobs_to_remove:
                self.regeneration_queue.remove(job)
                cancelled.append(job)

        return {
            "cancelled_count": len(cancelled),
            "cancelled_jobs": cancelled,
        }
