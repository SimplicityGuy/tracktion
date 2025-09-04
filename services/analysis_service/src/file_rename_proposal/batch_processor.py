"""Batch processing service for file rename proposals."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from shared.core_types.src.rename_proposal_repository import RenameProposalRepository
from shared.core_types.src.repositories import MetadataRepository, RecordingRepository

from .confidence_scorer import ConfidenceScorer
from .conflict_detector import ConflictDetector
from .proposal_generator import ProposalGenerator

logger = logging.getLogger(__name__)


class BatchProcessingJob:
    """Represents a batch processing job."""

    def __init__(
        self,
        job_id: str,
        recording_ids: list[UUID],
        options: dict[str, Any] | None = None,
    ) -> None:
        """Initialize batch job.

        Args:
            job_id: Unique identifier for the job
            recording_ids: List of recording UUIDs to process
            options: Optional processing options
        """
        self.job_id = job_id
        self.recording_ids = recording_ids
        self.options = options or {}
        self.created_at = datetime.now(UTC)
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.status = "pending"
        self.total_recordings = len(recording_ids)
        self.processed_recordings = 0
        self.successful_proposals = 0
        self.failed_recordings = 0
        self.errors: list[str] = []
        self.proposal_ids: list[UUID] = []


class BatchProcessor:
    """Handles batch processing of rename proposals."""

    def __init__(
        self,
        proposal_generator: ProposalGenerator,
        conflict_detector: ConflictDetector,
        confidence_scorer: ConfidenceScorer,
        proposal_repo: RenameProposalRepository,
        recording_repo: RecordingRepository,
        metadata_repo: MetadataRepository,
    ) -> None:
        """Initialize batch processor.

        Args:
            proposal_generator: Proposal generation service
            conflict_detector: Conflict detection service
            confidence_scorer: Confidence scoring service
            proposal_repo: Proposal repository
            recording_repo: Recording repository
            metadata_repo: Metadata repository
        """
        self.proposal_generator = proposal_generator
        self.conflict_detector = conflict_detector
        self.confidence_scorer = confidence_scorer
        self.proposal_repo = proposal_repo
        self.recording_repo = recording_repo
        self.metadata_repo = metadata_repo
        self.logger = logger

        # Active jobs tracking
        self.active_jobs: dict[str, BatchProcessingJob] = {}

    def submit_batch_job(
        self,
        recording_ids: list[UUID],
        job_id: str | None = None,
        max_workers: int = 4,
        chunk_size: int = 100,
        auto_approve_threshold: float = 0.9,
        enable_conflict_resolution: bool = True,
    ) -> BatchProcessingJob:
        """Submit a batch processing job.

        Args:
            recording_ids: List of recording UUIDs to process
            job_id: Optional job identifier (auto-generated if not provided)
            max_workers: Maximum number of parallel workers
            chunk_size: Number of recordings to process in each chunk
            auto_approve_threshold: Confidence threshold for auto-approval
            enable_conflict_resolution: Whether to attempt conflict resolution

        Returns:
            BatchProcessingJob instance
        """
        if not job_id:
            job_id = f"batch_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{len(recording_ids)}"

        job = BatchProcessingJob(
            job_id=job_id,
            recording_ids=recording_ids,
            options={
                "max_workers": max_workers,
                "chunk_size": chunk_size,
                "auto_approve_threshold": auto_approve_threshold,
                "enable_conflict_resolution": enable_conflict_resolution,
            },
        )

        self.active_jobs[job_id] = job
        logger.info(f"Submitted batch job {job_id} with {len(recording_ids)} recordings")

        return job

    def process_batch_job(self, job_id: str) -> BatchProcessingJob:
        """Process a batch job.

        Args:
            job_id: Job identifier

        Returns:
            Updated BatchProcessingJob instance

        Raises:
            ValueError: If job not found
        """
        job = self.active_jobs.get(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        job.started_at = datetime.now(UTC)
        job.status = "running"

        try:
            self._process_recordings_in_batches(job)
            job.status = "completed"
            job.completed_at = datetime.now(UTC)

            logger.info(
                f"Batch job {job_id} completed: "
                f"{job.successful_proposals} proposals created, "
                f"{job.failed_recordings} failures"
            )

        except Exception as e:
            job.status = "failed"
            job.errors.append(f"Job failed: {e!s}")
            logger.error(f"Batch job {job_id} failed: {e}")
            raise

        return job

    def _process_recordings_in_batches(self, job: BatchProcessingJob) -> None:
        """Process recordings in parallel batches.

        Args:
            job: Batch processing job
        """
        max_workers = job.options.get("max_workers", 4)
        chunk_size = job.options.get("chunk_size", 100)

        # Split recordings into chunks
        recording_chunks = [job.recording_ids[i : i + chunk_size] for i in range(0, len(job.recording_ids), chunk_size)]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit chunk processing tasks
            future_to_chunk = {
                executor.submit(self._process_recording_chunk, chunk, job): chunk for chunk in recording_chunks
            }

            # Process completed chunks
            for future in as_completed(future_to_chunk):
                try:
                    chunk_results = future.result()
                    self._update_job_progress(job, chunk_results)

                except Exception as e:
                    error_msg = f"Chunk processing failed: {e!s}"
                    job.errors.append(error_msg)
                    logger.error(f"Job {job.job_id}: {error_msg}")

    def _process_recording_chunk(self, recording_ids: list[UUID], job: BatchProcessingJob) -> dict[str, Any]:
        """Process a chunk of recordings.

        Args:
            recording_ids: Recording UUIDs in this chunk
            job: Parent batch job

        Returns:
            Dictionary with processing results
        """
        results: dict[str, Any] = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "proposal_ids": [],
            "errors": [],
        }

        # Get directory contents for conflict detection
        directory_contents = self._collect_directory_contents(recording_ids)

        for recording_id in recording_ids:
            try:
                result = self._process_single_recording(recording_id, job, directory_contents)

                results["processed"] += 1

                if result["success"]:
                    results["successful"] += 1
                    if result["proposal_id"]:
                        results["proposal_ids"].append(result["proposal_id"])
                else:
                    results["failed"] += 1
                    if result["error"]:
                        results["errors"].append(result["error"])

            except Exception as e:
                results["processed"] += 1
                results["failed"] += 1
                error_msg = f"Recording {recording_id}: {e!s}"
                results["errors"].append(error_msg)
                logger.error(f"Job {job.job_id}: {error_msg}")

        return results

    def _process_single_recording(
        self,
        recording_id: UUID,
        job: BatchProcessingJob,
        directory_contents: dict[str, set],
    ) -> dict[str, Any]:
        """Process a single recording.

        Args:
            recording_id: Recording UUID
            job: Batch job
            directory_contents: Directory contents for conflict detection

        Returns:
            Dictionary with processing result
        """
        # Get recording
        recording = self.recording_repo.get_by_id(recording_id)
        if not recording:
            return {
                "success": False,
                "error": f"Recording {recording_id} not found",
                "proposal_id": None,
            }

        try:
            # Validate recording has required fields
            if not recording.file_name or not recording.file_path:
                return {
                    "success": False,
                    "error": f"Recording {recording_id} missing file_name or file_path",
                    "proposal_id": None,
                }

            # Get metadata from MetadataRepository
            metadata_list = self.metadata_repo.get_by_recording(recording_id)
            metadata_dict = {m.key: m.value for m in metadata_list} if metadata_list else {}

            # Derive file extension
            file_path = Path(recording.file_name)
            file_extension = file_path.suffix[1:].lower() if file_path.suffix else "mp3"
            proposal = self.proposal_generator.generate_proposal(
                recording_id=recording_id,
                original_path=recording.file_path,
                metadata=metadata_dict,
                file_extension=file_extension,
            )

            if not proposal:
                return {
                    "success": False,
                    "error": f"Could not generate proposal for {recording.file_name}",
                    "proposal_id": None,
                }

            # Detect conflicts
            directory_path = str(Path(proposal.full_proposed_path).parent)
            existing_files = directory_contents.get(directory_path, set())

            # Get other proposals for conflict detection
            other_proposals = [
                {
                    "full_proposed_path": p.full_proposed_path,
                    "recording_id": str(p.recording_id),
                }
                for p in self.proposal_repo.get_pending_proposals()
                if p.recording_id != recording_id
            ]

            conflicts_result = self.conflict_detector.detect_conflicts(
                proposal.full_proposed_path, existing_files, other_proposals
            )

            conflicts = conflicts_result["conflicts"]
            warnings = conflicts_result["warnings"]

            # Attempt conflict resolution if enabled
            resolved_path = proposal.full_proposed_path
            resolved_filename = proposal.proposed_filename
            if conflicts and job.options.get("enable_conflict_resolution", True):
                alternative = self.conflict_detector.resolve_conflicts(proposal.full_proposed_path, conflicts)

                if alternative:
                    resolved_path = alternative
                    resolved_filename = Path(alternative).name

                    # Re-check conflicts for the alternative
                    conflicts_result = self.conflict_detector.detect_conflicts(
                        alternative, existing_files, other_proposals
                    )
                    conflicts = conflicts_result["conflicts"]
                    warnings = conflicts_result["warnings"]

            # Calculate confidence score
            confidence, components = self.confidence_scorer.calculate_confidence(
                metadata=cast("dict[str, str | None]", metadata_dict),  # Metadata values are non-null in our model
                original_filename=recording.file_name,  # Already validated above
                proposed_filename=resolved_filename,
                conflicts=conflicts,
                warnings=warnings,
                pattern_used=proposal.pattern_used,
                source=proposal.metadata_source,
            )

            # Determine status
            auto_approve_threshold = job.options.get("auto_approve_threshold", 0.9)
            if conflicts:
                status = "rejected"  # Auto-reject if conflicts remain
            elif confidence >= auto_approve_threshold:
                status = "approved"
            else:
                status = "pending"

            # Create proposal in database
            created_proposal = self.proposal_repo.create(
                recording_id=recording_id,
                original_path=str(Path(recording.file_path).parent),  # Already validated above
                original_filename=recording.file_name,  # Already validated above
                proposed_filename=resolved_filename,
                full_proposed_path=resolved_path,
                confidence_score=confidence,
                status=status,
                conflicts=conflicts if conflicts else None,
                warnings=warnings if warnings else None,
                metadata_source=proposal.metadata_source,
                pattern_used=proposal.pattern_used,
            )

            logger.debug(
                f"Created proposal {created_proposal.id} for {recording.file_name} "
                f"(confidence: {confidence:.3f}, status: {status})"
            )

            return {"success": True, "error": None, "proposal_id": created_proposal.id}

        except Exception as e:
            logger.error(f"Failed to process recording {recording_id}: {e}")
            return {"success": False, "error": str(e), "proposal_id": None}

    def _collect_directory_contents(self, recording_ids: list[UUID]) -> dict[str, set]:
        """Collect directory contents for conflict detection.

        Args:
            recording_ids: Recording UUIDs to check

        Returns:
            Dictionary mapping directory paths to sets of filenames
        """
        directory_contents: dict[str, set[str]] = {}

        for recording_id in recording_ids:
            recording = self.recording_repo.get_by_id(recording_id)
            if recording and recording.file_path:
                directory = str(Path(recording.file_path).parent)

                if directory not in directory_contents:
                    try:
                        dir_path = Path(directory)
                        if dir_path.exists():
                            files = {f.name for f in dir_path.iterdir()}
                            directory_contents[directory] = files
                        else:
                            directory_contents[directory] = set()
                    except (OSError, PermissionError) as e:
                        logger.warning(f"Could not list directory {directory}: {e}")
                        directory_contents[directory] = set()

        return directory_contents

    def _update_job_progress(self, job: BatchProcessingJob, chunk_results: dict[str, Any]) -> None:
        """Update job progress with chunk results.

        Args:
            job: Batch processing job
            chunk_results: Results from processing a chunk
        """
        job.processed_recordings += chunk_results["processed"]
        job.successful_proposals += chunk_results["successful"]
        job.failed_recordings += chunk_results["failed"]
        job.proposal_ids.extend(chunk_results["proposal_ids"])
        job.errors.extend(chunk_results["errors"])

        # Log progress
        progress = (job.processed_recordings / job.total_recordings) * 100
        logger.info(
            f"Job {job.job_id} progress: {job.processed_recordings}/{job.total_recordings} "
            f"({progress:.1f}%) - {job.successful_proposals} successful, {job.failed_recordings} failed"
        )

    def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get status of a batch job.

        Args:
            job_id: Job identifier

        Returns:
            Job status dictionary or None if not found
        """
        job = self.active_jobs.get(job_id)
        if not job:
            return None

        progress = (job.processed_recordings / job.total_recordings) * 100 if job.total_recordings > 0 else 0

        return {
            "job_id": job.job_id,
            "status": job.status,
            "total_recordings": job.total_recordings,
            "processed_recordings": job.processed_recordings,
            "successful_proposals": job.successful_proposals,
            "failed_recordings": job.failed_recordings,
            "progress_percentage": progress,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "error_count": len(job.errors),
            "proposal_count": len(job.proposal_ids),
            "options": job.options,
        }

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a batch job.

        Args:
            job_id: Job identifier

        Returns:
            True if job was cancelled successfully
        """
        job = self.active_jobs.get(job_id)
        if not job:
            return False

        if job.status in ["completed", "failed", "cancelled"]:
            return False

        job.status = "cancelled"
        job.completed_at = datetime.now(UTC)

        logger.info(f"Cancelled batch job {job_id}")
        return True

    def cleanup_completed_jobs(self, max_age_hours: int = 24) -> int:
        """Clean up old completed jobs.

        Args:
            max_age_hours: Maximum age in hours for keeping completed jobs

        Returns:
            Number of jobs cleaned up
        """
        cutoff_time = datetime.now(UTC) - timedelta(hours=max_age_hours)
        jobs_to_remove = []

        for job_id, job in self.active_jobs.items():
            if (
                job.status in ["completed", "failed", "cancelled"]
                and job.completed_at
                and job.completed_at < cutoff_time
            ):
                jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            del self.active_jobs[job_id]

        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} old batch jobs")

        return len(jobs_to_remove)

    def list_active_jobs(self) -> list[dict[str, Any]]:
        """List all active jobs.

        Returns:
            List of job status dictionaries
        """
        return [status for job_id in self.active_jobs if (status := self.get_job_status(job_id)) is not None]
