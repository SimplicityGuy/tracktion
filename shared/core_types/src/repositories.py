"""Repository pattern implementations for database operations."""

import logging
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, cast
from uuid import UUID, uuid4

from sqlalchemy import select

from .database import DatabaseManager
from .models import Job, Metadata, Recording, Tracklist

logger = logging.getLogger(__name__)


class RecordingRepository:
    """Repository for Recording entity operations."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initialize repository with database manager.

        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager

    def create(
        self,
        file_path: str,
        file_name: str,
        sha256_hash: str | None = None,
        xxh128_hash: str | None = None,
    ) -> Recording:
        """Create a new recording.

        Args:
            file_path: Full path to the file
            file_name: Name of the file
            sha256_hash: Optional SHA256 hash of file
            xxh128_hash: Optional XXH128 hash of file

        Returns:
            Created Recording instance

        Raises:
            IntegrityError: If recording with same hash already exists
        """
        with self.db.get_db_session() as session:
            recording = Recording(
                file_path=file_path,
                file_name=file_name,
                sha256_hash=sha256_hash,
                xxh128_hash=xxh128_hash,
            )
            session.add(recording)
            session.flush()
            session.refresh(recording)
            return recording

    def get_by_id(self, recording_id: UUID) -> Recording | None:
        """Get recording by ID.

        Args:
            recording_id: UUID of the recording

        Returns:
            Recording instance or None if not found
        """
        with self.db.get_db_session() as session:
            stmt = select(Recording).where(Recording.id == recording_id)
            result = session.execute(stmt)
            return cast("Recording | None", result.scalar_one_or_none())

    def get_by_file_path(self, file_path: str) -> Recording | None:
        """Get recording by file path.

        Args:
            file_path: Full path to the file

        Returns:
            Recording instance or None if not found
        """
        with self.db.get_db_session() as session:
            stmt = select(Recording).where(Recording.file_path == file_path)
            result = session.execute(stmt)
            return cast("Recording | None", result.scalar_one_or_none())

    def get_all(self, limit: int | None = None, offset: int | None = None) -> list[Recording]:
        """Get all recordings with optional pagination.

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of Recording instances
        """
        with self.db.get_db_session() as session:
            stmt = select(Recording)
            if offset:
                stmt = stmt.offset(offset)
            if limit:
                stmt = stmt.limit(limit)
            result = session.execute(stmt)
            return list(result.scalars().all())

    def update(self, recording_id: UUID, **kwargs: Any) -> Recording | None:
        """Update a recording.

        Args:
            recording_id: UUID of the recording
            **kwargs: Fields to update

        Returns:
            Updated Recording instance or None if not found
        """
        with self.db.get_db_session() as session:
            stmt = select(Recording).where(Recording.id == recording_id)
            result = session.execute(stmt)
            recording = cast("Recording | None", result.scalar_one_or_none())

            if not recording:
                return None

            for key, value in kwargs.items():
                if hasattr(recording, key):
                    setattr(recording, key, value)

            session.flush()
            session.refresh(recording)
            return recording

    def delete(self, recording_id: UUID) -> bool:
        """Delete a recording.

        Args:
            recording_id: UUID of the recording

        Returns:
            True if deleted, False if not found
        """
        with self.db.get_db_session() as session:
            stmt = select(Recording).where(Recording.id == recording_id)
            result = session.execute(stmt)
            recording = cast("Recording | None", result.scalar_one_or_none())

            if not recording:
                return False

            session.delete(recording)
            return True

    def bulk_create(self, recordings: list[dict[str, Any]]) -> list[Recording]:
        """Create multiple recordings in a single transaction.

        Args:
            recordings: List of recording data dictionaries

        Returns:
            List of created Recording instances
        """
        with self.db.get_db_session() as session:
            recording_objects = [Recording(**data) for data in recordings]
            session.bulk_save_objects(recording_objects, return_defaults=True)
            return recording_objects


class MetadataRepository:
    """Repository for Metadata entity operations."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initialize repository with database manager.

        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager

    def create(self, recording_id: UUID, key: str, value: str) -> Metadata:
        """Create metadata for a recording.

        Args:
            recording_id: UUID of the recording
            key: Metadata key
            value: Metadata value

        Returns:
            Created Metadata instance
        """
        with self.db.get_db_session() as session:
            metadata = Metadata(recording_id=recording_id, key=key, value=value)
            session.add(metadata)
            session.flush()
            session.refresh(metadata)
            return metadata

    def get_by_recording(self, recording_id: UUID) -> list[Metadata]:
        """Get all metadata for a recording.

        Args:
            recording_id: UUID of the recording

        Returns:
            List of Metadata instances
        """
        with self.db.get_db_session() as session:
            stmt = select(Metadata).where(Metadata.recording_id == recording_id)
            result = session.execute(stmt)
            return list(result.scalars().all())

    def get_by_key(self, recording_id: UUID, key: str) -> Metadata | None:
        """Get specific metadata by key for a recording.

        Args:
            recording_id: UUID of the recording
            key: Metadata key

        Returns:
            Metadata instance or None if not found
        """
        with self.db.get_db_session() as session:
            stmt = select(Metadata).where(Metadata.recording_id == recording_id, Metadata.key == key)
            result = session.execute(stmt)
            return cast("Metadata | None", result.scalar_one_or_none())

    def update(self, metadata_id: UUID, value: str) -> Metadata | None:
        """Update metadata value.

        Args:
            metadata_id: UUID of the metadata
            value: New value

        Returns:
            Updated Metadata instance or None if not found
        """
        with self.db.get_db_session() as session:
            stmt = select(Metadata).where(Metadata.id == metadata_id)
            result = session.execute(stmt)
            metadata = cast("Metadata | None", result.scalar_one_or_none())

            if not metadata:
                return None

            metadata.value = value
            session.flush()
            session.refresh(metadata)
            return metadata

    def delete(self, metadata_id: UUID) -> bool:
        """Delete metadata.

        Args:
            metadata_id: UUID of the metadata

        Returns:
            True if deleted, False if not found
        """
        with self.db.get_db_session() as session:
            stmt = select(Metadata).where(Metadata.id == metadata_id)
            result = session.execute(stmt)
            metadata = cast("Metadata | None", result.scalar_one_or_none())

            if not metadata:
                return False

            session.delete(metadata)
            return True

    def bulk_create(self, recording_id: UUID, metadata_items: list[dict[str, str]]) -> list[Metadata]:
        """Create multiple metadata entries for a recording.

        Args:
            recording_id: UUID of the recording
            metadata_items: List of dictionaries with 'key' and 'value'

        Returns:
            List of created Metadata instances
        """
        with self.db.get_db_session() as session:
            metadata_objects = [
                Metadata(recording_id=recording_id, key=item["key"], value=item["value"]) for item in metadata_items
            ]
            session.bulk_save_objects(metadata_objects, return_defaults=True)
            return metadata_objects


class TracklistRepository:
    """Repository for Tracklist entity operations."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initialize repository with database manager.

        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager

    def create(
        self,
        recording_id: UUID,
        source: str,
        tracks: list[dict[str, Any]] | None = None,
        cue_file_path: str | None = None,
    ) -> Tracklist:
        """Create a tracklist for a recording.

        Args:
            recording_id: UUID of the recording
            source: Source of the tracklist
            tracks: List of track dictionaries
            cue_file_path: Path to CUE file

        Returns:
            Created Tracklist instance

        Raises:
            IntegrityError: If tracklist already exists for recording
        """
        with self.db.get_db_session() as session:
            tracklist = Tracklist(
                recording_id=recording_id,
                source=source,
                tracks=tracks,
                cue_file_path=cue_file_path,
            )

            # Validate tracks structure
            if not tracklist.validate_tracks():
                raise ValueError("Invalid tracks structure")

            session.add(tracklist)
            session.flush()
            session.refresh(tracklist)
            return tracklist

    def get_by_recording(self, recording_id: UUID) -> Tracklist | None:
        """Get tracklist for a recording.

        Args:
            recording_id: UUID of the recording

        Returns:
            Tracklist instance or None if not found
        """
        with self.db.get_db_session() as session:
            stmt = select(Tracklist).where(Tracklist.recording_id == recording_id)
            result = session.execute(stmt)
            return cast("Tracklist | None", result.scalar_one_or_none())

    def get_by_id(self, tracklist_id: UUID) -> Tracklist | None:
        """Get tracklist by ID.

        Args:
            tracklist_id: UUID of the tracklist

        Returns:
            Tracklist instance or None if not found
        """
        with self.db.get_db_session() as session:
            stmt = select(Tracklist).where(Tracklist.id == tracklist_id)
            result = session.execute(stmt)
            return cast("Tracklist | None", result.scalar_one_or_none())

    def update(self, tracklist_id: UUID, **kwargs: Any) -> Tracklist | None:
        """Update a tracklist.

        Args:
            tracklist_id: UUID of the tracklist
            **kwargs: Fields to update

        Returns:
            Updated Tracklist instance or None if not found
        """
        with self.db.get_db_session() as session:
            stmt = select(Tracklist).where(Tracklist.id == tracklist_id)
            result = session.execute(stmt)
            tracklist = cast("Tracklist | None", result.scalar_one_or_none())

            if not tracklist:
                return None

            for key, value in kwargs.items():
                if hasattr(tracklist, key):
                    setattr(tracklist, key, value)

            # Validate if tracks were updated
            if "tracks" in kwargs and not tracklist.validate_tracks():
                raise ValueError("Invalid tracks structure")

            session.flush()
            session.refresh(tracklist)
            return tracklist

    def delete(self, tracklist_id: UUID) -> bool:
        """Delete a tracklist.

        Args:
            tracklist_id: UUID of the tracklist

        Returns:
            True if deleted, False if not found
        """
        with self.db.get_db_session() as session:
            stmt = select(Tracklist).where(Tracklist.id == tracklist_id)
            result = session.execute(stmt)
            tracklist = cast("Tracklist | None", result.scalar_one_or_none())

            if not tracklist:
                return False

            session.delete(tracklist)
            return True


class JobStatus(Enum):
    """Job status enumeration."""

    CREATED = "created"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class JobRepository:
    """Repository for Job entity operations."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initialize repository with database manager.

        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager

    def create(
        self,
        job_type: str,
        service_name: str | None = None,
        context: dict[str, Any] | None = None,
        correlation_id: UUID | None = None,
        parent_job_id: UUID | None = None,
    ) -> Job:
        """Create a new job.

        Args:
            job_type: Type of job (e.g., 'analysis', 'tracklist_generation')
            service_name: Name of the service creating the job
            context: Job-specific context data
            correlation_id: ID for correlating related jobs
            parent_job_id: Parent job ID for hierarchical jobs

        Returns:
            Created Job instance
        """
        with self.db.get_db_session() as session:
            job = Job(
                job_type=job_type,
                status=JobStatus.CREATED.value,
                service_name=service_name,
                context=context,
                correlation_id=correlation_id or uuid4(),
                parent_job_id=parent_job_id,
                created_at=datetime.now(UTC),
            )
            session.add(job)
            session.flush()
            session.refresh(job)
            return job

    def get_by_id(self, job_id: UUID) -> Job | None:
        """Get job by ID.

        Args:
            job_id: UUID of the job

        Returns:
            Job instance or None if not found
        """
        with self.db.get_db_session() as session:
            stmt = select(Job).where(Job.id == job_id)
            result = session.execute(stmt)
            return cast("Job | None", result.scalar_one_or_none())

    def get_by_correlation_id(self, correlation_id: UUID) -> list[Job]:
        """Get all jobs with a specific correlation ID.

        Args:
            correlation_id: Correlation UUID

        Returns:
            List of Job instances
        """
        with self.db.get_db_session() as session:
            stmt = select(Job).where(Job.correlation_id == correlation_id)
            result = session.execute(stmt)
            return list(result.scalars().all())

    def get_by_status(self, status: JobStatus, service_name: str | None = None) -> list[Job]:
        """Get jobs by status and optionally service name.

        Args:
            status: Job status
            service_name: Optional service name filter

        Returns:
            List of Job instances
        """
        with self.db.get_db_session() as session:
            stmt = select(Job).where(Job.status == status.value)
            if service_name:
                stmt = stmt.where(Job.service_name == service_name)
            result = session.execute(stmt)
            return list(result.scalars().all())

    def update_status(
        self,
        job_id: UUID,
        status: JobStatus,
        result: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> Job | None:
        """Update job status.

        Args:
            job_id: UUID of the job
            status: New status
            result: Optional result data
            error_message: Optional error message

        Returns:
            Updated Job instance or None if not found
        """
        with self.db.get_db_session() as session:
            stmt = select(Job).where(Job.id == job_id)
            result_query = session.execute(stmt)
            job = cast("Job | None", result_query.scalar_one_or_none())
            if not job:
                return None

            job.status = status.value
            job.updated_at = datetime.now(UTC)

            if status == JobStatus.RUNNING and not job.started_at:
                job.started_at = datetime.now(UTC)
            elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                job.completed_at = datetime.now(UTC)

            if result is not None:
                job.result = result
            if error_message is not None:
                job.error_message = error_message

            session.flush()
            session.refresh(job)
            return job

    def update_progress(self, job_id: UUID, progress: int, total_items: int | None = None) -> Job | None:
        """Update job progress.

        Args:
            job_id: UUID of the job
            progress: Current progress (e.g., items processed)
            total_items: Optional total number of items

        Returns:
            Updated Job instance or None if not found
        """
        with self.db.get_db_session() as session:
            stmt = select(Job).where(Job.id == job_id)
            result = session.execute(stmt)
            job = cast("Job | None", result.scalar_one_or_none())
            if not job:
                return None

            job.progress = progress
            if total_items is not None:
                job.total_items = total_items
            job.updated_at = datetime.now(UTC)

            session.flush()
            session.refresh(job)
            return job

    def create_child_job(
        self,
        parent_job_id: UUID,
        job_type: str,
        service_name: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> Job | None:
        """Create a child job.

        Args:
            parent_job_id: Parent job UUID
            job_type: Type of child job
            service_name: Name of the service creating the job
            context: Job-specific context data

        Returns:
            Created Job instance or None if parent not found
        """
        with self.db.get_db_session() as session:
            stmt = select(Job).where(Job.id == parent_job_id)
            result = session.execute(stmt)
            parent_job = cast("Job | None", result.scalar_one_or_none())
            if not parent_job:
                return None

            child_job = Job(
                job_type=job_type,
                status=JobStatus.CREATED.value,
                service_name=service_name,
                context=context,
                correlation_id=parent_job.correlation_id,
                parent_job_id=parent_job_id,
                created_at=datetime.now(UTC),
            )
            session.add(child_job)
            session.flush()
            session.refresh(child_job)
            return child_job

    def get_child_jobs(self, parent_job_id: UUID) -> list[Job]:
        """Get all child jobs for a parent job.

        Args:
            parent_job_id: Parent job UUID

        Returns:
            List of child Job instances
        """
        with self.db.get_db_session() as session:
            stmt = select(Job).where(Job.parent_job_id == parent_job_id)
            result = session.execute(stmt)
            return list(result.scalars().all())

    def cleanup_old_jobs(self, days: int = 30) -> int:
        """Clean up old completed/failed jobs.

        Args:
            days: Number of days to keep jobs (default 30)

        Returns:
            Number of jobs deleted
        """
        with self.db.get_db_session() as session:
            cutoff_date = datetime.now(UTC) - timedelta(days=days)
            stmt = select(Job).where(
                Job.status.in_([JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value]),
                Job.completed_at < cutoff_date,
            )
            result = session.execute(stmt)
            jobs_to_delete = list(result.scalars().all())
            count = len(jobs_to_delete)
            for job in jobs_to_delete:
                session.delete(job)
            return count
