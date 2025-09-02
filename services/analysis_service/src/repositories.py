"""Async repository pattern implementations for Analysis Service."""

import logging
from typing import Any, cast
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.orm import selectinload

from shared.core_types.src.async_database import AsyncDatabaseManager
from shared.core_types.src.models import AnalysisResult, Metadata, Recording, Tracklist

logger = logging.getLogger(__name__)


class AsyncRecordingRepository:
    """Async repository for Recording entity operations."""

    def __init__(self, db_manager: AsyncDatabaseManager) -> None:
        """Initialize repository with async database manager.

        Args:
            db_manager: Async database manager instance
        """
        self.db = db_manager

    async def get_by_id(self, recording_id: UUID) -> Recording | None:
        """Get recording by ID with relationships.

        Args:
            recording_id: UUID of the recording

        Returns:
            Recording instance or None if not found
        """
        async with self.db.get_db_session() as session:
            result = await session.execute(
                select(Recording)
                .where(Recording.id == recording_id)
                .options(selectinload(Recording.metadata_items))
                .options(selectinload(Recording.tracklist))
            )
            return cast("Recording | None", result.scalar_one_or_none())

    async def get_by_file_path(self, file_path: str) -> Recording | None:
        """Get recording by file path.

        Args:
            file_path: Full path to the file

        Returns:
            Recording instance or None if not found
        """
        async with self.db.get_db_session() as session:
            result = await session.execute(select(Recording).where(Recording.file_path == file_path))
            return cast("Recording | None", result.scalar_one_or_none())

    async def list_paginated(
        self,
        page: int = 1,
        limit: int = 20,
        status_filter: str | None = None,
    ) -> tuple[list[Recording], int]:
        """Get paginated list of recordings.

        Args:
            page: Page number (1-based)
            limit: Maximum records per page
            status_filter: Optional status filter

        Returns:
            Tuple of (recordings list, total count)
        """
        async with self.db.get_db_session() as session:
            # Build query
            query = select(Recording).where(Recording.deleted_at.is_(None))

            if status_filter:
                query = query.where(Recording.processing_status == status_filter)

            # Get total count
            count_query = select(func.count()).select_from(query.subquery())
            total_result = await session.execute(count_query)
            total_count = total_result.scalar_one()

            # Get paginated results
            offset = (page - 1) * limit
            result = await session.execute(
                query.offset(offset)
                .limit(limit)
                .options(selectinload(Recording.metadata_items))
                .order_by(Recording.created_at.desc())
            )
            recordings = list(result.scalars().all())

            return recordings, total_count

    async def update_status(self, recording_id: UUID, status: str, error: str | None = None) -> bool:
        """Update recording processing status.

        Args:
            recording_id: UUID of the recording
            status: New processing status
            error: Optional error message

        Returns:
            True if updated, False if not found
        """
        async with self.db.get_db_session() as session:
            result = await session.execute(
                update(Recording)
                .where(Recording.id == recording_id)
                .values(processing_status=status, processing_error=error, updated_at=func.current_timestamp())
            )
            return cast("bool", result.rowcount > 0)

    async def create(
        self,
        file_path: str,
        file_name: str,
        file_size: int | None = None,
        sha256_hash: str | None = None,
        xxh128_hash: str | None = None,
    ) -> Recording:
        """Create a new recording.

        Args:
            file_path: Full path to the file
            file_name: Name of the file
            file_size: Size of the file in bytes
            sha256_hash: Optional SHA256 hash of file
            xxh128_hash: Optional XXH128 hash of file

        Returns:
            Created Recording instance
        """
        async with self.db.get_db_session() as session:
            recording = Recording(
                file_path=file_path,
                file_name=file_name,
                file_size=file_size,
                sha256_hash=sha256_hash,
                xxh128_hash=xxh128_hash,
                processing_status="pending",
            )
            session.add(recording)
            await session.flush()
            await session.refresh(recording)
            await session.commit()  # Ensure transaction is committed
            return recording


class AsyncMetadataRepository:
    """Async repository for Metadata entity operations."""

    def __init__(self, db_manager: AsyncDatabaseManager) -> None:
        """Initialize repository with async database manager.

        Args:
            db_manager: Async database manager instance
        """
        self.db = db_manager

    async def get_by_recording_id(self, recording_id: UUID) -> list[Metadata]:
        """Get all metadata for a recording.

        Args:
            recording_id: UUID of the recording

        Returns:
            List of metadata items
        """
        async with self.db.get_db_session() as session:
            result = await session.execute(select(Metadata).where(Metadata.recording_id == recording_id))
            return list(result.scalars().all())

    async def get_by_key(self, recording_id: UUID, key: str) -> Metadata | None:
        """Get specific metadata by key.

        Args:
            recording_id: UUID of the recording
            key: Metadata key to look up

        Returns:
            Metadata instance or None if not found
        """
        async with self.db.get_db_session() as session:
            result = await session.execute(
                select(Metadata).where(Metadata.recording_id == recording_id, Metadata.key == key)
            )
            return cast("Metadata | None", result.scalar_one_or_none())

    async def create_batch(self, recording_id: UUID, metadata_items: dict[str, str]) -> list[Metadata]:
        """Create multiple metadata items for a recording.

        Args:
            recording_id: UUID of the recording
            metadata_items: Dict of key-value pairs

        Returns:
            List of created Metadata instances
        """
        async with self.db.get_db_session() as session:
            metadata_list = []
            for key, value in metadata_items.items():
                metadata = Metadata(recording_id=recording_id, key=key, value=str(value))
                session.add(metadata)
                metadata_list.append(metadata)

            await session.flush()
            for metadata in metadata_list:
                await session.refresh(metadata)
            await session.commit()  # Ensure transaction is committed

            return metadata_list

    async def update_by_key(self, recording_id: UUID, key: str, value: str) -> Metadata | None:
        """Update metadata value by key.

        Args:
            recording_id: UUID of the recording
            key: Metadata key to update
            value: New value

        Returns:
            Updated Metadata instance or None if not found
        """
        async with self.db.get_db_session() as session:
            result = await session.execute(
                update(Metadata)
                .where(Metadata.recording_id == recording_id, Metadata.key == key)
                .values(value=value)
                .returning(Metadata)
            )
            return cast("Metadata | None", result.scalar_one_or_none())


class AsyncTracklistRepository:
    """Async repository for Tracklist entity operations."""

    def __init__(self, db_manager: AsyncDatabaseManager) -> None:
        """Initialize repository with async database manager.

        Args:
            db_manager: Async database manager instance
        """
        self.db = db_manager

    async def get_by_recording_id(self, recording_id: UUID) -> Tracklist | None:
        """Get tracklist for a recording.

        Args:
            recording_id: UUID of the recording

        Returns:
            Tracklist instance or None if not found
        """
        async with self.db.get_db_session() as session:
            result = await session.execute(select(Tracklist).where(Tracklist.recording_id == recording_id))
            return cast("Tracklist | None", result.scalar_one_or_none())

    async def create(
        self,
        recording_id: UUID,
        source: str,
        tracks: list[dict[str, Any]],
        cue_file_path: str | None = None,
    ) -> Tracklist:
        """Create a new tracklist.

        Args:
            recording_id: UUID of the recording
            source: Source of the tracklist (e.g., "manual", "1001tracklists")
            tracks: List of track information dicts
            cue_file_path: Optional path to CUE file

        Returns:
            Created Tracklist instance
        """
        async with self.db.get_db_session() as session:
            tracklist = Tracklist(
                recording_id=recording_id,
                source=source,
                tracks=tracks,
                cue_file_path=cue_file_path,
            )
            session.add(tracklist)
            await session.flush()
            await session.refresh(tracklist)
            await session.commit()  # Ensure transaction is committed
            return tracklist

    async def update_tracks(self, recording_id: UUID, tracks: list[dict[str, Any]]) -> Tracklist | None:
        """Update tracks for a tracklist.

        Args:
            recording_id: UUID of the recording
            tracks: Updated track information

        Returns:
            Updated Tracklist instance or None if not found
        """
        async with self.db.get_db_session() as session:
            result = await session.execute(
                update(Tracklist)
                .where(Tracklist.recording_id == recording_id)
                .values(tracks=tracks)
                .returning(Tracklist)
            )
            return cast("Tracklist | None", result.scalar_one_or_none())


class AsyncAnalysisResultRepository:
    """Async repository for AnalysisResult entity operations."""

    def __init__(self, db_manager: AsyncDatabaseManager) -> None:
        """Initialize repository with async database manager.

        Args:
            db_manager: Async database manager instance
        """
        self.db = db_manager

    async def get_by_recording_id(self, recording_id: UUID, analysis_type: str | None = None) -> list[AnalysisResult]:
        """Get analysis results for a recording.

        Args:
            recording_id: UUID of the recording
            analysis_type: Optional filter by analysis type

        Returns:
            List of analysis results
        """
        async with self.db.get_db_session() as session:
            query = select(AnalysisResult).where(AnalysisResult.recording_id == recording_id)

            if analysis_type:
                query = query.where(AnalysisResult.analysis_type == analysis_type)

            result = await session.execute(query.order_by(AnalysisResult.created_at.desc()))
            return list(result.scalars().all())

    async def get_latest_by_type(self, recording_id: UUID, analysis_type: str) -> AnalysisResult | None:
        """Get latest analysis result of a specific type.

        Args:
            recording_id: UUID of the recording
            analysis_type: Type of analysis

        Returns:
            Latest AnalysisResult or None
        """
        async with self.db.get_db_session() as session:
            result = await session.execute(
                select(AnalysisResult)
                .where(AnalysisResult.recording_id == recording_id, AnalysisResult.analysis_type == analysis_type)
                .order_by(AnalysisResult.created_at.desc())
                .limit(1)
            )
            return cast("AnalysisResult | None", result.scalar_one_or_none())

    async def create(
        self,
        recording_id: UUID,
        analysis_type: str,
        result_data: dict[str, Any],
        confidence_score: float | None = None,
        processing_time_ms: int | None = None,
    ) -> AnalysisResult:
        """Create a new analysis result.

        Args:
            recording_id: UUID of the recording
            analysis_type: Type of analysis performed
            result_data: Analysis result data
            confidence_score: Confidence score (0.0-1.0)
            processing_time_ms: Processing time in milliseconds

        Returns:
            Created AnalysisResult instance
        """
        async with self.db.get_db_session() as session:
            analysis_result = AnalysisResult(
                recording_id=recording_id,
                analysis_type=analysis_type,
                result_data=result_data,
                confidence_score=confidence_score,
                processing_time_ms=processing_time_ms,
                status="completed",
            )
            session.add(analysis_result)
            await session.flush()
            await session.refresh(analysis_result)
            await session.commit()  # Ensure transaction is committed
            return analysis_result

    async def update_status(
        self,
        result_id: UUID,
        status: str,
        error_message: str | None = None,
    ) -> bool:
        """Update analysis result status.

        Args:
            result_id: UUID of the analysis result
            status: New status
            error_message: Optional error message

        Returns:
            True if updated, False if not found
        """
        async with self.db.get_db_session() as session:
            result = await session.execute(
                update(AnalysisResult)
                .where(AnalysisResult.id == result_id)
                .values(status=status, error_message=error_message, updated_at=func.current_timestamp())
            )
            return cast("bool", result.rowcount > 0)
