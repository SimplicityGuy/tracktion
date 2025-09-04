"""Async repository pattern implementations for database operations."""

import logging
from collections.abc import AsyncGenerator
from typing import Any, cast
from uuid import UUID

from sqlalchemy import String, delete, select, update
from sqlalchemy.orm import selectinload

from .async_database import AsyncDatabaseManager
from .models import Metadata, Recording, Tracklist

logger = logging.getLogger(__name__)


class AsyncRecordingRepository:
    """Async repository for Recording entity operations."""

    def __init__(self, db_manager: AsyncDatabaseManager) -> None:
        """Initialize repository with async database manager.

        Args:
            db_manager: Async database manager instance
        """
        self.db = db_manager

    async def create(
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
        """
        async with self.db.get_db_session() as session:
            recording = Recording(
                file_path=file_path,
                file_name=file_name,
                sha256_hash=sha256_hash,
                xxh128_hash=xxh128_hash,
            )
            session.add(recording)
            await session.flush()
            await session.refresh(recording)
            return recording

    async def get_by_id(self, recording_id: UUID) -> Recording | None:
        """Get recording by ID.

        Args:
            recording_id: UUID of the recording

        Returns:
            Recording instance or None if not found
        """
        async with self.db.get_db_session() as session:
            stmt = select(Recording).where(Recording.id == recording_id)
            result = await session.execute(stmt)
            return cast("Recording | None", result.scalar_one_or_none())

    async def get_by_file_path(self, file_path: str) -> Recording | None:
        """Get recording by file path.

        Args:
            file_path: Full path to the file

        Returns:
            Recording instance or None if not found
        """
        async with self.db.get_db_session() as session:
            stmt = select(Recording).where(Recording.file_path == file_path)
            result = await session.execute(stmt)
            return cast("Recording | None", result.scalar_one_or_none())

    async def get_by_hash(self, sha256_hash: str) -> Recording | None:
        """Get recording by SHA256 hash.

        Args:
            sha256_hash: SHA256 hash of the file

        Returns:
            Recording instance or None if not found
        """
        async with self.db.get_db_session() as session:
            stmt = select(Recording).where(Recording.sha256_hash == sha256_hash)
            result = await session.execute(stmt)
            return cast("Recording | None", result.scalar_one_or_none())

    async def get_all(self, limit: int | None = None, offset: int | None = None) -> list[Recording]:
        """Get all recordings with optional pagination.

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip

        Returns:
            List of Recording instances
        """
        async with self.db.get_db_session() as session:
            stmt = select(Recording)
            if offset:
                stmt = stmt.offset(offset)
            if limit:
                stmt = stmt.limit(limit)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def update(self, recording_id: UUID, **kwargs: Any) -> Recording | None:
        """Update a recording.

        Args:
            recording_id: UUID of the recording
            **kwargs: Fields to update

        Returns:
            Updated Recording instance or None if not found
        """
        async with self.db.get_db_session() as session:
            stmt = update(Recording).where(Recording.id == recording_id).values(**kwargs).returning(Recording)
            result = await session.execute(stmt)
            await session.commit()
            return cast("Recording | None", result.scalar_one_or_none())

    async def delete(self, recording_id: UUID) -> bool:
        """Delete a recording.

        Args:
            recording_id: UUID of the recording

        Returns:
            True if deleted, False if not found
        """
        async with self.db.get_db_session() as session:
            stmt = delete(Recording).where(Recording.id == recording_id)
            result = await session.execute(stmt)
            await session.commit()
            return cast("int", result.rowcount) > 0

    async def batch_create(self, recordings: list[dict[str, Any]]) -> list[Recording]:
        """Create multiple recordings in a batch.

        Args:
            recordings: List of recording data dictionaries

        Returns:
            List of created Recording instances
        """
        async with self.db.get_db_session() as session:
            recording_objs = [Recording(**data) for data in recordings]
            session.add_all(recording_objs)
            await session.flush()
            for recording in recording_objs:
                await session.refresh(recording)
            return recording_objs


class AsyncMetadataRepository:
    """Async repository for Metadata entity operations."""

    def __init__(self, db_manager: AsyncDatabaseManager) -> None:
        """Initialize repository with async database manager.

        Args:
            db_manager: Async database manager instance
        """
        self.db = db_manager

    async def create(self, recording_id: UUID, key: str, value: str) -> Metadata:
        """Create a new metadata entry.

        Args:
            recording_id: UUID of the associated recording
            key: Metadata key
            value: Metadata value

        Returns:
            Created Metadata instance
        """
        async with self.db.get_db_session() as session:
            metadata = Metadata(recording_id=recording_id, key=key, value=value)
            session.add(metadata)
            await session.flush()
            await session.refresh(metadata)
            return metadata

    async def get_by_recording(self, recording_id: UUID) -> list[Metadata]:
        """Get all metadata for a recording.

        Args:
            recording_id: UUID of the recording

        Returns:
            List of Metadata instances
        """
        async with self.db.get_db_session() as session:
            stmt = select(Metadata).where(Metadata.recording_id == recording_id)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def get_by_key(self, recording_id: UUID, key: str) -> Metadata | None:
        """Get metadata by recording ID and key.

        Args:
            recording_id: UUID of the recording
            key: Metadata key

        Returns:
            Metadata instance or None if not found
        """
        async with self.db.get_db_session() as session:
            stmt = select(Metadata).where(Metadata.recording_id == recording_id, Metadata.key == key)
            result = await session.execute(stmt)
            return cast("Metadata | None", result.scalar_one_or_none())

    async def update(self, metadata_id: UUID, value: str) -> Metadata | None:
        """Update a metadata value.

        Args:
            metadata_id: ID of the metadata entry
            value: New value

        Returns:
            Updated Metadata instance or None if not found
        """
        async with self.db.get_db_session() as session:
            stmt = update(Metadata).where(Metadata.id == metadata_id).values(value=value).returning(Metadata)
            result = await session.execute(stmt)
            await session.commit()
            return cast("Metadata | None", result.scalar_one_or_none())

    async def delete(self, metadata_id: UUID) -> bool:
        """Delete a metadata entry.

        Args:
            metadata_id: ID of the metadata entry

        Returns:
            True if deleted, False if not found
        """
        async with self.db.get_db_session() as session:
            stmt = delete(Metadata).where(Metadata.id == metadata_id)
            result = await session.execute(stmt)
            await session.commit()
            return cast("int", result.rowcount) > 0

    async def batch_create(self, recording_id: UUID, metadata_dict: dict[str, str]) -> list[Metadata]:
        """Create multiple metadata entries for a recording.

        Args:
            recording_id: UUID of the recording
            metadata_dict: Dictionary of key-value pairs

        Returns:
            List of created Metadata instances
        """
        async with self.db.get_db_session() as session:
            metadata_objs = [Metadata(recording_id=recording_id, key=k, value=v) for k, v in metadata_dict.items()]
            session.add_all(metadata_objs)
            await session.flush()
            for metadata in metadata_objs:
                await session.refresh(metadata)
            return metadata_objs


class AsyncTracklistRepository:
    """Async repository for Tracklist entity operations."""

    def __init__(self, db_manager: AsyncDatabaseManager) -> None:
        """Initialize repository with async database manager.

        Args:
            db_manager: Async database manager instance
        """
        self.db = db_manager

    async def create(
        self,
        recording_id: UUID,
        source: str,
        tracks: list[dict[str, Any]] | None,
        cue_file_path: str | None = None,
    ) -> Tracklist:
        """Create a new tracklist.

        Args:
            recording_id: UUID of the associated recording
            source: Source of the tracklist
            tracks: Track data as list of dictionaries or None
            cue_file_path: Optional path to cue file

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
            return tracklist

    async def get_by_recording(self, recording_id: UUID) -> Tracklist | None:
        """Get tracklist for a recording.

        Args:
            recording_id: UUID of the recording

        Returns:
            Tracklist instance or None if not found
        """
        async with self.db.get_db_session() as session:
            stmt = select(Tracklist).where(Tracklist.recording_id == recording_id)
            result = await session.execute(stmt)
            return cast("Tracklist | None", result.scalar_one_or_none())

    async def get_with_recording(self, tracklist_id: UUID) -> Tracklist | None:
        """Get tracklist with its associated recording.

        Args:
            tracklist_id: ID of the tracklist

        Returns:
            Tracklist instance with recording loaded or None
        """
        async with self.db.get_db_session() as session:
            stmt = select(Tracklist).options(selectinload(Tracklist.recording)).where(Tracklist.id == tracklist_id)
            result = await session.execute(stmt)
            return cast("Tracklist | None", result.scalar_one_or_none())

    async def update(self, tracklist_id: UUID, **kwargs: Any) -> Tracklist | None:
        """Update a tracklist.

        Args:
            tracklist_id: ID of the tracklist
            **kwargs: Fields to update

        Returns:
            Updated Tracklist instance or None if not found
        """
        async with self.db.get_db_session() as session:
            stmt = update(Tracklist).where(Tracklist.id == tracklist_id).values(**kwargs).returning(Tracklist)
            result = await session.execute(stmt)
            await session.commit()
            return cast("Tracklist | None", result.scalar_one_or_none())

    async def delete(self, tracklist_id: UUID) -> bool:
        """Delete a tracklist.

        Args:
            tracklist_id: ID of the tracklist

        Returns:
            True if deleted, False if not found
        """
        async with self.db.get_db_session() as session:
            stmt = delete(Tracklist).where(Tracklist.id == tracklist_id)
            result = await session.execute(stmt)
            await session.commit()
            return cast("int", result.rowcount) > 0

    async def search_by_track(self, track_name: str) -> list[Tracklist]:
        """Search tracklists by track name.

        Args:
            track_name: Name to search for in tracks

        Returns:
            List of matching Tracklist instances
        """
        async with self.db.get_db_session() as session:
            # Using PostgreSQL JSONB containment operator
            stmt = select(Tracklist).where(Tracklist.tracks.cast(String).contains(track_name))  # type: ignore[attr-defined]  # SQLAlchemy Column methods not recognized by mypy
            result = await session.execute(stmt)
            return list(result.scalars().all())


class AsyncBatchOperations:
    """Async batch operations for efficient bulk processing."""

    def __init__(self, db_manager: AsyncDatabaseManager) -> None:
        """Initialize with async database manager.

        Args:
            db_manager: Async database manager instance
        """
        self.db = db_manager

    async def bulk_insert_recordings(self, recordings_data: list[dict[str, Any]]) -> int:
        """Bulk insert recordings efficiently.

        Args:
            recordings_data: List of recording data dictionaries

        Returns:
            Number of recordings inserted
        """
        async with self.db.get_db_session() as session:
            # Use bulk_insert_mappings for efficiency
            await session.execute(Recording.__table__.insert(), recordings_data)
            await session.commit()
            return len(recordings_data)

    async def bulk_update_recordings(self, updates: list[dict[str, Any]]) -> int:
        """Bulk update recordings efficiently.

        Args:
            updates: List of dicts with 'id' and fields to update

        Returns:
            Number of recordings updated
        """
        async with self.db.get_db_session() as session:
            for update_data in updates:
                recording_id = update_data.pop("id")
                stmt = update(Recording).where(Recording.id == recording_id).values(**update_data)
                await session.execute(stmt)
            await session.commit()
            return len(updates)

    async def stream_large_dataset(self, query_limit: int = 1000) -> AsyncGenerator[list[Recording]]:
        """Stream large dataset in chunks for memory efficiency.

        Args:
            query_limit: Number of records per chunk

        Yields:
            Chunks of Recording instances
        """
        offset = 0
        async with self.db.get_db_session() as session:
            while True:
                stmt = select(Recording).offset(offset).limit(query_limit)
                result = await session.execute(stmt)
                recordings = list(result.scalars().all())

                if not recordings:
                    break

                yield recordings
                offset += query_limit
