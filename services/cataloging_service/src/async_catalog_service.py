"""Async catalog service for managing audio file records."""

import logging
from typing import Any
from uuid import UUID

from shared.core_types.src.async_database import AsyncDatabaseManager
from shared.core_types.src.async_repositories import (
    AsyncBatchOperations,
    AsyncMetadataRepository,
    AsyncRecordingRepository,
    AsyncTracklistRepository,
)
from shared.core_types.src.models import Metadata, Recording, Tracklist

logger = logging.getLogger(__name__)


class AsyncCatalogService:
    """Async service for cataloging audio files and their metadata."""

    def __init__(self, db_manager: AsyncDatabaseManager) -> None:
        """Initialize the async catalog service.

        Args:
            db_manager: Async database manager instance
        """
        self.db_manager = db_manager
        self.recording_repo = AsyncRecordingRepository(db_manager)
        self.metadata_repo = AsyncMetadataRepository(db_manager)
        self.tracklist_repo = AsyncTracklistRepository(db_manager)
        self.batch_ops = AsyncBatchOperations(db_manager)

    async def catalog_file(
        self,
        file_path: str,
        file_name: str,
        sha256_hash: str | None = None,
        xxh128_hash: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Recording:
        """Catalog a new audio file with its metadata.

        Args:
            file_path: Full path to the audio file
            file_name: Name of the file
            sha256_hash: SHA256 hash of the file
            xxh128_hash: XXH128 hash of the file
            metadata: Optional metadata dictionary

        Returns:
            Created Recording instance
        """
        try:
            # Check if file already exists
            existing = await self.recording_repo.get_by_file_path(file_path)
            if existing:
                logger.info(f"File already cataloged: {file_path}")
                return existing

            # Create recording
            recording = await self.recording_repo.create(
                file_path=file_path,
                file_name=file_name,
                sha256_hash=sha256_hash,
                xxh128_hash=xxh128_hash,
            )

            # Add metadata if provided
            if metadata:
                if recording.id is None:
                    raise ValueError("Recording creation failed - ID is None")
                await self.metadata_repo.batch_create(recording_id=recording.id, metadata_dict=metadata)

            logger.info(f"Successfully cataloged file: {file_path}")
            return recording

        except Exception as e:
            logger.error(f"Error cataloging file {file_path}: {e}")
            raise

    async def update_file_metadata(self, recording_id: UUID, metadata: dict[str, str]) -> list[Metadata]:
        """Update metadata for a recording.

        Args:
            recording_id: UUID of the recording
            metadata: Dictionary of metadata key-value pairs

        Returns:
            List of created/updated Metadata instances
        """
        try:
            updated_metadata = []

            for key, value in metadata.items():
                # Check if metadata key exists
                existing = await self.metadata_repo.get_by_key(recording_id, key)

                if existing:
                    # Update existing metadata
                    if existing.id is None:
                        raise ValueError(f"Metadata entry has invalid ID for key '{key}'")
                    updated = await self.metadata_repo.update(existing.id, value)
                    if updated:
                        updated_metadata.append(updated)
                else:
                    # Create new metadata
                    created = await self.metadata_repo.create(recording_id, key, value)
                    updated_metadata.append(created)

            logger.info(f"Updated {len(updated_metadata)} metadata entries for recording {recording_id}")
            return updated_metadata

        except Exception as e:
            logger.error(f"Error updating metadata for recording {recording_id}: {e}")
            raise

    async def add_tracklist(
        self,
        recording_id: UUID,
        source: str,
        tracks: dict[str, Any],
        cue_file_path: str | None = None,
    ) -> Tracklist:
        """Add a tracklist to a recording.

        Args:
            recording_id: UUID of the recording
            source: Source of the tracklist (e.g., "1001tracklists", "manual")
            tracks: Track information as dictionary
            cue_file_path: Optional path to generated cue file

        Returns:
            Created Tracklist instance
        """
        try:
            # Check if tracklist already exists
            existing = await self.tracklist_repo.get_by_recording(recording_id)
            if existing:
                logger.info(f"Updating existing tracklist for recording {recording_id}")
                if existing.id is None:
                    raise ValueError("Existing tracklist has invalid ID")
                updated = await self.tracklist_repo.update(
                    tracklist_id=existing.id,
                    source=source,
                    tracks=tracks,
                    cue_file_path=cue_file_path,
                )
                return updated or existing

            # Create new tracklist
            tracklist = await self.tracklist_repo.create(
                recording_id=recording_id,
                source=source,
                tracks=tracks if isinstance(tracks, list) else None,
                cue_file_path=cue_file_path,
            )

            logger.info(f"Added tracklist to recording {recording_id}")
            return tracklist

        except Exception as e:
            logger.error(f"Error adding tracklist to recording {recording_id}: {e}")
            raise

    async def handle_file_deleted(self, file_path: str) -> bool:
        """Handle file deletion event.

        Args:
            file_path: Path to the deleted file

        Returns:
            True if recording was deleted, False if not found
        """
        try:
            recording = await self.recording_repo.get_by_file_path(file_path)
            if not recording:
                logger.info(f"No recording found for deleted file: {file_path}")
                return False

            # Delete the recording (cascades to metadata and tracklist)
            if recording.id is None:
                raise ValueError("Recording has invalid ID - cannot delete")
            deleted = await self.recording_repo.delete(recording.id)

            if deleted:
                logger.info(f"Deleted recording for file: {file_path}")

            return deleted

        except Exception as e:
            logger.error(f"Error handling file deletion for {file_path}: {e}")
            raise

    async def handle_file_moved(self, old_path: str, new_path: str, new_name: str) -> Recording | None:
        """Handle file move/rename event.

        Args:
            old_path: Original file path
            new_path: New file path
            new_name: New file name

        Returns:
            Updated Recording instance or None if not found
        """
        try:
            recording = await self.recording_repo.get_by_file_path(old_path)
            if not recording:
                logger.info(f"No recording found for moved file: {old_path}")
                return None

            # Update the recording with new path and name
            if recording.id is None:
                raise ValueError("Recording has invalid ID - cannot update")
            updated = await self.recording_repo.update(
                recording_id=recording.id, file_path=new_path, file_name=new_name
            )

            if updated:
                logger.info(f"Updated recording path from {old_path} to {new_path}")

            return updated

        except Exception as e:
            logger.error(f"Error handling file move from {old_path} to {new_path}: {e}")
            raise

    async def batch_catalog_files(self, files_data: list[dict[str, Any]]) -> list[Recording]:
        """Catalog multiple files in a batch operation.

        Args:
            files_data: List of file data dictionaries

        Returns:
            List of created Recording instances
        """
        try:
            # Filter out existing files
            new_files = []
            for file_data in files_data:
                existing = await self.recording_repo.get_by_file_path(file_data["file_path"])
                if not existing:
                    new_files.append(file_data)

            if not new_files:
                logger.info("All files already cataloged")
                return []

            # Batch create recordings
            recordings = await self.recording_repo.batch_create(new_files)

            logger.info(f"Batch cataloged {len(recordings)} files")
            return recordings

        except Exception as e:
            logger.error(f"Error in batch catalog operation: {e}")
            raise

    async def search_recordings(
        self, file_name: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[Recording]:
        """Search for recordings.

        Args:
            file_name: Optional file name to search for
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of matching Recording instances
        """
        try:
            # For now, just get all with pagination
            # TODO: Implement actual search with filters
            recordings = await self.recording_repo.get_all(limit=limit, offset=offset)

            # Filter by file name if provided
            if file_name:
                recordings = [r for r in recordings if r.file_name and file_name.lower() in r.file_name.lower()]

            return recordings

        except Exception as e:
            logger.error(f"Error searching recordings: {e}")
            raise

    async def get_recording_details(self, recording_id: UUID) -> dict[str, Any] | None:
        """Get detailed information about a recording.

        Args:
            recording_id: UUID of the recording

        Returns:
            Dictionary with recording details or None if not found
        """
        try:
            recording = await self.recording_repo.get_by_id(recording_id)
            if not recording:
                return None

            # Get metadata
            metadata_list = await self.metadata_repo.get_by_recording(recording_id)
            metadata_dict = {m.key: m.value for m in metadata_list}

            # Get tracklist
            tracklist = await self.tracklist_repo.get_by_recording(recording_id)

            return {
                "recording": {
                    "id": str(recording.id),
                    "file_path": recording.file_path,
                    "file_name": recording.file_name,
                    "sha256_hash": recording.sha256_hash,
                    "xxh128_hash": recording.xxh128_hash,
                    "created_at": (recording.created_at.isoformat() if recording.created_at else None),
                },
                "metadata": metadata_dict,
                "tracklist": (
                    {
                        "source": tracklist.source,
                        "tracks": tracklist.tracks,
                        "cue_file_path": tracklist.cue_file_path,
                    }
                    if tracklist
                    else None
                ),
            }

        except Exception as e:
            logger.error(f"Error getting recording details for {recording_id}: {e}")
            raise
