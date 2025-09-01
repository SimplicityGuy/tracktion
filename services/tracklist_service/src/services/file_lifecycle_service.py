"""Service for handling file lifecycle events (created, modified, deleted, moved, renamed)."""

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.core_types.src.models import Recording

logger = logging.getLogger(__name__)


class FileLifecycleService:
    """Service for handling file lifecycle events."""

    def __init__(self, session: AsyncSession):
        """Initialize file lifecycle service.

        Args:
            session: Database session
        """
        self.session = session

    async def handle_file_created(
        self,
        file_path: str,
        sha256_hash: str | None = None,
        xxh128_hash: str | None = None,
        file_size: int | None = None,
    ) -> tuple[bool, str | None]:
        """Handle file creation event.

        Args:
            file_path: Path to the created file
            sha256_hash: SHA256 hash of the file
            xxh128_hash: XXH128 hash of the file
            file_size: Size of the file in bytes

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Check if file already exists (by hash)
            existing = None
            if sha256_hash:
                query = select(Recording).where(Recording.sha256_hash == sha256_hash, Recording.deleted_at.is_(None))
                result = await self.session.execute(query)
                existing = result.scalar_one_or_none()

            if existing:
                # File already exists, update path if different
                if existing.file_path != file_path:
                    existing.file_path = file_path
                    existing.file_name = Path(file_path).name
                    existing.updated_at = datetime.now(UTC)
                    logger.info(f"Updated existing recording path: {file_path}")
            else:
                # Create new recording
                recording = Recording(
                    file_path=file_path,
                    file_name=Path(file_path).name,
                    sha256_hash=sha256_hash,
                    xxh128_hash=xxh128_hash,
                    file_size=file_size,
                    processing_status="pending",
                )
                self.session.add(recording)
                logger.info(f"Created new recording for: {file_path}")

            await self.session.commit()
            return True, None

        except Exception as e:
            logger.error(f"Failed to handle file creation for {file_path}: {e}")
            await self.session.rollback()
            return False, str(e)

    async def handle_file_modified(
        self,
        file_path: str,
        sha256_hash: str | None = None,
        xxh128_hash: str | None = None,
        file_size: int | None = None,
    ) -> tuple[bool, str | None]:
        """Handle file modification event.

        Args:
            file_path: Path to the modified file
            sha256_hash: New SHA256 hash of the file
            xxh128_hash: New XXH128 hash of the file
            file_size: New size of the file in bytes

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Find recording by path
            query = select(Recording).where(Recording.file_path == file_path, Recording.deleted_at.is_(None))
            result = await self.session.execute(query)
            recording = result.scalar_one_or_none()

            if recording:
                # Update hashes and size
                if sha256_hash:
                    recording.sha256_hash = sha256_hash
                if xxh128_hash:
                    recording.xxh128_hash = xxh128_hash
                if file_size is not None:
                    recording.file_size = file_size

                recording.updated_at = datetime.now(UTC)
                recording.processing_status = "pending"  # Mark for reprocessing

                logger.info(f"Updated recording for modified file: {file_path}")
            else:
                # File not in database, treat as new file
                return await self.handle_file_created(file_path, sha256_hash, xxh128_hash, file_size)

            await self.session.commit()
            return True, None

        except Exception as e:
            logger.error(f"Failed to handle file modification for {file_path}: {e}")
            await self.session.rollback()
            return False, str(e)

    async def handle_file_deleted(self, file_path: str, soft_delete: bool = True) -> tuple[bool, str | None]:
        """Handle file deletion event.

        Args:
            file_path: Path to the deleted file
            soft_delete: If True, perform soft delete; otherwise hard delete

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Find recording by path
            query = select(Recording).where(Recording.file_path == file_path, Recording.deleted_at.is_(None))
            result = await self.session.execute(query)
            recording = result.scalar_one_or_none()

            if recording:
                if soft_delete:
                    # Soft delete - just mark as deleted
                    recording.deleted_at = datetime.now(UTC)
                    recording.processing_status = "deleted"
                    logger.info(f"Soft deleted recording: {file_path}")
                else:
                    # Hard delete - remove from database
                    # This will cascade delete related metadata and tracklists
                    await self.session.delete(recording)
                    logger.info(f"Hard deleted recording and related data: {file_path}")
            else:
                logger.warning(f"Recording not found for deletion: {file_path}")

            await self.session.commit()
            return True, None

        except Exception as e:
            logger.error(f"Failed to handle file deletion for {file_path}: {e}")
            await self.session.rollback()
            return False, str(e)

    async def handle_file_moved(
        self,
        old_path: str,
        new_path: str,
        sha256_hash: str | None = None,
        xxh128_hash: str | None = None,
    ) -> tuple[bool, str | None]:
        """Handle file move event.

        Args:
            old_path: Previous path of the file
            new_path: New path of the file
            sha256_hash: SHA256 hash of the file
            xxh128_hash: XXH128 hash of the file

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Find recording by old path
            query = select(Recording).where(Recording.file_path == old_path, Recording.deleted_at.is_(None))
            result = await self.session.execute(query)
            recording = result.scalar_one_or_none()

            if recording:
                # Update path
                recording.file_path = new_path
                recording.file_name = Path(new_path).name
                recording.updated_at = datetime.now(UTC)

                # Update hashes if provided (file content might have changed)
                if sha256_hash:
                    recording.sha256_hash = sha256_hash
                if xxh128_hash:
                    recording.xxh128_hash = xxh128_hash

                logger.info(f"Updated recording for moved file: {old_path} -> {new_path}")
            else:
                logger.warning(f"Recording not found for move: {old_path}")
                # Create new recording at new path
                return await self.handle_file_created(new_path, sha256_hash, xxh128_hash)

            await self.session.commit()
            return True, None

        except Exception as e:
            logger.error(f"Failed to handle file move from {old_path} to {new_path}: {e}")
            await self.session.rollback()
            return False, str(e)

    async def handle_file_renamed(
        self,
        old_path: str,
        new_path: str,
        sha256_hash: str | None = None,
        xxh128_hash: str | None = None,
    ) -> tuple[bool, str | None]:
        """Handle file rename event.

        This is essentially the same as move, but kept separate for clarity.

        Args:
            old_path: Previous path of the file
            new_path: New path of the file
            sha256_hash: SHA256 hash of the file
            xxh128_hash: XXH128 hash of the file

        Returns:
            Tuple of (success, error_message)
        """
        return await self.handle_file_moved(old_path, new_path, sha256_hash, xxh128_hash)

    async def recover_soft_deleted(self, file_path: str) -> tuple[bool, str | None]:
        """Recover a soft-deleted file.

        Args:
            file_path: Path of the file to recover

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Find soft-deleted recording
            query = select(Recording).where(Recording.file_path == file_path, Recording.deleted_at.is_not(None))
            result = await self.session.execute(query)
            recording = result.scalar_one_or_none()

            if recording:
                # Restore the recording
                recording.deleted_at = None
                recording.processing_status = "pending"
                recording.updated_at = datetime.now(UTC)
                logger.info(f"Recovered soft-deleted recording: {file_path}")
            else:
                logger.warning(f"No soft-deleted recording found for: {file_path}")
                return False, "Recording not found"

            await self.session.commit()
            return True, None

        except Exception as e:
            logger.error(f"Failed to recover soft-deleted file {file_path}: {e}")
            await self.session.rollback()
            return False, str(e)

    async def cleanup_old_soft_deletes(self, days_old: int = 30) -> int:
        """Clean up old soft-deleted records.

        Args:
            days_old: Number of days after which to permanently delete soft-deleted records

        Returns:
            Number of records deleted
        """
        try:
            cutoff_date = datetime.now(UTC) - timedelta(days=days_old)

            # Find old soft-deleted recordings
            query = select(Recording).where(Recording.deleted_at.is_not(None), Recording.deleted_at < cutoff_date)
            result = await self.session.execute(query)
            old_recordings = result.scalars().all()

            count = len(old_recordings)

            # Hard delete old soft-deleted recordings
            for recording in old_recordings:
                await self.session.delete(recording)
                logger.info(f"Permanently deleted old soft-deleted recording: {recording.file_path}")

            await self.session.commit()
            logger.info(f"Cleaned up {count} old soft-deleted recordings")
            return count

        except Exception as e:
            logger.error(f"Failed to cleanup old soft-deletes: {e}")
            await self.session.rollback()
            return 0

    async def get_active_recordings_query(self) -> Any:
        """Get query for active (non-deleted) recordings.

        Returns:
            SQLAlchemy query for active recordings
        """
        return select(Recording).where(Recording.deleted_at.is_(None))
