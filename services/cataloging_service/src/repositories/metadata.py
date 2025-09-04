"""Metadata repository implementation."""

from collections.abc import Sequence
from uuid import UUID

from services.cataloging_service.src.models.metadata import Metadata
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from .base import BaseRepository


class MetadataRepository(BaseRepository[Metadata]):
    """Repository for Metadata model operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the metadata repository.

        Args:
            session: The database session
        """
        super().__init__(Metadata, session)

    async def get_by_recording_id(self, recording_id: UUID) -> Sequence[Metadata]:
        """Get all metadata for a recording.

        Args:
            recording_id: The recording ID

        Returns:
            List of metadata entries for the recording
        """
        result = await self.session.execute(select(Metadata).where(Metadata.recording_id == recording_id))
        return result.scalars().all()  # type: ignore[no-any-return]  # SQLAlchemy exec() returns Any at runtime

    async def get_by_key(self, recording_id: UUID, key: str) -> Metadata | None:
        """Get metadata by recording ID and key.

        Args:
            recording_id: The recording ID
            key: The metadata key

        Returns:
            The metadata entry if found, None otherwise
        """
        result = await self.session.execute(
            select(Metadata).where((Metadata.recording_id == recording_id) & (Metadata.key == key))
        )
        return result.scalar_one_or_none()  # type: ignore[no-any-return]  # SQLAlchemy exec() returns Any at runtime

    async def upsert(self, recording_id: UUID, key: str, value: str) -> Metadata:
        """Insert or update metadata entry.

        Args:
            recording_id: The recording ID
            key: The metadata key
            value: The metadata value

        Returns:
            The created or updated metadata entry
        """
        existing = await self.get_by_key(recording_id, key)
        if existing:
            existing.value = value
            await self.session.flush()
            return existing
        return await self.create(recording_id=recording_id, key=key, value=value)

    async def bulk_create(self, recording_id: UUID, metadata_dict: dict[str, str]) -> list[Metadata]:
        """Create multiple metadata entries for a recording.

        Args:
            recording_id: The recording ID
            metadata_dict: Dictionary of key-value pairs

        Returns:
            List of created metadata entries
        """
        metadata_entries = []
        for key, value in metadata_dict.items():
            entry = Metadata(recording_id=recording_id, key=key, value=value)
            self.session.add(entry)
            metadata_entries.append(entry)

        await self.session.flush()
        return metadata_entries

    async def delete_by_recording_id(self, recording_id: UUID) -> int:
        """Delete all metadata for a recording.

        Args:
            recording_id: The recording ID

        Returns:
            Number of deleted entries
        """
        result = await self.session.execute(delete(Metadata).where(Metadata.recording_id == recording_id))
        return result.rowcount  # type: ignore[no-any-return]  # SQLAlchemy exec() returns Any at runtime

    async def search_by_key_value(self, key: str, value: str, limit: int = 100) -> Sequence[Metadata]:
        """Search metadata by key and value pattern.

        Args:
            key: The metadata key
            value: The value pattern to search for (case-insensitive)
            limit: Maximum number of results

        Returns:
            List of matching metadata entries
        """
        result = await self.session.execute(
            select(Metadata).where((Metadata.key == key) & (Metadata.value.ilike(f"%{value}%"))).limit(limit)  # type: ignore[attr-defined]  # value is non-nullable SQLAlchemy column, mypy can't infer .ilike() method availability at runtime
        )
        return result.scalars().all()  # type: ignore[no-any-return]  # SQLAlchemy exec() returns Any at runtime
