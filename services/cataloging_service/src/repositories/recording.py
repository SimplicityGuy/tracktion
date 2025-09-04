"""Recording repository implementation."""

from collections.abc import Sequence
from uuid import UUID

from services.cataloging_service.src.models.recording import Recording
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .base import BaseRepository


class RecordingRepository(BaseRepository[Recording]):
    """Repository for Recording model operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the recording repository.

        Args:
            session: The database session
        """
        super().__init__(Recording, session)

    async def get_by_file_path(self, file_path: str) -> Recording | None:
        """Get a recording by file path.

        Args:
            file_path: The file path to search for

        Returns:
            The recording if found, None otherwise
        """
        result = await self.session.execute(select(Recording).where(Recording.file_path == file_path))
        return result.scalar_one_or_none()  # type: ignore[no-any-return]  # SQLAlchemy exec() returns Any at runtime

    async def get_by_sha256_hash(self, sha256_hash: str) -> Recording | None:
        """Get a recording by SHA256 hash.

        Args:
            sha256_hash: The SHA256 hash to search for

        Returns:
            The recording if found, None otherwise
        """
        result = await self.session.execute(select(Recording).where(Recording.sha256_hash == sha256_hash))
        return result.scalar_one_or_none()  # type: ignore[no-any-return]  # SQLAlchemy exec() returns Any at runtime

    async def get_by_xxh128_hash(self, xxh128_hash: str) -> Recording | None:
        """Get a recording by XXH128 hash.

        Args:
            xxh128_hash: The XXH128 hash to search for

        Returns:
            The recording if found, None otherwise
        """
        result = await self.session.execute(select(Recording).where(Recording.xxh128_hash == xxh128_hash))
        return result.scalar_one_or_none()  # type: ignore[no-any-return]  # SQLAlchemy exec() returns Any at runtime

    async def get_with_metadata(self, id: UUID) -> Recording | None:
        """Get a recording with its metadata.

        Args:
            id: The recording ID

        Returns:
            The recording with metadata if found, None otherwise
        """
        result = await self.session.execute(
            select(Recording).options(selectinload(Recording.metadata_items)).where(Recording.id == id)
        )
        return result.scalar_one_or_none()  # type: ignore[no-any-return]  # SQLAlchemy exec() returns Any at runtime

    async def get_with_tracklists(self, id: UUID) -> Recording | None:
        """Get a recording with its tracklists.

        Args:
            id: The recording ID

        Returns:
            The recording with tracklists if found, None otherwise
        """
        result = await self.session.execute(
            select(Recording).options(selectinload(Recording.tracklists)).where(Recording.id == id)
        )
        return result.scalar_one_or_none()  # type: ignore[no-any-return]  # SQLAlchemy exec() returns Any at runtime

    async def get_with_all_relations(self, id: UUID) -> Recording | None:
        """Get a recording with all its related data.

        Args:
            id: The recording ID

        Returns:
            The recording with all relations if found, None otherwise
        """
        result = await self.session.execute(
            select(Recording)
            .options(selectinload(Recording.metadata_items), selectinload(Recording.tracklists))
            .where(Recording.id == id)
        )
        return result.scalar_one_or_none()  # type: ignore[no-any-return]  # SQLAlchemy exec() returns Any at runtime

    async def search_by_file_name(self, file_name: str, limit: int = 100) -> Sequence[Recording]:
        """Search recordings by file name pattern.

        Args:
            file_name: The file name pattern to search for (case-insensitive)
            limit: Maximum number of results

        Returns:
            List of matching recordings
        """
        result = await self.session.execute(
            select(Recording).where(Recording.file_name.ilike(f"%{file_name}%")).limit(limit)  # type: ignore[attr-defined]  # file_name is non-nullable SQLAlchemy column, mypy can't infer .ilike() method availability at runtime
        )
        return result.scalars().all()  # type: ignore[no-any-return]  # SQLAlchemy exec() returns Any at runtime
