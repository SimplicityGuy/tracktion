"""Tracklist repository implementation."""

from collections.abc import Sequence
from typing import Any, cast
from uuid import UUID

from services.cataloging_service.src.models.tracklist import Tracklist
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .base import BaseRepository


class TracklistRepository(BaseRepository[Tracklist]):
    """Repository for Tracklist model operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the tracklist repository.

        Args:
            session: The database session
        """
        super().__init__(Tracklist, session)

    async def get_by_recording_id(self, recording_id: UUID) -> Sequence[Tracklist]:
        """Get all tracklists for a recording.

        Args:
            recording_id: The recording ID

        Returns:
            List of tracklists for the recording
        """
        result = await self.session.execute(select(Tracklist).where(Tracklist.recording_id == recording_id))
        return cast("Sequence[Tracklist]", result.scalars().all())

    async def get_by_source(self, recording_id: UUID, source: str) -> Tracklist | None:
        """Get tracklist by recording ID and source.

        Args:
            recording_id: The recording ID
            source: The tracklist source

        Returns:
            The tracklist if found, None otherwise
        """
        result = await self.session.execute(
            select(Tracklist).where((Tracklist.recording_id == recording_id) & (Tracklist.source == source))
        )
        return cast("Tracklist | None", result.scalar_one_or_none())

    async def upsert(
        self, recording_id: UUID, source: str, tracks: list[dict[str, Any]], cue_file_path: str | None = None
    ) -> Tracklist:
        """Insert or update tracklist.

        Args:
            recording_id: The recording ID
            source: The tracklist source
            tracks: List of track dictionaries
            cue_file_path: Optional path to the cue file

        Returns:
            The created or updated tracklist
        """
        existing = await self.get_by_source(recording_id, source)
        if existing:
            existing.tracks = tracks
            if cue_file_path is not None:
                existing.cue_file_path = cue_file_path
            await self.session.flush()
            return existing
        return await self.create(recording_id=recording_id, source=source, tracks=tracks, cue_file_path=cue_file_path)

    async def delete_by_recording_id(self, recording_id: UUID) -> int:
        """Delete all tracklists for a recording.

        Args:
            recording_id: The recording ID

        Returns:
            Number of deleted tracklists
        """
        result = await self.session.execute(delete(Tracklist).where(Tracklist.recording_id == recording_id))
        return cast("int", result.rowcount)

    async def search_by_track_title(self, title: str, limit: int = 100) -> Sequence[Tracklist]:
        """Search tracklists containing tracks with matching titles.

        Args:
            title: The track title pattern to search for
            limit: Maximum number of results

        Returns:
            List of tracklists containing matching tracks
        """
        # Using PostgreSQL JSONB containment operator
        # This searches for tracks where any track's title contains the search pattern
        result = await self.session.execute(
            select(Tracklist)
            .where(func.jsonb_path_exists(Tracklist.tracks, f'$[*] ? (@.title like_regex "{title}" flag "i")'))
            .limit(limit)
        )
        return cast("Sequence[Tracklist]", result.scalars().all())

    async def get_tracks_by_recording_id(self, recording_id: UUID) -> list[dict[str, Any]]:
        """Get all tracks from all tracklists for a recording.

        Args:
            recording_id: The recording ID

        Returns:
            Combined list of all tracks from all tracklists
        """
        tracklists = await self.get_by_recording_id(recording_id)
        all_tracks = []
        for tracklist in tracklists:
            all_tracks.extend(tracklist.tracks)
        return all_tracks
