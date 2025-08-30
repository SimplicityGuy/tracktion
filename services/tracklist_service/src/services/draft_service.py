"""Draft management service for manual tracklist creation.

This service handles draft tracklist operations including creation,
version management, saving, retrieval, and publishing to final versions.
"""

import json
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

import redis
from sqlalchemy.orm import Session

from services.tracklist_service.src.models.tracklist import (
    Tracklist,
    TracklistDB,
    TrackEntry,
)


class DraftService:
    """Service for managing draft tracklists."""

    def __init__(self, db_session: Session, redis_client: Optional[redis.Redis] = None):
        """Initialize draft service.

        Args:
            db_session: SQLAlchemy database session.
            redis_client: Optional Redis client for caching.
        """
        self.db = db_session
        self.redis = redis_client
        self.cache_ttl = 3600  # 1 hour cache TTL
        self.batch_size = 100  # For batch operations

    def create_draft(
        self,
        audio_file_id: UUID,
        tracks: Optional[List[TrackEntry]] = None,
    ) -> Tracklist:
        """Create a new draft tracklist.

        Args:
            audio_file_id: ID of the associated audio file.
            tracks: Optional initial list of tracks.

        Returns:
            Created draft tracklist.
        """
        # Get the latest draft version for this audio file
        latest_version = self._get_latest_draft_version(audio_file_id)

        # Create new draft
        draft = Tracklist(
            id=uuid4(),
            audio_file_id=audio_file_id,
            source="manual",
            is_draft=True,
            draft_version=(latest_version + 1) if latest_version else 1,
            tracks=tracks or [],
            confidence_score=1.0,
            cue_file_id=None,
            parent_tracklist_id=None,
            default_cue_format=None,
        )

        # Save to database
        db_draft = TracklistDB.from_model(draft)
        self.db.add(db_draft)
        self.db.commit()

        # Cache in Redis if available
        if self.redis:
            self._cache_draft(draft)

        return draft

    def save_draft(
        self,
        draft_id: UUID,
        tracks: List[TrackEntry],
        auto_version: bool = True,
    ) -> Tracklist:
        """Save or update a draft tracklist.

        Args:
            draft_id: ID of the draft to update.
            tracks: Updated list of tracks.
            auto_version: Whether to create a new version automatically.

        Returns:
            Updated draft tracklist.

        Raises:
            ValueError: If draft not found or is not a draft.
        """
        # Fetch existing draft
        draft_db = self.db.query(TracklistDB).filter_by(id=draft_id).first()
        if not draft_db:
            raise ValueError(f"Draft with ID {draft_id} not found")

        if not draft_db.is_draft:
            raise ValueError(f"Tracklist {draft_id} is not a draft")

        draft = draft_db.to_model()

        # If auto-versioning and significant changes, create new version
        if auto_version and self._has_significant_changes(draft.tracks, tracks):
            new_draft = self.create_draft(draft.audio_file_id, tracks)
            new_draft_db = self.db.query(TracklistDB).filter_by(id=new_draft.id).first()
            if new_draft_db:
                new_draft_db.parent_tracklist_id = draft_id
            self.db.commit()

            # Cache new version
            if self.redis:
                self._cache_draft(new_draft)

            return new_draft

        # Update existing draft
        draft_db.tracks = [track.to_dict() for track in tracks]  # type: ignore
        draft_db.updated_at = datetime.utcnow()  # type: ignore
        self.db.commit()

        # Update cache
        draft.tracks = tracks
        draft.updated_at = draft_db.updated_at  # type: ignore
        if self.redis:
            self._cache_draft(draft)

        return draft  # type: ignore[no-any-return]

    def get_draft(self, draft_id: UUID) -> Optional[Tracklist]:
        """Retrieve a draft tracklist.

        Args:
            draft_id: ID of the draft to retrieve.

        Returns:
            Draft tracklist if found, None otherwise.
        """
        # Check cache first
        if self.redis:
            cached = self._get_cached_draft(draft_id)
            if cached:
                return cached

        # Query database
        draft_db = (
            self.db.query(TracklistDB)
            .filter_by(
                id=draft_id,
                is_draft=True,
            )
            .first()
        )

        if not draft_db:
            return None

        draft = draft_db.to_model()

        # Cache result
        if self.redis:
            self._cache_draft(draft)

        return draft  # type: ignore[no-any-return]

    def list_drafts(
        self,
        audio_file_id: UUID,
        include_versions: bool = False,
    ) -> List[Tracklist]:
        """List all draft tracklists for an audio file.

        Args:
            audio_file_id: ID of the audio file.
            include_versions: Whether to include all versions.

        Returns:
            List of draft tracklists.
        """
        query = self.db.query(TracklistDB).filter_by(
            audio_file_id=audio_file_id,
            is_draft=True,
        )

        if not include_versions:
            # Only get latest version of each draft lineage
            query = query.filter_by(parent_tracklist_id=None)

        drafts_db = query.order_by(TracklistDB.draft_version.desc()).all()  # type: ignore[union-attr]

        return [draft_db.to_model() for draft_db in drafts_db]

    def publish_draft(self, draft_id: UUID) -> Tracklist:
        """Publish a draft as a final tracklist.

        Args:
            draft_id: ID of the draft to publish.

        Returns:
            Published tracklist.

        Raises:
            ValueError: If draft not found or is not a draft.
        """
        # Get the draft
        draft_db = self.db.query(TracklistDB).filter_by(id=draft_id).first()
        if not draft_db:
            raise ValueError(f"Draft with ID {draft_id} not found")

        if not draft_db.is_draft:
            raise ValueError(f"Tracklist {draft_id} is already published")

        # Archive any existing published version for this audio file
        existing_published = (
            self.db.query(TracklistDB)
            .filter_by(
                audio_file_id=draft_db.audio_file_id,
                is_draft=False,
            )
            .first()
        )

        if existing_published:
            # Create a backup/archive entry
            archive = TracklistDB.from_model(existing_published.to_model())
            archive.id = uuid4()  # type: ignore
            archive.parent_tracklist_id = existing_published.id  # type: ignore
            self.db.add(archive)

        # Convert draft to published
        draft_db.is_draft = False  # type: ignore
        draft_db.updated_at = datetime.utcnow()  # type: ignore

        # Clear draft version since it's now published
        draft_db.draft_version = None  # type: ignore

        self.db.commit()

        # Clear cache for this draft
        if self.redis:
            self._invalidate_cache(draft_id)

        return draft_db.to_model()  # type: ignore[no-any-return]

    def delete_draft(self, draft_id: UUID) -> bool:
        """Delete a draft tracklist.

        Args:
            draft_id: ID of the draft to delete.

        Returns:
            True if deleted, False if not found.
        """
        draft_db = (
            self.db.query(TracklistDB)
            .filter_by(
                id=draft_id,
                is_draft=True,
            )
            .first()
        )

        if not draft_db:
            return False

        self.db.delete(draft_db)
        self.db.commit()

        # Clear cache
        if self.redis:
            self._invalidate_cache(draft_id)

        return True

    def batch_update_tracks(
        self,
        draft_id: UUID,
        track_updates: List[dict],
    ) -> Tracklist:
        """Batch update multiple tracks efficiently.

        Args:
            draft_id: ID of the draft.
            track_updates: List of track update dictionaries.

        Returns:
            Updated tracklist.

        Raises:
            ValueError: If draft not found.
        """
        # Get current draft
        draft_db = self.db.query(TracklistDB).filter_by(id=draft_id).first()
        if not draft_db:
            raise ValueError(f"Draft with ID {draft_id} not found")

        draft = draft_db.to_model()

        # Apply updates in batches
        tracks_dict = {track.position: track for track in draft.tracks}

        for update in track_updates:
            position = update.get("position")
            if position in tracks_dict:
                track = tracks_dict[position]
                # Update only provided fields
                for field, value in update.items():
                    if field != "position" and hasattr(track, field):
                        setattr(track, field, value)

        # Convert back to list and save
        updated_tracks = list(tracks_dict.values())
        return self.save_draft(draft_id, updated_tracks, auto_version=False)

    def bulk_create_drafts(
        self,
        draft_data: List[dict],
    ) -> List[Tracklist]:
        """Create multiple drafts efficiently in batch.

        Args:
            draft_data: List of draft creation data.

        Returns:
            List of created drafts.
        """
        created_drafts = []

        # Process in batches to avoid memory issues
        for i in range(0, len(draft_data), self.batch_size):
            batch = draft_data[i : i + self.batch_size]

            for data in batch:
                draft = self.create_draft(
                    audio_file_id=data["audio_file_id"],
                    tracks=data.get("tracks", []),
                )
                created_drafts.append(draft)

            # Commit batch
            self.db.commit()

        return created_drafts

    def _get_latest_draft_version(self, audio_file_id: UUID) -> Optional[int]:
        """Get the latest draft version number for an audio file.

        Args:
            audio_file_id: ID of the audio file.

        Returns:
            Latest version number or None if no drafts exist.
        """
        latest = (
            self.db.query(TracklistDB)
            .filter_by(
                audio_file_id=audio_file_id,
                is_draft=True,
            )
            .order_by(TracklistDB.draft_version.desc())  # type: ignore[union-attr]
            .first()
        )

        return latest.draft_version if latest else None  # type: ignore

    def _has_significant_changes(
        self,
        old_tracks: List[TrackEntry],
        new_tracks: List[TrackEntry],
    ) -> bool:
        """Check if track changes are significant enough for new version.

        Args:
            old_tracks: Previous track list.
            new_tracks: New track list.

        Returns:
            True if changes are significant.
        """
        # Different number of tracks is significant
        if len(old_tracks) != len(new_tracks):
            return True

        # Check for major changes in track order or content
        for old, new in zip(old_tracks, new_tracks):
            # Position change is significant
            if old.position != new.position:
                return True

            # Artist/title change is significant
            if old.artist != new.artist or old.title != new.title:
                return True

            # Large timing change is significant (>10 seconds)
            if abs((old.start_time - new.start_time).total_seconds()) > 10:
                return True

        return False

    def _cache_draft(self, draft: Tracklist) -> None:
        """Cache a draft in Redis.

        Args:
            draft: Draft to cache.
        """
        if not self.redis:
            return

        key = f"draft:{draft.id}"
        value = draft.model_dump_json()
        self.redis.setex(key, self.cache_ttl, value)

    def _get_cached_draft(self, draft_id: UUID) -> Optional[Tracklist]:
        """Get a draft from Redis cache.

        Args:
            draft_id: ID of the draft.

        Returns:
            Cached draft or None.
        """
        if not self.redis:
            return None

        key = f"draft:{draft_id}"
        value = self.redis.get(key)

        if value:
            data = json.loads(value if isinstance(value, (str, bytes)) else value.decode("utf-8"))  # type: ignore
            return Tracklist.model_validate(data)  # type: ignore[no-any-return]

        return None

    def _invalidate_cache(self, draft_id: UUID) -> None:
        """Invalidate Redis cache for a draft.

        Args:
            draft_id: ID of the draft.
        """
        if not self.redis:
            return

        key = f"draft:{draft_id}"
        self.redis.delete(key)
