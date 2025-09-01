"""Unit tests for draft management service."""

from datetime import timedelta
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
import redis

from services.tracklist_service.src.models.tracklist import (
    TrackEntry,
    Tracklist,
    TracklistDB,
)
from services.tracklist_service.src.services.draft_service import DraftService


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    return MagicMock()


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client."""
    return MagicMock(spec=redis.Redis)


@pytest.fixture
def draft_service(mock_db_session, mock_redis_client):
    """Create a DraftService instance with mocks."""
    return DraftService(mock_db_session, mock_redis_client)


@pytest.fixture
def sample_tracks():
    """Create sample track entries."""
    return [
        TrackEntry(
            position=1,
            start_time=timedelta(seconds=0),
            artist="Artist 1",
            title="Track 1",
            is_manual_entry=True,
        ),
        TrackEntry(
            position=2,
            start_time=timedelta(seconds=180),
            artist="Artist 2",
            title="Track 2",
            is_manual_entry=True,
        ),
    ]


class TestDraftService:
    """Test DraftService class."""

    def test_create_draft_first_version(self, draft_service, mock_db_session):
        """Test creating the first draft version."""
        audio_file_id = uuid4()

        # Mock no existing drafts
        mock_db_session.query().filter_by().order_by().first.return_value = None

        # Create draft
        draft = draft_service.create_draft(audio_file_id)

        assert draft.audio_file_id == audio_file_id
        assert draft.source == "manual"
        assert draft.is_draft is True
        assert draft.draft_version == 1
        assert len(draft.tracks) == 0
        assert draft.confidence_score == 1.0

        # Verify database operations
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()

    def test_create_draft_with_tracks(self, draft_service, mock_db_session, sample_tracks):
        """Test creating a draft with initial tracks."""
        audio_file_id = uuid4()

        mock_db_session.query().filter_by().order_by().first.return_value = None

        draft = draft_service.create_draft(audio_file_id, sample_tracks)

        assert len(draft.tracks) == 2
        assert draft.tracks[0].artist == "Artist 1"
        assert draft.tracks[1].artist == "Artist 2"

    def test_create_draft_increments_version(self, draft_service, mock_db_session):
        """Test that creating a new draft increments the version."""
        audio_file_id = uuid4()

        # Mock existing draft with version 2
        existing_draft = MagicMock()
        existing_draft.draft_version = 2
        mock_db_session.query().filter_by().order_by().first.return_value = existing_draft

        draft = draft_service.create_draft(audio_file_id)

        assert draft.draft_version == 3

    def test_create_draft_caches_in_redis(self, draft_service, mock_db_session, mock_redis_client):
        """Test that created draft is cached in Redis."""
        audio_file_id = uuid4()

        mock_db_session.query().filter_by().order_by().first.return_value = None

        draft = draft_service.create_draft(audio_file_id)

        # Verify Redis caching
        mock_redis_client.setex.assert_called_once()
        call_args = mock_redis_client.setex.call_args
        assert call_args[0][0] == f"draft:{draft.id}"
        assert call_args[0][1] == 3600  # TTL

    def test_save_draft_updates_existing(self, draft_service, mock_db_session, sample_tracks):
        """Test saving updates to an existing draft."""
        draft_id = uuid4()
        audio_file_id = uuid4()

        # Mock existing draft
        mock_draft_db = MagicMock(spec=TracklistDB)
        mock_draft_db.id = draft_id
        mock_draft_db.audio_file_id = audio_file_id
        mock_draft_db.is_draft = True
        mock_draft_db.tracks = []
        mock_draft_db.to_model.return_value = Tracklist(
            id=draft_id,
            audio_file_id=audio_file_id,
            source="manual",
            is_draft=True,
            draft_version=1,
            tracks=[],
        )

        mock_db_session.query().filter_by().first.return_value = mock_draft_db

        # Save draft with new tracks
        updated = draft_service.save_draft(draft_id, sample_tracks, auto_version=False)

        assert updated.id == draft_id
        assert len(updated.tracks) == 2
        mock_db_session.commit.assert_called()

    def test_save_draft_creates_new_version_on_significant_changes(self, draft_service, mock_db_session, sample_tracks):
        """Test that significant changes create a new version."""
        draft_id = uuid4()
        audio_file_id = uuid4()

        # Mock existing draft with different tracks
        old_tracks = [
            TrackEntry(
                position=1,
                start_time=timedelta(seconds=0),
                artist="Old Artist",
                title="Old Track",
            )
        ]

        mock_draft_db = MagicMock(spec=TracklistDB)
        mock_draft_db.is_draft = True
        mock_draft_db.to_model.return_value = Tracklist(
            id=draft_id,
            audio_file_id=audio_file_id,
            source="manual",
            is_draft=True,
            draft_version=1,
            tracks=old_tracks,
        )

        # Mock query for existing draft
        mock_query1 = MagicMock()
        mock_query1.filter_by.return_value = mock_query1
        mock_query1.first.return_value = mock_draft_db

        # Mock query for version check (returns None - no other versions)
        mock_query2 = MagicMock()
        mock_query2.filter_by.return_value = mock_query2
        mock_query2.order_by.return_value = mock_query2
        mock_query2.first.return_value = None

        # Mock query for new draft update
        new_draft_db = MagicMock()
        mock_query3 = MagicMock()
        mock_query3.filter_by.return_value = mock_query3
        mock_query3.first.return_value = new_draft_db

        # Set up side effects for each query
        call_count = [0]

        def query_side_effect(x):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_query1
            if call_count[0] == 2:
                return mock_query2
            return mock_query3

        mock_db_session.query.side_effect = query_side_effect

        # Save with auto-versioning
        updated = draft_service.save_draft(draft_id, sample_tracks, auto_version=True)

        # Should create a new draft
        assert updated.id != draft_id  # New ID
        assert updated.draft_version == 1  # New version

    def test_save_draft_raises_if_not_draft(self, draft_service, mock_db_session, sample_tracks):
        """Test that saving a non-draft raises an error."""
        draft_id = uuid4()

        mock_draft_db = MagicMock()
        mock_draft_db.is_draft = False
        mock_db_session.query().filter_by().first.return_value = mock_draft_db

        with pytest.raises(ValueError, match="is not a draft"):
            draft_service.save_draft(draft_id, sample_tracks)

    def test_save_draft_raises_if_not_found(self, draft_service, mock_db_session, sample_tracks):
        """Test that saving a non-existent draft raises an error."""
        draft_id = uuid4()

        mock_db_session.query().filter_by().first.return_value = None

        with pytest.raises(ValueError, match="not found"):
            draft_service.save_draft(draft_id, sample_tracks)

    def test_get_draft_from_cache(self, draft_service, mock_redis_client):
        """Test retrieving a draft from Redis cache."""
        draft_id = uuid4()
        audio_file_id = uuid4()

        cached_data = Tracklist(
            id=draft_id,
            audio_file_id=audio_file_id,
            source="manual",
            is_draft=True,
            draft_version=1,
        ).model_dump_json()

        mock_redis_client.get.return_value = cached_data.encode()

        draft = draft_service.get_draft(draft_id)

        assert draft is not None
        assert draft.id == draft_id
        mock_redis_client.get.assert_called_once_with(f"draft:{draft_id}")

    def test_get_draft_from_database(self, draft_service, mock_db_session, mock_redis_client):
        """Test retrieving a draft from database when not cached."""
        draft_id = uuid4()
        audio_file_id = uuid4()

        mock_redis_client.get.return_value = None  # Not in cache

        mock_draft_db = MagicMock()
        mock_draft_db.to_model.return_value = Tracklist(
            id=draft_id,
            audio_file_id=audio_file_id,
            source="manual",
            is_draft=True,
            draft_version=1,
        )

        mock_db_session.query().filter_by().first.return_value = mock_draft_db

        draft = draft_service.get_draft(draft_id)

        assert draft is not None
        assert draft.id == draft_id

        # Should cache the result
        mock_redis_client.setex.assert_called_once()

    def test_get_draft_returns_none_if_not_found(self, draft_service, mock_db_session, mock_redis_client):
        """Test that get_draft returns None if draft doesn't exist."""
        draft_id = uuid4()

        mock_redis_client.get.return_value = None
        mock_db_session.query().filter_by().first.return_value = None

        draft = draft_service.get_draft(draft_id)

        assert draft is None

    def test_list_drafts_for_audio_file(self, draft_service, mock_db_session):
        """Test listing all drafts for an audio file."""
        audio_file_id = uuid4()

        # Mock draft database records
        mock_drafts = []
        for i in range(3):
            mock_draft = MagicMock()
            mock_draft.to_model.return_value = Tracklist(
                id=uuid4(),
                audio_file_id=audio_file_id,
                source="manual",
                is_draft=True,
                draft_version=i + 1,
            )
            mock_drafts.append(mock_draft)

        mock_query = MagicMock()
        mock_query.filter_by.return_value = mock_query  # For chaining
        mock_query.order_by.return_value = mock_query  # For chaining
        mock_query.all.return_value = mock_drafts
        mock_db_session.query().filter_by.return_value = mock_query

        drafts = draft_service.list_drafts(audio_file_id, include_versions=True)

        assert len(drafts) == 3
        assert all(d.audio_file_id == audio_file_id for d in drafts)

    def test_list_drafts_exclude_versions(self, draft_service, mock_db_session):
        """Test listing drafts excluding old versions."""
        audio_file_id = uuid4()

        # Only latest versions (no parent)
        mock_draft = MagicMock()
        mock_draft.to_model.return_value = Tracklist(
            id=uuid4(),
            audio_file_id=audio_file_id,
            source="manual",
            is_draft=True,
            draft_version=3,
        )

        mock_query = MagicMock()
        mock_query.filter_by().order_by().all.return_value = [mock_draft]
        mock_db_session.query().filter_by.return_value = mock_query

        drafts = draft_service.list_drafts(audio_file_id, include_versions=False)

        assert len(drafts) == 1
        # Should filter by parent_tracklist_id=None
        mock_query.filter_by.assert_called_with(parent_tracklist_id=None)

    def test_publish_draft_success(self, draft_service, mock_db_session):
        """Test publishing a draft to final version."""
        draft_id = uuid4()
        audio_file_id = uuid4()

        # Mock draft
        mock_draft_db = MagicMock()
        mock_draft_db.id = draft_id
        mock_draft_db.audio_file_id = audio_file_id
        mock_draft_db.is_draft = True
        mock_draft_db.to_model.return_value = Tracklist(
            id=draft_id,
            audio_file_id=audio_file_id,
            source="manual",
            is_draft=False,  # Will be published
            draft_version=None,
        )

        # No existing published version
        mock_db_session.query().filter_by.side_effect = [
            MagicMock(first=lambda: mock_draft_db),  # Draft query
            MagicMock(first=lambda: None),  # No existing published
        ]

        published = draft_service.publish_draft(draft_id)

        assert published.id == draft_id
        assert published.is_draft is False
        assert mock_draft_db.is_draft is False
        assert mock_draft_db.draft_version is None
        mock_db_session.commit.assert_called()

    def test_publish_draft_archives_existing(self, draft_service, mock_db_session):
        """Test that publishing archives existing published version."""
        draft_id = uuid4()
        audio_file_id = uuid4()
        existing_id = uuid4()

        # Mock draft
        mock_draft_db = MagicMock()
        mock_draft_db.id = draft_id
        mock_draft_db.audio_file_id = audio_file_id
        mock_draft_db.is_draft = True
        mock_draft_db.to_model.return_value = Tracklist(
            id=draft_id,
            audio_file_id=audio_file_id,
            source="manual",
            is_draft=False,
        )

        # Mock existing published version
        mock_existing = MagicMock()
        mock_existing.id = existing_id
        mock_existing.to_model.return_value = Tracklist(
            id=existing_id,
            audio_file_id=audio_file_id,
            source="manual",
            is_draft=False,
        )

        mock_db_session.query().filter_by.side_effect = [
            MagicMock(first=lambda: mock_draft_db),  # Draft query
            MagicMock(first=lambda: mock_existing),  # Existing published
        ]

        _ = draft_service.publish_draft(draft_id)

        # Should add archive record
        mock_db_session.add.assert_called_once()
        archive_call = mock_db_session.add.call_args[0][0]
        assert archive_call.parent_tracklist_id == existing_id

    def test_publish_draft_raises_if_not_found(self, draft_service, mock_db_session):
        """Test that publishing a non-existent draft raises an error."""
        draft_id = uuid4()

        mock_db_session.query().filter_by().first.return_value = None

        with pytest.raises(ValueError, match="not found"):
            draft_service.publish_draft(draft_id)

    def test_publish_draft_raises_if_already_published(self, draft_service, mock_db_session):
        """Test that publishing an already published tracklist raises an error."""
        draft_id = uuid4()

        mock_draft_db = MagicMock()
        mock_draft_db.is_draft = False
        mock_db_session.query().filter_by().first.return_value = mock_draft_db

        with pytest.raises(ValueError, match="already published"):
            draft_service.publish_draft(draft_id)

    def test_delete_draft_success(self, draft_service, mock_db_session, mock_redis_client):
        """Test deleting a draft."""
        draft_id = uuid4()

        mock_draft_db = MagicMock()
        mock_db_session.query().filter_by().first.return_value = mock_draft_db

        result = draft_service.delete_draft(draft_id)

        assert result is True
        mock_db_session.delete.assert_called_once_with(mock_draft_db)
        mock_db_session.commit.assert_called_once()

        # Should clear cache
        mock_redis_client.delete.assert_called_once_with(f"draft:{draft_id}")

    def test_delete_draft_returns_false_if_not_found(self, draft_service, mock_db_session):
        """Test that deleting a non-existent draft returns False."""
        draft_id = uuid4()

        mock_db_session.query().filter_by().first.return_value = None

        result = draft_service.delete_draft(draft_id)

        assert result is False
        mock_db_session.delete.assert_not_called()

    def test_has_significant_changes_different_count(self, draft_service, sample_tracks):
        """Test that different track count is significant."""
        old_tracks = sample_tracks[:1]  # Only one track
        new_tracks = sample_tracks  # Two tracks

        result = draft_service._has_significant_changes(old_tracks, new_tracks)

        assert result is True

    def test_has_significant_changes_position_change(self, draft_service):
        """Test that position change is significant."""
        old_tracks = [
            TrackEntry(position=1, start_time=timedelta(0), artist="A", title="T1"),
            TrackEntry(position=2, start_time=timedelta(180), artist="B", title="T2"),
        ]
        new_tracks = [
            TrackEntry(position=2, start_time=timedelta(0), artist="A", title="T1"),
            TrackEntry(position=1, start_time=timedelta(180), artist="B", title="T2"),
        ]

        result = draft_service._has_significant_changes(old_tracks, new_tracks)

        assert result is True

    def test_has_significant_changes_artist_title(self, draft_service):
        """Test that artist/title change is significant."""
        old_tracks = [
            TrackEntry(position=1, start_time=timedelta(0), artist="Old Artist", title="T1"),
        ]
        new_tracks = [
            TrackEntry(
                position=1,
                start_time=timedelta(0),
                artist="Different Artist",
                title="T1",
            ),
        ]

        result = draft_service._has_significant_changes(old_tracks, new_tracks)

        assert result is True

    def test_has_significant_changes_large_timing(self, draft_service):
        """Test that large timing change is significant."""
        old_tracks = [
            TrackEntry(position=1, start_time=timedelta(seconds=0), artist="A", title="T1"),
        ]
        new_tracks = [
            TrackEntry(position=1, start_time=timedelta(seconds=15), artist="A", title="T1"),
        ]

        result = draft_service._has_significant_changes(old_tracks, new_tracks)

        assert result is True

    def test_has_significant_changes_minor_changes(self, draft_service):
        """Test that minor changes are not significant."""
        old_tracks = [
            TrackEntry(position=1, start_time=timedelta(seconds=0), artist="A", title="T1"),
        ]
        new_tracks = [
            TrackEntry(position=1, start_time=timedelta(seconds=2), artist="A", title="T1"),
        ]

        result = draft_service._has_significant_changes(old_tracks, new_tracks)

        assert result is False
