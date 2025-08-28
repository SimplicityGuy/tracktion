"""Unit tests for manual tracklist API endpoints."""

from datetime import timedelta
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from fastapi import HTTPException

from services.tracklist_service.src.api.manual_endpoints import (
    AddTrackRequest,
    CreateManualTracklistRequest,
    UpdateTracklistRequest,
    UpdateTrackRequest,
    UpdateTrackTimingRequest,
    add_track,
    create_manual_tracklist,
    delete_track,
    list_drafts,
    parse_time_string,
    publish_draft,
    update_track,
    update_track_timing,
    update_tracklist,
)
from services.tracklist_service.src.models.tracklist import TrackEntry, Tracklist


class TestTimeStringParsing:
    """Test time string parsing."""

    def test_parse_mm_ss_format(self):
        """Test parsing MM:SS format."""
        assert parse_time_string("03:45") == 225  # 3 minutes 45 seconds
        assert parse_time_string("00:30") == 30
        assert parse_time_string("10:00") == 600

    def test_parse_hh_mm_ss_format(self):
        """Test parsing HH:MM:SS format."""
        assert parse_time_string("1:23:45") == 5025  # 1 hour 23 minutes 45 seconds
        assert parse_time_string("0:00:30") == 30
        assert parse_time_string("2:00:00") == 7200

    def test_parse_invalid_format(self):
        """Test parsing invalid format."""
        with pytest.raises(ValueError):
            parse_time_string("123")
        with pytest.raises(ValueError):
            parse_time_string("1:2:3:4")


class TestManualTracklistEndpoints:
    """Test manual tracklist API endpoints."""

    @pytest.fixture
    def mock_db_session(self):
        """Create mock database session."""
        return MagicMock()

    @pytest.fixture
    def mock_draft_service(self):
        """Create mock draft service."""
        with patch("services.tracklist_service.src.api.manual_endpoints.DraftService") as mock_cls:
            yield mock_cls.return_value

    def test_create_manual_tracklist_as_draft(self, mock_db_session, mock_draft_service):
        """Test creating a manual tracklist as draft."""
        audio_file_id = uuid4()
        request = CreateManualTracklistRequest(
            audio_file_id=audio_file_id,
            is_draft=True,
        )

        expected_tracklist = Tracklist(
            id=uuid4(),
            audio_file_id=audio_file_id,
            source="manual",
            is_draft=True,
        )
        mock_draft_service.create_draft.return_value = expected_tracklist

        result = create_manual_tracklist(request, mock_db_session)

        assert result == expected_tracklist
        mock_draft_service.create_draft.assert_called_once_with(audio_file_id, [])

    def test_create_manual_tracklist_published(self, mock_db_session, mock_draft_service):
        """Test creating a manual tracklist as published."""
        audio_file_id = uuid4()
        draft_id = uuid4()
        request = CreateManualTracklistRequest(
            audio_file_id=audio_file_id,
            is_draft=False,
        )

        draft = Tracklist(
            id=draft_id,
            audio_file_id=audio_file_id,
            source="manual",
            is_draft=True,
        )
        published = Tracklist(
            id=draft_id,
            audio_file_id=audio_file_id,
            source="manual",
            is_draft=False,
        )

        mock_draft_service.create_draft.return_value = draft
        mock_draft_service.publish_draft.return_value = published

        result = create_manual_tracklist(request, mock_db_session)

        assert result == published
        mock_draft_service.create_draft.assert_called_once()
        mock_draft_service.publish_draft.assert_called_once_with(draft_id)

    def test_update_tracklist_success(self, mock_db_session, mock_draft_service):
        """Test updating a tracklist."""
        tracklist_id = uuid4()
        tracks = [
            TrackEntry(
                position=1,
                start_time=timedelta(0),
                artist="Artist 1",
                title="Track 1",
            )
        ]
        request = UpdateTracklistRequest(tracks=tracks, is_draft=True)

        updated = Tracklist(
            id=tracklist_id,
            audio_file_id=uuid4(),
            source="manual",
            tracks=tracks,
        )
        mock_draft_service.save_draft.return_value = updated

        result = update_tracklist(tracklist_id, request, mock_db_session)

        assert result == updated
        mock_draft_service.save_draft.assert_called_once_with(tracklist_id, tracks)

    def test_update_tracklist_not_found(self, mock_db_session, mock_draft_service):
        """Test updating non-existent tracklist."""
        tracklist_id = uuid4()
        request = UpdateTracklistRequest(tracks=[], is_draft=True)

        mock_draft_service.save_draft.side_effect = ValueError("Not found")

        with pytest.raises(HTTPException) as exc_info:
            update_tracklist(tracklist_id, request, mock_db_session)

        assert exc_info.value.status_code == 404

    def test_add_track_to_tracklist(self, mock_db_session, mock_draft_service):
        """Test adding a track to a tracklist."""
        tracklist_id = uuid4()
        request = AddTrackRequest(
            position=2,
            artist="New Artist",
            title="New Track",
            start_time="3:00",
        )

        existing_tracks = [
            TrackEntry(
                position=1,
                start_time=timedelta(0),
                artist="Artist 1",
                title="Track 1",
            )
        ]

        draft = Tracklist(
            id=tracklist_id,
            audio_file_id=uuid4(),
            source="manual",
            tracks=existing_tracks,
        )
        mock_draft_service.get_draft.return_value = draft

        result = add_track(tracklist_id, request, mock_db_session)

        assert result.position == 2
        assert result.artist == "New Artist"
        assert result.title == "New Track"
        assert result.start_time == timedelta(seconds=180)
        assert result.is_manual_entry is True

        # Should save the updated draft
        mock_draft_service.save_draft.assert_called_once()

    def test_add_track_position_conflict(self, mock_db_session, mock_draft_service):
        """Test adding a track with position conflict shifts existing tracks."""
        tracklist_id = uuid4()
        request = AddTrackRequest(
            position=1,
            artist="New Artist",
            title="New Track",
            start_time="0:00",
        )

        existing_tracks = [
            TrackEntry(
                position=1,
                start_time=timedelta(0),
                artist="Artist 1",
                title="Track 1",
            )
        ]

        draft = Tracklist(
            id=tracklist_id,
            audio_file_id=uuid4(),
            source="manual",
            tracks=existing_tracks,
        )
        mock_draft_service.get_draft.return_value = draft

        result = add_track(tracklist_id, request, mock_db_session)

        assert result.position == 1
        # The existing track should be shifted to position 2
        assert draft.tracks[0].position == 1  # New track
        assert draft.tracks[1].position == 2  # Shifted existing track

    def test_update_track_in_tracklist(self, mock_db_session, mock_draft_service):
        """Test updating a track in a tracklist."""
        tracklist_id = uuid4()
        position = 1
        request = UpdateTrackRequest(
            artist="Updated Artist",
            title="Updated Title",
        )

        track = TrackEntry(
            position=1,
            start_time=timedelta(0),
            artist="Original Artist",
            title="Original Title",
        )

        draft = Tracklist(
            id=tracklist_id,
            audio_file_id=uuid4(),
            source="manual",
            tracks=[track],
        )
        mock_draft_service.get_draft.return_value = draft

        result = update_track(tracklist_id, position, request, mock_db_session)

        assert result.artist == "Updated Artist"
        assert result.title == "Updated Title"
        mock_draft_service.save_draft.assert_called_once()

    def test_update_track_not_found(self, mock_db_session, mock_draft_service):
        """Test updating non-existent track."""
        tracklist_id = uuid4()
        request = UpdateTrackRequest(artist="New")

        draft = Tracklist(
            id=tracklist_id,
            audio_file_id=uuid4(),
            source="manual",
            tracks=[],
        )
        mock_draft_service.get_draft.return_value = draft

        with pytest.raises(HTTPException) as exc_info:
            update_track(tracklist_id, 1, request, mock_db_session)

        assert exc_info.value.status_code == 404
        assert "Track at position 1 not found" in str(exc_info.value.detail)

    def test_delete_track_from_tracklist(self, mock_db_session, mock_draft_service):
        """Test deleting a track from a tracklist."""
        tracklist_id = uuid4()
        position = 2

        tracks = [
            TrackEntry(position=1, start_time=timedelta(0), artist="A1", title="T1"),
            TrackEntry(position=2, start_time=timedelta(180), artist="A2", title="T2"),
            TrackEntry(position=3, start_time=timedelta(360), artist="A3", title="T3"),
        ]

        draft = Tracklist(
            id=tracklist_id,
            audio_file_id=uuid4(),
            source="manual",
            tracks=tracks,
        )
        mock_draft_service.get_draft.return_value = draft

        delete_track(tracklist_id, position, mock_db_session)

        # Check that save was called with the right tracks
        saved_tracks = mock_draft_service.save_draft.call_args[0][1]
        assert len(saved_tracks) == 2
        assert saved_tracks[0].artist == "A1"
        assert saved_tracks[1].artist == "A3"
        assert saved_tracks[1].position == 2  # Position adjusted

    def test_delete_track_not_found(self, mock_db_session, mock_draft_service):
        """Test deleting non-existent track."""
        tracklist_id = uuid4()

        draft = Tracklist(
            id=tracklist_id,
            audio_file_id=uuid4(),
            source="manual",
            tracks=[],
        )
        mock_draft_service.get_draft.return_value = draft

        with pytest.raises(HTTPException) as exc_info:
            delete_track(tracklist_id, 1, mock_db_session)

        assert exc_info.value.status_code == 404

    def test_update_track_timing(self, mock_db_session, mock_draft_service):
        """Test updating track timing."""
        tracklist_id = uuid4()
        position = 1
        request = UpdateTrackTimingRequest(
            start_time="0:30",
            end_time="3:30",
        )

        track = TrackEntry(
            position=1,
            start_time=timedelta(0),
            artist="Artist",
            title="Title",
        )

        draft = Tracklist(
            id=tracklist_id,
            audio_file_id=uuid4(),
            source="manual",
            tracks=[track],
        )
        mock_draft_service.get_draft.return_value = draft

        result = update_track_timing(tracklist_id, position, request, mock_db_session)

        assert result.start_time == timedelta(seconds=30)
        assert result.end_time == timedelta(seconds=210)
        mock_draft_service.save_draft.assert_called_once()

    def test_update_track_timing_conflict(self, mock_db_session, mock_draft_service):
        """Test updating track timing with overlap conflict."""
        tracklist_id = uuid4()
        position = 1
        request = UpdateTrackTimingRequest(
            start_time="2:00",  # 120 seconds
            end_time="4:00",  # 240 seconds
        )

        # Track 1 will be updated to 2:00-4:00 (120-240)
        # Track 2 is at 3:00-6:00 (180-360)
        # These overlap from 3:00-4:00 (180-240)
        tracks = [
            TrackEntry(position=1, start_time=timedelta(0), artist="A1", title="T1"),
            TrackEntry(
                position=2,
                start_time=timedelta(seconds=180),
                end_time=timedelta(seconds=360),
                artist="A2",
                title="T2",
            ),
        ]

        draft = Tracklist(
            id=tracklist_id,
            audio_file_id=uuid4(),
            source="manual",
            tracks=tracks,
        )
        mock_draft_service.get_draft.return_value = draft

        with pytest.raises(HTTPException) as exc_info:
            update_track_timing(tracklist_id, position, request, mock_db_session)

        assert exc_info.value.status_code == 400
        assert "Timing conflict" in str(exc_info.value.detail)

    def test_list_drafts_for_audio_file(self, mock_db_session, mock_draft_service):
        """Test listing drafts for an audio file."""
        audio_file_id = uuid4()

        drafts = [
            Tracklist(id=uuid4(), audio_file_id=audio_file_id, source="manual", is_draft=True),
            Tracklist(id=uuid4(), audio_file_id=audio_file_id, source="manual", is_draft=True),
        ]
        mock_draft_service.list_drafts.return_value = drafts

        result = list_drafts(audio_file_id, False, mock_db_session)

        assert result == drafts
        mock_draft_service.list_drafts.assert_called_once_with(audio_file_id, False)

    def test_publish_draft_success(self, mock_db_session, mock_draft_service):
        """Test publishing a draft."""
        tracklist_id = uuid4()

        published = Tracklist(
            id=tracklist_id,
            audio_file_id=uuid4(),
            source="manual",
            is_draft=False,
        )
        mock_draft_service.publish_draft.return_value = published

        result = publish_draft(tracklist_id, mock_db_session)

        assert result == published
        assert result.is_draft is False
        mock_draft_service.publish_draft.assert_called_once_with(tracklist_id)

    def test_publish_draft_error(self, mock_db_session, mock_draft_service):
        """Test publishing error handling."""
        tracklist_id = uuid4()

        mock_draft_service.publish_draft.side_effect = ValueError("Already published")

        with pytest.raises(HTTPException) as exc_info:
            publish_draft(tracklist_id, mock_db_session)

        assert exc_info.value.status_code == 400
        assert "Already published" in str(exc_info.value.detail)
