"""Unit tests for tracklist models with manual creation support."""

from datetime import datetime, timedelta
from uuid import uuid4

from services.tracklist_service.src.models.tracklist import (
    TrackEntry,
    Tracklist,
    TracklistDB,
)


class TestTrackEntry:
    """Test TrackEntry model with manual entry support."""

    def test_track_entry_creation_with_manual_flag(self):
        """Test creating a track entry with manual entry flag."""
        track = TrackEntry(
            position=1,
            start_time=timedelta(seconds=0),
            end_time=timedelta(seconds=180),
            artist="Test Artist",
            title="Test Track",
            is_manual_entry=True,
        )

        assert track.position == 1
        assert track.start_time == timedelta(seconds=0)
        assert track.end_time == timedelta(seconds=180)
        assert track.artist == "Test Artist"
        assert track.title == "Test Track"
        assert track.is_manual_entry is True

    def test_track_entry_defaults_manual_flag_to_false(self):
        """Test that manual entry flag defaults to False."""
        track = TrackEntry(
            position=1,
            start_time=timedelta(seconds=0),
            artist="Test Artist",
            title="Test Track",
        )

        assert track.is_manual_entry is False

    def test_track_entry_to_dict_includes_manual_flag(self):
        """Test that to_dict includes manual entry flag."""
        track = TrackEntry(
            position=1,
            start_time=timedelta(seconds=0),
            artist="Test Artist",
            title="Test Track",
            is_manual_entry=True,
        )

        track_dict = track.to_dict()
        assert "is_manual_entry" in track_dict
        assert track_dict["is_manual_entry"] is True

    def test_track_entry_from_dict_reads_manual_flag(self):
        """Test that from_dict reads manual entry flag."""
        track_data = {
            "position": 1,
            "start_time": 0,
            "artist": "Test Artist",
            "title": "Test Track",
            "is_manual_entry": True,
        }

        track = TrackEntry.from_dict(track_data)
        assert track.is_manual_entry is True

    def test_track_entry_from_dict_defaults_manual_flag(self):
        """Test that from_dict defaults manual flag when missing."""
        track_data = {
            "position": 1,
            "start_time": 0,
            "artist": "Test Artist",
            "title": "Test Track",
        }

        track = TrackEntry.from_dict(track_data)
        assert track.is_manual_entry is False


class TestTracklist:
    """Test Tracklist model with draft support."""

    def test_tracklist_creation_with_draft_fields(self):
        """Test creating a tracklist with draft fields."""
        audio_file_id = uuid4()
        parent_id = uuid4()

        tracklist = Tracklist(
            audio_file_id=audio_file_id,
            source="manual",
            is_draft=True,
            draft_version=1,
            parent_tracklist_id=parent_id,
        )

        assert tracklist.audio_file_id == audio_file_id
        assert tracklist.source == "manual"
        assert tracklist.is_draft is True
        assert tracklist.draft_version == 1
        assert tracklist.parent_tracklist_id == parent_id

    def test_tracklist_defaults_draft_fields(self):
        """Test that draft fields have proper defaults."""
        audio_file_id = uuid4()

        tracklist = Tracklist(
            audio_file_id=audio_file_id,
            source="manual",
        )

        assert tracklist.is_draft is False
        assert tracklist.draft_version is None
        assert tracklist.parent_tracklist_id is None

    def test_tracklist_manual_source_validation(self):
        """Test that 'manual' is a valid source."""
        audio_file_id = uuid4()

        tracklist = Tracklist(
            audio_file_id=audio_file_id,
            source="manual",
        )

        assert tracklist.source == "manual"

    def test_tracklist_with_manual_tracks(self):
        """Test tracklist with manually entered tracks."""
        audio_file_id = uuid4()

        tracks = [
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

        tracklist = Tracklist(
            audio_file_id=audio_file_id,
            source="manual",
            tracks=tracks,
            is_draft=True,
            draft_version=1,
        )

        assert len(tracklist.tracks) == 2
        assert all(track.is_manual_entry for track in tracklist.tracks)
        assert tracklist.is_draft is True


class TestTracklistDB:
    """Test SQLAlchemy TracklistDB model."""

    def test_tracklist_db_from_model_with_draft_fields(self):
        """Test converting from Pydantic model with draft fields."""
        audio_file_id = uuid4()
        parent_id = uuid4()
        tracklist_id = uuid4()

        model = Tracklist(
            id=tracklist_id,
            audio_file_id=audio_file_id,
            source="manual",
            is_draft=True,
            draft_version=2,
            parent_tracklist_id=parent_id,
        )

        db_model = TracklistDB.from_model(model)

        assert db_model.id == tracklist_id
        assert db_model.audio_file_id == audio_file_id
        assert db_model.source == "manual"
        assert db_model.is_draft is True
        assert db_model.draft_version == 2
        assert db_model.parent_tracklist_id == parent_id

    def test_tracklist_db_to_model_with_draft_fields(self):
        """Test converting to Pydantic model with draft fields."""
        audio_file_id = uuid4()
        parent_id = uuid4()
        tracklist_id = uuid4()

        db_model = TracklistDB()
        db_model.id = tracklist_id
        db_model.audio_file_id = audio_file_id
        db_model.source = "manual"
        db_model.created_at = datetime.utcnow()
        db_model.updated_at = datetime.utcnow()
        db_model.tracks = []
        db_model.confidence_score = 1.0
        db_model.is_draft = True
        db_model.draft_version = 3
        db_model.parent_tracklist_id = parent_id
        db_model.cue_file_id = None

        model = db_model.to_model()

        assert model.id == tracklist_id
        assert model.audio_file_id == audio_file_id
        assert model.source == "manual"
        assert model.is_draft is True
        assert model.draft_version == 3
        assert model.parent_tracklist_id == parent_id

    def test_tracklist_db_with_manual_tracks(self):
        """Test SQLAlchemy model with manual track entries."""
        audio_file_id = uuid4()
        tracklist_id = uuid4()

        track_data = [
            {
                "position": 1,
                "start_time": 0,
                "artist": "Artist 1",
                "title": "Track 1",
                "is_manual_entry": True,
            },
            {
                "position": 2,
                "start_time": 180,
                "artist": "Artist 2",
                "title": "Track 2",
                "is_manual_entry": True,
            },
        ]

        db_model = TracklistDB()
        db_model.id = tracklist_id
        db_model.audio_file_id = audio_file_id
        db_model.source = "manual"
        db_model.created_at = datetime.utcnow()
        db_model.updated_at = datetime.utcnow()
        db_model.tracks = track_data
        db_model.confidence_score = 1.0
        db_model.is_draft = True
        db_model.draft_version = 1
        db_model.parent_tracklist_id = None
        db_model.cue_file_id = None

        model = db_model.to_model()

        assert len(model.tracks) == 2
        assert all(track.is_manual_entry for track in model.tracks)
        assert model.is_draft is True
        assert model.draft_version == 1
