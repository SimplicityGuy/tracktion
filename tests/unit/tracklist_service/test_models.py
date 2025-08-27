"""
Unit tests for tracklist models.
"""

from datetime import timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from services.tracklist_service.src.models.tracklist import (
    ImportTracklistRequest,
    ImportTracklistResponse,
    TrackEntry,
    Tracklist,
)


class TestTrackEntry:
    """Test TrackEntry model validation and methods."""

    def test_valid_track_entry(self):
        """Test creating a valid TrackEntry."""
        track = TrackEntry(
            position=1,
            start_time=timedelta(minutes=0),
            end_time=timedelta(minutes=4, seconds=30),
            artist="Test Artist",
            title="Test Track",
            remix="Original Mix",
            label="Test Label",
            confidence=0.95,
            transition_type="blend",
        )
        assert track.position == 1
        assert track.artist == "Test Artist"
        assert track.confidence == 0.95

    def test_invalid_position(self):
        """Test that position must be positive."""
        with pytest.raises(ValidationError) as exc_info:
            TrackEntry(position=0, start_time=timedelta(minutes=0), artist="Test", title="Test")
        assert "Position must be positive" in str(exc_info.value)

    def test_empty_artist_title(self):
        """Test that artist and title cannot be empty."""
        with pytest.raises(ValidationError):
            TrackEntry(position=1, start_time=timedelta(minutes=0), artist="", title="Test")

        with pytest.raises(ValidationError):
            TrackEntry(position=1, start_time=timedelta(minutes=0), artist="Test", title="")

    def test_confidence_bounds(self):
        """Test confidence must be between 0 and 1."""
        # Valid confidence values
        track = TrackEntry(position=1, start_time=timedelta(minutes=0), artist="Test", title="Test", confidence=0.5)
        assert track.confidence == 0.5

        # Invalid confidence > 1
        with pytest.raises(ValidationError):
            TrackEntry(position=1, start_time=timedelta(minutes=0), artist="Test", title="Test", confidence=1.5)

        # Invalid confidence < 0
        with pytest.raises(ValidationError):
            TrackEntry(position=1, start_time=timedelta(minutes=0), artist="Test", title="Test", confidence=-0.1)

    def test_to_dict_from_dict(self):
        """Test conversion to and from dictionary."""
        track = TrackEntry(
            position=1,
            start_time=timedelta(minutes=2, seconds=30),
            end_time=timedelta(minutes=5, seconds=45),
            artist="Test Artist",
            title="Test Track",
            confidence=0.9,
        )

        # Convert to dict
        track_dict = track.to_dict()
        assert track_dict["position"] == 1
        assert track_dict["start_time"] == 150  # 2:30 in seconds
        assert track_dict["end_time"] == 345  # 5:45 in seconds
        assert track_dict["artist"] == "Test Artist"

        # Convert back from dict
        track_restored = TrackEntry.from_dict(track_dict)
        assert track_restored.position == track.position
        assert track_restored.start_time == track.start_time
        assert track_restored.end_time == track.end_time
        assert track_restored.artist == track.artist


class TestTracklist:
    """Test Tracklist model validation."""

    def test_valid_tracklist(self):
        """Test creating a valid Tracklist."""
        audio_file_id = uuid4()
        tracks = [
            TrackEntry(position=i, start_time=timedelta(minutes=i * 5), artist=f"Artist {i}", title=f"Track {i}")
            for i in range(1, 4)
        ]

        tracklist = Tracklist(
            audio_file_id=audio_file_id, source="1001tracklists", tracks=tracks, confidence_score=0.85
        )

        assert tracklist.audio_file_id == audio_file_id
        assert tracklist.source == "1001tracklists"
        assert len(tracklist.tracks) == 3
        assert tracklist.confidence_score == 0.85

    def test_invalid_source(self):
        """Test that source must be one of allowed values."""
        with pytest.raises(ValidationError) as exc_info:
            Tracklist(audio_file_id=uuid4(), source="invalid_source", tracks=[])
        assert "Source must be one of" in str(exc_info.value)

    def test_tracks_order_validation(self):
        """Test that tracks must be in sequential order."""
        # Out of order tracks
        tracks = [
            TrackEntry(position=2, start_time=timedelta(minutes=5), artist="A", title="T"),
            TrackEntry(position=1, start_time=timedelta(minutes=0), artist="B", title="T"),
        ]

        with pytest.raises(ValidationError) as exc_info:
            Tracklist(audio_file_id=uuid4(), source="manual", tracks=tracks)
        assert "Tracks must be in sequential order" in str(exc_info.value)

    def test_tracks_with_gaps(self):
        """Test that small gaps in track positions are allowed."""
        # Small gap is OK
        tracks = [
            TrackEntry(position=1, start_time=timedelta(minutes=0), artist="A", title="T1"),
            TrackEntry(position=3, start_time=timedelta(minutes=5), artist="B", title="T2"),
        ]

        tracklist = Tracklist(audio_file_id=uuid4(), source="manual", tracks=tracks)
        assert len(tracklist.tracks) == 2

        # Too many gaps should fail
        tracks_large_gap = [
            TrackEntry(position=1, start_time=timedelta(minutes=0), artist="A", title="T1"),
            TrackEntry(position=10, start_time=timedelta(minutes=5), artist="B", title="T2"),
        ]

        with pytest.raises(ValidationError) as exc_info:
            Tracklist(audio_file_id=uuid4(), source="manual", tracks=tracks_large_gap)
        assert "Too many gaps in track positions" in str(exc_info.value)


class TestImportRequest:
    """Test ImportTracklistRequest validation."""

    def test_valid_request(self):
        """Test creating a valid import request."""
        request = ImportTracklistRequest(url="https://1001tracklists.com/tracklist/abcd/test", audio_file_id=uuid4())
        assert request.force_refresh is False
        assert request.cue_format == "standard"

    def test_invalid_url(self):
        """Test that URL must be from 1001tracklists."""
        with pytest.raises(ValidationError) as exc_info:
            ImportTracklistRequest(url="https://example.com/tracklist", audio_file_id=uuid4())
        assert "URL must be from 1001tracklists.com" in str(exc_info.value)

    def test_valid_1001tracklists_urls(self):
        """Test various valid 1001tracklists URL formats."""
        valid_urls = [
            "http://1001tracklists.com/tracklist/test",
            "https://1001tracklists.com/tracklist/test",
            "http://www.1001tracklists.com/tracklist/test",
            "https://www.1001tracklists.com/tracklist/test",
        ]

        for url in valid_urls:
            request = ImportTracklistRequest(url=url, audio_file_id=uuid4())
            assert request.url == url


class TestImportResponse:
    """Test ImportTracklistResponse model."""

    def test_successful_response(self):
        """Test creating a successful import response."""
        tracklist = Tracklist(audio_file_id=uuid4(), source="1001tracklists", tracks=[])

        response = ImportTracklistResponse(
            success=True, tracklist=tracklist, cue_file_path="/path/to/cue/file.cue", cached=False
        )

        assert response.success is True
        assert response.tracklist == tracklist
        assert response.error is None

    def test_error_response(self):
        """Test creating an error response."""
        response = ImportTracklistResponse(success=False, error="Failed to fetch tracklist from 1001tracklists")

        assert response.success is False
        assert response.tracklist is None
        assert response.error == "Failed to fetch tracklist from 1001tracklists"
