"""
Unit tests for tracklist data models.

Tests all models including Track, CuePoint, Transition, and Tracklist
with various edge cases and validation scenarios.
"""

from datetime import date
from uuid import UUID

import pytest
from pydantic import ValidationError

from services.tracklist_service.src.models.tracklist_models import (
    CuePoint,
    Track,
    Tracklist,
    TracklistMetadata,
    TracklistRequest,
    TracklistResponse,
    Transition,
    TransitionType,
)


class TestCuePoint:
    """Test CuePoint model."""

    def test_valid_cue_point_mmss(self):
        """Test valid cue point with MM:SS format."""
        cue = CuePoint(track_number=1, timestamp_ms=65000, formatted_time="01:05")
        assert cue.track_number == 1
        assert cue.timestamp_ms == 65000
        assert cue.formatted_time == "01:05"

    def test_valid_cue_point_hhmmss(self):
        """Test valid cue point with HH:MM:SS format."""
        cue = CuePoint(track_number=5, timestamp_ms=3665000, formatted_time="01:01:05")
        assert cue.track_number == 5
        assert cue.formatted_time == "01:01:05"

    def test_invalid_track_number(self):
        """Test invalid track number (less than 1)."""
        with pytest.raises(ValidationError) as exc_info:
            CuePoint(track_number=0, timestamp_ms=1000, formatted_time="00:01")
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_invalid_timestamp_negative(self):
        """Test invalid negative timestamp."""
        with pytest.raises(ValidationError) as exc_info:
            CuePoint(track_number=1, timestamp_ms=-1000, formatted_time="00:01")
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_invalid_time_format(self):
        """Test invalid time format."""
        with pytest.raises(ValidationError) as exc_info:
            CuePoint(track_number=1, timestamp_ms=1000, formatted_time="1:2:3:4")
        assert "Time must be in MM:SS or HH:MM:SS format" in str(exc_info.value)

    def test_invalid_time_format_non_numeric(self):
        """Test time format with non-numeric values."""
        with pytest.raises(ValidationError) as exc_info:
            CuePoint(track_number=1, timestamp_ms=1000, formatted_time="aa:bb")
        assert "Invalid time format" in str(exc_info.value)


class TestTrack:
    """Test Track model."""

    def test_valid_track_minimal(self):
        """Test valid track with minimal information."""
        track = Track(number=1, artist="Carl Cox", title="The Revolution Continues")
        assert track.number == 1
        assert track.artist == "Carl Cox"
        assert track.title == "The Revolution Continues"
        assert track.is_id is False
        assert track.remix is None

    def test_valid_track_complete(self):
        """Test valid track with all fields."""
        cue = CuePoint(track_number=1, timestamp_ms=0, formatted_time="00:00")
        track = Track(
            number=1,
            timestamp=cue,
            artist="Amelie Lens",
            title="In Silence",
            remix="Kobosil 44 Rush Mix",
            label="Second State",
            is_id=False,
            bpm=130.5,
            key="Am",
            genre="Techno",
            notes="Opening track",
        )
        assert track.remix == "Kobosil 44 Rush Mix"
        assert track.label == "Second State"
        assert track.bpm == 130.5
        assert track.key == "Am"

    def test_id_track(self):
        """Test ID/unknown track."""
        track = Track(number=3, artist="ID", title="ID", is_id=True)
        assert track.is_id is True
        assert track.artist == "ID"
        assert track.title == "ID"

    def test_track_with_multiple_artists(self):
        """Test track with multiple artists."""
        track = Track(number=2, artist="Adam Beyer & Cirez D", title="Interchange")
        assert "Adam Beyer" in track.artist
        assert "Cirez D" in track.artist

    def test_invalid_track_number(self):
        """Test invalid track number."""
        with pytest.raises(ValidationError) as exc_info:
            Track(number=0, artist="Test", title="Test")
        assert "greater than or equal to 1" in str(exc_info.value)

    def test_empty_artist_non_id(self):
        """Test empty artist for non-ID track."""
        with pytest.raises(ValidationError) as exc_info:
            Track(number=1, artist="", title="Test Track")
        assert "Artist name cannot be empty" in str(exc_info.value)

    def test_invalid_bpm(self):
        """Test invalid BPM values."""
        with pytest.raises(ValidationError) as exc_info:
            Track(
                number=1,
                artist="Test",
                title="Test",
                bpm=250.0,  # Too high
            )
        assert "less than or equal to 200" in str(exc_info.value)


class TestTransition:
    """Test Transition model."""

    def test_valid_transition_minimal(self):
        """Test valid transition with minimal info."""
        transition = Transition(from_track=1, to_track=2)
        assert transition.from_track == 1
        assert transition.to_track == 2
        assert transition.transition_type == TransitionType.UNKNOWN

    def test_valid_transition_complete(self):
        """Test valid transition with all fields."""
        transition = Transition(
            from_track=3,
            to_track=4,
            transition_type=TransitionType.BLEND,
            timestamp_ms=180000,
            duration_ms=8000,
            notes="Long blend transition",
        )
        assert transition.transition_type == TransitionType.BLEND
        assert transition.timestamp_ms == 180000
        assert transition.duration_ms == 8000

    def test_invalid_track_order(self):
        """Test invalid track order (to_track <= from_track)."""
        with pytest.raises(ValidationError) as exc_info:
            Transition(from_track=5, to_track=4)
        assert "to_track must be greater than from_track" in str(exc_info.value)

    def test_same_track_transition(self):
        """Test transition to same track (invalid)."""
        with pytest.raises(ValidationError) as exc_info:
            Transition(from_track=3, to_track=3)
        assert "to_track must be greater than from_track" in str(exc_info.value)

    def test_all_transition_types(self):
        """Test all transition types."""
        for trans_type in TransitionType:
            transition = Transition(from_track=1, to_track=2, transition_type=trans_type)
            assert transition.transition_type == trans_type


class TestTracklistMetadata:
    """Test TracklistMetadata model."""

    def test_valid_metadata_minimal(self):
        """Test valid metadata with minimal info."""
        metadata = TracklistMetadata()
        assert metadata.recording_type is None
        assert metadata.tags == []

    def test_valid_metadata_complete(self):
        """Test valid metadata with all fields."""
        metadata = TracklistMetadata(
            recording_type="DJ Set",
            duration_minutes=120,
            play_count=50000,
            favorite_count=1500,
            comment_count=250,
            download_url="https://example.com/download",
            stream_url="https://example.com/stream",
            soundcloud_url="https://soundcloud.com/example",
            mixcloud_url="https://mixcloud.com/example",
            youtube_url="https://youtube.com/watch?v=example",
            tags=["Techno", "Peak Time", "Festival"],
        )
        assert metadata.duration_minutes == 120
        assert metadata.play_count == 50000
        assert len(metadata.tags) == 3


class TestTracklist:
    """Test Tracklist model."""

    def test_valid_tracklist_minimal(self):
        """Test valid tracklist with minimal info."""
        tracklist = Tracklist(url="https://www.1001tracklists.com/tracklist/example", dj_name="Charlotte de Witte")
        assert tracklist.dj_name == "Charlotte de Witte"
        assert tracklist.tracks == []
        assert tracklist.transitions == []

    def test_valid_tracklist_with_tracks(self):
        """Test valid tracklist with tracks."""
        tracks = [
            Track(number=1, artist="Artist 1", title="Track 1"),
            Track(number=2, artist="Artist 2", title="Track 2"),
            Track(number=3, artist="Artist 3", title="Track 3"),
        ]
        tracklist = Tracklist(
            url="https://1001tracklists.com/tracklist/test",
            dj_name="Test DJ",
            event_name="Test Festival",
            venue="Test Venue",
            date=date(2024, 7, 15),
            tracks=tracks,
        )
        assert len(tracklist.tracks) == 3
        assert tracklist.event_name == "Test Festival"
        assert tracklist.date == date(2024, 7, 15)

    def test_tracklist_with_transitions(self):
        """Test tracklist with transitions."""
        tracks = [Track(number=1, artist="A1", title="T1"), Track(number=2, artist="A2", title="T2")]
        transitions = [Transition(from_track=1, to_track=2, transition_type=TransitionType.FADE)]
        tracklist = Tracklist(
            url="https://1001tracklists.com/test", dj_name="DJ", tracks=tracks, transitions=transitions
        )
        assert len(tracklist.transitions) == 1
        assert tracklist.transitions[0].transition_type == TransitionType.FADE

    def test_invalid_url(self):
        """Test invalid URL (not from 1001tracklists)."""
        with pytest.raises(ValidationError) as exc_info:
            Tracklist(url="https://example.com/tracklist", dj_name="Test DJ")
        assert "URL must be from 1001tracklists.com" in str(exc_info.value)

    def test_track_order_with_gaps(self):
        """Test tracks with gaps in numbering (allowed within reason)."""
        tracks = [
            Track(number=1, artist="A1", title="T1"),
            Track(number=3, artist="A3", title="T3"),  # Gap at 2
            Track(number=4, artist="A4", title="T4"),
        ]
        tracklist = Tracklist(url="https://1001tracklists.com/test", dj_name="DJ", tracks=tracks)
        assert len(tracklist.tracks) == 3

    def test_track_order_too_many_gaps(self):
        """Test tracks with too many gaps (invalid)."""
        tracks = [
            Track(number=1, artist="A1", title="T1"),
            Track(number=50, artist="A50", title="T50"),  # Too big gap
        ]
        with pytest.raises(ValidationError) as exc_info:
            Tracklist(url="https://1001tracklists.com/test", dj_name="DJ", tracks=tracks)
        assert "Track numbers have too many gaps" in str(exc_info.value)


class TestTracklistRequest:
    """Test TracklistRequest model."""

    def test_valid_request_with_url(self):
        """Test valid request with URL."""
        request = TracklistRequest(url="https://1001tracklists.com/tracklist/example")
        assert request.url == "https://1001tracklists.com/tracklist/example"
        assert request.force_refresh is False
        assert request.include_transitions is True

    def test_valid_request_with_id(self):
        """Test valid request with tracklist ID."""
        request = TracklistRequest(tracklist_id="abc123")
        assert request.tracklist_id == "abc123"
        assert request.url is None

    def test_request_with_options(self):
        """Test request with additional options."""
        request = TracklistRequest(url="https://1001tracklists.com/test", force_refresh=True, include_transitions=False)
        assert request.force_refresh is True
        assert request.include_transitions is False

    def test_invalid_request_no_identifier(self):
        """Test invalid request with no URL or ID."""
        with pytest.raises(ValidationError) as exc_info:
            TracklistRequest()
        assert "Either url or tracklist_id must be provided" in str(exc_info.value)


class TestTracklistResponse:
    """Test TracklistResponse model."""

    def test_successful_response(self):
        """Test successful response with tracklist."""
        tracklist = Tracklist(url="https://1001tracklists.com/test", dj_name="Test DJ")
        response = TracklistResponse(
            success=True,
            tracklist=tracklist,
            cached=True,
            processing_time_ms=150,
            correlation_id=UUID("12345678-1234-5678-1234-567812345678"),
        )
        assert response.success is True
        assert response.tracklist is not None
        assert response.error is None
        assert response.cached is True

    def test_error_response(self):
        """Test error response without tracklist."""
        response = TracklistResponse(
            success=False,
            error="Failed to parse tracklist page",
            correlation_id=UUID("12345678-1234-5678-1234-567812345678"),
        )
        assert response.success is False
        assert response.tracklist is None
        assert response.error == "Failed to parse tracklist page"


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_unicode_in_track_names(self):
        """Test Unicode characters in track names."""
        track = Track(number=1, artist="Âme", title="Für Dich (ft. Jürgën)")
        assert track.artist == "Âme"
        assert "Jürgën" in track.title

    def test_very_long_set(self):
        """Test tracklist with many tracks."""
        tracks = [
            Track(number=i, artist=f"Artist {i}", title=f"Track {i}")
            for i in range(1, 201)  # 200 tracks
        ]
        tracklist = Tracklist(url="https://1001tracklists.com/long-set", dj_name="Marathon DJ", tracks=tracks)
        assert len(tracklist.tracks) == 200

    def test_special_characters_in_labels(self):
        """Test special characters in label names."""
        track = Track(number=1, artist="Test", title="Test", label="!K7 Records / R&S")
        assert "!K7" in track.label
        assert "R&S" in track.label

    def test_guest_mix_in_tracklist(self):
        """Test handling of guest mix notation."""
        track = Track(number=10, artist="Guest DJ", title="Guest Mix", notes="Start of guest mix by Guest DJ")
        assert track.notes is not None
        assert "guest mix" in track.notes.lower()
