"""
Unit tests for the timing service.
"""

from datetime import timedelta

import pytest

from services.tracklist_service.src.models.tracklist import TrackEntry
from services.tracklist_service.src.services.timing_service import TimingService


class TestTimingService:
    """Test the timing service functionality."""

    @pytest.fixture
    def timing_service(self):
        """Create a timing service instance."""
        return TimingService()

    @pytest.fixture
    def sample_tracks(self):
        """Create sample track entries."""
        return [
            TrackEntry(
                position=1,
                start_time=timedelta(minutes=0),
                end_time=timedelta(minutes=5),
                artist="Artist 1",
                title="Track 1",
            ),
            TrackEntry(
                position=2,
                start_time=timedelta(minutes=5),
                end_time=timedelta(minutes=10),
                artist="Artist 2",
                title="Track 2",
            ),
            TrackEntry(
                position=3,
                start_time=timedelta(minutes=10),
                end_time=None,  # Last track, no end time
                artist="Artist 3",
                title="Track 3",
            ),
        ]

    def test_parse_timing_format_mm_ss(self, timing_service):
        """Test parsing MM:SS format."""
        result = timing_service.parse_timing_format("5:30")
        assert result == timedelta(minutes=5, seconds=30)

        result = timing_service.parse_timing_format("65:45")  # Over 60 minutes
        assert result == timedelta(minutes=65, seconds=45)

    def test_parse_timing_format_hh_mm_ss(self, timing_service):
        """Test parsing HH:MM:SS format."""
        result = timing_service.parse_timing_format("1:05:30")
        assert result == timedelta(hours=1, minutes=5, seconds=30)

        result = timing_service.parse_timing_format("2:30:00")
        assert result == timedelta(hours=2, minutes=30)

    def test_parse_timing_format_decimal(self, timing_service):
        """Test parsing decimal minutes format."""
        result = timing_service.parse_timing_format("5.5")
        assert result == timedelta(minutes=5.5)

        result = timing_service.parse_timing_format("10.25")
        assert result == timedelta(minutes=10.25)

    def test_parse_timing_format_invalid(self, timing_service):
        """Test parsing invalid format returns zero."""
        result = timing_service.parse_timing_format("invalid")
        assert result == timedelta(0)

        result = timing_service.parse_timing_format("")
        assert result == timedelta(0)

        result = timing_service.parse_timing_format(None)
        assert result == timedelta(0)

    def test_calculate_offset_from_start(self, timing_service):
        """Test calculating offset for mix start time."""
        first_track = timedelta(minutes=2)
        mix_start = timedelta(minutes=0)

        offset = timing_service.calculate_offset_from_start(first_track, mix_start)
        assert offset == timedelta(minutes=-2)

        # Mix starts later than first track time
        mix_start2 = timedelta(minutes=5)
        offset2 = timing_service.calculate_offset_from_start(first_track, mix_start2)
        assert offset2 == timedelta(minutes=3)

    def test_apply_offset(self, timing_service, sample_tracks):
        """Test applying offset to tracks."""
        offset = timedelta(minutes=2)
        tracks = timing_service._apply_offset(sample_tracks.copy(), offset)

        assert tracks[0].start_time == timedelta(minutes=2)
        assert tracks[0].end_time == timedelta(minutes=7)
        assert tracks[1].start_time == timedelta(minutes=7)
        assert tracks[2].start_time == timedelta(minutes=12)

    def test_fix_timing_gaps(self, timing_service):
        """Test fixing gaps in track timings."""
        tracks = [
            TrackEntry(
                position=1,
                start_time=timedelta(minutes=0),
                end_time=None,  # Missing end time
                artist="A1",
                title="T1",
            ),
            TrackEntry(
                position=2,
                start_time=timedelta(minutes=5),
                end_time=timedelta(minutes=12),  # Overlaps with next
                artist="A2",
                title="T2",
            ),
            TrackEntry(
                position=3,
                start_time=timedelta(minutes=10),  # Overlap
                end_time=None,
                artist="A3",
                title="T3",
            ),
        ]

        fixed = timing_service._fix_timing_gaps(tracks)

        # First track should get end time from second track start
        assert fixed[0].end_time == timedelta(minutes=5)

        # Overlap should be fixed
        assert fixed[1].end_time <= fixed[2].start_time

    def test_ensure_within_duration(self, timing_service):
        """Test ensuring tracks fit within audio duration."""
        tracks = [
            TrackEntry(
                position=1,
                start_time=timedelta(minutes=0),
                end_time=timedelta(minutes=5),
                artist="Artist 1",
                title="Track 1",
            ),
            TrackEntry(
                position=2,
                start_time=timedelta(minutes=5),
                end_time=timedelta(minutes=10),
                artist="Artist 2",
                title="Track 2",
            ),
            TrackEntry(
                position=3,
                start_time=timedelta(minutes=10),
                end_time=None,  # Last track, no end time initially
                artist="Artist 3",
                title="Track 3",
            ),
        ]

        audio_duration = timedelta(minutes=12)  # Reasonable duration

        result = timing_service._ensure_within_duration(tracks, audio_duration)

        # Last track should get end time set to audio duration
        assert result[-1].end_time == audio_duration

    def test_ensure_within_duration_scaling(self, timing_service):
        """Test scaling tracks when they exceed duration."""
        tracks = [
            TrackEntry(
                position=1, start_time=timedelta(minutes=0), end_time=timedelta(minutes=10), artist="A1", title="T1"
            ),
            TrackEntry(
                position=2, start_time=timedelta(minutes=10), end_time=timedelta(minutes=20), artist="A2", title="T2"
            ),
            TrackEntry(position=3, start_time=timedelta(minutes=20), end_time=None, artist="A3", title="T3"),
        ]

        audio_duration = timedelta(minutes=10)  # Half the expected duration

        tracks = timing_service._ensure_within_duration(tracks, audio_duration)

        # All timestamps should be scaled down
        assert tracks[0].start_time == timedelta(0)
        assert tracks[1].start_time <= timedelta(minutes=10)
        assert tracks[2].start_time <= timedelta(minutes=10)

    def test_validate_timing_consistency_valid(self, timing_service, sample_tracks):
        """Test validating consistent timings."""
        audio_duration = timedelta(minutes=15)

        is_valid, issues = timing_service.validate_timing_consistency(sample_tracks, audio_duration)

        assert is_valid is True
        assert len(issues) == 0

    def test_validate_timing_consistency_overlap(self, timing_service):
        """Test detecting timing overlaps."""
        tracks = [
            TrackEntry(
                position=1,
                start_time=timedelta(minutes=0),
                end_time=timedelta(minutes=6),  # Overlaps with next
                artist="A1",
                title="T1",
            ),
            TrackEntry(
                position=2,
                start_time=timedelta(minutes=5),  # Overlap
                end_time=timedelta(minutes=10),
                artist="A2",
                title="T2",
            ),
        ]

        audio_duration = timedelta(minutes=15)

        is_valid, issues = timing_service.validate_timing_consistency(tracks, audio_duration)

        assert is_valid is False
        assert len(issues) > 0
        assert "overlaps" in issues[0].lower()

    def test_validate_timing_consistency_exceeds_duration(self, timing_service):
        """Test detecting tracks exceeding audio duration."""
        tracks = [
            TrackEntry(
                position=1, start_time=timedelta(minutes=0), end_time=timedelta(minutes=5), artist="A1", title="T1"
            ),
            TrackEntry(
                position=2,
                start_time=timedelta(minutes=5),
                end_time=timedelta(minutes=12),  # Exceeds duration
                artist="A2",
                title="T2",
            ),
        ]

        audio_duration = timedelta(minutes=10)

        is_valid, issues = timing_service.validate_timing_consistency(tracks, audio_duration)

        assert is_valid is False
        assert len(issues) > 0
        assert "after audio duration" in issues[0]

    def test_validate_timing_consistency_short_tracks(self, timing_service):
        """Test detecting very short tracks."""
        tracks = [
            TrackEntry(
                position=1,
                start_time=timedelta(seconds=0),
                end_time=timedelta(seconds=15),  # Too short (< 30 seconds)
                artist="A1",
                title="T1",
            ),
            TrackEntry(
                position=2, start_time=timedelta(seconds=15), end_time=timedelta(minutes=5), artist="A2", title="T2"
            ),
        ]

        audio_duration = timedelta(minutes=10)

        is_valid, issues = timing_service.validate_timing_consistency(tracks, audio_duration)

        assert is_valid is False
        assert len(issues) > 0
        assert "too short" in issues[0]

    def test_adjust_track_timings_complete(self, timing_service):
        """Test complete timing adjustment workflow."""
        tracks = [
            TrackEntry(
                position=1,
                start_time=timedelta(minutes=2),  # Starts at 2:00
                end_time=None,
                artist="A1",
                title="T1",
            ),
            TrackEntry(
                position=2,
                start_time=timedelta(minutes=7),
                end_time=timedelta(minutes=15),  # Exceeds duration
                artist="A2",
                title="T2",
            ),
        ]

        audio_duration = timedelta(minutes=10)
        offset = timedelta(minutes=-2)  # Shift to start at 0:00

        adjusted = timing_service.adjust_track_timings(tracks, audio_duration, offset)

        # First track should start at 0:00
        assert adjusted[0].start_time == timedelta(0)
        # First track should have end time set
        assert adjusted[0].end_time is not None
        # Last track shouldn't exceed duration
        assert adjusted[-1].end_time <= audio_duration

    def test_validate_track_durations(self, timing_service):
        """Test validating individual track durations."""
        tracks = [
            TrackEntry(
                position=1,
                start_time=timedelta(seconds=0),
                end_time=timedelta(seconds=10),  # Too short
                artist="A1",
                title="T1",
            ),
            TrackEntry(
                position=2,
                start_time=timedelta(seconds=40),  # Leave room for track 1 extension
                end_time=timedelta(minutes=30),  # Too long
                artist="A2",
                title="T2",
            ),
            TrackEntry(
                position=3,
                start_time=timedelta(minutes=30),
                end_time=timedelta(minutes=35),  # Normal
                artist="A3",
                title="T3",
            ),
        ]

        validated = timing_service._validate_track_durations(tracks)

        # First track should be extended to minimum if possible
        duration1 = validated[0].end_time - validated[0].start_time
        # Since next track is at 40 seconds, track can be extended to 30 seconds
        assert duration1 >= timing_service.min_track_duration or duration1 == timedelta(seconds=40)

        # Second track should be truncated to maximum
        duration2 = validated[1].end_time - validated[1].start_time
        assert duration2 <= timing_service.max_track_duration
