"""
Unit tests for audio validation service.
"""

from datetime import timedelta
from unittest.mock import patch
from uuid import uuid4

import pytest

from services.tracklist_service.src.models.tracklist import TrackEntry, Tracklist
from services.tracklist_service.src.services.audio_validation_service import AudioValidationService


class TestAudioValidationService:
    """Test AudioValidationService class."""

    @pytest.fixture
    def service(self):
        """Create AudioValidationService instance."""
        return AudioValidationService()

    @pytest.fixture
    def sample_tracklist(self):
        """Create a sample tracklist with proper timings."""
        tracks = [
            TrackEntry(
                position=1,
                start_time=timedelta(minutes=0),
                end_time=timedelta(minutes=3, seconds=30),
                artist="Artist 1",
                title="Track 1",
            ),
            TrackEntry(
                position=2,
                start_time=timedelta(minutes=3, seconds=30),
                end_time=timedelta(minutes=7),
                artist="Artist 2",
                title="Track 2",
            ),
        ]

        return Tracklist(audio_file_id=uuid4(), source="manual", tracks=tracks)

    @pytest.fixture
    def gapped_tracklist(self):
        """Create tracklist with gaps between tracks."""
        tracks = [
            TrackEntry(
                position=1,
                start_time=timedelta(minutes=0),
                end_time=timedelta(minutes=3),
                artist="Artist 1",
                title="Track 1",
            ),
            TrackEntry(
                position=2,
                start_time=timedelta(minutes=5),  # 2 minute gap
                end_time=timedelta(minutes=8),
                artist="Artist 2",
                title="Track 2",
            ),
        ]

        return Tracklist(audio_file_id=uuid4(), source="manual", tracks=tracks)

    @pytest.mark.asyncio
    async def test_get_audio_duration_success(self, service):
        """Test successful audio duration detection."""
        with patch("pathlib.Path.exists", return_value=True):
            duration = await service.get_audio_duration("test.mp3")
            assert duration == 420.0  # Mock duration

    @pytest.mark.asyncio
    async def test_get_audio_duration_file_not_found(self, service):
        """Test audio duration with missing file."""
        with patch("pathlib.Path.exists", return_value=False):
            duration = await service.get_audio_duration("missing.mp3")
            assert duration is None

    @pytest.mark.asyncio
    async def test_get_audio_duration_unsupported_format(self, service):
        """Test audio duration with unsupported format."""
        with patch("pathlib.Path.exists", return_value=True):
            duration = await service.get_audio_duration("test.txt")
            assert duration is None

    @pytest.mark.asyncio
    async def test_validate_audio_duration_success(self, service, sample_tracklist):
        """Test successful audio duration validation."""
        with patch.object(service, "get_audio_duration", return_value=420.0):  # 7 minutes
            result = await service.validate_audio_duration("test.mp3", sample_tracklist)

            assert result.valid is True
            assert len(result.warnings) == 0
            assert result.metadata["actual_duration_seconds"] == 420.0
            assert result.metadata["tracklist_duration_seconds"] == 420.0  # 7 minutes

    @pytest.mark.asyncio
    async def test_validate_audio_duration_mismatch(self, service, sample_tracklist):
        """Test audio duration validation with mismatch."""
        with patch.object(service, "get_audio_duration", return_value=300.0):  # 5 minutes audio, 7 minute tracklist
            result = await service.validate_audio_duration("test.mp3", sample_tracklist, tolerance_seconds=1.0)

            assert result.valid is False  # Significant mismatch
            assert len(result.warnings) >= 1
            assert any("Duration mismatch" in w for w in result.warnings)
            assert result.metadata["duration_difference_seconds"] == 120.0  # 2 minute difference

    @pytest.mark.asyncio
    async def test_validate_audio_duration_within_tolerance(self, service, sample_tracklist):
        """Test audio duration validation within tolerance."""
        with patch.object(service, "get_audio_duration", return_value=421.0):  # 1 second difference
            result = await service.validate_audio_duration("test.mp3", sample_tracklist, tolerance_seconds=2.0)

            assert result.valid is True  # Within tolerance
            assert len(result.warnings) == 0

    @pytest.mark.asyncio
    async def test_validate_audio_duration_no_audio_file(self, service, sample_tracklist):
        """Test validation when audio file cannot be read."""
        with patch.object(service, "get_audio_duration", return_value=None):
            result = await service.validate_audio_duration("missing.mp3", sample_tracklist)

            assert result.valid is False
            assert "Could not determine duration" in result.error

    @pytest.mark.asyncio
    async def test_validate_audio_duration_no_tracks(self, service):
        """Test validation with empty tracklist."""
        empty_tracklist = Tracklist(audio_file_id=uuid4(), source="manual", tracks=[])

        result = await service.validate_audio_duration("test.mp3", empty_tracklist)

        assert result.valid is False
        assert "no tracks to validate" in result.error

    @pytest.mark.asyncio
    async def test_validate_track_timings(self, service, sample_tracklist):
        """Test individual track timing validation."""
        warnings = await service.validate_track_timings(sample_tracklist, 420.0)

        # Should have no warnings for well-formed tracklist
        assert len(warnings) == 0

    @pytest.mark.asyncio
    async def test_validate_track_timings_beyond_audio(self, service):
        """Test track timing validation when tracks exceed audio duration."""
        tracks = [
            TrackEntry(
                position=1,
                start_time=timedelta(minutes=8),  # Starts after 7 minute audio
                end_time=timedelta(minutes=10),
                artist="Artist 1",
                title="Track 1",
            )
        ]

        tracklist = Tracklist(audio_file_id=uuid4(), source="manual", tracks=tracks)

        warnings = await service.validate_track_timings(tracklist, 420.0, tolerance_seconds=1.0)

        assert len(warnings) >= 1
        assert any("beyond audio duration" in w for w in warnings)

    @pytest.mark.asyncio
    async def test_validate_track_timings_very_short_track(self, service):
        """Test validation of very short tracks."""
        tracks = [
            TrackEntry(
                position=1,
                start_time=timedelta(seconds=0),
                end_time=timedelta(seconds=2),  # 2 second track
                artist="Artist 1",
                title="Short Track",
            )
        ]

        tracklist = Tracklist(audio_file_id=uuid4(), source="manual", tracks=tracks)

        warnings = await service.validate_track_timings(tracklist, 420.0)

        assert len(warnings) >= 1
        assert any("very short" in w for w in warnings)

    def test_validate_track_sequence_no_issues(self, service, sample_tracklist):
        """Test track sequence validation with no issues."""
        warnings = service.validate_track_sequence(sample_tracklist)
        assert len(warnings) == 0

    def test_validate_track_sequence_with_gap(self, service, gapped_tracklist):
        """Test track sequence validation with gap."""
        warnings = service.validate_track_sequence(gapped_tracklist)

        assert len(warnings) >= 1
        assert any("Gap of" in w for w in warnings)

    def test_validate_track_sequence_with_overlap(self, service):
        """Test track sequence validation with overlap."""
        tracks = [
            TrackEntry(
                position=1,
                start_time=timedelta(minutes=0),
                end_time=timedelta(minutes=4),  # Overlaps with next
                artist="Artist 1",
                title="Track 1",
            ),
            TrackEntry(
                position=2,
                start_time=timedelta(minutes=3),  # Starts before previous ends
                end_time=timedelta(minutes=6),
                artist="Artist 2",
                title="Track 2",
            ),
        ]

        tracklist = Tracklist(audio_file_id=uuid4(), source="manual", tracks=tracks)

        warnings = service.validate_track_sequence(tracklist)

        assert len(warnings) >= 1
        assert any("Overlap of" in w for w in warnings)

    @pytest.mark.asyncio
    async def test_suggest_timing_corrections(self, service, sample_tracklist):
        """Test timing correction suggestions."""
        # Modify tracklist to have issues
        sample_tracklist.tracks[1].end_time = timedelta(minutes=10)  # Beyond 7 minute audio

        with patch.object(service, "get_audio_duration", return_value=420.0):
            suggestions = await service.suggest_timing_corrections(sample_tracklist, "test.mp3")

            assert len(suggestions["corrections"]) >= 1
            correction = suggestions["corrections"][0]
            assert correction["issue"] == "extends_beyond_audio"
            assert correction["track_position"] == 2
            assert correction["suggested_end_time"] == 420.0

    def test_estimate_track_durations(self, service):
        """Test track duration estimation."""
        # Create tracklist with missing end times
        tracks = [
            TrackEntry(
                position=1,
                start_time=timedelta(minutes=0),
                # No end time
                artist="Artist 1",
                title="Track 1",
            ),
            TrackEntry(
                position=2,
                start_time=timedelta(minutes=4),
                end_time=timedelta(minutes=7),  # Has end time
                artist="Artist 2",
                title="Track 2",
            ),
        ]

        tracklist = Tracklist(audio_file_id=uuid4(), source="manual", tracks=tracks)

        estimates = service.estimate_track_durations(tracklist, 420.0)

        assert len(estimates) == 2
        assert estimates[0]["method"] == "next_track_start"  # Uses next track start
        assert estimates[1]["method"] == "provided"  # Has end time
        assert estimates[0]["end_time"] == 240.0  # 4 minutes (next track start)
        assert estimates[1]["duration"] == 180.0  # 3 minutes
