"""Tests for real audio duration detection functionality."""

from datetime import timedelta
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from mutagen import MutagenError
from pydub.exceptions import CouldntDecodeError

from services.tracklist_service.src.models.tracklist import TrackEntry, Tracklist
from services.tracklist_service.src.services.audio_validation_service import AudioValidationService
from services.tracklist_service.src.services.cue_integration import CueIntegrationService


class TestAudioDurationDetection:
    """Test real audio duration detection using mutagen and pydub."""

    @pytest.fixture
    def audio_service(self):
        """Create audio validation service instance."""
        return AudioValidationService()

    @pytest.fixture
    def sample_tracklist(self):
        """Create a sample tracklist for testing."""
        tracks = [
            TrackEntry(
                id=uuid4(),
                position=1,
                title="Track 1",
                artist="Artist 1",
                start_time=timedelta(seconds=0),
                end_time=timedelta(seconds=180),
            ),
            TrackEntry(
                id=uuid4(),
                position=2,
                title="Track 2",
                artist="Artist 2",
                start_time=timedelta(seconds=180),
                end_time=timedelta(seconds=360),
            ),
            TrackEntry(
                id=uuid4(),
                position=3,
                title="Track 3",
                artist="Artist 3",
                start_time=timedelta(seconds=360),
                end_time=timedelta(seconds=420),
            ),
        ]
        return Tracklist(
            id=uuid4(),
            title="Test Mix",
            artist="Test DJ",
            tracks=tracks,
            audio_file_id=uuid4(),
            source="manual",
        )

    @pytest.mark.asyncio
    async def test_get_audio_duration_with_mutagen_success(self, audio_service):
        """Test successful audio duration detection using mutagen."""
        test_file = "/test/audio.mp3"
        expected_duration = 420.5

        with patch("pathlib.Path.exists", return_value=True):
            mock_audio_file = Mock()
            mock_audio_file.info.length = expected_duration

            with patch("mutagen.File", return_value=mock_audio_file):
                duration = await audio_service.get_audio_duration(test_file)

                assert duration == expected_duration

    @pytest.mark.asyncio
    async def test_get_audio_duration_mutagen_fallback_to_pydub(self, audio_service):
        """Test fallback to pydub when mutagen fails."""
        test_file = "/test/audio.wav"
        expected_duration = 300.0

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("mutagen.File", side_effect=MutagenError("Unsupported format")),
        ):
            # Pydub succeeds
            mock_audio_segment = Mock()
            mock_audio_segment.__len__ = Mock(return_value=300000)  # 300 seconds in milliseconds

            with patch("pydub.AudioSegment.from_file", return_value=mock_audio_segment):
                duration = await audio_service.get_audio_duration(test_file)

                assert duration == expected_duration

    @pytest.mark.asyncio
    async def test_get_audio_duration_both_libraries_fail(self, audio_service):
        """Test when both mutagen and pydub fail to get duration."""
        test_file = "/test/unsupported.xyz"

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("mutagen.File", side_effect=MutagenError("Unsupported")),
            patch("pydub.AudioSegment.from_file", side_effect=CouldntDecodeError("Unsupported")),
        ):
            duration = await audio_service.get_audio_duration(test_file)

            assert duration is None

    @pytest.mark.asyncio
    async def test_get_audio_duration_file_not_found(self, audio_service):
        """Test audio duration detection when file doesn't exist."""
        test_file = "/nonexistent/file.mp3"

        with patch("pathlib.Path.exists", return_value=False):
            duration = await audio_service.get_audio_duration(test_file)

            assert duration is None

    @pytest.mark.asyncio
    async def test_get_audio_duration_mutagen_no_length_attribute(self, audio_service):
        """Test when mutagen file has no length attribute."""
        test_file = "/test/audio.mp3"
        expected_duration = 250.0

        with patch("pathlib.Path.exists", return_value=True):
            # Mutagen returns file but no length
            mock_audio_file = Mock()
            del mock_audio_file.info.length  # Remove length attribute

            with patch("mutagen.File", return_value=mock_audio_file):
                # Should fallback to pydub
                mock_audio_segment = Mock()
                mock_audio_segment.__len__ = Mock(return_value=250000)

                with patch("pydub.AudioSegment.from_file", return_value=mock_audio_segment):
                    duration = await audio_service.get_audio_duration(test_file)

                    assert duration == expected_duration

    @pytest.mark.asyncio
    async def test_validate_audio_duration_with_real_detection(self, audio_service, sample_tracklist):
        """Test audio duration validation with real duration detection."""
        test_file = "/test/audio.mp3"

        # Mock the actual duration detection
        with patch.object(audio_service, "get_audio_duration", return_value=420.0):
            result = await audio_service.validate_audio_duration(test_file, sample_tracklist, 2.0)

            assert result.valid is True
            assert result.audio_duration == 420.0
            assert result.tracklist_duration == 420.0
            assert result.error is None

    @pytest.mark.asyncio
    async def test_validate_audio_duration_mismatch(self, audio_service, sample_tracklist):
        """Test validation when audio and tracklist durations don't match."""
        test_file = "/test/audio.mp3"

        # Mock shorter audio duration than tracklist
        with patch.object(audio_service, "get_audio_duration", return_value=300.0):
            result = await audio_service.validate_audio_duration(test_file, sample_tracklist, 2.0)

            assert result.valid is False  # Track extends beyond audio
            assert result.audio_duration == 300.0
            assert result.tracklist_duration == 420.0
            assert len(result.warnings) > 0
            assert any("Duration mismatch" in w for w in result.warnings)

    @pytest.mark.asyncio
    async def test_validate_track_beyond_audio(self, audio_service, sample_tracklist):
        """Test validation when tracks extend beyond audio duration."""
        warnings = await audio_service.validate_track_timings(sample_tracklist, 300.0, 2.0)

        # Should have warnings for tracks that extend beyond audio
        assert len(warnings) > 0
        assert any("beyond audio duration" in w for w in warnings)

    @pytest.mark.asyncio
    async def test_suggest_timing_corrections_with_real_duration(self, audio_service, sample_tracklist):
        """Test timing correction suggestions with real audio duration."""
        test_file = "/test/audio.mp3"

        # Mock audio duration shorter than tracklist
        with patch.object(audio_service, "get_audio_duration", return_value=300.0):
            suggestions = await audio_service.suggest_timing_corrections(sample_tracklist, test_file)

            assert "corrections" in suggestions
            assert len(suggestions["corrections"]) > 0
            assert suggestions["metadata"]["audio_duration"] == 300.0

    @pytest.mark.asyncio
    async def test_estimate_track_durations_with_real_audio(self, audio_service):
        """Test track duration estimation with real audio duration."""
        # Create tracklist with missing end times
        tracks = [
            TrackEntry(
                id=uuid4(),
                position=1,
                title="Track 1",
                artist="Artist 1",
                start_time=timedelta(seconds=0),
                end_time=None,  # Missing end time
            ),
            TrackEntry(
                id=uuid4(),
                position=2,
                title="Track 2",
                artist="Artist 2",
                start_time=timedelta(seconds=180),
                end_time=None,  # Missing end time
            ),
        ]
        tracklist = Tracklist(
            id=uuid4(),
            title="Test Mix",
            artist="Test DJ",
            tracks=tracks,
            audio_file_id=uuid4(),
            source="manual",
        )

        # Estimate with real audio duration
        estimates = audio_service.estimate_track_durations(tracklist, 420.0)

        assert len(estimates) == 2
        assert estimates[0]["end_time"] == 180.0  # First track ends when second starts
        assert estimates[0]["method"] == "next_track_start"
        assert estimates[1]["end_time"] == 420.0  # Last track ends at audio end
        assert estimates[1]["method"] == "audio_end"


class TestCueIntegrationServiceDurationExtraction:
    """Test CUE content duration extraction functionality."""

    @pytest.mark.asyncio
    async def test_cue_duration_extraction(self):
        """Test extracting duration from CUE content."""
        cue_content = """
FILE "audio.wav" WAVE
  TRACK 01 AUDIO
    TITLE "Track 1"
    PERFORMER "Artist 1"
    INDEX 01 00:00:00
  TRACK 02 AUDIO
    TITLE "Track 2"
    PERFORMER "Artist 2"
    INDEX 01 03:00:00
  TRACK 03 AUDIO
    TITLE "Track 3"
    PERFORMER "Artist 3"
    INDEX 01 06:00:00
"""

        # Mock the CUE parser
        with patch("services.tracklist_service.src.services.cue_integration.CueParser") as mock_parser_class:
            mock_parser = Mock()
            mock_parser_class.return_value = mock_parser

            # Create mock parsed data
            mock_track1 = Mock(start_time_ms=0)
            mock_track2 = Mock(start_time_ms=180000)  # 3 minutes
            mock_track3 = Mock(start_time_ms=360000)  # 6 minutes

            mock_file = Mock(tracks=[mock_track1, mock_track2, mock_track3])
            mock_cue_data = Mock(files=[mock_file])

            mock_parser.parse.return_value = mock_cue_data

            # Test with audio duration provided
            cue_integration = CueIntegrationService()
            result = cue_integration.validate_cue_content(cue_content, audio_duration_seconds=420.0)

            # Should extract tracklist duration
            assert result.tracklist_duration == 360.0  # Last track start time
            assert result.audio_duration == 420.0

    @pytest.mark.asyncio
    async def test_cue_duration_extraction_error_handling(self):
        """Test error handling in CUE duration extraction."""
        cue_content = "INVALID CUE CONTENT"

        # Mock parser to raise exception
        with patch("services.tracklist_service.src.services.cue_integration.CueParser") as mock_parser_class:
            mock_parser = Mock()
            mock_parser_class.return_value = mock_parser
            mock_parser.parse.side_effect = Exception("Parse error")

            cue_integration = CueIntegrationService()
            result = cue_integration.validate_cue_content(cue_content, audio_duration_seconds=420.0)

            # Should handle error gracefully
            assert result.audio_duration == 420.0
            assert result.tracklist_duration is None
            assert any("Could not extract duration" in w for w in result.warnings)
