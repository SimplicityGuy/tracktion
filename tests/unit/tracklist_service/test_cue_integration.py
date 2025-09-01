"""
Unit tests for CUE handler integration service.
"""

from datetime import timedelta
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

from services.tracklist_service.src.models.cue_file import CueFormat
from services.tracklist_service.src.models.tracklist import TrackEntry, Tracklist
from services.tracklist_service.src.services.cue_integration import (
    CueFormatMapper,
    CueIntegrationService,
    TracklistToCueMapper,
)


class TestCueFormatMapper:
    """Test CueFormatMapper class."""

    def test_format_mapping(self):
        """Test format mapping between enums."""
        # Test all supported formats
        test_cases = [
            CueFormat.STANDARD,
            CueFormat.CDJ,
            CueFormat.TRAKTOR,
            CueFormat.SERATO,
            CueFormat.REKORDBOX,
            CueFormat.KODI,
        ]

        for format_val in test_cases:
            # Forward mapping
            handler_format = CueFormatMapper.to_cue_handler_format(format_val)
            assert handler_format is not None

            # Reverse mapping
            back_to_original = CueFormatMapper.from_cue_handler_format(handler_format)
            assert back_to_original == format_val

    def test_unsupported_format_error(self):
        """Test error handling for unsupported formats."""
        # This should not happen in practice but test error handling
        with (
            patch.dict(CueFormatMapper.FORMAT_MAPPING, {}, clear=True),
            pytest.raises(ValueError, match="Unsupported format"),
        ):
            CueFormatMapper.to_cue_handler_format(CueFormat.STANDARD)


class TestTracklistToCueMapper:
    """Test TracklistToCueMapper class."""

    @pytest.fixture
    def sample_tracklist(self):
        """Create a sample tracklist."""
        tracks = [
            TrackEntry(
                position=1,
                start_time=timedelta(minutes=0),
                end_time=timedelta(minutes=3, seconds=30),
                artist="Test Artist 1",
                title="Test Track 1",
                remix="Original Mix",
                label="Test Label 1",
                catalog_track_id=uuid4(),
                confidence=0.95,
                transition_type="cut",
                is_manual_entry=True,
            ),
            TrackEntry(
                position=2,
                start_time=timedelta(minutes=3, seconds=30),
                end_time=timedelta(minutes=7, seconds=15),
                artist="Test Artist 2",
                title="Test Track 2",
                confidence=0.8,
                is_manual_entry=False,
            ),
        ]

        return Tracklist(
            audio_file_id=uuid4(),
            source="manual",
            tracks=tracks,
            confidence_score=0.87,
            is_draft=True,
            draft_version=1,
        )

    def test_timedelta_to_milliseconds(self):
        """Test converting timedelta to milliseconds."""
        td = timedelta(minutes=3, seconds=30, milliseconds=500)
        ms = TracklistToCueMapper.timedelta_to_milliseconds(td)

        expected_ms = (3 * 60 + 30) * 1000 + 500
        assert ms == expected_ms

    def test_milliseconds_to_cue_time(self):
        """Test converting milliseconds to CUE time."""
        ms = 210000  # 3 minutes 30 seconds
        cue_time = TracklistToCueMapper.milliseconds_to_cue_time(ms)

        assert cue_time.minutes == 3
        assert cue_time.seconds == 30
        assert cue_time.frames >= 0

    def test_tracklist_to_cue_tracks(self, sample_tracklist):
        """Test converting tracklist to CUE tracks."""
        cue_tracks = TracklistToCueMapper.tracklist_to_cue_tracks(sample_tracklist)

        assert len(cue_tracks) == 2

        # Check first track
        track1 = cue_tracks[0]
        assert track1.number == 1
        assert track1.title == "Test Track 1 (Original Mix)"
        assert track1.performer == "Test Artist 1"
        assert 1 in track1.indices  # INDEX 01
        assert "LABEL" in track1.rem_fields
        assert "CATALOG_ID" in track1.rem_fields
        assert "CONFIDENCE" in track1.rem_fields
        assert "TRANSITION" in track1.rem_fields
        assert "MANUAL_ENTRY" in track1.rem_fields

        # Check second track
        track2 = cue_tracks[1]
        assert track2.number == 2
        assert track2.title == "Test Track 2"  # No remix
        assert "MANUAL_ENTRY" not in track2.rem_fields  # False, so not included


class TestCueIntegrationService:
    """Test CueIntegrationService class."""

    @pytest.fixture
    def service(self):
        """Create CueIntegrationService instance."""
        return CueIntegrationService()

    @pytest.fixture
    def sample_tracklist(self):
        """Create a sample tracklist."""
        tracks = [
            TrackEntry(
                position=1,
                start_time=timedelta(minutes=0),
                artist="Artist 1",
                title="Track 1",
            ),
            TrackEntry(
                position=2,
                start_time=timedelta(minutes=4),
                artist="Artist 2",
                title="Track 2",
            ),
        ]

        return Tracklist(audio_file_id=uuid4(), source="manual", tracks=tracks)

    def test_initialization(self, service):
        """Test service initialization."""
        assert service.generator is not None
        assert service.validator is not None
        assert service.converter is not None
        assert service.format_mapper is not None
        assert service.tracklist_mapper is not None

    @patch("services.tracklist_service.src.services.cue_integration.get_generator")
    def test_generate_cue_content_success(self, mock_get_generator, service, sample_tracklist):
        """Test successful CUE content generation."""
        # Setup mock
        mock_generator = MagicMock()
        mock_generator.generate.return_value = "MOCK CUE CONTENT"
        mock_get_generator.return_value = mock_generator

        # Test
        success, content, error = service.generate_cue_content(sample_tracklist, CueFormat.STANDARD, "test_audio.wav")

        assert success is True
        assert content == "MOCK CUE CONTENT"
        assert error is None
        assert mock_get_generator.called

        # Verify generator was called with disc and files list
        mock_generator.generate.assert_called_once()
        args = mock_generator.generate.call_args[0]
        assert len(args) == 2  # disc and files list
        cue_disc, cue_files = args
        assert cue_disc.title == f"Tracklist {sample_tracklist.id}"
        assert len(cue_files) == 1
        assert cue_files[0].filename == "test_audio.wav"
        assert cue_files[0].file_type == "WAVE"
        assert len(cue_files[0].tracks) == 2

    @patch("services.tracklist_service.src.services.cue_integration.get_generator")
    def test_generate_cue_content_failure(self, mock_get_generator, service, sample_tracklist):
        """Test CUE content generation failure."""
        # Setup mock to raise exception
        mock_get_generator.side_effect = Exception("Generation failed")

        # Test
        success, content, error = service.generate_cue_content(sample_tracklist, CueFormat.CDJ)

        assert success is False
        assert content == ""
        assert "Generation failed" in error

    def test_validate_cue_content_valid(self, service):
        """Test CUE content validation with valid content."""
        with patch.object(service.validator, "validate") as mock_validate:
            # Setup mock validation result
            mock_validation = MagicMock()
            mock_validation.is_valid = True
            mock_validation.issues = []
            mock_validate.return_value = mock_validation

            # Test
            result = service.validate_cue_content("VALID CUE CONTENT")

            assert result.valid is True
            assert result.error is None
            assert len(result.warnings) == 0

    def test_get_supported_formats(self, service):
        """Test getting list of supported formats."""
        formats = service.get_supported_formats()

        assert len(formats) == 6
        assert CueFormat.STANDARD in formats
        assert CueFormat.CDJ in formats
        assert CueFormat.TRAKTOR in formats
