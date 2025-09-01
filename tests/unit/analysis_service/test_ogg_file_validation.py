"""Unit tests for OGG file validation."""

import stat
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from services.analysis_service.src.metadata_extractor import (
    InvalidAudioFileError,
    MetadataExtractionError,
    MetadataExtractor,
)


class TestOggFileValidation:
    """Tests for OGG file validation and error handling."""

    def test_corrupted_ogg_file_detection(self):
        """Test detection of corrupted OGG files."""
        # Create a fake corrupted OGG file
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            # Write invalid data (not a valid OGG file)
            tmp.write(b"This is not a valid OGG file content")
            tmp.flush()
            tmp_path = tmp.name

        try:
            extractor = MetadataExtractor()
            # Should raise MetadataExtractionError for corrupted file
            with pytest.raises(MetadataExtractionError) as exc_info:
                extractor.extract(tmp_path)
            assert "Failed to extract metadata" in str(exc_info.value)
        finally:
            # Clean up
            Path(tmp_path).unlink(missing_ok=True)

    def test_empty_ogg_file_handling(self):
        """Test handling of empty OGG files."""
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            # Create empty file
            tmp.flush()
            tmp_path = tmp.name

        try:
            extractor = MetadataExtractor()
            # Should raise error for empty file
            with pytest.raises(MetadataExtractionError):
                extractor.extract(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_non_audio_file_with_ogg_extension(self):
        """Test handling of non-audio files with OGG extension."""
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            # Write text content
            tmp.write(b"Just some text content, not audio data")
            tmp.flush()
            tmp_path = tmp.name

        try:
            extractor = MetadataExtractor()
            # Should fail to extract metadata
            with pytest.raises(MetadataExtractionError):
                extractor.extract(tmp_path)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    @patch("services.analysis_service.src.metadata_extractor.logger")
    def test_ogg_extraction_error_logging(self, mock_logger):
        """Test that OGG extraction errors are properly logged."""
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp.write(b"Invalid OGG content")
            tmp.flush()
            tmp_path = tmp.name

        try:
            extractor = MetadataExtractor()
            with pytest.raises(MetadataExtractionError):
                extractor.extract(tmp_path)

            # Verify error was logged
            assert mock_logger.error.called
            error_call = mock_logger.error.call_args[0][0]
            assert "Failed to extract metadata" in error_call
            assert tmp_path in error_call
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_ogg_file_permissions_error(self):
        """Test handling of OGG files with permission issues."""
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
            tmp.write(b"Some content")
            tmp.flush()
            tmp_path = tmp.name

        tmp_path_obj = Path(tmp_path)
        try:
            # Remove read permissions
            tmp_path_obj.chmod(stat.S_IWRITE)

            extractor = MetadataExtractor()
            # Should raise error due to permission denied
            with pytest.raises((MetadataExtractionError, PermissionError)):
                extractor.extract(tmp_path)
        finally:
            # Restore permissions and clean up
            tmp_path_obj.chmod(stat.S_IREAD | stat.S_IWRITE)
            tmp_path_obj.unlink(missing_ok=True)

    def test_validate_ogg_extension_case_insensitive(self):
        """Test that OGG extension validation is case-insensitive."""
        extractor = MetadataExtractor()

        # Test various case combinations
        test_cases = [".ogg", ".OGG", ".Ogg", ".oGg", ".oga", ".OGA", ".Oga"]

        for ext in test_cases:
            assert ext.lower() in extractor.SUPPORTED_FORMATS

    @patch("services.analysis_service.src.metadata_extractor.OggVorbis")
    def test_ogg_file_with_missing_info(self, mock_oggvorbis):
        """Test handling of OGG files with missing info section."""
        # Setup mock with no info
        mock_audio = MagicMock()
        mock_audio.tags = {"title": ["Test"]}
        mock_audio.info = None
        mock_audio.get = lambda key, default: mock_audio.tags.get(key, default)

        mock_oggvorbis.return_value = mock_audio

        extractor = MetadataExtractor()
        metadata = extractor._extract_ogg("/test/noinfo.ogg")

        # Should handle missing info gracefully
        assert metadata["title"] == "Test"
        assert metadata.get("duration") is None
        assert metadata.get("bitrate") is None

    @patch("services.analysis_service.src.metadata_extractor.Path")
    def test_validate_path_is_file(self, mock_path):
        """Test validation that path is a file, not directory."""

        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_file.return_value = False  # It's a directory
        mock_path_obj.suffix = ".ogg"
        mock_path.return_value = mock_path_obj

        extractor = MetadataExtractor()

        with pytest.raises(InvalidAudioFileError) as exc_info:
            extractor.extract("/some/directory.ogg")
        assert "Not a file" in str(exc_info.value)
