"""Unit tests for OGG Vorbis metadata extraction."""

import tempfile
from unittest.mock import MagicMock, patch

import pytest

from services.analysis_service.src.metadata_extractor import (
    InvalidAudioFileError,
    MetadataExtractor,
)


class TestOggMetadataExtraction:
    """Tests for OGG Vorbis metadata extraction."""

    def test_supported_formats_includes_ogg(self):
        """Test that OGG formats are in supported formats."""
        extractor = MetadataExtractor()
        assert ".ogg" in extractor.SUPPORTED_FORMATS
        assert ".oga" in extractor.SUPPORTED_FORMATS

    def test_format_handlers_include_ogg(self):
        """Test that OGG format handlers are registered."""
        extractor = MetadataExtractor()
        assert ".ogg" in extractor._format_handlers
        assert ".oga" in extractor._format_handlers
        assert extractor._format_handlers[".ogg"] == extractor._extract_ogg
        assert extractor._format_handlers[".oga"] == extractor._extract_ogg

    @patch("services.analysis_service.src.metadata_extractor.OggVorbis")
    def test_extract_ogg_metadata(self, mock_oggvorbis):
        """Test extracting metadata from OGG file."""
        # Setup mock
        mock_audio = MagicMock()
        mock_audio.tags = {
            "title": ["Test Song"],
            "artist": ["Test Artist"],
            "album": ["Test Album"],
            "date": ["2024"],
            "genre": ["Electronic"],
            "tracknumber": ["5"],
            "albumartist": ["Album Artist"],
            "comment": ["Test comment"],
            "encoder": ["Xiph.Org libVorbis"],
            "organization": ["Test Org"],
        }
        mock_audio.info.length = 180.5
        mock_audio.info.bitrate = 192000
        mock_audio.info.sample_rate = 44100
        mock_audio.info.channels = 2
        mock_audio.info.bitrate_nominal = 192000
        mock_audio.info.encoder_version = "libVorbis 1.3.7"

        mock_audio.get = lambda key, default: mock_audio.tags.get(key, default)
        mock_oggvorbis.return_value = mock_audio

        # Test extraction
        extractor = MetadataExtractor()
        metadata = extractor._extract_ogg("/test/file.ogg")

        # Verify metadata
        assert metadata["title"] == "Test Song"
        assert metadata["artist"] == "Test Artist"
        assert metadata["album"] == "Test Album"
        assert metadata["date"] == "2024"
        assert metadata["genre"] == "Electronic"
        assert metadata["track"] == "5"
        assert metadata["albumartist"] == "Album Artist"
        assert metadata["comment"] == "Test comment"
        assert metadata["encoder"] == "Xiph.Org libVorbis"
        assert metadata["organization"] == "Test Org"
        assert metadata["duration"] == "180.5"
        assert metadata["bitrate"] == 192000
        assert metadata["sample_rate"] == 44100
        assert metadata["channels"] == 2
        assert metadata["bitrate_nominal"] == 192000
        # encoder_version is an attribute of info, not stored directly
        # It's only stored if getattr finds it

    @patch("services.analysis_service.src.metadata_extractor.OggVorbis")
    def test_extract_ogg_minimal_metadata(self, mock_oggvorbis):
        """Test extracting metadata from OGG file with minimal tags."""
        # Setup mock with minimal data
        mock_audio = MagicMock()
        mock_audio.tags = {
            "title": ["Minimal Song"],
            "artist": ["Minimal Artist"],
        }
        mock_audio.info.length = 60.0
        mock_audio.info.bitrate = 128000
        mock_audio.info.sample_rate = 48000
        mock_audio.info.channels = 1
        mock_audio.info.bitrate_nominal = None
        mock_audio.info.encoder_version = None

        mock_audio.get = lambda key, default: mock_audio.tags.get(key, default)
        mock_oggvorbis.return_value = mock_audio

        # Test extraction
        extractor = MetadataExtractor()
        metadata = extractor._extract_ogg("/test/minimal.ogg")

        # Verify metadata - fields without values aren't included in output
        assert metadata["title"] == "Minimal Song"
        assert metadata["artist"] == "Minimal Artist"
        # Fields not present in tags won't be in the output
        assert "album" not in metadata
        assert "date" not in metadata
        assert "genre" not in metadata
        assert metadata["duration"] == "60.0"
        assert metadata["bitrate"] == 128000
        assert metadata["sample_rate"] == 48000
        assert metadata["channels"] == 1
        assert metadata["bitrate_nominal"] is None

    @patch("services.analysis_service.src.metadata_extractor.OggVorbis")
    def test_extract_ogg_no_tags(self, mock_oggvorbis):
        """Test extracting metadata from OGG file with no tags."""
        # Setup mock with no tags
        mock_audio = MagicMock()
        mock_audio.tags = None
        mock_audio.info.length = 120.0
        mock_audio.info.bitrate = 256000
        mock_audio.info.sample_rate = 44100
        mock_audio.info.channels = 2

        mock_oggvorbis.return_value = mock_audio

        # Test extraction
        extractor = MetadataExtractor()
        metadata = extractor._extract_ogg("/test/notags.ogg")

        # Verify only technical metadata is present
        assert metadata.get("title") is None
        assert metadata.get("artist") is None
        assert metadata["duration"] == "120.0"
        assert metadata["bitrate"] == 256000
        assert metadata["sample_rate"] == 44100
        assert metadata["channels"] == 2

    @patch("services.analysis_service.src.metadata_extractor.Path")
    @patch("services.analysis_service.src.metadata_extractor.OggVorbis")
    def test_extract_ogg_file_integration(self, mock_oggvorbis, mock_path):
        """Test full extraction flow for OGG file."""
        # Setup path mock
        mock_path_obj = MagicMock()
        mock_path_obj.exists.return_value = True
        mock_path_obj.is_file.return_value = True
        mock_path_obj.suffix = ".ogg"
        mock_path.return_value = mock_path_obj

        # Setup OggVorbis mock
        mock_audio = MagicMock()
        mock_audio.tags = {
            "title": ["Integration Test"],
            "artist": ["Test Artist"],
        }
        mock_audio.info.length = 90.0
        mock_audio.info.bitrate = 192000
        mock_audio.info.sample_rate = 44100
        mock_audio.info.channels = 2

        mock_audio.get = lambda key, default: mock_audio.tags.get(key, default)
        mock_oggvorbis.return_value = mock_audio

        # Test full extraction
        extractor = MetadataExtractor()
        metadata = extractor.extract("/test/integration.ogg")

        # Verify metadata includes format
        assert metadata["title"] == "Integration Test"
        assert metadata["artist"] == "Test Artist"
        assert metadata["format"] == "ogg"
        assert metadata["duration"] == "90.0"

    def test_extract_unsupported_rejects_non_ogg(self):
        """Test that unsupported formats are still rejected."""
        extractor = MetadataExtractor()

        with tempfile.NamedTemporaryFile(suffix=".xyz") as tmp:
            with pytest.raises(InvalidAudioFileError) as exc_info:
                extractor.extract(tmp.name)
            assert "Unsupported format: .xyz" in str(exc_info.value)

    def test_extract_ogg_file_not_found(self):
        """Test handling of non-existent OGG file."""
        extractor = MetadataExtractor()

        with pytest.raises(InvalidAudioFileError) as exc_info:
            extractor.extract("/nonexistent/file.ogg")
        assert "File not found" in str(exc_info.value)

    @patch("services.analysis_service.src.metadata_extractor.OggVorbis")
    def test_extract_ogg_corrupted_file(self, mock_oggvorbis):
        """Test handling of corrupted OGG file."""
        # Setup mock to raise exception
        mock_oggvorbis.side_effect = Exception("Corrupted file")

        extractor = MetadataExtractor()

        # Should handle the error gracefully
        with pytest.raises(Exception) as exc_info:
            extractor._extract_ogg("/test/corrupted.ogg")
        assert "Corrupted file" in str(exc_info.value)

    def test_get_supported_formats_includes_ogg(self):
        """Test that get_supported_formats includes OGG."""
        extractor = MetadataExtractor()
        formats = extractor.get_supported_formats()

        assert ".ogg" in formats
        assert ".oga" in formats
        # Ensure it's a copy, not the original
        formats.add(".test")
        assert ".test" not in extractor.SUPPORTED_FORMATS
