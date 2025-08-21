"""Integration tests for OGG Vorbis metadata extraction and storage."""

import json
import uuid
from unittest.mock import Mock, patch

import pytest

from services.analysis_service.src.metadata_extractor import MetadataExtractor
from services.analysis_service.src.storage_handler import StorageHandler


class TestOggMetadataIntegration:
    """Test OGG metadata extraction and storage integration."""

    @pytest.fixture
    def storage_handler(self):
        """Create a mock storage handler."""
        handler = Mock(spec=StorageHandler)
        handler.store_metadata = Mock(return_value=True)
        return handler

    @pytest.fixture
    def extractor(self):
        """Create a metadata extractor instance."""
        return MetadataExtractor()

    def test_ogg_metadata_storage(self, extractor, storage_handler):
        """Test that OGG metadata is properly stored."""
        recording_id = str(uuid.uuid4())

        # Create mock OGG file data
        with patch("services.analysis_service.src.metadata_extractor.OggVorbis") as mock_ogg:
            mock_audio = Mock()
            mock_audio.tags = {
                "TITLE": ["Test Song"],
                "ARTIST": ["Test Artist"],
                "ALBUM": ["Test Album"],
                "DATE": ["2024"],
                "GENRE": ["Electronic"],
                "CUSTOM_TAG": ["Custom Value"],
                "BPM": ["128"],
                "KEY": ["Am"],
            }
            mock_audio.info = Mock(
                length=180.5,
                bitrate=192000,
                sample_rate=44100,
                channels=2,
                bitrate_nominal=192000,
            )
            mock_ogg.return_value = mock_audio

            # Extract metadata
            with patch("os.path.getsize", return_value=5242880):
                metadata = extractor._extract_ogg("/tmp/test.ogg")

            # Store metadata
            storage_handler.store_metadata(recording_id, metadata)

            # Verify storage was called with correct data
            storage_handler.store_metadata.assert_called_once()
            call_args = storage_handler.store_metadata.call_args[0]

            assert call_args[0] == recording_id
            stored_metadata = call_args[1]

            # Verify standard fields
            assert stored_metadata["title"] == "Test Song"
            assert stored_metadata["artist"] == "Test Artist"
            assert stored_metadata["album"] == "Test Album"
            assert stored_metadata["date"] == "2024"
            assert stored_metadata["genre"] == "Electronic"

            # Verify extended fields
            assert stored_metadata["bpm"] == "128"
            assert stored_metadata["key"] == "Am"

            # Verify custom tags are stored as JSON
            assert "custom_tags" in stored_metadata
            custom_tags = json.loads(stored_metadata["custom_tags"])
            assert custom_tags["CUSTOM_TAG"] == "Custom Value"

            # Verify technical metadata
            assert float(stored_metadata["duration"]) == 180.5
            assert stored_metadata["bitrate"] == 192000
            assert stored_metadata["sample_rate"] == 44100
            assert stored_metadata["channels"] == 2

    def test_ogg_multiple_artists_storage(self, extractor, storage_handler):
        """Test that multiple artists are properly stored."""
        recording_id = str(uuid.uuid4())

        with patch("services.analysis_service.src.metadata_extractor.OggVorbis") as mock_ogg:
            mock_audio = Mock()
            mock_audio.tags = {
                "TITLE": ["Collaboration Song"],
                "ARTIST": ["Artist One", "Artist Two", "Artist Three"],
                "ALBUM": ["Various Artists"],
            }
            mock_audio.info = Mock(
                length=240,
                bitrate=256000,
                sample_rate=48000,
                channels=2,
            )
            mock_ogg.return_value = mock_audio

            # Extract and store
            metadata = extractor._extract_ogg("/tmp/test_collab.ogg")
            storage_handler.store_metadata(recording_id, metadata)

            # Verify multiple artists are joined
            call_args = storage_handler.store_metadata.call_args[0]
            stored_metadata = call_args[1]
            assert stored_metadata["artist"] == "Artist One; Artist Two; Artist Three"

    def test_ogg_replaygain_storage(self, extractor, storage_handler):
        """Test that ReplayGain tags are properly stored."""
        recording_id = str(uuid.uuid4())

        with patch("services.analysis_service.src.metadata_extractor.OggVorbis") as mock_ogg:
            mock_audio = Mock()
            mock_audio.tags = {
                "TITLE": ["Test Song"],
                "REPLAYGAIN_TRACK_GAIN": ["-3.21 dB"],
                "REPLAYGAIN_TRACK_PEAK": ["0.988525"],
                "REPLAYGAIN_ALBUM_GAIN": ["-4.56 dB"],
                "REPLAYGAIN_ALBUM_PEAK": ["0.999969"],
            }
            mock_audio.info = Mock(
                length=180,
                bitrate=192000,
                sample_rate=44100,
                channels=2,
            )
            mock_ogg.return_value = mock_audio

            # Extract and store
            metadata = extractor._extract_ogg("/tmp/test_replaygain.ogg")
            storage_handler.store_metadata(recording_id, metadata)

            # Verify ReplayGain tags are stored
            call_args = storage_handler.store_metadata.call_args[0]
            stored_metadata = call_args[1]
            assert stored_metadata["replaygain_track_gain"] == "-3.21 dB"
            assert stored_metadata["replaygain_track_peak"] == "0.988525"
            assert stored_metadata["replaygain_album_gain"] == "-4.56 dB"
            assert stored_metadata["replaygain_album_peak"] == "0.999969"

    def test_ogg_custom_tags_limit(self, extractor, storage_handler):
        """Test that custom tags are limited to prevent DoS."""
        recording_id = str(uuid.uuid4())

        with patch("services.analysis_service.src.metadata_extractor.OggVorbis") as mock_ogg:
            mock_audio = Mock()
            # Create 150 custom tags
            tags = {"TITLE": ["Test"]}
            for i in range(150):
                tags[f"CUSTOM_{i}"] = [f"Value {i}"]
            mock_audio.tags = tags
            mock_audio.info = Mock(
                length=180,
                bitrate=192000,
                sample_rate=44100,
                channels=2,
            )
            mock_ogg.return_value = mock_audio

            # Extract and store
            metadata = extractor._extract_ogg("/tmp/test_many_tags.ogg")
            storage_handler.store_metadata(recording_id, metadata)

            # Verify custom tags are limited to 100
            call_args = storage_handler.store_metadata.call_args[0]
            stored_metadata = call_args[1]
            custom_tags = json.loads(stored_metadata["custom_tags"])
            assert len(custom_tags) == 100

    def test_full_extraction_pipeline(self, extractor):
        """Test the complete extraction pipeline through the public interface."""
        with patch("services.analysis_service.src.metadata_extractor.OggVorbis") as mock_ogg:
            mock_audio = Mock()
            mock_audio.tags = {
                "TITLE": ["Complete Test"],
                "ARTIST": ["Full Artist"],
                "ALBUM": ["Full Album"],
                "GENRE": ["Rock"],
                "COMPOSER": ["Bach"],
                "CUSTOM_FIELD": ["Custom"],
            }
            mock_audio.info = Mock(
                length=300,
                bitrate=320000,
                sample_rate=48000,
                channels=2,
                bitrate_nominal=320000,
                bitrate_lower=256000,
                bitrate_upper=384000,
            )
            mock_ogg.return_value = mock_audio

            # Mock file existence and use the public extract method
            with (
                patch("pathlib.Path.exists", return_value=True),
                patch("pathlib.Path.is_file", return_value=True),
                patch("os.path.getsize", return_value=12000000),
            ):
                result = extractor.extract("/tmp/test.ogg")

            # Verify all metadata is extracted
            assert result["title"] == "Complete Test"
            assert result["artist"] == "Full Artist"
            assert result["album"] == "Full Album"
            assert result["genre"] == "Rock"
            assert result["composer"] == "Bach"
            assert result["format"] == "ogg"

            # Verify custom tags
            assert "custom_tags" in result
            custom_tags = json.loads(result["custom_tags"])
            assert custom_tags["CUSTOM_FIELD"] == "Custom"

            # Verify technical metadata
            assert result["bitrate"] == "320000"  # Converted to string
            assert result["bitrate_mode"] == "VBR"  # Different lower/upper bounds
            assert result["file_size"] == "12000000"
