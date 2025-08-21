"""Enhanced tests for OGG Vorbis metadata extraction."""

import json
from unittest.mock import Mock, patch

import pytest

from services.analysis_service.src.metadata_extractor import (
    MetadataExtractionError,
    MetadataExtractor,
)


class TestOggMetadataEnhanced:
    """Test enhanced OGG Vorbis metadata extraction."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = MetadataExtractor()
        self.test_file = "/tmp/test.ogg"

    def test_extract_all_standard_vorbis_comments(self):
        """Test extraction of all standard Vorbis comment fields."""
        # Mock OggVorbis with comprehensive tags
        with patch("services.analysis_service.src.metadata_extractor.OggVorbis") as mock_ogg:
            mock_audio = Mock()
            mock_audio.tags = {
                "TITLE": ["Test Title"],
                "VERSION": ["1.0"],
                "ALBUM": ["Test Album"],
                "TRACKNUMBER": ["5"],
                "ARTIST": ["Test Artist"],
                "PERFORMER": ["Test Performer"],
                "COPYRIGHT": ["© 2024"],
                "LICENSE": ["CC BY 4.0"],
                "ORGANIZATION": ["Test Org"],
                "DESCRIPTION": ["Test Description"],
                "GENRE": ["Electronic"],
                "DATE": ["2024-01-15"],
                "LOCATION": ["Studio A"],
                "CONTACT": ["test@example.com"],
                "ISRC": ["USRC17607839"],
            }
            mock_audio.info = Mock(
                length=180.5,
                bitrate=192000,
                sample_rate=44100,
                channels=2,
                bitrate_nominal=192000,
            )
            mock_ogg.return_value = mock_audio

            with patch("os.path.getsize", return_value=5242880):
                result = self.extractor._extract_ogg(self.test_file)

            # Verify all standard fields are extracted
            assert result["title"] == "Test Title"
            assert result["version"] == "1.0"
            assert result["album"] == "Test Album"
            assert result["track"] == "5"
            assert result["artist"] == "Test Artist"
            assert result["performer"] == "Test Performer"
            assert result["copyright"] == "© 2024"
            assert result["license"] == "CC BY 4.0"
            assert result["organization"] == "Test Org"
            assert result["description"] == "Test Description"
            assert result["genre"] == "Electronic"
            assert result["date"] == "2024-01-15"
            assert result["location"] == "Studio A"
            assert result["contact"] == "test@example.com"
            assert result["isrc"] == "USRC17607839"

    def test_extract_extended_vorbis_comments(self):
        """Test extraction of extended Vorbis comment fields."""
        with patch("services.analysis_service.src.metadata_extractor.OggVorbis") as mock_ogg:
            mock_audio = Mock()
            mock_audio.tags = {
                "ALBUMARTIST": ["Various Artists"],
                "COMPOSER": ["J.S. Bach"],
                "CONDUCTOR": ["Herbert von Karajan"],
                "DISCNUMBER": ["2"],
                "DISCTOTAL": ["3"],
                "TOTALTRACKS": ["12"],
                "PUBLISHER": ["Test Records"],
                "LABEL": ["Test Label"],
                "COMPILATION": ["1"],
                "LYRICS": ["La la la"],
                "LANGUAGE": ["en"],
                "MOOD": ["Happy"],
                "BPM": ["128"],
                "KEY": ["Am"],
                "ENCODER": ["Lavf58.29.100"],
            }
            mock_audio.info = Mock(length=200, bitrate=256000, sample_rate=48000, channels=2)
            mock_ogg.return_value = mock_audio

            with patch("os.path.getsize", return_value=6400000):
                result = self.extractor._extract_ogg(self.test_file)

            # Verify extended fields
            assert result["albumartist"] == "Various Artists"
            assert result["composer"] == "J.S. Bach"
            assert result["conductor"] == "Herbert von Karajan"
            assert result["discnumber"] == "2"
            assert result["disctotal"] == "3"
            assert result["totaltracks"] == "12"
            assert result["publisher"] == "Test Records"
            assert result["label"] == "Test Label"
            assert result["compilation"] == "1"
            assert result["lyrics"] == "La la la"
            assert result["language"] == "en"
            assert result["mood"] == "Happy"
            assert result["bpm"] == "128"
            assert result["key"] == "Am"
            assert result["encoder"] == "Lavf58.29.100"

    def test_extract_replaygain_tags(self):
        """Test extraction of ReplayGain tags."""
        with patch("services.analysis_service.src.metadata_extractor.OggVorbis") as mock_ogg:
            mock_audio = Mock()
            mock_audio.tags = {
                "REPLAYGAIN_TRACK_GAIN": ["-3.21 dB"],
                "REPLAYGAIN_TRACK_PEAK": ["0.988525"],
                "REPLAYGAIN_ALBUM_GAIN": ["-4.56 dB"],
                "REPLAYGAIN_ALBUM_PEAK": ["0.999969"],
            }
            mock_audio.info = Mock(length=240, bitrate=320000, sample_rate=44100, channels=2)
            mock_ogg.return_value = mock_audio

            result = self.extractor._extract_ogg(self.test_file)

            assert result["replaygain_track_gain"] == "-3.21 dB"
            assert result["replaygain_track_peak"] == "0.988525"
            assert result["replaygain_album_gain"] == "-4.56 dB"
            assert result["replaygain_album_peak"] == "0.999969"

    def test_custom_tag_preservation(self):
        """Test that custom non-standard tags are preserved."""
        with patch("services.analysis_service.src.metadata_extractor.OggVorbis") as mock_ogg:
            mock_audio = Mock()
            mock_audio.tags = {
                "TITLE": ["Standard Title"],
                "ARTIST": ["Standard Artist"],
                "CUSTOM_FIELD": ["Custom Value"],
                "MySpecialTag": ["Special Value"],
                "rating": ["5 stars"],
                "purchase_date": ["2024-01-01"],
            }
            mock_audio.info = Mock(length=180, bitrate=192000, sample_rate=44100, channels=2)
            mock_ogg.return_value = mock_audio

            result = self.extractor._extract_ogg(self.test_file)

            # Standard fields should be in main metadata
            assert result["title"] == "Standard Title"
            assert result["artist"] == "Standard Artist"

            # Custom fields should be in custom_tags with original case (as JSON string)
            assert "custom_tags" in result
            custom_tags = json.loads(result["custom_tags"])
            assert custom_tags["CUSTOM_FIELD"] == "Custom Value"
            assert custom_tags["MySpecialTag"] == "Special Value"
            assert custom_tags["rating"] == "5 stars"
            assert custom_tags["purchase_date"] == "2024-01-01"

    def test_multiple_values_per_key(self):
        """Test handling of multiple values for the same key."""
        with patch("services.analysis_service.src.metadata_extractor.OggVorbis") as mock_ogg:
            mock_audio = Mock()
            mock_audio.tags = {
                "ARTIST": ["Artist 1", "Artist 2", "Artist 3"],
                "GENRE": ["Rock", "Alternative"],
                "PERFORMER": ["Singer", "Guitarist", "Drummer"],
            }
            mock_audio.info = Mock(length=200, bitrate=256000, sample_rate=44100, channels=2)
            mock_ogg.return_value = mock_audio

            result = self.extractor._extract_ogg(self.test_file)

            # Multiple values should be joined with semicolon
            assert result["artist"] == "Artist 1; Artist 2; Artist 3"
            assert result["genre"] == "Rock; Alternative"
            assert result["performer"] == "Singer; Guitarist; Drummer"

    def test_missing_metadata_graceful_handling(self):
        """Test graceful handling of missing metadata."""
        with patch("services.analysis_service.src.metadata_extractor.OggVorbis") as mock_ogg:
            mock_audio = Mock()
            # File with minimal tags
            mock_audio.tags = {
                "TITLE": ["Only Title"],
            }
            mock_audio.info = Mock(length=120, bitrate=128000, sample_rate=44100, channels=2)
            mock_ogg.return_value = mock_audio

            result = self.extractor._extract_ogg(self.test_file)

            # Should have title
            assert result["title"] == "Only Title"

            # Other standard fields should not be present if not in file
            assert "artist" not in result or result["artist"] is None
            assert "album" not in result or result["album"] is None

            # Technical metadata should still be present
            assert float(result["duration"]) == 120.0
            assert result["bitrate"] == 128000

    def test_empty_tags_handling(self):
        """Test handling of empty tag values."""
        with patch("services.analysis_service.src.metadata_extractor.OggVorbis") as mock_ogg:
            mock_audio = Mock()
            mock_audio.tags = {
                "TITLE": [""],  # Empty string
                "ARTIST": [None],  # None value
                "ALBUM": [],  # Empty list
                "GENRE": ["Rock"],  # Valid value
            }
            mock_audio.info = Mock(length=180, bitrate=192000, sample_rate=44100, channels=2)
            mock_ogg.return_value = mock_audio

            result = self.extractor._extract_ogg(self.test_file)

            # Empty values should be handled gracefully
            assert "title" not in result or result["title"] is None
            assert "artist" not in result or result["artist"] is None
            assert "album" not in result or result["album"] is None
            assert result["genre"] == "Rock"

    def test_malformed_ogg_file_handling(self):
        """Test handling of malformed OGG files."""
        with patch("services.analysis_service.src.metadata_extractor.OggVorbis") as mock_ogg:
            # Simulate mutagen raising an exception for corrupt file
            mock_ogg.side_effect = Exception("Invalid OGG file structure")

            with pytest.raises(MetadataExtractionError) as exc_info:
                self.extractor._extract_ogg(self.test_file)

            assert "Invalid OGG file" in str(exc_info.value)

    def test_technical_metadata_extraction(self):
        """Test extraction of technical metadata."""
        with patch("services.analysis_service.src.metadata_extractor.OggVorbis") as mock_ogg:
            mock_audio = Mock()
            mock_audio.tags = {"TITLE": ["Test"]}
            mock_audio.info = Mock(
                length=305.123,
                bitrate=256000,
                sample_rate=48000,
                channels=2,
                bitrate_nominal=256000,
                bitrate_lower=128000,
                bitrate_upper=320000,
            )
            mock_ogg.return_value = mock_audio

            with patch("os.path.getsize", return_value=9785344):
                result = self.extractor._extract_ogg(self.test_file)

            # Check technical metadata
            assert float(result["duration"]) == 305.123
            assert result["bitrate"] == 256000
            assert result["sample_rate"] == 48000
            assert result["channels"] == 2
            assert result["bitrate_nominal"] == 256000
            assert result["bitrate_lower"] == 128000
            assert result["bitrate_upper"] == 320000
            assert result["bitrate_mode"] == "VBR"  # Lower != Upper
            assert result["file_size"] == "9785344"

    def test_cbr_detection(self):
        """Test CBR (Constant Bitrate) detection."""
        with patch("services.analysis_service.src.metadata_extractor.OggVorbis") as mock_ogg:
            mock_audio = Mock()
            mock_audio.tags = {}
            mock_audio.info = Mock(
                length=200,
                bitrate=192000,
                sample_rate=44100,
                channels=2,
                bitrate_nominal=192000,
                bitrate_lower=192000,
                bitrate_upper=192000,
            )
            mock_ogg.return_value = mock_audio

            result = self.extractor._extract_ogg(self.test_file)

            assert result["bitrate_mode"] == "CBR"  # Lower == Upper

    def test_case_insensitive_tag_matching(self):
        """Test that tag matching is case-insensitive."""
        with patch("services.analysis_service.src.metadata_extractor.OggVorbis") as mock_ogg:
            mock_audio = Mock()
            mock_audio.tags = {
                "title": ["Lowercase Title"],
                "ARTIST": ["Uppercase Artist"],
                "AlBuM": ["Mixed Case Album"],
                "TrAcKnUmBeR": ["42"],
            }
            mock_audio.info = Mock(length=180, bitrate=192000, sample_rate=44100, channels=2)
            mock_ogg.return_value = mock_audio

            result = self.extractor._extract_ogg(self.test_file)

            # All variations should be recognized
            assert result["title"] == "Lowercase Title"
            assert result["artist"] == "Uppercase Artist"
            assert result["album"] == "Mixed Case Album"
            assert result["track"] == "42"

    def test_maximum_custom_tags_limit(self):
        """Test that custom tags are limited to prevent DoS."""
        with patch("services.analysis_service.src.metadata_extractor.OggVorbis") as mock_ogg:
            mock_audio = Mock()
            # Create more than 100 custom tags
            tags = {"TITLE": ["Test"]}
            for i in range(150):
                tags[f"CUSTOM_{i}"] = [f"Value {i}"]
            mock_audio.tags = tags
            mock_audio.info = Mock(length=180, bitrate=192000, sample_rate=44100, channels=2)
            mock_ogg.return_value = mock_audio

            with patch("services.analysis_service.src.metadata_extractor.logger") as mock_logger:
                result = self.extractor._extract_ogg(self.test_file)

                # Should have custom_tags but limited to 100
                assert "custom_tags" in result
                custom_tags = json.loads(result["custom_tags"])
                assert len(custom_tags) == 100

                # Should have logged warning about limit
                mock_logger.warning.assert_called()

    def test_extremely_long_tag_value_truncation(self):
        """Test that extremely long tag values are truncated."""
        with patch("services.analysis_service.src.metadata_extractor.OggVorbis") as mock_ogg:
            mock_audio = Mock()
            # Create a very long lyrics field
            long_lyrics = "A" * 10000
            mock_audio.tags = {
                "TITLE": ["Normal Title"],
                "LYRICS": [long_lyrics],
            }
            mock_audio.info = Mock(length=180, bitrate=192000, sample_rate=44100, channels=2)
            mock_ogg.return_value = mock_audio

            with patch("services.analysis_service.src.metadata_extractor.logger") as mock_logger:
                result = self.extractor._extract_ogg(self.test_file)

                assert result["title"] == "Normal Title"
                assert len(result["lyrics"]) == 5000  # Max length enforced
                # Should log truncation warning
                mock_logger.warning.assert_called()

    def test_null_bytes_and_control_characters_sanitization(self):
        """Test that null bytes and control characters are sanitized."""
        with patch("services.analysis_service.src.metadata_extractor.OggVorbis") as mock_ogg:
            mock_audio = Mock()
            mock_audio.tags = {
                "TITLE": ["Test\x00Title"],  # Null byte
                "ARTIST": ["Test\x01\x02\x03Artist"],  # Control characters
                "ALBUM": ["Normal\nNewline\tTab"],  # Allowed whitespace
            }
            mock_audio.info = Mock(length=180, bitrate=192000, sample_rate=44100, channels=2)
            mock_ogg.return_value = mock_audio

            result = self.extractor._extract_ogg(self.test_file)

            # Null bytes and control chars should be removed
            assert result["title"] == "TestTitle"
            assert result["artist"] == "TestArtist"
            # Newlines and tabs should be preserved
            assert result["album"] == "Normal\nNewline\tTab"

    def test_file_size_error_handling(self):
        """Test graceful handling when file size cannot be determined."""
        with patch("services.analysis_service.src.metadata_extractor.OggVorbis") as mock_ogg:
            mock_audio = Mock()
            mock_audio.tags = {"TITLE": ["Test"]}
            mock_audio.info = Mock(length=180, bitrate=192000, sample_rate=44100, channels=2)
            mock_ogg.return_value = mock_audio

            with patch("os.path.getsize", side_effect=OSError("Permission denied")):
                result = self.extractor._extract_ogg(self.test_file)

                # Should still extract other metadata
                assert result["title"] == "Test"
                assert float(result["duration"]) == 180.0
                # file_size should not be present or be None
                assert "file_size" not in result or result["file_size"] is None
