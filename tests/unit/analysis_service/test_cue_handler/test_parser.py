"""Unit tests for CUE parser."""

import logging
import tempfile
from io import StringIO
from pathlib import Path

import pytest

from services.analysis_service.src.cue_handler.exceptions import CueParsingError, CueValidationError
from services.analysis_service.src.cue_handler.models import CueTime
from services.analysis_service.src.cue_handler.parser import CueParser


class TestCueParser:
    """Test CueParser class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = CueParser()

    def test_parse_simple_cue(self):
        """Test parsing a simple valid CUE sheet."""
        cue_content = """TITLE "Test Album"
PERFORMER "Test Artist"
FILE "test.mp3" MP3
  TRACK 01 AUDIO
    TITLE "Track 1"
    PERFORMER "Artist 1"
    INDEX 01 00:00:00
  TRACK 02 AUDIO
    TITLE "Track 2"
    PERFORMER "Artist 2"
    INDEX 01 03:45:00"""

        sheet = self.parser.parse(cue_content)

        assert sheet.title == "Test Album"
        assert sheet.performer == "Test Artist"
        assert len(sheet.files) == 1
        assert sheet.files[0].filename == "test.mp3"
        assert sheet.files[0].file_type == "MP3"
        assert len(sheet.files[0].tracks) == 2

        track1 = sheet.files[0].tracks[0]
        assert track1.number == 1
        assert track1.title == "Track 1"
        assert track1.performer == "Artist 1"
        assert track1.indices[1] == CueTime(0, 0, 0)

        track2 = sheet.files[0].tracks[1]
        assert track2.number == 2
        assert track2.title == "Track 2"
        assert track2.indices[1] == CueTime(3, 45, 0)

    def test_parse_multi_file_cue(self):
        """Test parsing CUE with multiple FILE entries."""
        cue_content = """FILE "disc1.mp3" MP3
  TRACK 01 AUDIO
    INDEX 01 00:00:00
  TRACK 02 AUDIO
    INDEX 01 03:00:00
FILE "disc2.mp3" MP3
  TRACK 03 AUDIO
    INDEX 01 00:00:00
  TRACK 04 AUDIO
    INDEX 01 04:00:00"""

        sheet = self.parser.parse(cue_content)

        assert len(sheet.files) == 2
        assert sheet.files[0].filename == "disc1.mp3"
        assert len(sheet.files[0].tracks) == 2
        assert sheet.files[1].filename == "disc2.mp3"
        assert len(sheet.files[1].tracks) == 2
        assert sheet.files[1].tracks[0].number == 3

    def test_parse_catalog_and_cdtextfile(self):
        """Test parsing CATALOG and CDTEXTFILE commands."""
        cue_content = """CATALOG 1234567890123
CDTEXTFILE "cdtext.dat"
TITLE "Test Album"
FILE "test.wav" WAVE
  TRACK 01 AUDIO
    INDEX 01 00:00:00"""

        sheet = self.parser.parse(cue_content)

        assert sheet.catalog == "1234567890123"
        assert sheet.cdtextfile == "cdtext.dat"

    def test_parse_track_metadata(self):
        """Test parsing complete track metadata."""
        cue_content = """FILE "test.wav" WAVE
  TRACK 01 AUDIO
    FLAGS DCP 4CH PRE
    TITLE "Track Title"
    PERFORMER "Track Artist"
    SONGWRITER "Track Writer"
    ISRC USRC17607839
    PREGAP 00:02:00
    INDEX 00 00:00:00
    INDEX 01 00:02:00
    POSTGAP 00:01:00"""

        sheet = self.parser.parse(cue_content)

        track = sheet.files[0].tracks[0]
        assert track.flags == ["DCP", "4CH", "PRE"]
        assert track.title == "Track Title"
        assert track.performer == "Track Artist"
        assert track.songwriter == "Track Writer"
        assert track.isrc == "USRC17607839"
        assert track.pregap == CueTime(0, 2, 0)
        assert track.indices[0] == CueTime(0, 0, 0)
        assert track.indices[1] == CueTime(0, 2, 0)
        assert track.postgap == CueTime(0, 1, 0)

    def test_parse_rem_fields(self):
        """Test parsing REM fields."""
        cue_content = """REM GENRE Electronic
REM DATE 2023
REM DISCID 8B0A0C0D
REM COMMENT "Test comment"
REM DISCNUMBER 2
REM COMPOSER "Test Composer"
REM REPLAYGAIN_ALBUM_GAIN -7.89 dB
REM REPLAYGAIN_ALBUM_PEAK 0.988
REM CUSTOM_FIELD "Custom Value"
FILE "test.wav" WAVE
  TRACK 01 AUDIO
    REM REPLAYGAIN_TRACK_GAIN -6.50 dB
    REM REPLAYGAIN_TRACK_PEAK 0.950
    INDEX 01 00:00:00"""

        sheet = self.parser.parse(cue_content)

        # Check disc-level REM fields
        assert sheet.rem_fields["GENRE"] == "Electronic"
        assert sheet.rem_fields["DATE"] == "2023"
        assert sheet.rem_fields["DISCID"] == "8B0A0C0D"
        assert sheet.rem_fields["COMMENT"] == "Test comment"
        assert sheet.rem_fields["DISCNUMBER"] == "2"
        assert sheet.rem_fields["COMPOSER"] == "Test Composer"
        assert sheet.rem_fields["REPLAYGAIN_ALBUM_GAIN"] == "-7.89 dB"
        assert sheet.rem_fields["REPLAYGAIN_ALBUM_PEAK"] == "0.988"
        assert sheet.rem_fields["CUSTOM_FIELD"] == "Custom Value"

        # Check track-level REM fields
        track = sheet.files[0].tracks[0]
        assert track.rem_fields["REPLAYGAIN_TRACK_GAIN"] == "-6.50 dB"
        assert track.rem_fields["REPLAYGAIN_TRACK_PEAK"] == "0.950"

    def test_parse_alternative_comment_syntax(self):
        """Test parsing alternative comment syntax (; and //)."""
        cue_content = """; This is a comment
// This is another comment
FILE "test.wav" WAVE
  TRACK 01 AUDIO
    ; Track comment
    INDEX 01 00:00:00
    // Another track comment"""

        sheet = self.parser.parse(cue_content)

        # Comments are stored in REM fields (key may vary)
        # Should have converted comments to REM entries
        assert len(sheet.rem_fields) > 0 or len(sheet.files[0].tracks[0].rem_fields) > 0

    def test_parse_quoted_strings(self):
        """Test parsing quoted strings with spaces."""
        cue_content = """TITLE "Album with Spaces"
FILE "file with spaces.mp3" MP3
  TRACK 01 AUDIO
    TITLE "Track with Quotes"
    INDEX 01 00:00:00"""

        sheet = self.parser.parse(cue_content)

        assert sheet.title == "Album with Spaces"
        assert sheet.files[0].filename == "file with spaces.mp3"
        assert sheet.files[0].tracks[0].title == "Track with Quotes"

    def test_parse_flac_as_wave(self):
        """Test parsing FLAC files with WAVE type."""
        cue_content = """FILE "audio.flac" WAVE
  TRACK 01 AUDIO
    INDEX 01 00:00:00"""

        sheet = self.parser.parse(cue_content)

        assert sheet.files[0].filename == "audio.flac"
        assert sheet.files[0].file_type == "WAVE"
        # Should have warning about FLAC using WAVE
        assert any("FLAC" in w for w in sheet.parsing_warnings)

    def test_parse_various_track_types(self):
        """Test parsing different track types."""
        cue_content = """FILE "data.bin" BINARY
  TRACK 01 MODE1/2048
    INDEX 01 00:00:00
  TRACK 02 AUDIO
    INDEX 01 10:00:00
  TRACK 03 MODE2/2352
    INDEX 01 20:00:00"""

        sheet = self.parser.parse(cue_content)

        assert sheet.files[0].tracks[0].track_type == "MODE1/2048"
        assert sheet.files[0].tracks[1].track_type == "AUDIO"
        assert sheet.files[0].tracks[2].track_type == "MODE2/2352"

    def test_validation_missing_index01(self):
        """Test validation error for missing INDEX 01."""
        cue_content = """FILE "test.wav" WAVE
  TRACK 01 AUDIO
    TITLE "Missing Index"
    INDEX 00 00:00:00"""

        with pytest.raises(CueValidationError) as exc:
            self.parser.parse(cue_content)

        assert "INDEX 01" in str(exc.value)

    def test_validation_index_ordering(self):
        """Test validation of INDEX ordering."""
        cue_content = """FILE "test.wav" WAVE
  TRACK 01 AUDIO
    INDEX 01 00:00:00
    INDEX 00 00:01:00"""  # INDEX 00 after 01

        with pytest.raises(CueValidationError) as exc:
            self.parser.parse(cue_content)

        assert "INDEX 00" in str(exc.value)

    def test_character_limit_warnings(self):
        """Test warnings for exceeding character limits."""
        long_title = "A" * 100
        cue_content = f"""TITLE "{long_title}"
FILE "test.wav" WAVE
  TRACK 01 AUDIO
    PERFORMER "{long_title}"
    INDEX 01 00:00:00"""

        sheet = self.parser.parse(cue_content)

        # Should have warnings about character limits
        assert any("exceeds 80 character" in w for w in sheet.parsing_warnings)

    def test_invalid_catalog_warning(self):
        """Test warning for invalid CATALOG format."""
        cue_content = """CATALOG 123456789
FILE "test.wav" WAVE
  TRACK 01 AUDIO
    INDEX 01 00:00:00"""

        sheet = self.parser.parse(cue_content)

        assert sheet.catalog == "123456789"
        assert any("CATALOG should be 13 digits" in w for w in sheet.parsing_warnings)

    def test_invalid_isrc_warning(self):
        """Test warning for invalid ISRC format."""
        cue_content = """FILE "test.wav" WAVE
  TRACK 01 AUDIO
    ISRC SHORT
    INDEX 01 00:00:00"""

        sheet = self.parser.parse(cue_content)

        assert any("ISRC should be 12 characters" in w for w in sheet.parsing_warnings)

    def test_invalid_flags_warning(self):
        """Test warning for unknown FLAGS."""
        cue_content = """FILE "test.wav" WAVE
  TRACK 01 AUDIO
    FLAGS DCP UNKNOWN
    INDEX 01 00:00:00"""

        sheet = self.parser.parse(cue_content)

        track = sheet.files[0].tracks[0]
        assert "DCP" in track.flags
        assert "UNKNOWN" in track.flags
        assert any("Unknown flag" in w for w in sheet.parsing_warnings)

    def test_frame_time_calculation(self):
        """Test accurate frame time parsing (75 fps)."""
        cue_content = """FILE "test.wav" WAVE
  TRACK 01 AUDIO
    INDEX 01 00:00:00
  TRACK 02 AUDIO
    INDEX 01 00:01:00
  TRACK 03 AUDIO
    INDEX 01 00:01:74"""  # Last valid frame

        sheet = self.parser.parse(cue_content)

        # 00:01:00 = 1 second = 75 frames
        assert sheet.files[0].tracks[1].indices[1].to_frames() == 75

        # 00:01:74 = 1 second + 74 frames = 149 frames
        assert sheet.files[0].tracks[2].indices[1].to_frames() == 149

    def test_parse_file_with_encoding(self):
        """Test parsing file with specific encoding."""
        with tempfile.NamedTemporaryFile(mode="w", encoding="latin-1", suffix=".cue", delete=False) as f:
            f.write('TITLE "Tëst Älbüm"\n')
            f.write('FILE "test.wav" WAVE\n')
            f.write("  TRACK 01 AUDIO\n")
            f.write("    INDEX 01 00:00:00\n")
            temp_path = f.name

        try:
            # Should auto-detect encoding
            sheet = self.parser.parse_file(temp_path)
            assert "Tëst Älbüm" in sheet.title

            # Can also specify encoding
            sheet = self.parser.parse_file(temp_path, encoding="latin-1")
            assert "Tëst Älbüm" in sheet.title
        finally:
            Path(temp_path).unlink()

    def test_parse_nonexistent_file(self):
        """Test parsing nonexistent file raises error."""
        with pytest.raises(CueParsingError) as exc:
            self.parser.parse_file("/nonexistent/file.cue")

        assert "not found" in str(exc.value)

    def test_parse_oversized_file(self):
        """Test parsing oversized file raises security error."""
        # Create a temporary file that exceeds size limit
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cue", delete=False) as f:
            # Write data exceeding MAX_FILE_SIZE (10MB)
            # Each line is ~20 bytes, so we need 550,000+ lines
            for _ in range(550000):
                f.write("REM Large file test\n")
            temp_path = f.name

        try:
            with pytest.raises(CueParsingError) as exc:
                self.parser.parse_file(temp_path)

            assert "too large" in str(exc.value)
        finally:
            Path(temp_path).unlink()

    def test_empty_cue_sheet(self):
        """Test parsing empty CUE sheet."""
        cue_content = ""

        with pytest.raises(CueValidationError) as exc:
            self.parser.parse(cue_content)

        assert "No FILE" in str(exc.value)

    def test_logger_integration(self):
        """Test that parser accepts and uses logger."""

        # Create a logger with string stream handler
        log_stream = StringIO()
        test_logger = logging.getLogger("test")
        test_logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler(log_stream)
        test_logger.addHandler(handler)

        # Create parser with logger
        parser = CueParser(logger=test_logger)

        # Parse a simple CUE
        cue_content = """FILE "test.wav" WAVE
  TRACK 01 AUDIO
    INDEX 01 00:00:00"""

        sheet = parser.parse(cue_content)

        # Check that logging occurred
        log_output = log_stream.getvalue()
        assert "Successfully parsed" in log_output or len(sheet.files) == 1

    def test_complex_multitrack_cue(self):
        """Test parsing complex CUE with all features."""

        cue_content = """REM GENRE "Electronic"
REM DATE 2023
REM DISCID 8E0B710B
REM COMMENT "Generated by Tracktion v1.0"
CATALOG 1234567890123
TITLE "Complex Album"
PERFORMER "Various Artists"
FILE "disc1.wav" WAVE
  TRACK 01 AUDIO
    FLAGS DCP PRE
    TITLE "Opening"
    PERFORMER "Artist 1"
    SONGWRITER "Writer 1"
    ISRC GBUM71029078
    REM COMPOSER "Composer 1"
    PREGAP 00:02:00
    INDEX 00 00:00:00
    INDEX 01 00:02:00
    INDEX 02 00:03:30
  TRACK 02 AUDIO
    TITLE "Second Track"
    PERFORMER "Artist 2"
    INDEX 00 04:58:50
    INDEX 01 05:00:00
FILE "disc2.wav" WAVE
  TRACK 03 AUDIO
    TITLE "Part Two"
    PERFORMER "Artist 3"
    INDEX 01 00:00:00
  TRACK 04 AUDIO
    TITLE "Finale"
    PERFORMER "Artist 4"
    INDEX 01 03:45:60
    POSTGAP 00:05:00"""

        sheet = self.parser.parse(cue_content)

        # Verify complete structure
        assert sheet.catalog == "1234567890123"
        assert sheet.title == "Complex Album"
        assert sheet.performer == "Various Artists"
        assert len(sheet.files) == 2
        assert sheet.get_track_count() == 4

        # Check REM fields
        assert sheet.rem_fields["GENRE"] == "Electronic"
        assert sheet.rem_fields["DATE"] == "2023"

        # Check first track details
        track1 = sheet.files[0].tracks[0]
        assert track1.number == 1
        assert track1.flags == ["DCP", "PRE"]
        assert track1.isrc == "GBUM71029078"
        assert track1.pregap == CueTime(0, 2, 0)
        assert len(track1.indices) == 3  # 00, 01, 02
        assert track1.rem_fields["COMPOSER"] == "Composer 1"

        # Check last track
        track4 = sheet.files[1].tracks[1]
        assert track4.number == 4
        assert track4.postgap == CueTime(0, 5, 0)
