"""Unit tests for CUE file validator."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from services.analysis_service.src.cue_handler.exceptions import CueParsingError
from services.analysis_service.src.cue_handler.models import (
    CueSheet,
    CueTime,
    FileReference,
    Track,
)
from services.analysis_service.src.cue_handler.validator import (
    CueValidator,
    Severity,
    ValidationIssue,
    ValidationResult,
)


class TestCueValidator:
    """Test CUE file validation functionality."""

    @pytest.fixture
    def validator(self):
        """Create a CueValidator instance."""
        return CueValidator()

    @pytest.fixture
    def valid_cue_content(self):
        """Valid CUE file content."""
        return """REM GENRE Electronic
REM DATE 2024
PERFORMER "DJ Test"
TITLE "Test Mix"
FILE "audio.wav" WAVE
  TRACK 01 AUDIO
    TITLE "Track One"
    PERFORMER "Artist One"
    INDEX 01 00:00:00
  TRACK 02 AUDIO
    TITLE "Track Two"
    PERFORMER "Artist Two"
    INDEX 01 03:45:00"""

    @pytest.fixture
    def valid_cue_sheet(self):
        """Create a valid CueSheet object."""
        sheet = CueSheet()
        sheet.performer = "DJ Test"
        sheet.title = "Test Mix"
        sheet.rem_fields = {"GENRE": "Electronic", "DATE": "2024"}

        file_ref = FileReference(filename="audio.wav", file_type="WAVE")

        track1 = Track(number=1, track_type="AUDIO")
        track1.title = "Track One"
        track1.performer = "Artist One"
        track1.indices[1] = CueTime(0, 0, 0)

        track2 = Track(number=2, track_type="AUDIO")
        track2.title = "Track Two"
        track2.performer = "Artist Two"
        track2.indices[1] = CueTime(3, 45, 0)

        file_ref.tracks = [track1, track2]
        sheet.files = [file_ref]

        return sheet

    def test_validate_file_not_found(self, validator):
        """Test validation when CUE file doesn't exist."""
        result = validator.validate("/nonexistent/file.cue")

        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].category == "file"
        assert "not found" in result.errors[0].message

    def test_validate_valid_cue(self, validator, valid_cue_content, valid_cue_sheet):
        """Test validation of a valid CUE file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create CUE file
            cue_path = Path(tmpdir) / "test.cue"
            cue_path.write_text(valid_cue_content)

            # Create audio file
            audio_path = Path(tmpdir) / "audio.wav"
            audio_path.write_bytes(b"fake audio content")

            # Mock parser to return valid sheet
            with patch.object(validator.parser, "parse", return_value=valid_cue_sheet):
                result = validator.validate(str(cue_path))

            assert result.is_valid
            assert len(result.errors) == 0

    def test_validate_missing_audio_file(self, validator, valid_cue_content, valid_cue_sheet):
        """Test validation when referenced audio file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create CUE file without audio file
            cue_path = Path(tmpdir) / "test.cue"
            cue_path.write_text(valid_cue_content)

            # Mock parser
            with patch.object(validator.parser, "parse", return_value=valid_cue_sheet):
                result = validator.validate(str(cue_path))

            assert not result.is_valid
            assert any(e.category == "file" and "not found" in e.message for e in result.errors)

    def test_validate_parse_error(self, validator):
        """Test validation when CUE parsing fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create invalid CUE file
            cue_path = Path(tmpdir) / "test.cue"
            cue_path.write_text("INVALID CUE CONTENT")

            # Mock parser to raise error

            with patch.object(validator.parser, "parse", side_effect=CueParsingError("Parse failed")):
                result = validator.validate(str(cue_path))

            assert not result.is_valid
            assert len(result.errors) == 1
            assert result.errors[0].category == "syntax"
            assert "Parse error" in result.errors[0].message

    def test_validate_batch(self, validator, valid_cue_content):
        """Test batch validation of multiple CUE files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple CUE files
            paths = []
            for i in range(3):
                cue_path = Path(tmpdir) / f"test{i}.cue"
                cue_path.write_text(valid_cue_content)
                paths.append(str(cue_path))

            results = validator.validate_batch(paths)

            assert len(results) == 3
            assert all(isinstance(r, ValidationResult) for r in results)


class TestValidationResult:
    """Test ValidationResult class."""

    def test_add_issue(self):
        """Test adding issues to result."""
        result = ValidationResult(file_path="test.cue", is_valid=True)

        # Add error
        error = ValidationIssue(
            severity=Severity.ERROR,
            line_number=10,
            category="syntax",
            message="Test error",
        )
        result.add_issue(error)
        assert not result.is_valid
        assert len(result.errors) == 1

        # Add warning
        warning = ValidationIssue(
            severity=Severity.WARNING,
            line_number=20,
            category="timing",
            message="Test warning",
        )
        result.add_issue(warning)
        assert len(result.warnings) == 1

        # Add info
        info = ValidationIssue(
            severity=Severity.INFO,
            line_number=30,
            category="compatibility",
            message="Test info",
        )
        result.add_issue(info)
        assert len(result.info) == 1

    def test_to_report(self):
        """Test generating human-readable report."""
        result = ValidationResult(file_path="test.cue", is_valid=False)

        # Add various issues
        result.add_issue(
            ValidationIssue(
                severity=Severity.ERROR,
                line_number=10,
                category="syntax",
                message="Missing INDEX 01",
                suggestion="Add INDEX 01",
            )
        )

        result.add_issue(
            ValidationIssue(
                severity=Severity.WARNING,
                line_number=20,
                category="timing",
                message="Time overlap",
                suggestion="Fix timing",
            )
        )

        result.audio_duration_ms = 180000  # 3 minutes
        result.cue_duration_ms = 190000  # 3:10
        result.format_compatibility = {"CDJ": True, "Traktor": False}

        report = result.to_report()

        assert "test.cue" in report
        assert "❌ INVALID" in report
        assert "Missing INDEX 01" in report
        assert "Add INDEX 01" in report
        assert "Time overlap" in report
        assert "Audio: 180.00s" in report
        assert "CUE: 190.00s" in report
        assert "CDJ: ✅" in report
        assert "Traktor: ❌" in report

    def test_to_json(self):
        """Test generating JSON report."""
        result = ValidationResult(file_path="test.cue", is_valid=False)

        result.add_issue(
            ValidationIssue(
                severity=Severity.ERROR,
                line_number=10,
                category="syntax",
                message="Test error",
                suggestion="Fix it",
            )
        )

        result.audio_duration_ms = 180000
        result.cue_duration_ms = 190000
        result.format_compatibility = {"CDJ": True}

        json_report = result.to_json()

        assert json_report["file_path"] == "test.cue"
        assert not json_report["is_valid"]
        assert json_report["summary"]["errors"] == 1
        assert json_report["summary"]["warnings"] == 0
        assert json_report["summary"]["info"] == 0
        assert json_report["duration"]["audio_ms"] == 180000
        assert json_report["duration"]["cue_ms"] == 190000
        assert json_report["format_compatibility"]["CDJ"] is True
        assert len(json_report["errors"]) == 1
        assert json_report["errors"][0]["message"] == "Test error"


class TestSyntaxValidation:
    """Test syntax validation rules."""

    @pytest.fixture
    def validator(self):
        """Create a CueValidator instance."""
        return CueValidator()

    def test_missing_file_command(self, validator):
        """Test detection of missing FILE command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cue_path = Path(tmpdir) / "test.cue"
            cue_path.write_text('TITLE "Test"\nTRACK 01 AUDIO\n  INDEX 01 00:00:00')

            # Create empty CueSheet without files
            sheet = CueSheet()
            sheet.title = "Test"

            with patch.object(validator.parser, "parse", return_value=sheet):
                result = validator.validate(str(cue_path))

            assert not result.is_valid
            assert any(e.message == "No FILE command found in CUE sheet" for e in result.errors)

    def test_missing_index01(self, validator):
        """Test detection of missing INDEX 01."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cue_path = Path(tmpdir) / "test.cue"
            cue_content = """FILE "audio.wav" WAVE
  TRACK 01 AUDIO
    INDEX 00 00:00:00"""
            cue_path.write_text(cue_content)

            # Create sheet with track missing INDEX 01
            sheet = CueSheet()
            file_ref = FileReference(filename="audio.wav", file_type="WAVE")
            track = Track(number=1, track_type="AUDIO")
            track.indices[0] = CueTime(0, 0, 0)
            file_ref.tracks = [track]
            sheet.files = [file_ref]

            # Create audio file
            audio_path = Path(tmpdir) / "audio.wav"
            audio_path.write_bytes(b"fake")

            with patch.object(validator.parser, "parse", return_value=sheet):
                result = validator.validate(str(cue_path))

            assert not result.is_valid
            assert any("missing required INDEX 01" in e.message for e in result.errors)

    def test_character_limit_warning(self, validator):
        """Test detection of character limit violations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cue_path = Path(tmpdir) / "test.cue"
            long_title = "A" * 100
            cue_content = f"""FILE "audio.wav" WAVE
  TRACK 01 AUDIO
    TITLE "{long_title}"
    INDEX 01 00:00:00"""
            cue_path.write_text(cue_content)

            # Create sheet with long title
            sheet = CueSheet()
            file_ref = FileReference(filename="audio.wav", file_type="WAVE")
            track = Track(number=1, track_type="AUDIO")
            track.title = long_title
            track.indices[1] = CueTime(0, 0, 0)
            file_ref.tracks = [track]
            sheet.files = [file_ref]

            # Create audio file
            audio_path = Path(tmpdir) / "audio.wav"
            audio_path.write_bytes(b"fake")

            with patch.object(validator.parser, "parse", return_value=sheet):
                result = validator.validate(str(cue_path))

            assert any("exceeds 80 character limit" in w.message for w in result.warnings)


class TestAudioDurationValidation:
    """Test audio duration validation."""

    @pytest.fixture
    def validator(self):
        """Create a CueValidator instance."""
        return CueValidator()

    def test_duration_mismatch(self, validator):
        """Test detection of CUE vs audio duration mismatch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cue_path = Path(tmpdir) / "test.cue"
            cue_content = """FILE "audio.wav" WAVE
  TRACK 01 AUDIO
    INDEX 01 00:00:00
  TRACK 02 AUDIO
    INDEX 01 10:00:00"""  # 10 minutes - likely beyond audio
            cue_path.write_text(cue_content)

            sheet = CueSheet()
            file_ref = FileReference(filename="audio.wav", file_type="WAVE")

            track1 = Track(number=1, track_type="AUDIO")
            track1.indices[1] = CueTime(0, 0, 0)

            track2 = Track(number=2, track_type="AUDIO")
            track2.indices[1] = CueTime(10, 0, 0)

            file_ref.tracks = [track1, track2]
            sheet.files = [file_ref]

            # Create audio file
            audio_path = Path(tmpdir) / "audio.wav"
            audio_path.write_bytes(b"fake")

            # Mock audio analyzer to return 3 minutes duration
            with (
                patch.object(validator.parser, "parse", return_value=sheet),
                patch.object(
                    validator.audio_analyzer,
                    "analyze_durations",
                    return_value=(180000, 600000),
                ),
            ):  # 3 min audio, 10 min CUE
                result = validator.validate(str(cue_path))

            assert any("duration" in e.message.lower() and "differs" in e.message for e in result.errors)


class TestCompatibilityChecks:
    """Test software compatibility validation."""

    @pytest.fixture
    def validator(self):
        """Create a CueValidator instance."""
        return CueValidator()

    def test_multi_file_compatibility(self, validator):
        """Test multi-file CUE compatibility warnings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cue_path = Path(tmpdir) / "test.cue"
            cue_content = """FILE "audio1.wav" WAVE
  TRACK 01 AUDIO
    INDEX 01 00:00:00
FILE "audio2.wav" WAVE
  TRACK 02 AUDIO
    INDEX 01 00:00:00"""
            cue_path.write_text(cue_content)

            sheet = CueSheet()

            file_ref1 = FileReference(filename="audio1.wav", file_type="WAVE")
            track1 = Track(number=1, track_type="AUDIO")
            track1.indices[1] = CueTime(0, 0, 0)
            file_ref1.tracks = [track1]

            file_ref2 = FileReference(filename="audio2.wav", file_type="WAVE")
            track2 = Track(number=2, track_type="AUDIO")
            track2.indices[1] = CueTime(0, 0, 0)
            file_ref2.tracks = [track2]

            sheet.files = [file_ref1, file_ref2]

            # Create audio files
            (Path(tmpdir) / "audio1.wav").write_bytes(b"fake")
            (Path(tmpdir) / "audio2.wav").write_bytes(b"fake")

            with patch.object(validator.parser, "parse", return_value=sheet):
                result = validator.validate(str(cue_path))

            assert any("Multi-file CUE" in i.message for i in result.info)

    def test_format_compatibility_report(self, validator):
        """Test format compatibility reporting."""

        with tempfile.TemporaryDirectory() as tmpdir:
            cue_path = Path(tmpdir) / "test.cue"
            cue_path.write_text('FILE "audio.wav" WAVE\n  TRACK 01 AUDIO\n    INDEX 01 00:00:00')

            sheet = CueSheet()
            file_ref = FileReference(filename="audio.wav", file_type="WAVE")
            track = Track(number=1, track_type="AUDIO")
            track.indices[1] = CueTime(0, 0, 0)
            file_ref.tracks = [track]
            sheet.files = [file_ref]

            # Create audio file
            (Path(tmpdir) / "audio.wav").write_bytes(b"fake")

            with patch.object(validator.parser, "parse", return_value=sheet):
                result = validator.validate(str(cue_path))

            assert "CDJ" in result.format_compatibility
            assert "Traktor" in result.format_compatibility
            assert "Serato" in result.format_compatibility
            assert "Rekordbox" in result.format_compatibility
            assert "Kodi" in result.format_compatibility
