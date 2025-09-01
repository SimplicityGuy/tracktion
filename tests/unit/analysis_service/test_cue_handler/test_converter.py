"""Unit tests for the CUE format converter."""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from services.analysis_service.src.cue_handler import (
    CONVERSION_RULES,
    FORMAT_CAPABILITIES,
    BatchConversionReport,
    CompatibilityChecker,
    CompatibilityLevel,
    ConversionChange,
    ConversionMode,
    ConversionReport,
    CueConverter,
    CueFormat,
    ValidationResult,
    get_conversion_rules,
    get_format_capabilities,
    get_format_from_string,
    get_lossy_warnings,
)


class TestCueConverter:
    """Test the CueConverter class."""

    @pytest.fixture
    def converter(self):
        """Create a CueConverter instance."""
        return CueConverter()

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_cue_content(self):
        """Sample CUE file content."""
        return """REM GENRE "Electronic"
REM DATE 2024
PERFORMER "Test Artist"
TITLE "Test Album"
FILE "test.wav" WAVE
  TRACK 01 AUDIO
    TITLE "Track 1"
    PERFORMER "Test Artist"
    INDEX 01 00:00:00
  TRACK 02 AUDIO
    TITLE "Track 2"
    PERFORMER "Test Artist"
    INDEX 01 03:30:00"""

    @pytest.fixture
    def sample_cue_file(self, temp_dir, sample_cue_content):
        """Create a sample CUE file."""
        cue_file = temp_dir / "test.cue"
        cue_file.write_text(sample_cue_content)
        return cue_file

    def test_converter_initialization(self):
        """Test CueConverter initialization."""
        # Default initialization
        converter = CueConverter()
        assert converter.mode == ConversionMode.STANDARD
        assert converter.validate_output is True
        assert converter.verbose is False

        # Custom initialization
        converter = CueConverter(mode=ConversionMode.STRICT, validate_output=False, verbose=True)
        assert converter.mode == ConversionMode.STRICT
        assert converter.validate_output is False
        assert converter.verbose is True

    def test_convert_file_not_found(self, converter):
        """Test conversion with non-existent file."""
        report = converter.convert("nonexistent.cue", CueFormat.CDJ)

        assert report.success is False
        assert len(report.errors) > 0
        assert "does not exist" in report.errors[0]

    def test_format_detection(self, converter, temp_dir):
        """Test auto-detection of CUE format."""
        # Create Traktor format CUE
        traktor_content = """REM BPM 128
REM KEY Am
REM ENERGY 7
TITLE "Test"
FILE "test.wav" WAVE
  TRACK 01 AUDIO
    INDEX 01 00:00:00"""
        traktor_file = temp_dir / "traktor.cue"
        traktor_file.write_text(traktor_content)

        # Mock parser and format detection
        mock_cuesheet = Mock()
        mock_cuesheet.rem_comments = {"BPM": "128", "KEY": "Am", "ENERGY": "7"}
        mock_cuesheet.tracks = []

        with patch.object(converter.parser, "parse", return_value=mock_cuesheet):
            detected = converter._detect_format(mock_cuesheet)
            assert detected == CueFormat.TRAKTOR

    def test_standard_to_cdj_conversion(self, converter, sample_cue_file, temp_dir):
        """Test converting from Standard to CDJ format."""
        output_file = temp_dir / "output.cdj.cue"

        # Mock the parser and generator
        mock_cuesheet = Mock()
        mock_cuesheet.tracks = [Mock(), Mock()]
        mock_cuesheet.files = [Mock()]
        mock_cuesheet.rem_comments = {}
        mock_cuesheet.title = "Test Album"
        mock_cuesheet.performer = "Test Artist"
        mock_cuesheet.encoding = "UTF-8"

        with (
            patch.object(converter.parser, "parse", return_value=mock_cuesheet),
            patch("services.analysis_service.src.cue_handler.converter.get_generator") as mock_get_gen,
        ):
            mock_generator = Mock()
            mock_generator.generate_from_cuesheet.return_value = "CDJ formatted content"
            mock_get_gen.return_value = Mock(return_value=mock_generator)

            report = converter.convert(sample_cue_file, CueFormat.CDJ, output_file)

            assert report.source_file == str(sample_cue_file)
            assert report.target_format == CueFormat.CDJ
            assert report.target_file == str(output_file)

    def test_batch_conversion(self, converter, temp_dir):
        """Test batch conversion of multiple files."""
        # Create multiple CUE files
        files = []
        for i in range(3):
            cue_file = temp_dir / f"test{i}.cue"
            cue_file.write_text(
                f"""TITLE "Album {i}"
FILE "test{i}.wav" WAVE
  TRACK 01 AUDIO
    INDEX 01 00:00:00"""
            )
            files.append(cue_file)

        # Mock parser and generator
        mock_cuesheet = Mock()
        mock_cuesheet.tracks = [Mock()]
        mock_cuesheet.files = [Mock()]
        mock_cuesheet.rem_comments = {}
        mock_cuesheet.title = "Test"
        mock_cuesheet.performer = "Artist"

        with (
            patch.object(converter.parser, "parse", return_value=mock_cuesheet),
            patch("services.analysis_service.src.cue_handler.converter.get_generator") as mock_get_gen,
        ):
            mock_generator = Mock()
            mock_generator.generate_from_cuesheet.return_value = "converted content"
            mock_get_gen.return_value = Mock(return_value=mock_generator)

            # Use list of files instead of glob pattern
            batch_report = converter.batch_convert(
                files,
                CueFormat.TRAKTOR,
                output_dir=temp_dir / "output",
                parallel=False,
            )

            assert batch_report.total_files == 3
            assert len(batch_report.reports) == 3

    def test_dry_run_mode(self, converter, sample_cue_file, temp_dir):
        """Test dry-run mode doesn't write files."""
        output_file = temp_dir / "dry_run_output.cue"

        mock_cuesheet = Mock()
        mock_cuesheet.tracks = []
        mock_cuesheet.files = []
        mock_cuesheet.rem_comments = {}

        with patch.object(converter.parser, "parse", return_value=mock_cuesheet):
            report = converter.convert(sample_cue_file, CueFormat.SERATO, output_file, dry_run=True)

            assert not output_file.exists()
            assert report.target_file == str(output_file)

    def test_conversion_with_validation(self, converter, sample_cue_file, temp_dir):
        """Test conversion with output validation."""
        output_file = temp_dir / "validated.cue"

        mock_cuesheet = Mock()
        mock_cuesheet.tracks = []
        mock_cuesheet.files = []
        mock_cuesheet.rem_comments = {}

        mock_validation = ValidationResult(file_path=str(output_file), is_valid=True)

        with (
            patch.object(converter.parser, "parse", return_value=mock_cuesheet),
            patch("services.analysis_service.src.cue_handler.converter.get_generator") as mock_get_gen,
        ):
            mock_generator = Mock()
            mock_generator.generate_from_cuesheet.return_value = "content"
            mock_get_gen.return_value = Mock(return_value=mock_generator)

            with patch.object(converter.validator, "validate", return_value=mock_validation):
                report = converter.convert(sample_cue_file, CueFormat.REKORDBOX, output_file)

                assert report.validation_result == mock_validation

    def test_metadata_preservation_calculation(self):
        """Test metadata preservation calculation."""
        report = ConversionReport(
            source_file="test.cue",
            source_format=CueFormat.STANDARD,
            target_file="output.cue",
            target_format=CueFormat.CDJ,
            success=True,
        )

        # No changes - 100% preserved
        report.calculate_metadata_preservation()
        assert report.metadata_preserved == 100.0

        # Add some changes
        report.add_change(ConversionChange(change_type="removed", command="PREGAP", original_value="00:02:00"))
        report.add_change(
            ConversionChange(
                change_type="modified",
                command="FLAGS",
                original_value="DCP",
                new_value="COPY",
            )
        )

        report.calculate_metadata_preservation()
        assert report.metadata_preserved < 100.0
        assert report.metadata_preserved > 0.0

    def test_custom_conversion_rules(self, converter, sample_cue_file, temp_dir):
        """Test conversion with custom rules."""
        output_file = temp_dir / "custom.cue"
        custom_rules = {
            "limit_tracks": 50,
            "encoding": "UTF-16",
            "custom_field": "test",
        }

        mock_cuesheet = Mock()
        mock_cuesheet.tracks = []
        mock_cuesheet.files = []
        mock_cuesheet.rem_comments = {}

        with (
            patch.object(converter.parser, "parse", return_value=mock_cuesheet),
            patch("services.analysis_service.src.cue_handler.converter.get_generator") as mock_get_gen,
        ):
            mock_generator = Mock()
            mock_generator.generate_from_cuesheet.return_value = "content"
            mock_get_gen.return_value = Mock(return_value=mock_generator)

            report = converter.convert(
                sample_cue_file,
                CueFormat.TRAKTOR,
                output_file,
                custom_rules=custom_rules,
            )

            assert report.source_file == str(sample_cue_file)

    def test_get_supported_formats(self, converter):
        """Test getting list of supported formats."""
        formats = converter.get_supported_formats()

        assert "standard" in formats
        assert "cdj" in formats
        assert "traktor" in formats
        assert "serato" in formats
        assert "rekordbox" in formats
        assert "kodi" in formats

    def test_preview_conversion(self, converter, sample_cue_file):
        """Test preview conversion (dry-run shortcut)."""
        mock_cuesheet = Mock()
        mock_cuesheet.tracks = []
        mock_cuesheet.files = []
        mock_cuesheet.rem_comments = {}

        with patch.object(converter.parser, "parse", return_value=mock_cuesheet):
            report = converter.preview_conversion(sample_cue_file, CueFormat.CDJ)

            assert isinstance(report, ConversionReport)


class TestCompatibilityChecker:
    """Test the CompatibilityChecker class."""

    @pytest.fixture
    def checker(self):
        """Create a CompatibilityChecker instance."""
        return CompatibilityChecker()

    def test_full_compatibility(self, checker):
        """Test checking full compatibility between same formats."""
        report = checker.check_compatibility(CueFormat.STANDARD, CueFormat.STANDARD)

        assert report.compatibility_level == CompatibilityLevel.FULL
        assert report.can_convert is True
        assert report.metadata_preservation_estimate == 100.0

    def test_track_limit_compatibility(self, checker):
        """Test track count limit checking."""
        mock_cuesheet = Mock()
        mock_cuesheet.tracks = [Mock() for _ in range(150)]

        report = checker.check_compatibility(CueFormat.STANDARD, CueFormat.CDJ, mock_cuesheet)

        # CDJ supports more tracks than standard, but check if warning exists
        assert report.can_convert is True

    def test_multi_file_compatibility(self, checker):
        """Test multi-file support checking."""
        mock_cuesheet = Mock()
        mock_cuesheet.files = [Mock(), Mock()]  # Multiple files

        report = checker.check_compatibility(
            CueFormat.STANDARD,
            CueFormat.SERATO,  # Serato doesn't support multi-file
            mock_cuesheet,
        )

        assert report.can_convert is True
        assert any(issue.feature == "multi_file" for issue in report.issues)

    def test_lossy_conversion_warnings(self, checker):
        """Test that lossy conversions generate warnings."""
        report = checker.check_compatibility(CueFormat.STANDARD, CueFormat.TRAKTOR)

        assert len(report.warnings) > 0
        assert any("ISRC" in warning for warning in report.warnings)

    def test_compatibility_matrix(self, checker):
        """Test generation of compatibility matrix."""
        matrix = checker.get_compatibility_matrix()

        # Same format should always be fully compatible
        assert matrix[(CueFormat.STANDARD, CueFormat.STANDARD)] == CompatibilityLevel.FULL
        assert matrix[(CueFormat.CDJ, CueFormat.CDJ)] == CompatibilityLevel.FULL

        # Check some known conversions
        assert (CueFormat.STANDARD, CueFormat.CDJ) in matrix
        assert (CueFormat.TRAKTOR, CueFormat.SERATO) in matrix

    def test_conversion_path_recommendation(self, checker):
        """Test optimal conversion path recommendation."""
        # Direct path for compatible formats
        path = checker.recommend_conversion_path(CueFormat.STANDARD, CueFormat.KODI)
        assert len(path) == 2
        assert path[0] == CueFormat.STANDARD
        assert path[1] == CueFormat.KODI

        # May suggest intermediate format for complex conversions
        path = checker.recommend_conversion_path(CueFormat.TRAKTOR, CueFormat.CDJ)
        assert len(path) >= 2
        assert path[0] == CueFormat.TRAKTOR
        assert path[-1] == CueFormat.CDJ


class TestFormatMappings:
    """Test format mapping functions."""

    def test_get_format_from_string(self):
        """Test string to CueFormat conversion."""
        assert get_format_from_string("standard") == CueFormat.STANDARD
        assert get_format_from_string("CDJ") == CueFormat.CDJ
        assert get_format_from_string("TRAKTOR") == CueFormat.TRAKTOR

        with pytest.raises(ValueError):
            get_format_from_string("invalid_format")

    def test_get_format_capabilities(self):
        """Test getting format capabilities."""
        caps = get_format_capabilities(CueFormat.STANDARD)
        assert caps["max_tracks"] == 99
        assert caps["multi_file"] is True
        assert caps["pregap_postgap"] is True

        caps = get_format_capabilities(CueFormat.CDJ)
        assert caps["max_tracks"] == 999
        assert caps["multi_file"] is False

    def test_get_conversion_rules(self):
        """Test getting conversion rules."""
        rules = get_conversion_rules(CueFormat.STANDARD, CueFormat.CDJ)
        assert "remove_commands" in rules
        assert "limit_tracks" in rules
        assert rules["limit_tracks"] == 999

        rules = get_conversion_rules(CueFormat.STANDARD, CueFormat.TRAKTOR)
        assert "add_rem_fields" in rules
        assert "BPM" in rules["add_rem_fields"]

    def test_get_lossy_warnings(self):
        """Test getting lossy conversion warnings."""
        warnings = get_lossy_warnings(CueFormat.STANDARD, CueFormat.CDJ)
        assert len(warnings) > 0
        assert any("PREGAP" in w for w in warnings)

        warnings = get_lossy_warnings(CueFormat.TRAKTOR, CueFormat.STANDARD)
        assert len(warnings) > 0
        assert any("Beat grid" in w or "beat grid" in w for w in warnings)

    def test_format_capabilities_completeness(self):
        """Test that all formats have capabilities defined."""
        for format_type in CueFormat:
            caps = FORMAT_CAPABILITIES.get(format_type)
            assert caps is not None, f"Missing capabilities for {format_type}"
            assert "max_tracks" in caps
            assert "multi_file" in caps
            assert "rem_fields" in caps

    def test_conversion_rules_symmetry(self):
        """Test that major format pairs have conversion rules."""
        major_formats = [CueFormat.STANDARD, CueFormat.CDJ, CueFormat.TRAKTOR]

        for source in major_formats:
            for target in major_formats:
                if source != target:
                    key = (source.value, target.value)
                    # Not all pairs need rules, but check major ones exist
                    if source == CueFormat.STANDARD:
                        assert key in CONVERSION_RULES or target == CueFormat.STANDARD


class TestConversionReport:
    """Test ConversionReport class."""

    def test_report_initialization(self):
        """Test ConversionReport initialization."""
        report = ConversionReport(
            source_file="test.cue",
            source_format=CueFormat.STANDARD,
            target_file="output.cue",
            target_format=CueFormat.CDJ,
            success=True,
        )

        assert report.source_file == "test.cue"
        assert report.source_format == CueFormat.STANDARD
        assert report.target_format == CueFormat.CDJ
        assert report.success is True
        assert len(report.changes) == 0
        assert len(report.warnings) == 0
        assert len(report.errors) == 0

    def test_add_change(self):
        """Test adding changes to report."""
        report = ConversionReport(
            source_file="test.cue",
            source_format=CueFormat.STANDARD,
            target_file="output.cue",
            target_format=CueFormat.CDJ,
            success=True,
        )

        change = ConversionChange(
            change_type="removed",
            command="PREGAP",
            original_value="00:02:00",
            reason="Not supported",
        )

        report.add_change(change)
        assert len(report.changes) == 1
        assert report.changes[0] == change

    def test_add_error(self):
        """Test adding errors to report."""
        report = ConversionReport(
            source_file="test.cue",
            source_format=CueFormat.STANDARD,
            target_file="output.cue",
            target_format=CueFormat.CDJ,
            success=True,
        )

        report.add_error("Test error")
        assert len(report.errors) == 1
        assert report.errors[0] == "Test error"
        assert report.success is False  # Adding error sets success to False


class TestBatchConversionReport:
    """Test BatchConversionReport class."""

    def test_batch_report_initialization(self):
        """Test BatchConversionReport initialization."""
        batch = BatchConversionReport()
        assert batch.total_files == 0
        assert batch.successful == 0
        assert batch.failed == 0
        assert len(batch.reports) == 0

    def test_add_report(self):
        """Test adding reports to batch."""
        batch = BatchConversionReport()

        # Add successful report
        success_report = ConversionReport(
            source_file="test1.cue",
            source_format=CueFormat.STANDARD,
            target_file="out1.cue",
            target_format=CueFormat.CDJ,
            success=True,
        )
        batch.add_report(success_report)

        assert batch.total_files == 1
        assert batch.successful == 1
        assert batch.failed == 0

        # Add failed report
        fail_report = ConversionReport(
            source_file="test2.cue",
            source_format=CueFormat.STANDARD,
            target_file="out2.cue",
            target_format=CueFormat.CDJ,
            success=False,
        )
        batch.add_report(fail_report)

        assert batch.total_files == 2
        assert batch.successful == 1
        assert batch.failed == 1

    def test_get_summary(self):
        """Test batch summary generation."""
        batch = BatchConversionReport()

        for i in range(3):
            report = ConversionReport(
                source_file=f"test{i}.cue",
                source_format=CueFormat.STANDARD,
                target_file=f"out{i}.cue",
                target_format=CueFormat.CDJ,
                success=True,
            )
            report.metadata_preserved = 90.0
            batch.add_report(report)

        summary = batch.get_summary()
        assert "Total files: 3" in summary
        assert "Successful: 3" in summary
        assert "Failed: 0" in summary
        assert "90.0%" in summary
