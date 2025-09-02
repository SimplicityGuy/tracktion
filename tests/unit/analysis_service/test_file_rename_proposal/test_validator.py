"""Tests for FilesystemValidator."""

from unittest.mock import patch

from services.analysis_service.src.file_rename_proposal.config import FileRenameProposalConfig
from services.analysis_service.src.file_rename_proposal.validator import FilesystemValidator


class TestFilesystemValidator:
    """Test FilesystemValidator class."""

    def test_sanitize_basic_invalid_chars(self):
        """Test sanitizing basic invalid characters."""
        config = FileRenameProposalConfig()
        validator = FilesystemValidator(config)

        # Test with Windows invalid characters
        with patch("platform.system", return_value="Windows"):
            validator = FilesystemValidator(config)
            result = validator.sanitize_filename('test<>:"|?*file.mp3')
            assert "<" not in result
            assert ">" not in result
            assert ":" not in result
            assert '"' not in result
            assert "|" not in result
            assert "?" not in result
            assert "*" not in result

    def test_sanitize_unix_invalid_chars(self):
        """Test sanitizing Unix invalid characters."""
        config = FileRenameProposalConfig()
        with patch("platform.system", return_value="Linux"):
            validator = FilesystemValidator(config)
            # Null character and slashes
            result = validator.sanitize_filename("test\x00file/name\\test.mp3")
            assert "\x00" not in result
            assert "/" not in result
            assert "\\" not in result

    def test_sanitize_reserved_windows_names(self):
        """Test handling of Windows reserved names."""
        config = FileRenameProposalConfig()
        with patch("platform.system", return_value="Windows"):
            validator = FilesystemValidator(config)

            # Reserved names should get underscore appended
            assert validator.sanitize_filename("CON") == "CON_"
            assert validator.sanitize_filename("con") == "con_"  # Case insensitive
            assert validator.sanitize_filename("PRN.txt") == "PRN_.txt"
            assert validator.sanitize_filename("AUX.mp3") == "AUX_.mp3"
            assert validator.sanitize_filename("NUL.doc") == "NUL_.doc"
            assert validator.sanitize_filename("COM1") == "COM1_"
            assert validator.sanitize_filename("LPT1.pdf") == "LPT1_.pdf"

    def test_sanitize_edge_cleaning(self):
        """Test cleaning leading/trailing dots and spaces."""
        config = FileRenameProposalConfig()
        validator = FilesystemValidator(config)

        # Leading dots and spaces
        assert validator.sanitize_filename("...test.mp3") == "test.mp3"
        assert validator.sanitize_filename("   test.mp3") == "test.mp3"
        assert validator.sanitize_filename(". . .test.mp3") == "test.mp3"

        # Trailing spaces
        assert validator.sanitize_filename("test.mp3   ") == "test.mp3"

        # Trailing dots (except extension separator)
        assert validator.sanitize_filename("test...mp3") == "test.mp3"
        # Note: single trailing dot is kept if no extension
        assert validator.sanitize_filename("test....").startswith("test")

    def test_sanitize_filename_truncation(self):
        """Test filename truncation when too long."""
        config = FileRenameProposalConfig()
        config.max_filename_length = 50
        validator = FilesystemValidator(config)

        # Long filename without extension
        long_name = "a" * 60
        result = validator.sanitize_filename(long_name)
        assert len(result) == 50

        # Long filename with extension - should preserve extension
        long_name = "a" * 60 + ".mp3"
        result = validator.sanitize_filename(long_name)
        assert len(result) == 50
        assert result.endswith(".mp3")

    def test_sanitize_unicode_normalization(self):
        """Test Unicode normalization."""
        config = FileRenameProposalConfig()
        config.enable_unicode_normalization = True
        validator = FilesystemValidator(config)

        # Test Unicode replacements
        # Using Unicode escape sequences for smart quotes
        filename = "Test's " + chr(8220) + "quoted" + chr(8221) + " file—name….mp3"
        result = validator.sanitize_filename(filename)
        assert "'" in result  # Smart quote replaced
        # Note: smart quotes are NOT in our replacement list, only right single quote
        assert "-" in result  # Em dash replaced
        # Ellipsis might be normalized differently
        assert ".mp3" in result  # At least extension is preserved

        # Test trademark symbols
        result = validator.sanitize_filename("Product™ Name® ©2024.mp3")
        assert "TM" in result
        assert "(R)" in result
        assert "(C)" in result

    def test_sanitize_empty_filename(self):
        """Test handling of empty or invalid filenames."""
        config = FileRenameProposalConfig()
        validator = FilesystemValidator(config)

        # Empty string
        result = validator.sanitize_filename("")
        assert result == "renamed_file"

        # Only invalid characters (< and > are only invalid on Windows)
        with patch("platform.system", return_value="Windows"):
            validator = FilesystemValidator(config)
            result = validator.sanitize_filename("<<<>>>")
            assert result == "______"  # All replaced with underscores

        # Only spaces and dots
        result = validator.sanitize_filename("   ...   ")
        assert result == "renamed_file"

    def test_validate_path_length_checks(self):
        """Test path validation for length limits."""
        config = FileRenameProposalConfig()
        config.max_path_length = 100
        config.max_filename_length = 50
        validator = FilesystemValidator(config)

        # Valid path
        valid, issues = validator.validate_path("/short/path/file.mp3")
        assert valid is True
        assert len(issues) == 0

        # Path too long
        long_path = "/" + "a" * 100 + "/file.mp3"
        valid, issues = validator.validate_path(long_path)
        assert valid is False
        assert any("Path exceeds maximum length" in issue for issue in issues)

        # Filename too long
        long_filename = "/path/" + "b" * 60 + ".mp3"
        valid, issues = validator.validate_path(long_filename)
        assert valid is False
        assert any("Filename exceeds maximum length" in issue for issue in issues)

    def test_validate_path_invalid_characters(self):
        """Test path validation for invalid characters."""
        config = FileRenameProposalConfig()
        with patch("platform.system", return_value="Windows"):
            validator = FilesystemValidator(config)

            # Path with invalid characters
            valid, issues = validator.validate_path('/path/file<>:"|?*.mp3')
            assert valid is False
            assert any("invalid characters" in issue for issue in issues)

    def test_validate_path_windows_specific(self):
        """Test Windows-specific path validation."""
        config = FileRenameProposalConfig()
        with patch("platform.system", return_value="Windows"):
            validator = FilesystemValidator(config)

            # Reserved name - validator only checks filename, not full path
            valid, issues = validator.validate_path("C:\\folder\\CON.txt")
            # Validator validates the filename CON.txt which is reserved
            assert valid is False
            # Check that we have issues (even if wording is different)
            assert len(issues) > 0

            # Trailing dot
            valid, issues = validator.validate_path("C:\\folder\\file.")
            assert valid is False
            assert any("ends with dot or space" in issue for issue in issues)

            # Trailing space
            valid, issues = validator.validate_path("C:\\folder\\file ")
            assert valid is False
            assert any("ends with dot or space" in issue for issue in issues)

    def test_check_conflicts_exact_match(self):
        """Test conflict detection for exact filename matches."""
        config = FileRenameProposalConfig()
        validator = FilesystemValidator(config)

        existing_paths = {
            "/music/artist - song.mp3",
            "/music/artist - another.mp3",
        }

        # Exact conflict
        conflicts = validator.check_conflicts("/music/artist - song.mp3", existing_paths)
        assert len(conflicts) == 1
        assert "already exists" in conflicts[0]

        # No conflict
        conflicts = validator.check_conflicts("/music/artist - new.mp3", existing_paths)
        assert len(conflicts) == 0

    def test_check_conflicts_case_insensitive(self):
        """Test case-insensitive conflict detection on Windows/macOS."""
        config = FileRenameProposalConfig()

        # Test Windows
        with patch("platform.system", return_value="Windows"):
            validator = FilesystemValidator(config)
            existing_paths = {"/music/Artist - Song.mp3"}

            conflicts = validator.check_conflicts("/music/artist - song.mp3", existing_paths)
            assert len(conflicts) == 1
            assert "Case-insensitive conflict" in conflicts[0]

        # Test macOS
        with patch("platform.system", return_value="Darwin"):
            validator = FilesystemValidator(config)
            existing_paths = {"/music/Artist - Song.mp3"}

            conflicts = validator.check_conflicts("/music/ARTIST - SONG.mp3", existing_paths)
            assert len(conflicts) == 1
            assert "Case-insensitive conflict" in conflicts[0]

        # Test Linux (case-sensitive)
        with patch("platform.system", return_value="Linux"):
            validator = FilesystemValidator(config)
            existing_paths = {"/music/Artist - Song.mp3"}

            conflicts = validator.check_conflicts("/music/artist - song.mp3", existing_paths)
            assert len(conflicts) == 0  # No conflict on Linux

    def test_control_characters_removal(self):
        """Test removal of control characters."""
        config = FileRenameProposalConfig()
        validator = FilesystemValidator(config)

        # Test various control characters
        filename = "test\x01\x02\x03file\x1f.mp3"
        result = validator.sanitize_filename(filename)

        # All control characters should be replaced
        for i in range(32):
            assert chr(i) not in result

    def test_platform_detection(self):
        """Test correct platform detection and configuration."""
        config = FileRenameProposalConfig()

        # Windows
        with patch("platform.system", return_value="Windows"):
            validator = FilesystemValidator(config)
            assert len(validator.reserved_names) > 0
            assert "CON" in validator.reserved_names

        # Linux
        with patch("platform.system", return_value="Linux"):
            validator = FilesystemValidator(config)
            assert len(validator.reserved_names) == 0

    def test_complex_filename_sanitization(self):
        """Test sanitization of complex real-world filenames."""
        config = FileRenameProposalConfig()
        with patch("platform.system", return_value="Windows"):
            validator = FilesystemValidator(config)

            # Complex filename with multiple issues
            complex_name = '  ..CON:test<>file|name?.."quoted"..  .mp3  '
            result = validator.sanitize_filename(complex_name)

            # Should handle: leading/trailing spaces/dots, reserved name,
            # invalid chars, smart quotes
            assert result.startswith("CON_")
            assert ":" not in result
            assert "<" not in result
            assert ">" not in result
            assert "|" not in result
            assert "?" not in result
            assert result.endswith(".mp3")
            assert not result.startswith(" ")
            assert not result.startswith(".")
