"""Tests for the conflict resolution module."""

import pytest

from app.proposal.conflicts import (
    FilenameConflictResolver,
    check_duplicate,
    detect_conflicts,
    generate_unique_name,
    resolve_conflict,
    validate_filename,
)


class TestFilenameConflictResolver:
    """Test cases for FilenameConflictResolver."""

    def setup_method(self):
        """Set up test fixture."""
        self.resolver = FilenameConflictResolver()

    def test_validate_filename_valid_cases(self):
        """Test valid filename cases."""
        valid_filenames = [
            "test.txt",
            "my-file_name.pdf",
            "Document with spaces.docx",
            "file123.jpg",
            "测试文件.txt",  # Unicode
            "a" * 255,  # Max length
        ]

        for filename in valid_filenames:
            assert self.resolver.validate_filename(filename), f"Should be valid: {filename}"

    def test_validate_filename_invalid_cases(self):
        """Test invalid filename cases."""
        invalid_filenames = [
            "",  # Empty
            "   ",  # Whitespace only
            "file<test>.txt",  # Invalid character
            "file:name.txt",  # Invalid character
            "CON.txt",  # Reserved Windows name
            "NUL.pdf",  # Reserved Windows name
            ".hidden",  # Starts with dot
            "file.",  # Ends with dot
            " file.txt",  # Starts with space
            "file.txt ",  # Ends with space
            "a" * 256,  # Too long
        ]

        for filename in invalid_filenames:
            assert not self.resolver.validate_filename(filename), f"Should be invalid: {filename}"

    def test_check_duplicate(self):
        """Test duplicate detection."""
        # Case-insensitive duplicates
        assert self.resolver.check_duplicate("test.txt", "TEST.TXT")
        assert self.resolver.check_duplicate("file.pdf", "file.PDF")

        # Path normalization
        assert self.resolver.check_duplicate("/path/to/file.txt", "file.txt")
        assert self.resolver.check_duplicate("dir\\file.txt", "file.txt")

        # Not duplicates
        assert not self.resolver.check_duplicate("file1.txt", "file2.txt")
        assert not self.resolver.check_duplicate("test.txt", "test.pdf")

        # Empty/None cases
        assert not self.resolver.check_duplicate("", "test.txt")
        assert not self.resolver.check_duplicate("test.txt", "")

    def test_generate_unique_name(self):
        """Test unique name generation."""
        existing_files = ["test.txt", "document.pdf", "image.jpg"]

        # No conflict
        unique_name = self.resolver.generate_unique_name("new_file.txt", existing_files)
        assert unique_name == "new_file.txt"

        # With conflict - should append number
        unique_name = self.resolver.generate_unique_name("test.txt", existing_files)
        assert unique_name == "test (1).txt"

        # Multiple conflicts
        existing_with_numbered = existing_files + ["test (1).txt", "test (2).txt"]
        unique_name = self.resolver.generate_unique_name("test.txt", existing_with_numbered)
        assert unique_name == "test (3).txt"

    def test_generate_unique_name_errors(self):
        """Test error cases for unique name generation."""
        with pytest.raises(ValueError, match="Base name cannot be empty"):
            self.resolver.generate_unique_name("", [])

        with pytest.raises(ValueError, match="Invalid base filename"):
            self.resolver.generate_unique_name("CON.txt", [])

    def test_detect_conflicts_no_conflict(self):
        """Test conflict detection when no conflicts exist."""
        existing_files = ["file1.txt", "document.pdf"]
        conflict = self.resolver.detect_conflicts("new_file.txt", existing_files)
        assert conflict is None

    def test_detect_conflicts_with_duplicate(self):
        """Test conflict detection with duplicates."""
        existing_files = ["test.txt", "document.pdf"]
        conflict = self.resolver.detect_conflicts("TEST.TXT", existing_files)

        assert conflict is not None
        assert conflict.strategy == "append_number"
        assert conflict.existing_file == "test.txt"
        assert "existing file: test.txt" in conflict.proposed_action

    def test_detect_conflicts_invalid_filename(self):
        """Test conflict detection with invalid filename."""
        existing_files = ["test.txt"]
        conflict = self.resolver.detect_conflicts("CON.txt", existing_files)

        assert conflict is not None
        assert conflict.strategy == "skip"
        assert "invalid filename" in conflict.proposed_action.lower()

    def test_resolve_conflict_append_number(self):
        """Test conflict resolution with append_number strategy."""
        existing_files = ["test.txt", "document.pdf"]
        resolved = self.resolver.resolve_conflict("test.txt", existing_files, "append_number")
        assert resolved == "test (1).txt"

    def test_resolve_conflict_skip(self):
        """Test conflict resolution with skip strategy."""
        existing_files = ["test.txt"]
        resolved = self.resolver.resolve_conflict("test.txt", existing_files, "skip")
        assert resolved == ""

    def test_resolve_conflict_replace(self):
        """Test conflict resolution with replace strategy."""
        existing_files = ["test.txt"]
        resolved = self.resolver.resolve_conflict("test.txt", existing_files, "replace")
        assert resolved == "test.txt"

    def test_resolve_conflict_errors(self):
        """Test error cases for conflict resolution."""
        with pytest.raises(ValueError, match="Proposed name cannot be empty"):
            self.resolver.resolve_conflict("", [])

        with pytest.raises(ValueError, match="Existing files must be a list"):
            self.resolver.resolve_conflict("test.txt", "not_a_list")

        with pytest.raises(ValueError, match="Unsupported strategy"):
            self.resolver.resolve_conflict("test.txt", [], "invalid_strategy")

    def test_batch_resolve_conflicts(self):
        """Test batch conflict resolution."""
        proposed_names = ["new_file.txt", "test.txt", "another.pdf"]
        existing_files = ["test.txt", "existing.pdf"]

        results = self.resolver.batch_resolve_conflicts(proposed_names, existing_files)

        assert len(results) == 3
        assert results["new_file.txt"] == "new_file.txt"  # No conflict
        assert results["test.txt"] == "test (1).txt"  # Resolved conflict
        assert results["another.pdf"] == "another.pdf"  # No conflict

    def test_get_conflict_summary(self):
        """Test conflict summary generation."""
        proposed_names = ["new_file.txt", "test.txt", "CON.txt", "duplicate.pdf"]
        existing_files = ["test.txt", "duplicate.pdf"]

        summary = self.resolver.get_conflict_summary(proposed_names, existing_files)

        assert summary["total_proposed"] == 4
        assert summary["conflicts_detected"] == 2  # test.txt and duplicate.pdf
        assert summary["invalid_filenames"] == 1  # CON.txt
        assert len(summary["conflict_details"]) == 2
        assert len(summary["invalid_details"]) == 1


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_convenience_functions(self):
        """Test all convenience functions work correctly."""
        existing_files = ["test.txt"]

        # Test detect_conflicts
        conflict = detect_conflicts("TEST.TXT", existing_files)
        assert conflict is not None
        assert conflict.strategy == "append_number"

        # Test resolve_conflict
        resolved = resolve_conflict("test.txt", existing_files)
        assert resolved == "test (1).txt"

        # Test check_duplicate
        assert check_duplicate("test.txt", "TEST.TXT")

        # Test generate_unique_name
        unique = generate_unique_name("test.txt", existing_files)
        assert unique == "test (1).txt"

        # Test validate_filename
        assert validate_filename("valid_file.txt")
        assert not validate_filename("CON.txt")
