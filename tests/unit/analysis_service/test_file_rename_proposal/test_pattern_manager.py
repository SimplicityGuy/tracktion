"""Tests for PatternManager."""

import pytest

from services.analysis_service.src.file_rename_proposal.config import (
    FileRenameProposalConfig,
)
from services.analysis_service.src.file_rename_proposal.pattern_manager import (
    PatternManager,
)


class TestPatternManager:
    """Test PatternManager class."""

    def test_default_pattern_application(self):
        """Test applying default patterns."""
        config = FileRenameProposalConfig()
        manager = PatternManager(config)

        metadata = {
            "artist": "The Beatles",
            "title": "Hey Jude",
            "album": "Past Masters",
            "track": "1",
        }

        # Test MP3 default pattern
        result = manager.apply_pattern(metadata, "mp3")
        assert result == "The Beatles - Hey Jude"

        # Test FLAC pattern with track number
        result = manager.apply_pattern(metadata, "flac")
        assert result == "The Beatles - Past Masters - 01 - Hey Jude"

    def test_pattern_with_missing_metadata(self):
        """Test pattern application with missing metadata fields."""
        config = FileRenameProposalConfig()
        manager = PatternManager(config)

        metadata = {"artist": "Unknown Artist"}  # Missing title

        result = manager.apply_pattern(metadata, "mp3")
        assert result == "Unknown Artist - Untitled"

    def test_custom_pattern(self):
        """Test setting and using custom patterns."""
        config = FileRenameProposalConfig()
        manager = PatternManager(config)

        # Set custom pattern
        manager.set_custom_pattern("mp3", "{track:02d}. {artist} - {title}")

        metadata = {
            "artist": "Pink Floyd",
            "title": "Time",
            "track": "4",
        }

        result = manager.apply_pattern(metadata, "mp3")
        assert result == "04. Pink Floyd - Time"

    def test_invalid_pattern_validation(self):
        """Test validation of invalid patterns."""
        config = FileRenameProposalConfig()
        manager = PatternManager(config)

        # Empty pattern
        with pytest.raises(ValueError):
            manager.set_custom_pattern("mp3", "")

        # No placeholders
        with pytest.raises(ValueError):
            manager.set_custom_pattern("mp3", "static filename")

        # Unbalanced braces
        with pytest.raises(ValueError):
            manager.set_custom_pattern("mp3", "{artist - {title}")

        # Nested braces
        with pytest.raises(ValueError):
            manager.set_custom_pattern("mp3", "{{artist}} - {title}")

    def test_pattern_with_formatting(self):
        """Test patterns with format specifications."""
        config = FileRenameProposalConfig()
        manager = PatternManager(config)

        metadata = {
            "artist": "Led Zeppelin",
            "title": "Stairway to Heaven",
            "track": "4",
            "disc": "2",
        }

        # Test with zero-padded track number
        manager.set_custom_pattern("mp3", "CD{disc} Track {track:03d} - {artist} - {title}")
        result = manager.apply_pattern(metadata, "mp3")
        assert result == "CD2 Track 004 - Led Zeppelin - Stairway to Heaven"

    def test_pattern_with_special_characters(self):
        """Test pattern handling with special characters in metadata."""
        config = FileRenameProposalConfig()
        manager = PatternManager(config)

        metadata = {
            "artist": "AC/DC",
            "title": "T.N.T.",
            "album": "High Voltage",
        }

        result = manager.apply_pattern(metadata, "mp3")
        assert result == "AC/DC - T.N.T."

    def test_pattern_with_empty_values(self):
        """Test pattern handling with empty string values."""
        config = FileRenameProposalConfig()
        manager = PatternManager(config)

        metadata = {
            "artist": "",
            "title": "",
            "album": "Some Album",
        }

        result = manager.apply_pattern(metadata, "mp3")
        assert result == "Unknown Artist - Untitled"

    def test_pattern_with_bpm_and_key(self):
        """Test patterns with BPM and key metadata."""
        config = FileRenameProposalConfig()
        manager = PatternManager(config)

        metadata = {
            "artist": "Daft Punk",
            "title": "One More Time",
            "bpm": "123",
            "key": "F#m",
        }

        manager.set_custom_pattern("mp3", "{artist} - {title} [{bpm}BPM, {key}]")
        result = manager.apply_pattern(metadata, "mp3")
        assert result == "Daft Punk - One More Time [123BPM, F#m]"

    def test_get_available_fields(self):
        """Test getting list of available fields."""
        config = FileRenameProposalConfig()
        manager = PatternManager(config)

        fields = manager.get_available_fields()
        assert "artist" in fields
        assert "title" in fields
        assert "album" in fields
        assert "bpm" in fields
        assert "key" in fields

    def test_get_pattern_for_type(self):
        """Test retrieving active pattern for a file type."""
        config = FileRenameProposalConfig()
        manager = PatternManager(config)

        # Default pattern
        pattern = manager.get_pattern_for_type("mp3")
        assert pattern == "{artist} - {title}"

        # Custom pattern
        manager.set_custom_pattern("mp3", "{track}. {title}")
        pattern = manager.get_pattern_for_type("mp3")
        assert pattern == "{track}. {title}"

        # Unknown type falls back to default
        pattern = manager.get_pattern_for_type("unknown")
        assert pattern == "{artist} - {title}"

    def test_conditional_patterns(self):
        """Test that patterns handle conditional metadata gracefully."""
        config = FileRenameProposalConfig()
        manager = PatternManager(config)

        # Pattern with optional year
        manager.set_custom_pattern("mp3", "{artist} - {title} ({year})")

        # With year
        metadata = {
            "artist": "The Who",
            "title": "Pinball Wizard",
            "year": "1969",
        }
        result = manager.apply_pattern(metadata, "mp3")
        assert result == "The Who - Pinball Wizard (1969)"

        # Without year - should use default
        metadata = {
            "artist": "The Who",
            "title": "Pinball Wizard",
        }
        result = manager.apply_pattern(metadata, "mp3")
        assert result == "The Who - Pinball Wizard (0000)"

    def test_multiple_space_cleanup(self):
        """Test that multiple spaces are cleaned up."""
        config = FileRenameProposalConfig()
        manager = PatternManager(config)

        metadata = {
            "artist": "The   Beatles",  # Multiple spaces
            "title": "Let It   Be",  # Multiple spaces
        }

        result = manager.apply_pattern(metadata, "mp3")
        # Multiple spaces should be collapsed to single space
        assert result == "The Beatles - Let It Be"

    def test_explicit_pattern_override(self):
        """Test using explicit pattern parameter."""
        config = FileRenameProposalConfig()
        manager = PatternManager(config)

        metadata = {
            "artist": "Queen",
            "title": "Bohemian Rhapsody",
            "year": "1975",
        }

        # Use explicit pattern instead of file type default
        result = manager.apply_pattern(metadata, "mp3", pattern="{year} - {artist} - {title}")
        assert result == "1975 - Queen - Bohemian Rhapsody"
