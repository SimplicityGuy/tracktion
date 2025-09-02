"""Unit tests for OGG-specific pattern manager functionality."""

import pytest

from services.analysis_service.src.file_rename_proposal.config import FileRenameProposalConfig
from services.analysis_service.src.file_rename_proposal.pattern_manager import PatternManager


class TestPatternManagerOgg:
    """Test suite for OGG-specific pattern manager functionality."""

    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return FileRenameProposalConfig()

    @pytest.fixture
    def pattern_manager(self, config):
        """Create a pattern manager instance."""
        return PatternManager(config)

    def test_ogg_file_type_has_default_pattern(self, pattern_manager):
        """Test that OGG file type has a default pattern."""
        pattern = pattern_manager.get_pattern_for_type("ogg")
        assert pattern == "{artist} - {title}"

        pattern = pattern_manager.get_pattern_for_type("oga")
        assert pattern == "{artist} - {title}"

    def test_ogg_metadata_fields_available(self, pattern_manager):
        """Test that OGG-specific fields are available."""
        fields = pattern_manager.get_available_fields()
        assert "organization" in fields
        assert "encoder" in fields
        assert "custom_tags" in fields

    def test_apply_pattern_with_ogg_metadata(self, pattern_manager):
        """Test applying pattern with OGG-specific metadata."""
        metadata = {
            "artist": "Test Artist",
            "title": "Test Song",
            "organization": "Test Org",
            "encoder": "Xiph.Org libVorbis",
        }

        # Test with custom pattern using OGG fields
        pattern = "{artist} - {title} [{organization}]"
        result = pattern_manager.apply_pattern(metadata, "ogg", pattern)
        assert result == "Test Artist - Test Song [Test Org]"

    def test_apply_pattern_with_encoder_field(self, pattern_manager):
        """Test pattern with encoder field."""
        metadata = {
            "artist": "Artist",
            "title": "Title",
            "encoder": "libVorbis 1.3.7",
        }

        pattern = "{artist} - {title} (encoded with {encoder})"
        result = pattern_manager.apply_pattern(metadata, "ogg", pattern)
        assert result == "Artist - Title (encoded with libVorbis 1.3.7)"

    def test_ogg_missing_metadata_defaults(self, pattern_manager):
        """Test default values for missing OGG metadata."""
        metadata = {
            "artist": "Some Artist",
            "title": "Some Title",
        }

        pattern = "{artist} - {title} - {organization} - {encoder}"
        result = pattern_manager.apply_pattern(metadata, "ogg", pattern)
        assert result == "Some Artist - Some Title - Unknown - Unknown"

    def test_custom_pattern_for_ogg(self, pattern_manager):
        """Test setting custom pattern for OGG files."""
        custom_pattern = "{artist} - {album} - {track:02d} - {title} [{organization}]"
        pattern_manager.set_custom_pattern("ogg", custom_pattern)

        assert pattern_manager.get_pattern_for_type("ogg") == custom_pattern

        # Apply the custom pattern
        metadata = {
            "artist": "Artist",
            "album": "Album",
            "track": "5",
            "title": "Title",
            "organization": "Org",
        }
        result = pattern_manager.apply_pattern(metadata, "ogg")
        assert result == "Artist - Album - 05 - Title [Org]"

    def test_custom_pattern_for_oga(self, pattern_manager):
        """Test setting custom pattern for OGA files."""
        custom_pattern = "{artist} - {title} ({encoder})"
        pattern_manager.set_custom_pattern("oga", custom_pattern)

        assert pattern_manager.get_pattern_for_type("oga") == custom_pattern

    def test_ogg_pattern_with_custom_tags(self, pattern_manager):
        """Test pattern that references custom tags."""
        metadata = {
            "artist": "Artist",
            "title": "Title",
            "custom_tags": '{"label": "Test Label", "catalog": "CAT001"}',
        }

        # Custom tags as a whole field
        pattern = "{artist} - {title} - {custom_tags}"
        result = pattern_manager.apply_pattern(metadata, "ogg", pattern)
        assert "Artist - Title -" in result
        assert "Test Label" in result or "custom_tags" in result

    def test_ogg_pattern_validation(self, pattern_manager):
        """Test pattern validation for OGG patterns."""
        # Valid patterns
        assert pattern_manager._validate_pattern("{artist} - {title}")
        assert pattern_manager._validate_pattern("{organization} - {encoder}")
        assert pattern_manager._validate_pattern("{artist} - {title} [{custom_tags}]")

        # Invalid patterns
        assert not pattern_manager._validate_pattern("")
        assert not pattern_manager._validate_pattern("no placeholders")
        assert not pattern_manager._validate_pattern("{unclosed")

    def test_apply_pattern_ogg_with_formatting(self, pattern_manager):
        """Test OGG pattern with formatting specifications."""
        metadata = {
            "artist": "Artist",
            "title": "Title",
            "track": "3",
            "organization": "Very Long Organization Name",
        }

        # Test track number formatting
        pattern = "{artist} - {track:02d} - {title}"
        result = pattern_manager.apply_pattern(metadata, "ogg", pattern)
        assert result == "Artist - 03 - Title"

        # Test string truncation (though not directly supported, test behavior)
        pattern = "{artist} - {title} ({organization})"
        result = pattern_manager.apply_pattern(metadata, "ogg", pattern)
        assert result == "Artist - Title (Very Long Organization Name)"

    def test_oga_extension_treated_same_as_ogg(self, config):
        """Test that .oga extension is treated the same as .ogg."""
        assert "oga" in config.default_patterns
        assert "ogg" in config.default_patterns
        # Both should have patterns defined
