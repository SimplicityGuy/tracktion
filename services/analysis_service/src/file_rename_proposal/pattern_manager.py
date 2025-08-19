"""Manages naming patterns for file rename proposals."""

import logging
import re
from typing import Dict, Optional

from .config import FileRenameProposalConfig

logger = logging.getLogger(__name__)


class PatternManager:
    """Manages and applies naming patterns for filename generation."""

    def __init__(self, config: FileRenameProposalConfig):
        """Initialize the pattern manager.

        Args:
            config: Configuration containing default patterns
        """
        self.config = config
        self.patterns = config.default_patterns.copy()
        self.custom_patterns: Dict[str, str] = {}

    def apply_pattern(self, metadata: Dict[str, str], file_type: str, pattern: Optional[str] = None) -> str:
        """Apply a naming pattern to metadata to generate a filename.

        Args:
            metadata: Dictionary of metadata values
            file_type: File extension (mp3, flac, etc.)
            pattern: Optional custom pattern to use

        Returns:
            Generated filename (without extension)
        """
        # Select pattern to use
        if pattern:
            template = pattern
        elif file_type in self.custom_patterns:
            template = self.custom_patterns[file_type]
        elif file_type in self.patterns:
            template = self.patterns[file_type]
        else:
            template = self.patterns.get("default", "{artist} - {title}")

        # Process the template
        result = self._process_template(template, metadata)

        # Clean up multiple spaces and trim
        result = re.sub(r"\s+", " ", result).strip()

        return result

    def set_custom_pattern(self, file_type: str, pattern: str) -> None:
        """Set a custom pattern for a file type.

        Args:
            file_type: File extension
            pattern: Pattern string with placeholders
        """
        if self._validate_pattern(pattern):
            self.custom_patterns[file_type] = pattern
            logger.info(f"Set custom pattern for {file_type}: {pattern}")
        else:
            raise ValueError(f"Invalid pattern: {pattern}")

    def _process_template(self, template: str, metadata: Dict[str, str]) -> str:
        """Process a template string with metadata values.

        Args:
            template: Template string with {field} placeholders
            metadata: Metadata dictionary

        Returns:
            Processed string with placeholders replaced
        """
        result = template

        # Find all placeholders in the template
        placeholders = re.findall(r"\{([^}]+)\}", template)

        for placeholder in placeholders:
            # Check for formatting (e.g., track:02d)
            if ":" in placeholder:
                field, format_spec = placeholder.split(":", 1)
            else:
                field = placeholder
                format_spec = None

            # Get the value from metadata
            value = metadata.get(field, "")

            # Handle empty values
            if not value:
                # Use a default value based on field name
                value = self._get_default_value(field)

            # Apply formatting if specified
            if format_spec and value:
                try:
                    if "d" in format_spec:  # Integer formatting
                        try:
                            value = format(int(value), format_spec)
                        except (ValueError, TypeError):
                            pass  # Keep original value if conversion fails
                    else:
                        value = format(value, format_spec)
                except (ValueError, TypeError):
                    pass  # Keep original value if formatting fails

            # Replace the placeholder
            result = result.replace(f"{{{placeholder}}}", str(value))

        return result

    def _get_default_value(self, field: str) -> str:
        """Get a default value for a missing field.

        Args:
            field: Field name

        Returns:
            Default value for the field
        """
        defaults = {
            "artist": "Unknown Artist",
            "title": "Untitled",
            "album": "Unknown Album",
            "track": "00",
            "year": "0000",
            "genre": "Unknown",
            "bpm": "000",
            "key": "Unknown",
        }
        return defaults.get(field, "Unknown")

    def _validate_pattern(self, pattern: str) -> bool:
        """Validate a pattern string.

        Args:
            pattern: Pattern to validate

        Returns:
            True if pattern is valid
        """
        if not pattern:
            return False

        # Check for at least one placeholder
        if not re.search(r"\{[^}]+\}", pattern):
            return False

        # Check for balanced braces
        open_count = pattern.count("{")
        close_count = pattern.count("}")
        if open_count != close_count:
            return False

        # Check for nested braces
        if "{{" in pattern or "}}" in pattern:
            return False

        return True

    def get_available_fields(self) -> list:
        """Get a list of available metadata fields for patterns.

        Returns:
            List of field names that can be used in patterns
        """
        return [
            "artist",
            "title",
            "album",
            "track",
            "year",
            "genre",
            "bpm",
            "key",
            "mood",
            "energy",
            "composer",
            "albumartist",
            "disc",
            "comment",
        ]

    def get_pattern_for_type(self, file_type: str) -> str:
        """Get the active pattern for a file type.

        Args:
            file_type: File extension

        Returns:
            Active pattern string
        """
        if file_type in self.custom_patterns:
            return self.custom_patterns[file_type]
        return self.patterns.get(file_type, self.patterns.get("default", "{artist} - {title}"))
