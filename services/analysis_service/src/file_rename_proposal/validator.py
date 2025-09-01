"""Filesystem validation for rename proposals."""

import logging
import platform
import unicodedata
from pathlib import Path

from .config import FileRenameProposalConfig

logger = logging.getLogger(__name__)


class FilesystemValidator:
    """Validates and sanitizes filenames for filesystem compatibility."""

    def __init__(self, config: FileRenameProposalConfig):
        """Initialize the validator.

        Args:
            config: Configuration with filesystem limits and rules
        """
        self.config = config
        self.platform = platform.system().lower()

        # Determine which invalid characters to use based on platform
        if self.platform == "windows":
            self.invalid_chars = set(config.invalid_chars_windows)
            # Windows also reserves certain names
            self.reserved_names = {
                "CON",
                "PRN",
                "AUX",
                "NUL",
                "COM1",
                "COM2",
                "COM3",
                "COM4",
                "COM5",
                "COM6",
                "COM7",
                "COM8",
                "COM9",
                "LPT1",
                "LPT2",
                "LPT3",
                "LPT4",
                "LPT5",
                "LPT6",
                "LPT7",
                "LPT8",
                "LPT9",
            }
        else:
            self.invalid_chars = set(config.invalid_chars_unix)
            self.reserved_names = set()

        # Add forward slash and backslash as invalid for filenames
        self.invalid_chars.add("/")
        self.invalid_chars.add("\\")

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize a filename for filesystem compatibility.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename safe for the filesystem
        """
        # Normalize Unicode if enabled
        if self.config.enable_unicode_normalization:
            filename = self._normalize_unicode(filename)

        # Replace invalid characters
        sanitized = self._replace_invalid_chars(filename)

        # Handle reserved names (Windows)
        sanitized = self._handle_reserved_names(sanitized)

        # Remove leading/trailing dots and spaces
        sanitized = self._clean_edges(sanitized)

        # Truncate if too long
        sanitized = self._truncate_filename(sanitized)

        # Ensure filename is not empty
        if not sanitized:
            sanitized = "renamed_file"

        return sanitized

    def validate_path(self, path: str) -> tuple[bool, list[str]]:
        """Validate a full file path.

        Args:
            path: Full path to validate

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Check path length
        if len(path) > self.config.max_path_length:
            issues.append(f"Path exceeds maximum length ({self.config.max_path_length} chars)")

        # Check filename length
        filename = Path(path).name
        if len(filename) > self.config.max_filename_length:
            issues.append(f"Filename exceeds maximum length ({self.config.max_filename_length} chars)")

        # Check for invalid characters in filename
        invalid_found = self._find_invalid_chars(filename)
        if invalid_found:
            issues.append(f"Filename contains invalid characters: {invalid_found}")

        # Platform-specific checks
        if self.platform == "windows":
            # Check for reserved names
            name_without_ext = Path(filename).stem.upper()
            if name_without_ext in self.reserved_names:
                issues.append(f"Filename uses reserved name: {name_without_ext}")

            # Check for trailing dots or spaces
            if filename.endswith((".", " ")):
                issues.append("Filename ends with dot or space (invalid on Windows)")

        return len(issues) == 0, issues

    def check_conflicts(self, proposed_path: str, existing_paths: set[str]) -> list[str]:
        """Check for naming conflicts.

        Args:
            proposed_path: Proposed file path
            existing_paths: Set of existing file paths

        Returns:
            List of conflict descriptions
        """
        conflicts = []

        # Direct conflict
        if proposed_path in existing_paths:
            conflicts.append("File already exists with this exact name")

        # Case-insensitive conflict (important for Windows/macOS)
        if self.platform in ["windows", "darwin"]:  # darwin is macOS
            proposed_lower = proposed_path.lower()
            for existing in existing_paths:
                if existing.lower() == proposed_lower and existing != proposed_path:
                    conflicts.append(f"Case-insensitive conflict with: {Path(existing).name}")
                    break

        return conflicts

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        # Normalize to NFC (Canonical Decomposition, followed by Canonical Composition)
        normalized = unicodedata.normalize("NFC", text)

        # Replace some common Unicode characters with ASCII equivalents
        replacements = {
            "'": "'",  # Right single quotation mark
            """: '"',  # Left double quotation mark
            """: '"',  # Right double quotation mark
            "\u2013": "-",  # EN DASH
            "—": "-",
            "…": "...",
            "™": "TM",
            "®": "(R)",
            "©": "(C)",
        }

        for unicode_char, ascii_char in replacements.items():
            normalized = normalized.replace(unicode_char, ascii_char)

        return normalized

    def _replace_invalid_chars(self, filename: str) -> str:
        """Replace invalid characters in filename.

        Args:
            filename: Original filename

        Returns:
            Filename with invalid characters replaced
        """
        result = []
        for char in filename:
            if char in self.invalid_chars or ord(char) < 32:
                result.append(self.config.replacement_char)
            else:
                result.append(char)
        return "".join(result)

    def _handle_reserved_names(self, filename: str) -> str:
        """Handle reserved names (Windows).

        Args:
            filename: Filename to check

        Returns:
            Modified filename if it was reserved
        """
        if not self.reserved_names:
            return filename

        # Check without extension
        name_parts = filename.rsplit(".", 1)
        name_without_ext = name_parts[0]
        extension = name_parts[1] if len(name_parts) > 1 else ""

        if name_without_ext.upper() in self.reserved_names:
            # Append underscore to make it non-reserved
            name_without_ext = name_without_ext + "_"
            return f"{name_without_ext}.{extension}" if extension else name_without_ext

        return filename

    def _clean_edges(self, filename: str) -> str:
        """Remove leading/trailing dots and spaces.

        Args:
            filename: Filename to clean

        Returns:
            Cleaned filename
        """
        # Remove leading dots and spaces
        filename = filename.lstrip(". ")

        # Remove trailing spaces (but keep one trailing dot for extension)
        filename = filename.rstrip(" ")

        # Remove trailing dots except for the extension separator
        if "." in filename:
            parts = filename.rsplit(".", 1)
            parts[0] = parts[0].rstrip(".")
            filename = ".".join(parts)
        else:
            filename = filename.rstrip(".")

        return filename

    def _truncate_filename(self, filename: str) -> str:
        """Truncate filename if too long.

        Args:
            filename: Filename to check

        Returns:
            Truncated filename if necessary
        """
        if len(filename) <= self.config.max_filename_length:
            return filename

        # Try to preserve extension
        if "." in filename:
            name_parts = filename.rsplit(".", 1)
            name = name_parts[0]
            extension = name_parts[1]

            # Calculate how much we can keep of the name
            max_name_length = self.config.max_filename_length - len(extension) - 1  # -1 for dot

            if max_name_length > 0:
                return f"{name[:max_name_length]}.{extension}"

        # If no extension or still too long, just truncate
        return filename[: self.config.max_filename_length]

    def _find_invalid_chars(self, filename: str) -> set[str]:
        """Find invalid characters in a filename.

        Args:
            filename: Filename to check

        Returns:
            Set of invalid characters found
        """
        found = set()
        for char in filename:
            if char in self.invalid_chars or ord(char) < 32:
                found.add(char)
        return found
