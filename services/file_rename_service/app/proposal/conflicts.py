"""Conflict resolution system for filename conflicts during rename operations."""

import logging
import re
import time
from pathlib import Path
from typing import Any, ClassVar

from .models import ConflictResolution

logger = logging.getLogger(__name__)


class FilenameConflictResolver:
    """Handles filename conflicts and provides resolution strategies."""

    # Invalid filename characters for cross-platform compatibility
    INVALID_CHARS: ClassVar[str] = r'[<>:"/\\|?*\x00-\x1f]'

    # Maximum filename length (accounting for filesystem limitations)
    MAX_FILENAME_LENGTH: ClassVar[int] = 255

    # Reserved Windows filenames
    RESERVED_NAMES: ClassVar[set[str]] = {
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

    def __init__(self) -> None:
        """Initialize the conflict resolver."""
        logger.debug("Initializing FilenameConflictResolver")

    def validate_filename(self, filename: str) -> bool:
        """
        Validate if a filename is safe and compatible across platforms.

        Args:
            filename: The filename to validate

        Returns:
            True if filename is valid, False otherwise
        """
        if not filename or not filename.strip():
            logger.warning("Empty or whitespace-only filename provided")
            return False

        # Remove path separators to get just the filename
        filename = Path(filename.strip()).name

        # Check length
        if len(filename) > self.MAX_FILENAME_LENGTH:
            logger.warning(f"Filename too long: {len(filename)} > {self.MAX_FILENAME_LENGTH}")
            return False

        # Check for invalid characters
        if re.search(self.INVALID_CHARS, filename):
            logger.warning(f"Filename contains invalid characters: {filename}")
            return False

        # Check for reserved names (Windows compatibility)
        name_without_ext = Path(filename).stem.upper()
        if name_without_ext in self.RESERVED_NAMES:
            logger.warning(f"Filename uses reserved name: {name_without_ext}")
            return False

        # Check for filenames starting or ending with spaces or dots
        if filename.startswith((" ", ".")) or filename.endswith((" ", ".")):
            logger.warning(f"Filename starts or ends with space/dot: '{filename}'")
            return False

        return True

    def check_duplicate(self, filename1: str, filename2: str) -> bool:
        """
        Check if two filenames would be considered duplicates.

        This performs case-insensitive comparison for better compatibility.

        Args:
            filename1: First filename to compare
            filename2: Second filename to compare

        Returns:
            True if filenames are considered duplicates, False otherwise
        """
        if not filename1 or not filename2:
            return False

        # Normalize filenames by removing paths and comparing case-insensitively
        name1 = Path(filename1.strip()).name.lower()
        name2 = Path(filename2.strip()).name.lower()

        is_duplicate = name1 == name2

        if is_duplicate:
            logger.debug(f"Duplicate detected: '{filename1}' == '{filename2}'")

        return is_duplicate

    def detect_conflicts(self, proposed_name: str, existing_files: list[str]) -> ConflictResolution | None:
        """
        Detect conflicts between a proposed filename and existing files.

        Args:
            proposed_name: The proposed new filename
            existing_files: List of existing filenames to check against

        Returns:
            ConflictResolution object if conflicts are found, None otherwise
        """
        if not self.validate_filename(proposed_name):
            logger.error(f"Invalid proposed filename: {proposed_name}")
            return ConflictResolution(
                strategy="skip",
                existing_file=proposed_name,
                proposed_action=f"Skip rename due to invalid filename: {proposed_name}",
            )

        # Check for exact duplicates
        for existing_file in existing_files:
            if self.check_duplicate(proposed_name, existing_file):
                logger.info(f"Conflict detected: '{proposed_name}' conflicts with '{existing_file}'")
                return ConflictResolution(
                    strategy="append_number",
                    existing_file=existing_file,
                    proposed_action=f"Append number to resolve conflict with existing file: {existing_file}",
                )

        # Check against filesystem if files exist
        proposed_path = Path(proposed_name)
        if proposed_path.exists():
            logger.info(f"Filesystem conflict detected: '{proposed_name}' already exists")
            return ConflictResolution(
                strategy="append_number",
                existing_file=proposed_name,
                proposed_action=f"Append number to resolve filesystem conflict with: {proposed_name}",
            )

        logger.debug(f"No conflicts detected for: {proposed_name}")
        return None

    def generate_unique_name(self, base_name: str, existing_files: list[str]) -> str:
        """
        Generate a unique filename by appending numbers if necessary.

        Args:
            base_name: The base filename to make unique
            existing_files: List of existing filenames to avoid

        Returns:
            A unique filename
        """
        if not base_name:
            raise ValueError("Base name cannot be empty")

        # Validate base name first
        if not self.validate_filename(base_name):
            raise ValueError(f"Invalid base filename: {base_name}")

        # Create a set for faster lookups (case-insensitive)
        existing_lower = {Path(f.strip()).name.lower() for f in existing_files if f}

        # Split name and extension
        path_obj = Path(base_name)
        name_part = path_obj.stem
        extension = path_obj.suffix

        # Check if base name is already unique
        if base_name.lower() not in existing_lower and not Path(base_name).exists():
            logger.debug(f"Base name is already unique: {base_name}")
            return base_name

        # Generate numbered variations
        counter = 1
        max_attempts = 9999  # Prevent infinite loops

        while counter <= max_attempts:
            # Create new filename with counter
            candidate = f"{name_part} ({counter}){extension}"

            # Validate the candidate
            if not self.validate_filename(candidate):
                logger.warning(f"Generated candidate filename is invalid: {candidate}")
                counter += 1
                continue

            # Check if this candidate is unique
            if candidate.lower() not in existing_lower and not Path(candidate).exists():
                logger.info(f"Generated unique name: {candidate}")
                return candidate

            counter += 1

        # Fallback: append timestamp if we can't find a unique name
        timestamp = str(int(time.time()))
        fallback = f"{name_part}_{timestamp}{extension}"

        logger.warning(f"Could not generate unique name after {max_attempts} attempts, using timestamp: {fallback}")
        return fallback

    def resolve_conflict(self, proposed_name: str, existing_files: list[str], strategy: str = "append_number") -> str:
        """
        Resolve a filename conflict using the specified strategy.

        Args:
            proposed_name: The proposed filename that has a conflict
            existing_files: List of existing filenames
            strategy: Resolution strategy ("append_number", "skip", "replace")

        Returns:
            The resolved filename

        Raises:
            ValueError: If strategy is unsupported or parameters are invalid
        """
        if not proposed_name:
            raise ValueError("Proposed name cannot be empty")

        if not isinstance(existing_files, list):
            raise ValueError("Existing files must be a list")

        valid_strategies = {"append_number", "skip", "replace"}
        if strategy not in valid_strategies:
            raise ValueError(f"Unsupported strategy: {strategy}. Valid strategies: {valid_strategies}")

        logger.info(f"Resolving conflict for '{proposed_name}' using strategy: {strategy}")

        # Validate the proposed name first
        if not self.validate_filename(proposed_name):
            logger.error(f"Cannot resolve conflict for invalid filename: {proposed_name}")
            raise ValueError(f"Invalid proposed filename: {proposed_name}")

        if strategy == "skip":
            logger.info(f"Skipping rename for: {proposed_name}")
            return ""  # Empty string indicates skipping

        if strategy == "replace":
            logger.warning(f"Using replace strategy for: {proposed_name}")
            return proposed_name  # Return as-is, caller handles replacement

        if strategy == "append_number":
            return self.generate_unique_name(proposed_name, existing_files)

        # This should never be reached due to validation above
        raise ValueError(f"Unexpected strategy: {strategy}")

    def batch_resolve_conflicts(
        self, proposed_names: list[str], existing_files: list[str], strategy: str = "append_number"
    ) -> dict[str, str]:
        """
        Resolve conflicts for multiple proposed filenames.

        Args:
            proposed_names: List of proposed filenames
            existing_files: List of existing filenames
            strategy: Resolution strategy to use for all conflicts

        Returns:
            Dictionary mapping original proposed names to resolved names
        """
        if not isinstance(proposed_names, list):
            raise ValueError("Proposed names must be a list")

        results: dict[str, str] = {}
        working_existing = existing_files.copy()

        logger.info(f"Batch resolving {len(proposed_names)} filenames with strategy: {strategy}")

        for proposed_name in proposed_names:
            try:
                # Check for conflict
                conflict = self.detect_conflicts(proposed_name, working_existing)

                if conflict is None:
                    # No conflict, use original name
                    resolved_name = proposed_name
                else:
                    # Resolve the conflict
                    resolved_name = self.resolve_conflict(proposed_name, working_existing, strategy)

                results[proposed_name] = resolved_name

                # Add resolved name to working list to avoid conflicts with subsequent names
                if resolved_name and resolved_name not in working_existing:
                    working_existing.append(resolved_name)

            except Exception as e:
                logger.error(f"Error resolving conflict for '{proposed_name}': {e}")
                results[proposed_name] = ""  # Mark as failed/skipped

        logger.info(f"Batch resolution complete: {len([r for r in results.values() if r])} successful")
        return results

    def get_conflict_summary(self, proposed_names: list[str], existing_files: list[str]) -> dict[str, Any]:
        """
        Get a summary of potential conflicts without resolving them.

        Args:
            proposed_names: List of proposed filenames to analyze
            existing_files: List of existing filenames to check against

        Returns:
            Dictionary containing conflict analysis summary
        """
        summary: dict[str, Any] = {
            "total_proposed": len(proposed_names),
            "conflicts_detected": 0,
            "invalid_filenames": 0,
            "conflict_details": [],
            "invalid_details": [],
        }

        for proposed_name in proposed_names:
            # Check validity
            if not self.validate_filename(proposed_name):
                summary["invalid_filenames"] += 1
                summary["invalid_details"].append(
                    {"filename": proposed_name, "reason": "Invalid filename format or characters"}
                )
                continue

            # Check for conflicts
            conflict = self.detect_conflicts(proposed_name, existing_files)
            if conflict is not None:
                summary["conflicts_detected"] += 1
                summary["conflict_details"].append(
                    {
                        "proposed_name": proposed_name,
                        "conflicting_file": conflict.existing_file,
                        "suggested_strategy": conflict.strategy,
                    }
                )

        logger.info(
            f"Conflict summary: {summary['conflicts_detected']} conflicts, "
            f"{summary['invalid_filenames']} invalid filenames out of {summary['total_proposed']} total"
        )

        return summary


# Convenience functions for direct usage
def detect_conflicts(proposed_name: str, existing_files: list[str]) -> ConflictResolution | None:
    """Convenience function to detect conflicts."""
    resolver = FilenameConflictResolver()
    return resolver.detect_conflicts(proposed_name, existing_files)


def resolve_conflict(proposed_name: str, existing_files: list[str], strategy: str = "append_number") -> str:
    """Convenience function to resolve a single conflict."""
    resolver = FilenameConflictResolver()
    return resolver.resolve_conflict(proposed_name, existing_files, strategy)


def check_duplicate(filename1: str, filename2: str) -> bool:
    """Convenience function to check for duplicates."""
    resolver = FilenameConflictResolver()
    return resolver.check_duplicate(filename1, filename2)


def generate_unique_name(base_name: str, existing_files: list[str]) -> str:
    """Convenience function to generate unique names."""
    resolver = FilenameConflictResolver()
    return resolver.generate_unique_name(base_name, existing_files)


def validate_filename(filename: str) -> bool:
    """Convenience function to validate filenames."""
    resolver = FilenameConflictResolver()
    return resolver.validate_filename(filename)
