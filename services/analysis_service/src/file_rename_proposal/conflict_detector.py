"""Conflict detection for file rename proposals."""

import logging
import os
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class ConflictDetector:
    """Detects conflicts in file rename proposals."""

    def __init__(self) -> None:
        """Initialize the conflict detector."""
        self.logger = logger

    def detect_conflicts(
        self, proposed_path: str, existing_paths: Set[str], other_proposals: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, List[str]]:
        """Detect conflicts for a proposed file path.

        Args:
            proposed_path: The proposed new file path
            existing_paths: Set of existing file paths in the directory
            other_proposals: List of other pending proposals (optional)

        Returns:
            Dictionary containing detected conflicts and warnings
        """
        conflicts: List[str] = []
        warnings: List[str] = []

        # Check for exact path collision
        if proposed_path in existing_paths:
            conflicts.append(f"File already exists: {proposed_path}")

        # Check for case-insensitive collision (important for Windows/macOS)
        proposed_lower = proposed_path.lower()
        for existing in existing_paths:
            if existing.lower() == proposed_lower and existing != proposed_path:
                conflicts.append(f"Case-insensitive collision with: {existing}")

        # Check for collision with other proposals
        if other_proposals:
            for proposal in other_proposals:
                if proposal.get("full_proposed_path") == proposed_path:
                    conflicts.append(f"Conflicts with another proposal for recording: {proposal.get('recording_id')}")

        # Check for directory traversal attempts
        if self._has_directory_traversal(proposed_path):
            conflicts.append("Path contains directory traversal patterns")

        # Check for hidden file creation
        filename = os.path.basename(proposed_path)
        if filename.startswith("."):
            warnings.append("Creates a hidden file")

        # Check for backup file patterns
        if self._is_backup_pattern(filename):
            warnings.append("Filename matches backup file pattern")

        # Check for temporary file patterns
        if self._is_temp_pattern(filename):
            warnings.append("Filename matches temporary file pattern")

        # Check path length
        if len(proposed_path) > 255:
            conflicts.append(f"Path exceeds maximum length: {len(proposed_path)} > 255")

        # Check for special system files
        if self._is_system_file(filename):
            conflicts.append(f"Conflicts with system file: {filename}")

        return {"conflicts": conflicts, "warnings": warnings}

    def detect_batch_conflicts(
        self, proposals: List[Dict[str, str]], directory_contents: Dict[str, Set[str]]
    ) -> Dict[str, Dict[str, List[str]]]:
        """Detect conflicts for a batch of proposals.

        Args:
            proposals: List of proposal dictionaries with 'recording_id' and 'full_proposed_path'
            directory_contents: Dictionary mapping directory paths to sets of existing files

        Returns:
            Dictionary mapping recording IDs to conflict/warning lists
        """
        results = {}

        # Build a map of all proposed paths for cross-checking
        proposed_paths: Dict[str, List[str]] = {}
        for proposal in proposals:
            path = proposal.get("full_proposed_path", "")
            if path:
                if path not in proposed_paths:
                    proposed_paths[path] = []
                proposed_paths[path].append(proposal.get("recording_id", ""))

        # Check each proposal
        for proposal in proposals:
            recording_id = proposal.get("recording_id", "")
            proposed_path = proposal.get("full_proposed_path", "")

            if not proposed_path:
                results[recording_id] = {"conflicts": ["Missing proposed path"], "warnings": []}
                continue

            # Get directory contents for this path
            directory = os.path.dirname(proposed_path)
            existing_files_in_dir = directory_contents.get(directory, set())

            # Convert to full paths for comparison
            existing_files = set()
            for filename in existing_files_in_dir:
                # If it's already a full path, use it; otherwise join with directory
                if os.path.isabs(filename):
                    existing_files.add(filename)
                else:
                    existing_files.add(os.path.join(directory, filename))

            # Find other proposals targeting the same path
            other_proposals = []
            if proposed_path in proposed_paths:
                for other_id in proposed_paths[proposed_path]:
                    if other_id != recording_id:
                        other_proposals.append({"recording_id": other_id, "full_proposed_path": proposed_path})

            # Detect conflicts
            conflicts_warnings = self.detect_conflicts(proposed_path, existing_files, other_proposals)

            results[recording_id] = conflicts_warnings

        return results

    def resolve_conflicts(self, proposed_path: str, conflicts: List[str]) -> Optional[str]:
        """Attempt to resolve conflicts by suggesting alternative paths.

        Args:
            proposed_path: The original proposed path
            conflicts: List of detected conflicts

        Returns:
            Alternative path suggestion or None if unresolvable
        """
        # Check if the main conflict is file existence
        has_existence_conflict = any("already exists" in c for c in conflicts)

        if has_existence_conflict:
            # Try adding a number suffix
            base, ext = os.path.splitext(proposed_path)
            for i in range(1, 100):
                alternative = f"{base}_{i}{ext}"

                # Quick check if this alternative might work
                # In practice, we'd need to re-run full conflict detection
                if not os.path.exists(alternative):
                    return alternative

        # Check for case conflicts
        has_case_conflict = any("case-insensitive" in c.lower() for c in conflicts)

        if has_case_conflict:
            # Try different casing strategies
            directory = os.path.dirname(proposed_path)
            filename = os.path.basename(proposed_path)

            # Try title case
            title_case = filename.title()
            if title_case != filename:
                return os.path.join(directory, title_case)

            # Try lower case
            lower_case = filename.lower()
            if lower_case != filename:
                return os.path.join(directory, lower_case)

        return None

    def _has_directory_traversal(self, path: str) -> bool:
        """Check if path contains directory traversal patterns.

        Args:
            path: Path to check

        Returns:
            True if path contains traversal patterns
        """
        # Check for parent directory references (but not just two dots in a filename)
        # Look for /../ or /.. at the end or .. at the beginning
        if "/../" in path or path.endswith("/..") or path.startswith("../"):
            return True

        return False

    def _is_backup_pattern(self, filename: str) -> bool:
        """Check if filename matches common backup patterns.

        Args:
            filename: Filename to check

        Returns:
            True if matches backup pattern
        """
        backup_patterns = [".bak", ".backup", ".old", ".orig", "~", ".save", ".tmp", ".temp"]

        filename_lower = filename.lower()

        # Check extensions
        for pattern in backup_patterns:
            if filename_lower.endswith(pattern):
                return True

        # Check prefixes
        if filename_lower.startswith("backup_") or filename_lower.startswith("copy_"):
            return True

        # Check for numbered backups
        import re

        if re.match(r".*\.\d{1,3}$", filename):
            return True

        return False

    def _is_temp_pattern(self, filename: str) -> bool:
        """Check if filename matches temporary file patterns.

        Args:
            filename: Filename to check

        Returns:
            True if matches temporary pattern
        """
        temp_patterns = [".tmp", ".temp", ".swp", ".lock", "~$", ".~lock"]

        filename_lower = filename.lower()

        # Check extensions and patterns
        for pattern in temp_patterns:
            if pattern in filename_lower:
                return True

        # Check prefixes
        if filename_lower.startswith("tmp_") or filename_lower.startswith("temp_"):
            return True

        return False

    def _is_system_file(self, filename: str) -> bool:
        """Check if filename is a system file.

        Args:
            filename: Filename to check

        Returns:
            True if it's a system file
        """
        system_files = {
            # Windows
            "desktop.ini",
            "thumbs.db",
            "pagefile.sys",
            "hiberfil.sys",
            # macOS
            ".ds_store",
            ".localized",
            ".spotlight-v100",
            ".trashes",
            # Linux
            ".directory",
            "lost+found",
            # Common
            ".git",
            ".svn",
            ".hg",
            ".bzr",
        }

        return filename.lower() in system_files

    def validate_rename_safety(self, original_path: str, proposed_path: str) -> Tuple[bool, List[str]]:
        """Validate if a rename operation is safe to perform.

        Args:
            original_path: Original file path
            proposed_path: Proposed new path

        Returns:
            Tuple of (is_safe, list of safety issues)
        """
        issues = []

        # Check if original file exists
        if not os.path.exists(original_path):
            issues.append(f"Original file does not exist: {original_path}")

        # Check if we're moving across filesystems
        original_mount = self._get_mount_point(original_path)
        proposed_mount = self._get_mount_point(proposed_path)

        if original_mount != proposed_mount:
            issues.append("Rename crosses filesystem boundaries")

        # Check permissions on target directory
        target_dir = os.path.dirname(proposed_path)
        if os.path.exists(target_dir) and not os.access(target_dir, os.W_OK):
            issues.append(f"No write permission for directory: {target_dir}")

        # Check if target is a directory
        if os.path.isdir(proposed_path):
            issues.append(f"Target is a directory: {proposed_path}")

        # Check for circular rename
        if os.path.abspath(original_path) == os.path.abspath(proposed_path):
            issues.append("Original and proposed paths are the same")

        is_safe = len(issues) == 0
        return is_safe, issues

    def _get_mount_point(self, path: str) -> str:
        """Get the mount point for a path.

        Args:
            path: Path to check

        Returns:
            Mount point path
        """
        path = os.path.abspath(path)
        while not os.path.ismount(path):
            path = os.path.dirname(path)
            if path == os.path.dirname(path):  # Reached root
                break
        return path
