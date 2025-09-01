"""Backup management for CUE file editing."""

import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class BackupManager:
    """Manages backups of CUE files during editing operations."""

    def __init__(self, retention_limit: int = 5, enabled: bool = True):
        """Initialize backup manager.

        Args:
            retention_limit: Maximum number of backups to keep per file
            enabled: Whether automatic backups are enabled
        """
        self.retention_limit = retention_limit
        self.enabled = enabled

    def create_backup(self, filepath: Path) -> Path | None:
        """Create a backup of the specified file.

        Args:
            filepath: Path to file to backup

        Returns:
            Path to created backup file, or None if source doesn't exist or backups disabled
        """
        if not self.enabled:
            logger.debug("Backups disabled, skipping backup creation")
            return None

        if not filepath.exists():
            logger.warning(f"Cannot backup non-existent file: {filepath}")
            return None

        # Rotate existing backups if needed
        self._rotate_backups(filepath)

        # Find next available backup number after rotation
        backup_path = self._get_next_backup_path(filepath)

        # Create the backup
        shutil.copy2(filepath, backup_path)
        logger.info(f"Created backup: {backup_path}")

        return backup_path

    def restore_from_backup(self, filepath: Path, backup_number: int | None = None) -> bool:
        """Restore file from backup.

        Args:
            filepath: Original file path
            backup_number: Specific backup to restore (None for most recent)

        Returns:
            True if restored successfully
        """
        backup_path = self._get_backup_path(filepath, backup_number)

        if not backup_path.exists():
            logger.error(f"Backup not found: {backup_path}")
            return False

        # Create backup of current file before restoring
        if filepath.exists():
            timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
            temp_backup = filepath.with_suffix(f".{timestamp}.tmp")
            shutil.copy2(filepath, temp_backup)

        # Restore from backup
        shutil.copy2(backup_path, filepath)
        logger.info(f"Restored from backup: {backup_path}")

        return True

    def list_backups(self, filepath: Path) -> list[Path]:
        """List all backups for a file.

        Args:
            filepath: Original file path

        Returns:
            List of backup file paths, sorted newest first
        """
        backups = []

        # Check for unnumbered backup
        base_backup = filepath.with_suffix(filepath.suffix + ".bak")
        if base_backup.exists():
            backups.append(base_backup)

        # Check for numbered backups
        for i in range(1, self.retention_limit + 1):
            backup_path = filepath.with_suffix(f"{filepath.suffix}.bak.{i}")
            if backup_path.exists():
                backups.append(backup_path)

        return backups

    def cleanup_old_backups(self, filepath: Path) -> None:
        """Remove backups beyond retention limit.

        Args:
            filepath: Original file path
        """
        backups = self.list_backups(filepath)

        # Keep only up to retention_limit backups
        for backup in backups[self.retention_limit :]:
            backup.unlink()
            logger.info(f"Removed old backup: {backup}")

    def _get_backup_path(self, filepath: Path, number: int | None = None) -> Path:
        """Get path for a specific backup.

        Args:
            filepath: Original file path
            number: Backup number (None for most recent)

        Returns:
            Path to backup file
        """
        if number is None:
            return filepath.with_suffix(filepath.suffix + ".bak")
        return filepath.with_suffix(f"{filepath.suffix}.bak.{number}")

    def _get_next_backup_path(self, filepath: Path) -> Path:
        """Get path for the next backup to create.

        Args:
            filepath: Original file path

        Returns:
            Path for new backup
        """
        # First backup is unnumbered
        base_backup = filepath.with_suffix(filepath.suffix + ".bak")
        if not base_backup.exists():
            return base_backup

        # Find first available numbered slot
        for i in range(1, self.retention_limit):
            backup_path = filepath.with_suffix(f"{filepath.suffix}.bak.{i}")
            if not backup_path.exists():
                return backup_path

        # If all slots full, will overwrite oldest (highest number)
        return filepath.with_suffix(f"{filepath.suffix}.bak.{self.retention_limit}")

    def _rotate_backups(self, filepath: Path) -> None:
        """Rotate backups to make room for new one.

        Args:
            filepath: Original file path
        """
        # Check if we need to rotate
        base_backup = filepath.with_suffix(filepath.suffix + ".bak")
        if not base_backup.exists():
            return  # No rotation needed

        # Find highest numbered backup
        highest = 0
        for i in range(1, self.retention_limit + 1):
            if filepath.with_suffix(f"{filepath.suffix}.bak.{i}").exists():
                highest = i

        # If at limit, remove oldest
        if highest >= self.retention_limit - 1:
            oldest = filepath.with_suffix(f"{filepath.suffix}.bak.{self.retention_limit - 1}")
            if oldest.exists():
                oldest.unlink()
                logger.debug(f"Removed oldest backup: {oldest}")
            highest = self.retention_limit - 2

        # Rotate existing numbered backups (work backwards)
        for i in range(highest, 0, -1):
            source = filepath.with_suffix(f"{filepath.suffix}.bak.{i}")
            target = filepath.with_suffix(f"{filepath.suffix}.bak.{i + 1}")
            if source.exists():
                shutil.move(str(source), str(target))

        # Move base backup to .bak.1
        if base_backup.exists():
            target = filepath.with_suffix(f"{filepath.suffix}.bak.1")
            shutil.move(str(base_backup), str(target))
