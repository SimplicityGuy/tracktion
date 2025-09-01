"""File Rename Executor for performing actual file rename operations with metadata preservation."""

import logging
import shutil
from pathlib import Path
from uuid import UUID

from shared.core_types.src.database import DatabaseManager

from .metadata_preserver import MetadataPreserver
from .transaction_manager import TransactionManager

logger = logging.getLogger(__name__)


class FileRenameExecutor:
    """Executes file rename operations with metadata preservation and rollback support."""

    def __init__(self, db_manager: DatabaseManager):
        """Initialize the FileRenameExecutor.

        Args:
            db_manager: Database manager for handling database operations
        """
        self.db_manager = db_manager
        self.transaction_manager = TransactionManager(db_manager)
        self.metadata_preserver = MetadataPreserver()

    def execute_rename(self, proposal_id: UUID) -> tuple[bool, str | None]:
        """Execute a file rename based on an approved proposal.

        Args:
            proposal_id: The UUID of the rename proposal to execute

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Use transaction manager for atomic rename
            with self.transaction_manager.atomic_rename(proposal_id) as (
                session,
                proposal,
                recording,
            ):
                # Validate rename preconditions
                valid, error = self.transaction_manager.validate_rename_preconditions(session, proposal, recording)
                if not valid:
                    return False, error

                # Paths
                source_path = Path(proposal.original_path)
                dest_path = Path(proposal.full_proposed_path)

                # Ensure destination directory exists
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                # Preserve metadata for OGG files
                metadata_snapshot = None
                is_ogg = source_path.suffix.lower() in [".ogg", ".oga"]
                if is_ogg:
                    metadata_snapshot = self.metadata_preserver.snapshot_metadata(str(source_path))

                # Perform the rename
                try:
                    shutil.move(str(source_path), str(dest_path))
                except Exception as e:
                    logger.error(f"Failed to rename file: {e}")
                    raise RuntimeError(f"Failed to rename file: {e}") from e

                # Restore metadata for OGG files
                if is_ogg and metadata_snapshot:
                    success = self.metadata_preserver.restore_metadata(str(dest_path), metadata_snapshot)
                    if not success:
                        logger.warning("Failed to restore metadata after rename")
                        # Don't fail the rename, but log the issue

                    # Verify metadata was preserved
                    if not self.metadata_preserver.verify_metadata(str(source_path), str(dest_path), metadata_snapshot):
                        logger.warning("Metadata verification failed after rename")

                # Update database records
                self.transaction_manager.update_recording_path(session, recording, str(dest_path), dest_path.name)
                self.transaction_manager.update_proposal_status(session, proposal, "applied")

                # Store rollback info (for validation only, paths already in proposal)
                self.transaction_manager.store_rollback_info(session, proposal, str(source_path), source_path.name)

                logger.info(f"Successfully renamed file from {source_path} to {dest_path}")

                return True, None

        except Exception as e:
            logger.error(f"Error executing rename for proposal {proposal_id}: {e}")
            return False, str(e)

    def rollback_rename(self, proposal_id: UUID) -> tuple[bool, str | None]:
        """Rollback a previously executed rename operation.

        Args:
            proposal_id: The UUID of the rename proposal to rollback

        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Use transaction manager for atomic rollback
            with self.transaction_manager.atomic_rename(proposal_id) as (
                session,
                proposal,
                recording,
            ):
                # Validate rollback preconditions
                valid, error = self.transaction_manager.validate_rollback_preconditions(session, proposal, recording)
                if not valid:
                    return False, error

                # Paths
                current_path = Path(proposal.full_proposed_path)
                original_path = Path(proposal.original_path)

                # Ensure original directory exists
                original_path.parent.mkdir(parents=True, exist_ok=True)

                # Preserve metadata for OGG files
                metadata_snapshot = None
                is_ogg = current_path.suffix.lower() in [".ogg", ".oga"]
                if is_ogg:
                    metadata_snapshot = self.metadata_preserver.snapshot_metadata(str(current_path))

                # Perform the rollback rename
                try:
                    shutil.move(str(current_path), str(original_path))
                except Exception as e:
                    logger.error(f"Failed to rollback rename: {e}")
                    raise RuntimeError(f"Failed to rollback rename: {e}") from e

                # Restore metadata for OGG files
                if is_ogg and metadata_snapshot:
                    success = self.metadata_preserver.restore_metadata(str(original_path), metadata_snapshot)
                    if not success:
                        logger.warning("Failed to restore metadata during rollback")
                        # Don't fail the rollback, but log the issue

                    # Verify metadata was preserved
                    if not self.metadata_preserver.verify_metadata(
                        str(current_path), str(original_path), metadata_snapshot
                    ):
                        logger.warning("Metadata verification failed during rollback")

                # Update database records
                self.transaction_manager.update_recording_path(
                    session,
                    recording,
                    proposal.original_path,
                    proposal.original_filename,
                )
                self.transaction_manager.update_proposal_status(session, proposal, "rolled_back")

                logger.info(f"Successfully rolled back rename from {current_path} to {original_path}")

                return True, None

        except Exception as e:
            logger.error(f"Error rolling back rename for proposal {proposal_id}: {e}")
            return False, str(e)
