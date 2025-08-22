"""Transaction manager for atomic file rename operations."""

import logging
from contextlib import contextmanager
from typing import Optional, Tuple, Generator, Any
from uuid import UUID
from pathlib import Path

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from shared.core_types.src.database import DatabaseManager
from shared.core_types.src.models import RenameProposal, Recording

logger = logging.getLogger(__name__)


class TransactionManager:
    """Manages database transactions for file rename operations."""

    def __init__(self, db_manager: DatabaseManager):
        """Initialize the transaction manager.

        Args:
            db_manager: Database manager for handling database operations
        """
        self.db_manager = db_manager

    @contextmanager
    def atomic_rename(self, proposal_id: UUID) -> Generator[Tuple[Session, Any, Any], None, None]:
        """Context manager for atomic rename operations.

        Provides a transactional context that ensures database updates
        are only committed if the file rename succeeds.

        Args:
            proposal_id: The UUID of the rename proposal

        Yields:
            Tuple of (session, proposal, recording)

        Raises:
            ValueError: If proposal or recording not found
            SQLAlchemyError: If database operation fails
        """
        session = None
        try:
            # Start a new session
            with self.db_manager.get_db_session() as session:
                # Get the proposal
                proposal = session.query(RenameProposal).filter_by(id=proposal_id).first()
                if not proposal:
                    raise ValueError(f"Proposal {proposal_id} not found")

                # Get the recording
                recording = session.query(Recording).filter_by(id=proposal.recording_id).first()
                if not recording:
                    raise ValueError(f"Recording {proposal.recording_id} not found")

                # Store original values for potential rollback
                original_file_path = recording.file_path
                original_file_name = recording.file_name
                original_status = proposal.status

                try:
                    # Yield control to perform the rename
                    yield session, proposal, recording

                    # If we get here, the rename succeeded - commit the transaction
                    session.commit()
                    logger.info(f"Successfully committed rename transaction for proposal {proposal_id}")

                except Exception as e:
                    # Rename failed - rollback the transaction
                    logger.error(f"Rename operation failed, rolling back transaction: {e}")
                    session.rollback()

                    # Restore original values
                    recording.file_path = original_file_path
                    recording.file_name = original_file_name
                    proposal.status = original_status

                    # Re-raise the exception
                    raise

        except SQLAlchemyError as e:
            logger.error(f"Database error in atomic rename: {e}")
            if session:
                session.rollback()
            raise
        except Exception as e:
            logger.error(f"Unexpected error in atomic rename: {e}")
            if session:
                session.rollback()
            raise

    def update_recording_path(self, session: Session, recording: Recording, new_path: str, new_filename: str) -> None:
        """Update recording path in database.

        Args:
            session: Database session
            recording: Recording to update
            new_path: New file path
            new_filename: New filename
        """
        recording.file_path = new_path
        recording.file_name = new_filename
        logger.debug(f"Updated recording {recording.id} path to {new_path}")

    def update_proposal_status(self, session: Session, proposal: RenameProposal, status: str) -> None:
        """Update proposal status in database.

        Args:
            session: Database session
            proposal: Proposal to update
            status: New status ('applied', 'rolled_back', etc.)
        """
        proposal.status = status
        logger.debug(f"Updated proposal {proposal.id} status to {status}")

    def store_rollback_info(
        self, session: Session, proposal: RenameProposal, original_path: str, original_filename: str
    ) -> None:
        """Store information needed for rollback.

        The original path and filename are already stored in the proposal,
        but this method ensures they are not overwritten.

        Args:
            session: Database session
            proposal: Proposal to update
            original_path: Original file path
            original_filename: Original filename
        """
        # Verify original path info is preserved
        if proposal.original_path != original_path:
            logger.warning(
                f"Original path mismatch for proposal {proposal.id}: "
                f"stored={proposal.original_path}, actual={original_path}"
            )
        if proposal.original_filename != original_filename:
            logger.warning(
                f"Original filename mismatch for proposal {proposal.id}: "
                f"stored={proposal.original_filename}, actual={original_filename}"
            )

    def validate_rename_preconditions(
        self, session: Session, proposal: RenameProposal, recording: Recording
    ) -> Tuple[bool, Optional[str]]:
        """Validate preconditions for rename operation.

        Args:
            session: Database session
            proposal: Rename proposal
            recording: Recording to rename

        Returns:
            Tuple of (valid, error_message)
        """
        # Check proposal status
        if proposal.status != "approved":
            return False, f"Proposal status is '{proposal.status}', expected 'approved'"

        # Check that recording path matches proposal's original path
        if recording.file_path != proposal.original_path:
            return False, (
                f"Recording path mismatch: recording={recording.file_path}, proposal={proposal.original_path}"
            )

        # Check that the source file exists
        source_path = Path(proposal.original_path)
        if not source_path.exists():
            return False, f"Source file does not exist: {proposal.original_path}"

        # Check that destination doesn't exist
        dest_path = Path(proposal.full_proposed_path)
        if dest_path.exists():
            return False, f"Destination already exists: {proposal.full_proposed_path}"

        return True, None

    def validate_rollback_preconditions(
        self, session: Session, proposal: RenameProposal, recording: Recording
    ) -> Tuple[bool, Optional[str]]:
        """Validate preconditions for rollback operation.

        Args:
            session: Database session
            proposal: Rename proposal
            recording: Recording to rollback

        Returns:
            Tuple of (valid, error_message)
        """
        # Check proposal status
        if proposal.status != "applied":
            return False, f"Proposal status is '{proposal.status}', expected 'applied'"

        # Check that recording path matches proposal's new path
        if recording.file_path != proposal.full_proposed_path:
            return False, (
                f"Recording path mismatch: recording={recording.file_path}, proposal={proposal.full_proposed_path}"
            )

        # Check that the current file exists
        current_path = Path(proposal.full_proposed_path)
        if not current_path.exists():
            return False, f"Current file does not exist: {proposal.full_proposed_path}"

        # Check that original path doesn't exist
        original_path = Path(proposal.original_path)
        if original_path.exists():
            return False, f"Original path already exists: {proposal.original_path}"

        return True, None
