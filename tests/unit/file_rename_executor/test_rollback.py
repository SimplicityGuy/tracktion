"""Unit tests for File Rename Executor Rollback Mechanism."""

import uuid
from unittest.mock import MagicMock, Mock, patch

import pytest

from services.analysis_service.src.file_rename_executor.executor import FileRenameExecutor


class TestRollbackMechanism:
    """Test suite for FileRenameExecutor rollback functionality."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager."""
        return Mock()

    @pytest.fixture
    def executor(self, mock_db_manager):
        """Create a FileRenameExecutor instance with mocked dependencies."""
        with (
            patch("services.analysis_service.src.file_rename_executor.executor.TransactionManager") as mock_tm,
            patch("services.analysis_service.src.file_rename_executor.executor.MetadataPreserver") as mock_mp,
        ):
            executor = FileRenameExecutor(mock_db_manager)
            executor.transaction_manager = mock_tm.return_value
            executor.metadata_preserver = mock_mp.return_value
            return executor

    def test_rollback_rename_success(self, executor):
        """Test successful rollback of a rename operation."""
        proposal_id = uuid.uuid4()

        # Setup mocks
        session = Mock()
        proposal = Mock()
        proposal.full_proposed_path = "/tmp/new.ogg"
        proposal.original_path = "/tmp/old.ogg"
        proposal.original_filename = "old.ogg"

        recording = Mock()

        # Setup transaction manager context
        context_manager = MagicMock()
        context_manager.__enter__.return_value = (session, proposal, recording)
        context_manager.__exit__.return_value = None
        executor.transaction_manager.atomic_rename.return_value = context_manager
        executor.transaction_manager.validate_rollback_preconditions.return_value = (True, None)

        # Setup metadata preserver
        metadata = {"tags": {"title": "Test"}}
        executor.metadata_preserver.snapshot_metadata.return_value = metadata
        executor.metadata_preserver.restore_metadata.return_value = True
        executor.metadata_preserver.verify_metadata.return_value = True

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("shutil.move") as mock_move,
        ):
            result, error = executor.rollback_rename(proposal_id)

            assert result is True
            assert error is None

            # Verify move was called
            mock_move.assert_called_once_with("/tmp/new.ogg", "/tmp/old.ogg")

            # Verify metadata operations
            executor.metadata_preserver.snapshot_metadata.assert_called_once_with("/tmp/new.ogg")
            executor.metadata_preserver.restore_metadata.assert_called_once_with("/tmp/old.ogg", metadata)

            # Verify database updates
            executor.transaction_manager.update_recording_path.assert_called_once()
            executor.transaction_manager.update_proposal_status.assert_called_once_with(
                session, proposal, "rolled_back"
            )

    def test_rollback_rename_validation_failure(self, executor):
        """Test rollback fails when validation fails."""
        proposal_id = uuid.uuid4()

        # Setup mocks
        session = Mock()
        proposal = Mock()
        recording = Mock()

        context_manager = MagicMock()
        context_manager.__enter__.return_value = (session, proposal, recording)
        context_manager.__exit__.return_value = None
        executor.transaction_manager.atomic_rename.return_value = context_manager
        executor.transaction_manager.validate_rollback_preconditions.return_value = (
            False,
            "Proposal not in applied state",
        )

        result, error = executor.rollback_rename(proposal_id)

        assert result is False
        assert error == "Proposal not in applied state"

    def test_rollback_rename_file_move_failure(self, executor):
        """Test rollback fails when file move fails."""
        proposal_id = uuid.uuid4()

        # Setup mocks
        session = Mock()
        proposal = Mock()
        proposal.full_proposed_path = "/tmp/new.ogg"
        proposal.original_path = "/tmp/old.ogg"

        recording = Mock()

        context_manager = MagicMock()
        context_manager.__enter__.return_value = (session, proposal, recording)
        context_manager.__exit__.return_value = None
        executor.transaction_manager.atomic_rename.return_value = context_manager
        executor.transaction_manager.validate_rollback_preconditions.return_value = (True, None)

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("shutil.move", side_effect=OSError("Permission denied")),
        ):
            result, error = executor.rollback_rename(proposal_id)

            assert result is False
            assert "Failed to rollback rename" in error

    def test_rollback_rename_non_ogg_file(self, executor):
        """Test rollback for non-OGG files."""
        proposal_id = uuid.uuid4()

        # Setup mocks
        session = Mock()
        proposal = Mock()
        proposal.full_proposed_path = "/tmp/new.mp3"  # Non-OGG file
        proposal.original_path = "/tmp/old.mp3"
        proposal.original_filename = "old.mp3"

        recording = Mock()

        context_manager = MagicMock()
        context_manager.__enter__.return_value = (session, proposal, recording)
        context_manager.__exit__.return_value = None
        executor.transaction_manager.atomic_rename.return_value = context_manager
        executor.transaction_manager.validate_rollback_preconditions.return_value = (True, None)

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("shutil.move") as mock_move,
        ):
            result, error = executor.rollback_rename(proposal_id)

            assert result is True
            assert error is None

            # Verify move was called
            mock_move.assert_called_once_with("/tmp/new.mp3", "/tmp/old.mp3")

            # Verify metadata operations were NOT called for non-OGG
            executor.metadata_preserver.snapshot_metadata.assert_not_called()

    def test_rollback_rename_metadata_restore_failure(self, executor):
        """Test rollback continues even if metadata restore fails."""
        proposal_id = uuid.uuid4()

        # Setup mocks
        session = Mock()
        proposal = Mock()
        proposal.full_proposed_path = "/tmp/new.ogg"
        proposal.original_path = "/tmp/old.ogg"
        proposal.original_filename = "old.ogg"

        recording = Mock()

        context_manager = MagicMock()
        context_manager.__enter__.return_value = (session, proposal, recording)
        context_manager.__exit__.return_value = None
        executor.transaction_manager.atomic_rename.return_value = context_manager
        executor.transaction_manager.validate_rollback_preconditions.return_value = (True, None)

        # Setup metadata preserver to fail restore
        metadata = {"tags": {"title": "Test"}}
        executor.metadata_preserver.snapshot_metadata.return_value = metadata
        executor.metadata_preserver.restore_metadata.return_value = False  # Fail restore

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("shutil.move"),
            patch("services.analysis_service.src.file_rename_executor.executor.logger") as mock_logger,
        ):
            result, error = executor.rollback_rename(proposal_id)

            # Should still succeed even if metadata restore fails
            assert result is True
            assert error is None

            # Verify warning was logged
            mock_logger.warning.assert_called()

            # Verify database was still updated
            executor.transaction_manager.update_proposal_status.assert_called_once_with(
                session, proposal, "rolled_back"
            )

    def test_rollback_rename_exception_handling(self, executor):
        """Test rollback handles unexpected exceptions."""
        proposal_id = uuid.uuid4()

        # Setup transaction manager to raise exception
        executor.transaction_manager.atomic_rename.side_effect = Exception("Database error")

        result, error = executor.rollback_rename(proposal_id)

        assert result is False
        assert "Database error" in error

    def test_rollback_history_tracking(self, executor):
        """Test that rollback operations are properly tracked."""
        proposal_id = uuid.uuid4()

        # Setup mocks
        session = Mock()
        proposal = Mock()
        proposal.full_proposed_path = "/tmp/new.ogg"
        proposal.original_path = "/tmp/old.ogg"
        proposal.original_filename = "old.ogg"
        proposal.id = proposal_id

        recording = Mock()
        recording.id = uuid.uuid4()

        context_manager = MagicMock()
        context_manager.__enter__.return_value = (session, proposal, recording)
        context_manager.__exit__.return_value = None
        executor.transaction_manager.atomic_rename.return_value = context_manager
        executor.transaction_manager.validate_rollback_preconditions.return_value = (True, None)

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.mkdir"),
            patch("shutil.move"),
            patch("services.analysis_service.src.file_rename_executor.executor.logger") as mock_logger,
        ):
            result, error = executor.rollback_rename(proposal_id)

            assert result is True

            # Verify logging for history tracking
            mock_logger.info.assert_called_with("Successfully rolled back rename from /tmp/new.ogg to /tmp/old.ogg")

    def test_rollback_by_proposal_id(self, executor):
        """Test rollback can be executed by proposal ID."""
        proposal_id = uuid.uuid4()

        # Setup mocks
        session = Mock()
        proposal = Mock()
        proposal.id = proposal_id
        proposal.full_proposed_path = "/tmp/new.ogg"
        proposal.original_path = "/tmp/old.ogg"
        proposal.original_filename = "old.ogg"

        recording = Mock()

        context_manager = MagicMock()
        context_manager.__enter__.return_value = (session, proposal, recording)
        context_manager.__exit__.return_value = None
        executor.transaction_manager.atomic_rename.return_value = context_manager
        executor.transaction_manager.validate_rollback_preconditions.return_value = (True, None)

        with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.mkdir"), patch("shutil.move"):
            result, error = executor.rollback_rename(proposal_id)

            assert result is True

            # Verify atomic_rename was called with the correct proposal_id
            executor.transaction_manager.atomic_rename.assert_called_once_with(proposal_id)
