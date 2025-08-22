"""Updated unit tests for File Rename Executor with new architecture."""

import uuid
from unittest.mock import MagicMock, Mock, patch

import pytest

from services.analysis_service.src.file_rename_executor.executor import FileRenameExecutor


class TestFileRenameExecutorUpdated:
    """Test suite for FileRenameExecutor with new architecture."""

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

    def test_execute_rename_success(self, executor):
        """Test successful file rename execution."""
        proposal_id = uuid.uuid4()

        # Setup mocks
        session = Mock()
        proposal = Mock()
        proposal.original_path = "/tmp/old.ogg"
        proposal.full_proposed_path = "/tmp/new.ogg"

        recording = Mock()

        # Setup transaction manager context
        context_manager = MagicMock()
        context_manager.__enter__.return_value = (session, proposal, recording)
        context_manager.__exit__.return_value = None
        executor.transaction_manager.atomic_rename.return_value = context_manager
        executor.transaction_manager.validate_rename_preconditions.return_value = (True, None)

        # Setup metadata preserver
        metadata = {"tags": {"title": "Test"}}
        executor.metadata_preserver.snapshot_metadata.return_value = metadata
        executor.metadata_preserver.restore_metadata.return_value = True
        executor.metadata_preserver.verify_metadata.return_value = True

        with patch("pathlib.Path.mkdir"), patch("shutil.move") as mock_move:
            result, error = executor.execute_rename(proposal_id)

            assert result is True
            assert error is None

            # Verify move was called
            mock_move.assert_called_once_with("/tmp/old.ogg", "/tmp/new.ogg")

            # Verify metadata operations
            executor.metadata_preserver.snapshot_metadata.assert_called_once_with("/tmp/old.ogg")
            executor.metadata_preserver.restore_metadata.assert_called_once_with("/tmp/new.ogg", metadata)

            # Verify database updates
            executor.transaction_manager.update_recording_path.assert_called_once()
            executor.transaction_manager.update_proposal_status.assert_called_once_with(session, proposal, "applied")

    def test_execute_rename_validation_failure(self, executor):
        """Test rename fails when validation fails."""
        proposal_id = uuid.uuid4()

        # Setup mocks
        session = Mock()
        proposal = Mock()
        recording = Mock()

        context_manager = MagicMock()
        context_manager.__enter__.return_value = (session, proposal, recording)
        context_manager.__exit__.return_value = None
        executor.transaction_manager.atomic_rename.return_value = context_manager
        executor.transaction_manager.validate_rename_preconditions.return_value = (False, "Source file does not exist")

        result, error = executor.execute_rename(proposal_id)

        assert result is False
        assert error == "Source file does not exist"

    def test_execute_rename_file_move_failure(self, executor):
        """Test rename fails when file move fails."""
        proposal_id = uuid.uuid4()

        # Setup mocks
        session = Mock()
        proposal = Mock()
        proposal.original_path = "/tmp/old.ogg"
        proposal.full_proposed_path = "/tmp/new.ogg"

        recording = Mock()

        context_manager = MagicMock()
        context_manager.__enter__.return_value = (session, proposal, recording)
        context_manager.__exit__.return_value = None
        executor.transaction_manager.atomic_rename.return_value = context_manager
        executor.transaction_manager.validate_rename_preconditions.return_value = (True, None)

        with patch("pathlib.Path.mkdir"), patch("shutil.move", side_effect=OSError("Permission denied")):
            result, error = executor.execute_rename(proposal_id)

            assert result is False
            assert "Failed to rename file" in error

    def test_execute_rename_non_ogg_file(self, executor):
        """Test rename for non-OGG files."""
        proposal_id = uuid.uuid4()

        # Setup mocks
        session = Mock()
        proposal = Mock()
        proposal.original_path = "/tmp/old.mp3"  # Non-OGG file
        proposal.full_proposed_path = "/tmp/new.mp3"

        recording = Mock()

        context_manager = MagicMock()
        context_manager.__enter__.return_value = (session, proposal, recording)
        context_manager.__exit__.return_value = None
        executor.transaction_manager.atomic_rename.return_value = context_manager
        executor.transaction_manager.validate_rename_preconditions.return_value = (True, None)

        with patch("pathlib.Path.mkdir"), patch("shutil.move") as mock_move:
            result, error = executor.execute_rename(proposal_id)

            assert result is True
            assert error is None

            # Verify move was called
            mock_move.assert_called_once_with("/tmp/old.mp3", "/tmp/new.mp3")

            # Verify metadata operations were NOT called for non-OGG
            executor.metadata_preserver.snapshot_metadata.assert_not_called()

    def test_execute_rename_metadata_restore_failure(self, executor):
        """Test rename continues even if metadata restore fails."""
        proposal_id = uuid.uuid4()

        # Setup mocks
        session = Mock()
        proposal = Mock()
        proposal.original_path = "/tmp/old.ogg"
        proposal.full_proposed_path = "/tmp/new.ogg"

        recording = Mock()

        context_manager = MagicMock()
        context_manager.__enter__.return_value = (session, proposal, recording)
        context_manager.__exit__.return_value = None
        executor.transaction_manager.atomic_rename.return_value = context_manager
        executor.transaction_manager.validate_rename_preconditions.return_value = (True, None)

        # Setup metadata preserver to fail restore
        metadata = {"tags": {"title": "Test"}}
        executor.metadata_preserver.snapshot_metadata.return_value = metadata
        executor.metadata_preserver.restore_metadata.return_value = False  # Fail restore

        with (
            patch("pathlib.Path.mkdir"),
            patch("shutil.move"),
            patch("services.analysis_service.src.file_rename_executor.executor.logger") as mock_logger,
        ):
            result, error = executor.execute_rename(proposal_id)

            # Should still succeed even if metadata restore fails
            assert result is True
            assert error is None

            # Verify warning was logged
            mock_logger.warning.assert_called()

            # Verify database was still updated
            executor.transaction_manager.update_proposal_status.assert_called_once_with(session, proposal, "applied")

    def test_execute_rename_exception_handling(self, executor):
        """Test rename handles unexpected exceptions."""
        proposal_id = uuid.uuid4()

        # Setup transaction manager to raise exception
        executor.transaction_manager.atomic_rename.side_effect = Exception("Database error")

        result, error = executor.execute_rename(proposal_id)

        assert result is False
        assert "Database error" in error

    def test_rollback_rename_success(self, executor):
        """Test successful rollback operation."""
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

        # Setup metadata preserver
        metadata = {"tags": {"title": "Test"}}
        executor.metadata_preserver.snapshot_metadata.return_value = metadata
        executor.metadata_preserver.restore_metadata.return_value = True
        executor.metadata_preserver.verify_metadata.return_value = True

        with patch("pathlib.Path.mkdir"), patch("shutil.move") as mock_move:
            result, error = executor.rollback_rename(proposal_id)

            assert result is True
            assert error is None

            # Verify move was called
            mock_move.assert_called_once_with("/tmp/new.ogg", "/tmp/old.ogg")

            # Verify database updates
            executor.transaction_manager.update_recording_path.assert_called_once()
            executor.transaction_manager.update_proposal_status.assert_called_once_with(
                session, proposal, "rolled_back"
            )

    def test_architecture_integration(self, mock_db_manager):
        """Test that new architecture components are properly integrated."""
        with (
            patch("services.analysis_service.src.file_rename_executor.executor.TransactionManager") as mock_tm,
            patch("services.analysis_service.src.file_rename_executor.executor.MetadataPreserver") as mock_mp,
        ):
            executor = FileRenameExecutor(mock_db_manager)

            # Verify components are created
            mock_tm.assert_called_once_with(mock_db_manager)
            mock_mp.assert_called_once()

            # Verify they're accessible
            assert executor.transaction_manager is not None
            assert executor.metadata_preserver is not None
            assert executor.db_manager is mock_db_manager
