"""Unit tests for Transaction Manager."""

import uuid
from unittest.mock import MagicMock, Mock, patch

import pytest
from sqlalchemy.exc import SQLAlchemyError

from services.analysis_service.src.file_rename_executor.transaction_manager import TransactionManager
from shared.core_types.src.models import Recording, RenameProposal


class TestTransactionManager:
    """Test suite for TransactionManager."""

    @pytest.fixture
    def mock_db_manager(self):
        """Create a mock database manager."""
        db_manager = Mock()
        session = MagicMock()
        context_manager = MagicMock()
        context_manager.__enter__.return_value = session
        context_manager.__exit__.return_value = None
        db_manager.get_db_session.return_value = context_manager
        return db_manager, session

    @pytest.fixture
    def transaction_manager(self, mock_db_manager):
        """Create a TransactionManager instance with mock db."""
        db_manager, _ = mock_db_manager
        return TransactionManager(db_manager)

    def test_atomic_rename_success(self, transaction_manager, mock_db_manager):
        """Test successful atomic rename transaction."""
        _, session = mock_db_manager
        proposal_id = uuid.uuid4()
        recording_id = uuid.uuid4()

        # Create mock proposal and recording
        proposal = Mock(spec=RenameProposal)
        proposal.id = proposal_id
        proposal.recording_id = recording_id
        proposal.status = "approved"
        proposal.original_path = "/tmp/old.ogg"
        proposal.original_filename = "old.ogg"
        proposal.full_proposed_path = "/tmp/new.ogg"

        recording = Mock(spec=Recording)
        recording.id = recording_id
        recording.file_path = "/tmp/old.ogg"
        recording.file_name = "old.ogg"

        session.query.return_value.filter_by.return_value.first.side_effect = [
            proposal,
            recording,
        ]

        # Use the atomic rename context manager
        with transaction_manager.atomic_rename(proposal_id) as (sess, prop, rec):
            assert sess == session
            assert prop == proposal
            assert rec == recording

            # Simulate successful rename
            rec.file_path = "/tmp/new.ogg"
            rec.file_name = "new.ogg"
            prop.status = "applied"

        # Verify commit was called
        session.commit.assert_called_once()
        session.rollback.assert_not_called()

    def test_atomic_rename_failure_rollback(self, transaction_manager, mock_db_manager):
        """Test atomic rename rollback on failure."""
        _, session = mock_db_manager
        proposal_id = uuid.uuid4()
        recording_id = uuid.uuid4()

        # Create mock proposal and recording
        proposal = Mock(spec=RenameProposal)
        proposal.id = proposal_id
        proposal.recording_id = recording_id
        proposal.status = "approved"
        proposal.original_path = "/tmp/old.ogg"
        proposal.original_filename = "old.ogg"

        recording = Mock(spec=Recording)
        recording.id = recording_id
        recording.file_path = "/tmp/old.ogg"
        recording.file_name = "old.ogg"

        session.query.return_value.filter_by.return_value.first.side_effect = [
            proposal,
            recording,
        ]

        # Use the atomic rename context manager with simulated failure
        with (
            pytest.raises(RuntimeError),
            transaction_manager.atomic_rename(proposal_id) as (_sess, prop, rec),
        ):
            # Simulate rename failure
            rec.file_path = "/tmp/new.ogg"
            rec.file_name = "new.ogg"
            prop.status = "applied"
            raise RuntimeError("Rename failed")

        # Verify rollback was called (may be called twice due to nested exception handling)
        assert session.rollback.called
        assert session.rollback.call_count >= 1
        session.commit.assert_not_called()

        # Verify values were restored
        assert recording.file_path == "/tmp/old.ogg"
        assert recording.file_name == "old.ogg"
        assert proposal.status == "approved"

    def test_atomic_rename_proposal_not_found(self, transaction_manager, mock_db_manager):
        """Test atomic rename with non-existent proposal."""
        _, session = mock_db_manager
        proposal_id = uuid.uuid4()

        session.query.return_value.filter_by.return_value.first.return_value = None

        with pytest.raises(ValueError) as exc_info, transaction_manager.atomic_rename(proposal_id):
            pass

        assert f"Proposal {proposal_id} not found" in str(exc_info.value)

    def test_atomic_rename_recording_not_found(self, transaction_manager, mock_db_manager):
        """Test atomic rename with non-existent recording."""
        _, session = mock_db_manager
        proposal_id = uuid.uuid4()
        recording_id = uuid.uuid4()

        proposal = Mock(spec=RenameProposal)
        proposal.id = proposal_id
        proposal.recording_id = recording_id

        session.query.return_value.filter_by.return_value.first.side_effect = [
            proposal,
            None,
        ]

        with pytest.raises(ValueError) as exc_info, transaction_manager.atomic_rename(proposal_id):
            pass

        assert f"Recording {recording_id} not found" in str(exc_info.value)

    def test_update_recording_path(self, transaction_manager):
        """Test updating recording path."""
        session = Mock()
        recording = Mock(spec=Recording)
        recording.id = uuid.uuid4()

        transaction_manager.update_recording_path(session, recording, "/new/path.ogg", "path.ogg")

        assert recording.file_path == "/new/path.ogg"
        assert recording.file_name == "path.ogg"

    def test_update_proposal_status(self, transaction_manager):
        """Test updating proposal status."""
        session = Mock()
        proposal = Mock(spec=RenameProposal)
        proposal.id = uuid.uuid4()

        transaction_manager.update_proposal_status(session, proposal, "applied")

        assert proposal.status == "applied"

    def test_store_rollback_info(self, transaction_manager):
        """Test storing rollback information."""
        session = Mock()
        proposal = Mock(spec=RenameProposal)
        proposal.id = uuid.uuid4()
        proposal.original_path = "/tmp/old.ogg"
        proposal.original_filename = "old.ogg"

        # Test with matching values (no warning)
        with patch("services.analysis_service.src.file_rename_executor.transaction_manager.logger") as mock_logger:
            transaction_manager.store_rollback_info(session, proposal, "/tmp/old.ogg", "old.ogg")
            mock_logger.warning.assert_not_called()

        # Test with mismatched values (should log warning)
        with patch("services.analysis_service.src.file_rename_executor.transaction_manager.logger") as mock_logger:
            transaction_manager.store_rollback_info(session, proposal, "/tmp/different.ogg", "different.ogg")
            assert mock_logger.warning.call_count == 2

    def test_validate_rename_preconditions_success(self, transaction_manager):
        """Test successful rename precondition validation."""
        session = Mock()
        proposal = Mock(spec=RenameProposal)
        proposal.status = "approved"
        proposal.original_path = "/tmp/old.ogg"
        proposal.full_proposed_path = "/tmp/new.ogg"

        recording = Mock(spec=Recording)
        recording.file_path = "/tmp/old.ogg"

        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.side_effect = [True, False]  # source exists, dest doesn't

            valid, error = transaction_manager.validate_rename_preconditions(session, proposal, recording)

            assert valid is True
            assert error is None

    def test_validate_rename_preconditions_wrong_status(self, transaction_manager):
        """Test rename validation with wrong proposal status."""
        session = Mock()
        proposal = Mock(spec=RenameProposal)
        proposal.status = "pending"

        recording = Mock(spec=Recording)

        valid, error = transaction_manager.validate_rename_preconditions(session, proposal, recording)

        assert valid is False
        assert "expected 'approved'" in error

    def test_validate_rename_preconditions_path_mismatch(self, transaction_manager):
        """Test rename validation with path mismatch."""
        session = Mock()
        proposal = Mock(spec=RenameProposal)
        proposal.status = "approved"
        proposal.original_path = "/tmp/old.ogg"

        recording = Mock(spec=Recording)
        recording.file_path = "/tmp/different.ogg"

        valid, error = transaction_manager.validate_rename_preconditions(session, proposal, recording)

        assert valid is False
        assert "Recording path mismatch" in error

    def test_validate_rename_preconditions_source_not_exists(self, transaction_manager):
        """Test rename validation when source doesn't exist."""
        session = Mock()
        proposal = Mock(spec=RenameProposal)
        proposal.status = "approved"
        proposal.original_path = "/tmp/old.ogg"
        proposal.full_proposed_path = "/tmp/new.ogg"

        recording = Mock(spec=Recording)
        recording.file_path = "/tmp/old.ogg"

        with patch("pathlib.Path.exists", return_value=False):
            valid, error = transaction_manager.validate_rename_preconditions(session, proposal, recording)

            assert valid is False
            assert "Source file does not exist" in error

    def test_validate_rename_preconditions_dest_exists(self, transaction_manager):
        """Test rename validation when destination exists."""
        session = Mock()
        proposal = Mock(spec=RenameProposal)
        proposal.status = "approved"
        proposal.original_path = "/tmp/old.ogg"
        proposal.full_proposed_path = "/tmp/new.ogg"

        recording = Mock(spec=Recording)
        recording.file_path = "/tmp/old.ogg"

        with patch("pathlib.Path.exists", return_value=True):
            valid, error = transaction_manager.validate_rename_preconditions(session, proposal, recording)

            assert valid is False
            assert "Destination already exists" in error

    def test_validate_rollback_preconditions_success(self, transaction_manager):
        """Test successful rollback precondition validation."""
        session = Mock()
        proposal = Mock(spec=RenameProposal)
        proposal.status = "applied"
        proposal.full_proposed_path = "/tmp/new.ogg"
        proposal.original_path = "/tmp/old.ogg"

        recording = Mock(spec=Recording)
        recording.file_path = "/tmp/new.ogg"

        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.side_effect = [True, False]  # current exists, original doesn't

            valid, error = transaction_manager.validate_rollback_preconditions(session, proposal, recording)

            assert valid is True
            assert error is None

    def test_validate_rollback_preconditions_wrong_status(self, transaction_manager):
        """Test rollback validation with wrong proposal status."""
        session = Mock()
        proposal = Mock(spec=RenameProposal)
        proposal.status = "pending"

        recording = Mock(spec=Recording)

        valid, error = transaction_manager.validate_rollback_preconditions(session, proposal, recording)

        assert valid is False
        assert "expected 'applied'" in error

    def test_atomic_rename_database_error(self, transaction_manager, mock_db_manager):
        """Test atomic rename with database error."""
        _, session = mock_db_manager
        proposal_id = uuid.uuid4()

        # Simulate database error
        session.query.side_effect = SQLAlchemyError("Database error")

        with pytest.raises(SQLAlchemyError), transaction_manager.atomic_rename(proposal_id):
            pass

        session.rollback.assert_called()
