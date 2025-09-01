"""Unit tests for File Rename Executor."""

import uuid
from unittest.mock import MagicMock, Mock, patch

import pytest

from services.analysis_service.src.file_rename_executor import FileRenameExecutor
from shared.core_types.src.models import Recording, RenameProposal


class TestFileRenameExecutor:
    """Test suite for FileRenameExecutor."""

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
    def executor(self, mock_db_manager):
        """Create a FileRenameExecutor instance with mock db."""
        db_manager, _ = mock_db_manager
        return FileRenameExecutor(db_manager)

    def test_execute_rename_success(self, executor, mock_db_manager):
        """Test successful file rename execution."""
        _, session = mock_db_manager
        proposal_id = uuid.uuid4()
        recording_id = uuid.uuid4()

        # Create mock proposal and recording
        proposal = Mock(spec=RenameProposal)
        proposal.id = proposal_id
        proposal.status = "approved"
        proposal.recording_id = recording_id
        proposal.original_path = "/tmp/old_file.ogg"
        proposal.original_filename = "old_file.ogg"
        proposal.full_proposed_path = "/tmp/new_file.ogg"
        proposal.proposed_filename = "new_file.ogg"

        recording = Mock(spec=Recording)
        recording.id = recording_id
        recording.file_path = "/tmp/old_file.ogg"
        recording.file_name = "old_file.ogg"

        session.query.return_value.filter_by.return_value.first.side_effect = [
            proposal,
            recording,
        ]

        with (
            patch("pathlib.Path.exists") as mock_exists,
            patch("pathlib.Path.mkdir"),
            patch("shutil.move") as mock_move,
            patch.object(executor, "preserve_metadata", return_value={"tags": {"title": "Test"}}),
            patch.object(executor, "_restore_metadata", return_value=True),
        ):
            # Setup path existence checks
            mock_exists.side_effect = [True, False]  # source exists, dest doesn't

            success, error = executor.execute_rename(proposal_id)

            assert success is True
            assert error is None
            mock_move.assert_called_once_with("/tmp/old_file.ogg", "/tmp/new_file.ogg")
            assert proposal.status == "applied"
            assert recording.file_path == "/tmp/new_file.ogg"
            assert recording.file_name == "new_file.ogg"
            session.commit.assert_called_once()

    def test_execute_rename_proposal_not_found(self, executor, mock_db_manager):
        """Test rename execution with non-existent proposal."""
        _, session = mock_db_manager
        proposal_id = uuid.uuid4()

        session.query.return_value.filter_by.return_value.first.return_value = None

        success, error = executor.execute_rename(proposal_id)

        assert success is False
        assert f"Proposal {proposal_id} not found" in error

    def test_execute_rename_not_approved(self, executor, mock_db_manager):
        """Test rename execution with non-approved proposal."""
        _, session = mock_db_manager
        proposal_id = uuid.uuid4()

        proposal = Mock(spec=RenameProposal)
        proposal.status = "pending"

        session.query.return_value.filter_by.return_value.first.return_value = proposal

        success, error = executor.execute_rename(proposal_id)

        assert success is False
        assert "is not approved" in error

    def test_execute_rename_source_not_exists(self, executor, mock_db_manager):
        """Test rename execution when source file doesn't exist."""
        _, session = mock_db_manager
        proposal_id = uuid.uuid4()
        recording_id = uuid.uuid4()

        proposal = Mock(spec=RenameProposal)
        proposal.id = proposal_id
        proposal.status = "approved"
        proposal.recording_id = recording_id
        proposal.original_path = "/tmp/nonexistent.ogg"

        recording = Mock(spec=Recording)
        recording.id = recording_id

        session.query.return_value.filter_by.return_value.first.side_effect = [
            proposal,
            recording,
        ]

        with patch("pathlib.Path.exists", return_value=False):
            success, error = executor.execute_rename(proposal_id)

            assert success is False
            assert "Source file does not exist" in error

    def test_execute_rename_destination_exists(self, executor, mock_db_manager):
        """Test rename execution when destination file already exists."""
        _, session = mock_db_manager
        proposal_id = uuid.uuid4()
        recording_id = uuid.uuid4()

        proposal = Mock(spec=RenameProposal)
        proposal.id = proposal_id
        proposal.status = "approved"
        proposal.recording_id = recording_id
        proposal.original_path = "/tmp/old_file.ogg"
        proposal.full_proposed_path = "/tmp/new_file.ogg"

        recording = Mock(spec=Recording)
        recording.id = recording_id

        session.query.return_value.filter_by.return_value.first.side_effect = [
            proposal,
            recording,
        ]

        with patch("pathlib.Path.exists", return_value=True):
            success, error = executor.execute_rename(proposal_id)

            assert success is False
            assert "Destination file already exists" in error

    def test_preserve_metadata_ogg_file(self, executor):
        """Test metadata preservation for OGG files."""
        with patch("services.analysis_service.src.file_rename_executor.executor.MutagenFile") as mock_file:
            mock_audio = MagicMock()
            mock_audio.tags = {"title": ["Test Song"], "artist": ["Test Artist"]}
            mock_audio.info.length = 180.5
            mock_audio.info.bitrate = 192000
            mock_audio.info.sample_rate = 44100
            mock_audio.info.channels = 2
            mock_audio.info.bitrate_nominal = 192000

            mock_file.return_value = mock_audio
            # Make isinstance check work
            with patch("services.analysis_service.src.file_rename_executor.executor.isinstance") as mock_isinstance:
                mock_isinstance.return_value = True

                metadata = executor.preserve_metadata("/tmp/test.ogg")

                assert metadata is not None
                assert "tags" in metadata
                assert metadata["tags"]["title"] == ["Test Song"]
                assert metadata["info"]["bitrate"] == 192000

    def test_preserve_metadata_non_ogg_file(self, executor):
        """Test metadata preservation returns None for non-OGG files."""
        with patch("services.analysis_service.src.file_rename_executor.executor.MutagenFile") as mock_file:
            mock_audio = MagicMock()
            mock_file.return_value = mock_audio

            with patch("services.analysis_service.src.file_rename_executor.executor.isinstance") as mock_isinstance:
                mock_isinstance.return_value = False

                metadata = executor.preserve_metadata("/tmp/test.mp3")

                assert metadata is None

    def test_restore_metadata_success(self, executor):
        """Test successful metadata restoration."""
        metadata = {"tags": {"title": ["Test Song"], "artist": ["Test Artist"]}}

        with patch("services.analysis_service.src.file_rename_executor.executor.OggVorbis") as mock_ogg:
            mock_audio = MagicMock()
            mock_ogg.return_value = mock_audio

            result = executor._restore_metadata("/tmp/test.ogg", metadata)

            assert result is True
            mock_audio.save.assert_called_once()

    def test_restore_metadata_failure(self, executor):
        """Test metadata restoration failure handling."""
        metadata = {"tags": {"title": ["Test"]}}

        with patch(
            "services.analysis_service.src.file_rename_executor.executor.OggVorbis",
            side_effect=Exception("Test error"),
        ):
            result = executor._restore_metadata("/tmp/test.ogg", metadata)

            assert result is False

    def test_rollback_rename_success(self, executor, mock_db_manager):
        """Test successful rename rollback."""
        _, session = mock_db_manager
        proposal_id = uuid.uuid4()
        recording_id = uuid.uuid4()

        proposal = Mock(spec=RenameProposal)
        proposal.id = proposal_id
        proposal.status = "applied"
        proposal.recording_id = recording_id
        proposal.original_path = "/tmp/old_file.ogg"
        proposal.original_filename = "old_file.ogg"
        proposal.full_proposed_path = "/tmp/new_file.ogg"

        recording = Mock(spec=Recording)
        recording.id = recording_id
        recording.file_path = "/tmp/new_file.ogg"
        recording.file_name = "new_file.ogg"

        session.query.return_value.filter_by.return_value.first.side_effect = [
            proposal,
            recording,
        ]

        with (
            patch("pathlib.Path.exists") as mock_exists,
            patch("pathlib.Path.mkdir"),
            patch("shutil.move") as mock_move,
            patch.object(executor, "preserve_metadata", return_value={"tags": {"title": "Test"}}),
            patch.object(executor, "_restore_metadata", return_value=True),
        ):
            # Setup path existence checks
            mock_exists.side_effect = [True, False]  # current exists, original doesn't

            success, error = executor.rollback_rename(proposal_id)

            assert success is True
            assert error is None
            mock_move.assert_called_once_with("/tmp/new_file.ogg", "/tmp/old_file.ogg")
            assert proposal.status == "rolled_back"
            assert recording.file_path == "/tmp/old_file.ogg"
            assert recording.file_name == "old_file.ogg"
            session.commit.assert_called_once()

    def test_rollback_rename_not_applied(self, executor, mock_db_manager):
        """Test rollback of non-applied proposal."""
        _, session = mock_db_manager
        proposal_id = uuid.uuid4()

        proposal = Mock(spec=RenameProposal)
        proposal.status = "pending"

        session.query.return_value.filter_by.return_value.first.return_value = proposal

        success, error = executor.rollback_rename(proposal_id)

        assert success is False
        assert "has not been applied" in error

    def test_rollback_rename_current_file_not_exists(self, executor, mock_db_manager):
        """Test rollback when current file doesn't exist."""
        _, session = mock_db_manager
        proposal_id = uuid.uuid4()
        recording_id = uuid.uuid4()

        proposal = Mock(spec=RenameProposal)
        proposal.id = proposal_id
        proposal.status = "applied"
        proposal.recording_id = recording_id
        proposal.full_proposed_path = "/tmp/nonexistent.ogg"

        recording = Mock(spec=Recording)
        recording.id = recording_id

        session.query.return_value.filter_by.return_value.first.side_effect = [
            proposal,
            recording,
        ]

        with patch("pathlib.Path.exists", return_value=False):
            success, error = executor.rollback_rename(proposal_id)

            assert success is False
            assert "Current file does not exist" in error
