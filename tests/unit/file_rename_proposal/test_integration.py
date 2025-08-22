"""Unit tests for file rename proposal integration."""

from datetime import UTC
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from services.analysis_service.src.file_rename_proposal.config import FileRenameProposalConfig
from services.analysis_service.src.file_rename_proposal.integration import FileRenameProposalIntegration


class TestFileRenameProposalIntegration:
    """Test file rename proposal integration functionality."""

    @pytest.fixture
    def mock_repos(self):
        """Create mock repositories."""
        proposal_repo = Mock()
        recording_repo = Mock()
        return proposal_repo, recording_repo

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return FileRenameProposalConfig(
            auto_generate_proposals=True,
            auto_approve_threshold=0.9,
        )

    @pytest.fixture
    def integration(self, mock_repos, config):
        """Create integration instance with mocks."""
        proposal_repo, recording_repo = mock_repos
        return FileRenameProposalIntegration(
            proposal_repo=proposal_repo,
            recording_repo=recording_repo,
            config=config,
        )

    def test_initialization(self, integration):
        """Test proper initialization of integration."""
        assert integration.proposal_repo is not None
        assert integration.recording_repo is not None
        assert integration.config is not None
        assert integration.pattern_manager is not None
        assert integration.validator is not None
        assert integration.conflict_detector is not None
        assert integration.confidence_scorer is not None
        assert integration.proposal_generator is not None
        assert integration.batch_processor is not None

    def test_process_recording_metadata_disabled(self, integration, mock_repos):
        """Test processing when auto-generation is disabled."""
        proposal_repo, recording_repo = mock_repos
        integration.auto_generate_proposals = False

        recording_id = uuid4()
        metadata = {"artist": "Test Artist", "title": "Test Song"}
        correlation_id = "test_correlation"

        result = integration.process_recording_metadata(recording_id, metadata, correlation_id)

        assert result is None
        recording_repo.get_by_id.assert_not_called()
        proposal_repo.create.assert_not_called()

    def test_process_recording_metadata_recording_not_found(self, integration, mock_repos):
        """Test processing when recording is not found."""
        proposal_repo, recording_repo = mock_repos
        recording_repo.get_by_id.return_value = None

        recording_id = uuid4()
        metadata = {"artist": "Test Artist", "title": "Test Song"}
        correlation_id = "test_correlation"

        result = integration.process_recording_metadata(recording_id, metadata, correlation_id)

        assert result is None
        recording_repo.get_by_id.assert_called_once_with(recording_id)
        proposal_repo.create.assert_not_called()

    @patch("os.path.exists")
    @patch("os.listdir")
    def test_process_recording_metadata_success(self, mock_listdir, mock_exists, integration, mock_repos):
        """Test successful metadata processing."""
        proposal_repo, recording_repo = mock_repos

        # Setup recording mock
        recording = Mock()
        recording.file_name = "test_song.mp3"
        recording.file_path = "/music/test_song.mp3"
        recording_repo.get_by_id.return_value = recording

        # Setup directory mocking
        mock_exists.return_value = True
        mock_listdir.return_value = ["other_song.mp3"]

        # Setup proposal repo mocking
        proposal_repo.get_pending_proposals.return_value = []
        mock_proposal = Mock()
        mock_proposal.id = uuid4()
        proposal_repo.create.return_value = mock_proposal

        # Setup conflict detector and confidence scorer mocks
        with (
            patch.object(
                integration.conflict_detector, "detect_conflicts", return_value={"conflicts": [], "warnings": []}
            ),
            patch.object(integration.confidence_scorer, "calculate_confidence", return_value=(0.95, {"test": 0.95})),
        ):
            recording_id = uuid4()
            metadata = {
                "artist": "Test Artist",
                "title": "Test Song",
                "album": "Test Album",
                "date": "2024",
            }
            correlation_id = "test_correlation"

            result = integration.process_recording_metadata(recording_id, metadata, correlation_id)

            assert result == str(mock_proposal.id)
            recording_repo.get_by_id.assert_called_once_with(recording_id)
            proposal_repo.create.assert_called_once()

    @patch("os.path.exists")
    @patch("os.listdir")
    def test_process_recording_metadata_with_conflicts(self, mock_listdir, mock_exists, integration, mock_repos):
        """Test metadata processing with conflicts."""
        proposal_repo, recording_repo = mock_repos

        # Setup recording mock
        recording = Mock()
        recording.file_name = "test_song.mp3"
        recording.file_path = "/music/test_song.mp3"
        recording_repo.get_by_id.return_value = recording

        # Setup directory mocking
        mock_exists.return_value = True
        mock_listdir.return_value = ["Test Artist - Test Song.mp3"]  # Conflict

        # Setup proposal repo mocking
        proposal_repo.get_pending_proposals.return_value = []
        mock_proposal = Mock()
        mock_proposal.id = uuid4()
        proposal_repo.create.return_value = mock_proposal

        # Setup conflict detector to return conflicts, then resolve them
        with (
            patch.object(
                integration.conflict_detector,
                "detect_conflicts",
                side_effect=[
                    {"conflicts": ["File already exists"], "warnings": []},
                    {"conflicts": [], "warnings": []},  # After resolution
                ],
            ),
            patch.object(
                integration.conflict_detector, "resolve_conflicts", return_value="/music/Test Artist - Test Song_2.mp3"
            ),
            patch.object(integration.confidence_scorer, "calculate_confidence", return_value=(0.8, {"test": 0.8})),
        ):
            recording_id = uuid4()
            metadata = {"artist": "Test Artist", "title": "Test Song"}
            correlation_id = "test_correlation"

            result = integration.process_recording_metadata(recording_id, metadata, correlation_id)

            assert result == str(mock_proposal.id)
            assert integration.conflict_detector.resolve_conflicts.called

    def test_process_batch_recordings_disabled(self, integration):
        """Test batch processing when auto-generation is disabled."""
        integration.auto_generate_proposals = False

        recording_ids = [uuid4(), uuid4()]
        correlation_id = "test_correlation"

        result = integration.process_batch_recordings(recording_ids, correlation_id)

        assert result is None

    def test_process_batch_recordings_success(self, integration):
        """Test successful batch processing."""
        # Mock batch processor
        mock_job = Mock()
        mock_job.job_id = "test_job_123"

        with patch.object(integration.batch_processor, "submit_batch_job", return_value=mock_job):
            recording_ids = [uuid4(), uuid4()]
            correlation_id = "test_correlation"

            result = integration.process_batch_recordings(recording_ids, correlation_id)

            assert result == "test_job_123"
            integration.batch_processor.submit_batch_job.assert_called_once()

    def test_get_proposal_status_no_proposals(self, integration, mock_repos):
        """Test getting proposal status when no proposals exist."""
        proposal_repo, _ = mock_repos
        proposal_repo.get_by_recording.return_value = []

        recording_id = uuid4()
        result = integration.get_proposal_status(recording_id)

        assert result is None
        proposal_repo.get_by_recording.assert_called_once_with(recording_id)

    def test_get_proposal_status_with_proposals(self, integration, mock_repos):
        """Test getting proposal status with existing proposals."""
        from datetime import datetime

        proposal_repo, _ = mock_repos

        # Create mock proposal
        mock_proposal = Mock()
        mock_proposal.id = uuid4()
        mock_proposal.status = "pending"
        mock_proposal.proposed_filename = "Artist - Title.mp3"
        mock_proposal.confidence_score = 0.85
        mock_proposal.created_at = datetime.now(UTC)

        proposal_repo.get_by_recording.return_value = [mock_proposal]

        recording_id = uuid4()
        result = integration.get_proposal_status(recording_id)

        assert result is not None
        assert result["proposal_id"] == str(mock_proposal.id)
        assert result["status"] == "pending"
        assert result["proposed_filename"] == "Artist - Title.mp3"
        assert result["confidence_score"] == "0.85"

    def test_cleanup_old_proposals(self, integration, mock_repos):
        """Test cleaning up old proposals."""
        proposal_repo, _ = mock_repos
        proposal_repo.cleanup_old_proposals.return_value = 5

        result = integration.cleanup_old_proposals(days=30)

        assert result == 5
        proposal_repo.cleanup_old_proposals.assert_called_once_with(30)

    def test_cleanup_old_proposals_error(self, integration, mock_repos):
        """Test cleanup error handling."""
        proposal_repo, _ = mock_repos
        proposal_repo.cleanup_old_proposals.side_effect = Exception("Database error")

        result = integration.cleanup_old_proposals(days=30)

        assert result == 0

    def test_process_recording_metadata_error_handling(self, integration, mock_repos):
        """Test error handling in metadata processing."""
        proposal_repo, recording_repo = mock_repos
        recording_repo.get_by_id.side_effect = Exception("Database connection error")

        recording_id = uuid4()
        metadata = {"artist": "Test Artist", "title": "Test Song"}
        correlation_id = "test_correlation"

        result = integration.process_recording_metadata(recording_id, metadata, correlation_id)

        assert result is None
        proposal_repo.create.assert_not_called()

    def test_directory_permission_error(self, integration, mock_repos):
        """Test handling of directory permission errors."""
        proposal_repo, recording_repo = mock_repos

        # Setup recording mock
        recording = Mock()
        recording.file_name = "test_song.mp3"
        recording.file_path = "/restricted/test_song.mp3"
        recording_repo.get_by_id.return_value = recording

        # Setup proposal repo mocking
        proposal_repo.get_pending_proposals.return_value = []
        mock_proposal = Mock()
        mock_proposal.id = uuid4()
        proposal_repo.create.return_value = mock_proposal

        # Mock directory operations to raise permission error
        with (
            patch("os.path.exists", return_value=True),
            patch("os.listdir", side_effect=PermissionError("Permission denied")),
            patch.object(
                integration.conflict_detector, "detect_conflicts", return_value={"conflicts": [], "warnings": []}
            ),
            patch.object(integration.confidence_scorer, "calculate_confidence", return_value=(0.85, {"test": 0.85})),
        ):
            recording_id = uuid4()
            metadata = {"artist": "Test Artist", "title": "Test Song"}
            correlation_id = "test_correlation"

            result = integration.process_recording_metadata(recording_id, metadata, correlation_id)

            # Should still succeed despite directory listing error
            assert result == str(mock_proposal.id)


class TestRenameExecutorIntegration:
    """Test suite for rename proposal and executor integration."""

    @pytest.fixture
    def mock_components(self):
        """Create mock components for the message interface."""

        proposal_generator = Mock()
        conflict_detector = Mock()
        confidence_scorer = Mock()
        proposal_repo = Mock()
        recording_repo = Mock()
        batch_processor = Mock()
        rename_executor = Mock()

        return {
            "proposal_generator": proposal_generator,
            "conflict_detector": conflict_detector,
            "confidence_scorer": confidence_scorer,
            "proposal_repo": proposal_repo,
            "recording_repo": recording_repo,
            "batch_processor": batch_processor,
            "rename_executor": rename_executor,
        }

    @pytest.fixture
    def message_interface(self, mock_components):
        """Create a message interface with mocked components."""
        from services.analysis_service.src.file_rename_proposal.message_interface import (
            RenameProposalMessageInterface,
        )

        return RenameProposalMessageInterface(**mock_components)

    def test_execute_rename_message_success(self, message_interface, mock_components):
        """Test successful rename execution through message interface."""
        from services.analysis_service.src.file_rename_proposal.message_interface import MessageTypes

        proposal_id = str(uuid4())

        # Setup mock
        mock_components["rename_executor"].execute_rename.return_value = (True, None)

        # Create message
        message = {
            "type": MessageTypes.EXECUTE_RENAME,
            "proposal_id": proposal_id,
            "request_id": "test-request",
        }

        # Process message
        response = message_interface.process_message(message)

        # Verify response
        assert response["type"] == MessageTypes.RENAME_EXECUTED
        assert response["proposal_id"] == proposal_id
        assert response["success"] is True
        assert "request_id" in response
        assert "timestamp" in response

        # Verify executor was called
        mock_components["rename_executor"].execute_rename.assert_called_once()

    def test_execute_rename_message_failure(self, message_interface, mock_components):
        """Test failed rename execution through message interface."""
        from services.analysis_service.src.file_rename_proposal.message_interface import MessageTypes

        proposal_id = str(uuid4())

        # Setup mock
        mock_components["rename_executor"].execute_rename.return_value = (
            False,
            "File does not exist",
        )

        # Create message
        message = {
            "type": MessageTypes.EXECUTE_RENAME,
            "proposal_id": proposal_id,
            "request_id": "test-request",
        }

        # Process message
        response = message_interface.process_message(message)

        # Verify error response
        assert response["type"] == MessageTypes.ERROR
        assert response["error"]["code"] == "RENAME_FAILED"
        assert "File does not exist" in response["error"]["message"]

    def test_rollback_rename_message_success(self, message_interface, mock_components):
        """Test successful rollback through message interface."""
        from services.analysis_service.src.file_rename_proposal.message_interface import MessageTypes

        proposal_id = str(uuid4())

        # Setup mock
        mock_components["rename_executor"].rollback_rename.return_value = (True, None)

        # Create message
        message = {
            "type": MessageTypes.ROLLBACK_RENAME,
            "proposal_id": proposal_id,
            "request_id": "test-request",
        }

        # Process message
        response = message_interface.process_message(message)

        # Verify response
        assert response["type"] == MessageTypes.RENAME_ROLLED_BACK
        assert response["proposal_id"] == proposal_id
        assert response["success"] is True

        # Verify executor was called
        mock_components["rename_executor"].rollback_rename.assert_called_once()

    def test_execute_rename_missing_proposal_id(self, message_interface):
        """Test execute rename with missing proposal_id."""
        from services.analysis_service.src.file_rename_proposal.message_interface import MessageTypes

        message = {
            "type": MessageTypes.EXECUTE_RENAME,
            "request_id": "test-request",
        }

        response = message_interface.process_message(message)

        assert response["type"] == MessageTypes.ERROR
        assert response["error"]["code"] == "MISSING_PARAMETER"
        assert "Missing proposal_id" in response["error"]["message"]

    def test_execute_rename_no_executor_configured(self, mock_components):
        """Test execute rename when executor is not configured."""
        from services.analysis_service.src.file_rename_proposal.message_interface import (
            MessageTypes,
            RenameProposalMessageInterface,
        )

        # Create interface without executor
        interface = RenameProposalMessageInterface(
            proposal_generator=mock_components["proposal_generator"],
            conflict_detector=mock_components["conflict_detector"],
            confidence_scorer=mock_components["confidence_scorer"],
            proposal_repo=mock_components["proposal_repo"],
            recording_repo=mock_components["recording_repo"],
            batch_processor=mock_components["batch_processor"],
            rename_executor=None,  # No executor
        )

        message = {
            "type": MessageTypes.EXECUTE_RENAME,
            "proposal_id": str(uuid4()),
            "request_id": "test-request",
        }

        response = interface.process_message(message)

        assert response["type"] == MessageTypes.ERROR
        assert response["error"]["code"] == "EXECUTOR_NOT_CONFIGURED"
        assert "Rename executor not configured" in response["error"]["message"]
