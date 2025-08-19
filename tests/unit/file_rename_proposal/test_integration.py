"""Unit tests for file rename proposal integration."""

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
        integration.conflict_detector.detect_conflicts.return_value = {"conflicts": [], "warnings": []}
        integration.confidence_scorer.calculate_confidence.return_value = (0.95, {"test": 0.95})

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
        integration.conflict_detector.detect_conflicts.side_effect = [
            {"conflicts": ["File already exists"], "warnings": []},
            {"conflicts": [], "warnings": []},  # After resolution
        ]
        integration.conflict_detector.resolve_conflicts.return_value = "/music/Test Artist - Test Song_2.mp3"
        integration.confidence_scorer.calculate_confidence.return_value = (0.8, {"test": 0.8})

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
        integration.batch_processor.submit_batch_job.return_value = mock_job

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
        mock_proposal.created_at = datetime.utcnow()

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
        ):
            # Setup other mocks
            integration.conflict_detector.detect_conflicts.return_value = {"conflicts": [], "warnings": []}
            integration.confidence_scorer.calculate_confidence.return_value = (0.85, {"test": 0.85})

            recording_id = uuid4()
            metadata = {"artist": "Test Artist", "title": "Test Song"}
            correlation_id = "test_correlation"

            result = integration.process_recording_metadata(recording_id, metadata, correlation_id)

            # Should still succeed despite directory listing error
            assert result == str(mock_proposal.id)
