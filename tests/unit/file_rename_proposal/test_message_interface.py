"""Unit tests for message interface."""

from datetime import UTC, datetime
from unittest.mock import Mock
from uuid import uuid4

import pytest

from services.analysis_service.src.file_rename_proposal.message_interface import (
    MessageTypes,
    RenameProposalMessageInterface,
)
from services.analysis_service.src.file_rename_proposal.proposal_generator import (
    RenameProposal,
)


class TestMessageInterface:
    """Test message interface functionality."""

    @pytest.fixture
    def mock_services(self):
        """Create mock services."""
        return {
            "proposal_generator": Mock(),
            "conflict_detector": Mock(),
            "confidence_scorer": Mock(),
            "proposal_repo": Mock(),
            "recording_repo": Mock(),
            "batch_processor": Mock(),
        }

    @pytest.fixture
    def interface(self, mock_services):
        """Create message interface with mock services."""
        return RenameProposalMessageInterface(
            proposal_generator=mock_services["proposal_generator"],
            conflict_detector=mock_services["conflict_detector"],
            confidence_scorer=mock_services["confidence_scorer"],
            proposal_repo=mock_services["proposal_repo"],
            recording_repo=mock_services["recording_repo"],
            batch_processor=mock_services["batch_processor"],
        )

    def test_process_message_missing_type(self, interface):
        """Test processing message without type."""
        message = {"data": "test"}

        response = interface.process_message(message)

        assert response["type"] == MessageTypes.ERROR
        assert "Missing message type" in response["error"]["message"]
        assert response["error"]["code"] == "INVALID_MESSAGE"

    def test_process_message_unknown_type(self, interface):
        """Test processing message with unknown type."""
        message = {"type": "unknown_type", "request_id": "test_123"}

        response = interface.process_message(message)

        assert response["type"] == MessageTypes.ERROR
        assert "Unknown message type" in response["error"]["message"]
        assert response["error"]["code"] == "UNKNOWN_MESSAGE_TYPE"
        assert response["request_id"] == "test_123"

    def test_generate_proposal_success(self, interface, mock_services):
        """Test successful proposal generation."""
        recording_id = uuid4()
        request_id = "test_request"

        # Setup mocks
        mock_recording = Mock()
        mock_recording.file_name = "song.mp3"
        mock_recording.file_path = "/music/song.mp3"
        mock_services["recording_repo"].get_by_id.return_value = mock_recording

        mock_proposal_obj = RenameProposal(
            recording_id=recording_id,
            original_path="/music",
            original_filename="song.mp3",
            proposed_filename="Artist - Title.mp3",
            full_proposed_path="/music/Artist - Title.mp3",
            confidence_score=0.95,
            metadata_source="id3",
            pattern_used="{artist} - {title}",
        )
        mock_services["proposal_generator"].generate_proposal.return_value = mock_proposal_obj

        mock_services["conflict_detector"].detect_conflicts.return_value = {
            "conflicts": [],
            "warnings": ["Minor issue"],
        }

        mock_services["confidence_scorer"].calculate_confidence.return_value = (
            0.95,
            {"metadata_completeness": 0.9, "pattern_match": 1.0},
        )

        mock_proposal = Mock()
        mock_proposal.id = uuid4()
        mock_proposal.recording_id = recording_id
        mock_proposal.original_filename = "song.mp3"
        mock_proposal.proposed_filename = "Artist - Title.mp3"
        mock_proposal.full_proposed_path = "/music/Artist - Title.mp3"
        mock_proposal.confidence_score = 0.95
        mock_proposal.status = "pending"
        mock_proposal.conflicts = None
        mock_proposal.warnings = ["Minor issue"]
        mock_proposal.created_at = datetime.now(UTC)
        mock_services["proposal_repo"].create.return_value = mock_proposal

        message = {
            "type": MessageTypes.GENERATE_PROPOSAL,
            "request_id": request_id,
            "recording_id": str(recording_id),
        }

        response = interface.process_message(message)

        assert response["type"] == MessageTypes.PROPOSAL_GENERATED
        assert response["request_id"] == request_id
        assert "proposal" in response
        assert response["proposal"]["recording_id"] == str(recording_id)
        assert response["proposal"]["proposed_filename"] == "Artist - Title.mp3"
        assert response["proposal"]["confidence_score"] == 0.95
        assert response["proposal"]["warnings"] == ["Minor issue"]
        assert "confidence_components" in response["proposal"]

    def test_generate_proposal_recording_not_found(self, interface, mock_services):
        """Test proposal generation when recording not found."""
        recording_id = uuid4()
        mock_services["recording_repo"].get_by_id.return_value = None

        message = {
            "type": MessageTypes.GENERATE_PROPOSAL,
            "request_id": "test",
            "recording_id": str(recording_id),
        }

        response = interface.process_message(message)

        assert response["type"] == MessageTypes.ERROR
        assert response["error"]["code"] == "RECORDING_NOT_FOUND"
        assert str(recording_id) in response["error"]["message"]

    def test_generate_proposal_generation_failed(self, interface, mock_services):
        """Test proposal generation failure."""
        recording_id = uuid4()
        mock_recording = Mock()
        mock_services["recording_repo"].get_by_id.return_value = mock_recording
        mock_services["proposal_generator"].generate_proposal.side_effect = Exception("Generation failed")

        message = {
            "type": MessageTypes.GENERATE_PROPOSAL,
            "request_id": "test",
            "recording_id": str(recording_id),
        }

        response = interface.process_message(message)

        assert response["type"] == MessageTypes.ERROR
        assert response["error"]["code"] == "GENERATION_ERROR"

    def test_batch_process_success(self, interface, mock_services):
        """Test successful batch processing submission."""
        recording_ids = [uuid4(), uuid4()]

        mock_job = Mock()
        mock_job.job_id = "batch_123"
        mock_job.status = "pending"
        mock_job.total_recordings = 2
        mock_job.created_at = datetime.now(UTC)
        mock_job.options = {"max_workers": 4}

        mock_services["batch_processor"].submit_batch_job.return_value = mock_job

        message = {
            "type": MessageTypes.BATCH_PROCESS,
            "request_id": "test",
            "recording_ids": [str(rid) for rid in recording_ids],
            "options": {"max_workers": 4, "chunk_size": 50, "start_immediately": False},
        }

        response = interface.process_message(message)

        assert response["type"] == MessageTypes.BATCH_SUBMITTED
        assert response["job"]["job_id"] == "batch_123"
        assert response["job"]["total_recordings"] == 2
        assert response["job"]["status"] == "pending"

        # Verify batch processor was called with correct arguments
        mock_services["batch_processor"].submit_batch_job.assert_called_once()
        call_args = mock_services["batch_processor"].submit_batch_job.call_args
        assert len(call_args[1]["recording_ids"]) == 2
        assert call_args[1]["max_workers"] == 4
        assert call_args[1]["chunk_size"] == 50

    def test_batch_process_start_immediately(self, interface, mock_services):
        """Test batch processing with immediate start."""
        recording_ids = [uuid4()]

        mock_job = Mock()
        mock_job.job_id = "batch_123"
        mock_job.status = "pending"
        mock_job.total_recordings = 1
        mock_job.created_at = datetime.now(UTC)
        mock_job.options = {}

        mock_services["batch_processor"].submit_batch_job.return_value = mock_job

        message = {
            "type": MessageTypes.BATCH_PROCESS,
            "request_id": "test",
            "recording_ids": [str(recording_ids[0])],
            "options": {"start_immediately": True},
        }

        response = interface.process_message(message)

        assert response["type"] == MessageTypes.BATCH_SUBMITTED

        # Verify processing was started
        mock_services["batch_processor"].process_batch_job.assert_called_once_with("batch_123")

    def test_get_proposal_by_id(self, interface, mock_services):
        """Test getting proposal by ID."""
        proposal_id = uuid4()

        mock_proposal = Mock()
        mock_proposal.id = proposal_id
        mock_proposal.recording_id = uuid4()
        mock_proposal.original_filename = "song.mp3"
        mock_proposal.proposed_filename = "Artist - Title.mp3"
        mock_proposal.full_proposed_path = "/music/Artist - Title.mp3"
        mock_proposal.confidence_score = 0.85
        mock_proposal.status = "approved"
        mock_proposal.conflicts = []
        mock_proposal.warnings = []
        mock_proposal.metadata_source = "id3"
        mock_proposal.pattern_used = "{artist} - {title}"
        mock_proposal.created_at = datetime.now(UTC)
        mock_proposal.updated_at = datetime.now(UTC)

        mock_services["proposal_repo"].get.return_value = mock_proposal

        message = {
            "type": MessageTypes.GET_PROPOSAL,
            "request_id": "test",
            "proposal_id": str(proposal_id),
        }

        response = interface.process_message(message)

        assert response["type"] == MessageTypes.PROPOSAL_RETRIEVED
        assert response["count"] == 1
        assert len(response["proposals"]) == 1
        assert response["proposals"][0]["id"] == str(proposal_id)
        assert response["proposals"][0]["status"] == "approved"

    def test_get_proposal_by_recording_id(self, interface, mock_services):
        """Test getting proposals by recording ID."""
        recording_id = uuid4()

        mock_proposals = [Mock(), Mock()]
        for i, proposal in enumerate(mock_proposals):
            proposal.id = uuid4()
            proposal.recording_id = recording_id
            proposal.original_filename = f"song{i}.mp3"
            proposal.proposed_filename = f"Artist - Title {i}.mp3"
            proposal.full_proposed_path = f"/music/Artist - Title {i}.mp3"
            proposal.confidence_score = 0.8 + i * 0.1
            proposal.status = "pending"
            proposal.conflicts = []
            proposal.warnings = []
            proposal.metadata_source = "id3"
            proposal.pattern_used = "{artist} - {title}"
            proposal.created_at = datetime.now(UTC)
            proposal.updated_at = datetime.now(UTC)

        mock_services["proposal_repo"].get_by_recording.return_value = mock_proposals

        message = {
            "type": MessageTypes.GET_PROPOSAL,
            "request_id": "test",
            "recording_id": str(recording_id),
            "status": "pending",
        }

        response = interface.process_message(message)

        assert response["type"] == MessageTypes.PROPOSAL_RETRIEVED
        assert response["count"] == 2
        assert len(response["proposals"]) == 2

        # Verify repository was called with correct parameters
        mock_services["proposal_repo"].get_by_recording.assert_called_once_with(recording_id, "pending")

    def test_get_proposal_not_found(self, interface, mock_services):
        """Test getting non-existent proposal."""
        proposal_id = uuid4()
        mock_services["proposal_repo"].get.return_value = None

        message = {
            "type": MessageTypes.GET_PROPOSAL,
            "request_id": "test",
            "proposal_id": str(proposal_id),
        }

        response = interface.process_message(message)

        assert response["type"] == MessageTypes.ERROR
        assert response["error"]["code"] == "PROPOSAL_NOT_FOUND"

    def test_update_proposal_success(self, interface, mock_services):
        """Test successful proposal update."""
        proposal_id = uuid4()

        mock_proposal = Mock()
        mock_proposal.id = proposal_id
        mock_proposal.recording_id = uuid4()
        mock_proposal.status = "approved"
        mock_proposal.confidence_score = 0.95
        mock_proposal.updated_at = datetime.now(UTC)

        mock_services["proposal_repo"].update.return_value = mock_proposal

        message = {
            "type": MessageTypes.UPDATE_PROPOSAL,
            "request_id": "test",
            "proposal_id": str(proposal_id),
            "updates": {"status": "approved", "confidence_score": 0.95},
        }

        response = interface.process_message(message)

        assert response["type"] == MessageTypes.PROPOSAL_UPDATED
        assert response["proposal"]["id"] == str(proposal_id)
        assert response["proposal"]["status"] == "approved"
        assert response["proposal"]["confidence_score"] == 0.95

        # Verify update was called with correct parameters
        mock_services["proposal_repo"].update.assert_called_once_with(
            proposal_id, status="approved", confidence_score=0.95
        )

    def test_update_proposal_not_found(self, interface, mock_services):
        """Test updating non-existent proposal."""
        proposal_id = uuid4()
        mock_services["proposal_repo"].update.return_value = None

        message = {
            "type": MessageTypes.UPDATE_PROPOSAL,
            "request_id": "test",
            "proposal_id": str(proposal_id),
            "updates": {"status": "approved"},
        }

        response = interface.process_message(message)

        assert response["type"] == MessageTypes.ERROR
        assert response["error"]["code"] == "PROPOSAL_NOT_FOUND"

    def test_get_batch_status_specific_job(self, interface, mock_services):
        """Test getting specific batch job status."""
        job_id = "batch_123"

        mock_status = {
            "job_id": job_id,
            "status": "running",
            "progress_percentage": 50.0,
            "processed_recordings": 5,
            "total_recordings": 10,
        }

        mock_services["batch_processor"].get_job_status.return_value = mock_status

        message = {
            "type": MessageTypes.GET_BATCH_STATUS,
            "request_id": "test",
            "job_id": job_id,
        }

        response = interface.process_message(message)

        assert response["type"] == MessageTypes.BATCH_STATUS
        assert response["job"]["job_id"] == job_id
        assert response["job"]["status"] == "running"
        assert response["job"]["progress_percentage"] == 50.0

    def test_get_batch_status_all_jobs(self, interface, mock_services):
        """Test getting all active job statuses."""
        mock_jobs = [
            {"job_id": "job1", "status": "running"},
            {"job_id": "job2", "status": "completed"},
        ]

        mock_services["batch_processor"].list_active_jobs.return_value = mock_jobs

        message = {"type": MessageTypes.GET_BATCH_STATUS, "request_id": "test"}

        response = interface.process_message(message)

        assert response["type"] == MessageTypes.BATCH_STATUS
        assert response["count"] == 2
        assert len(response["jobs"]) == 2
        assert response["jobs"][0]["job_id"] == "job1"
        assert response["jobs"][1]["job_id"] == "job2"

    def test_cancel_batch_success(self, interface, mock_services):
        """Test successful batch cancellation."""
        job_id = "batch_123"
        mock_services["batch_processor"].cancel_job.return_value = True

        message = {
            "type": MessageTypes.CANCEL_BATCH,
            "request_id": "test",
            "job_id": job_id,
        }

        response = interface.process_message(message)

        assert response["type"] == MessageTypes.BATCH_CANCELLED
        assert response["job_id"] == job_id
        assert response["cancelled"] is True

        mock_services["batch_processor"].cancel_job.assert_called_once_with(job_id)

    def test_cancel_batch_failed(self, interface, mock_services):
        """Test failed batch cancellation."""
        job_id = "batch_123"
        mock_services["batch_processor"].cancel_job.return_value = False

        message = {
            "type": MessageTypes.CANCEL_BATCH,
            "request_id": "test",
            "job_id": job_id,
        }

        response = interface.process_message(message)

        assert response["type"] == MessageTypes.ERROR
        assert response["error"]["code"] == "CANCEL_FAILED"

    def test_get_statistics(self, interface, mock_services):
        """Test getting statistics."""
        mock_stats = {
            "total": 100,
            "by_status": {"pending": 30, "approved": 50, "rejected": 20},
            "average_confidence": 0.75,
            "with_conflicts": 5,
            "with_warnings": 15,
        }

        mock_services["proposal_repo"].get_statistics.return_value = mock_stats

        message = {"type": MessageTypes.GET_STATISTICS, "request_id": "test"}

        response = interface.process_message(message)

        assert response["type"] == MessageTypes.STATISTICS
        assert response["statistics"]["total"] == 100
        assert response["statistics"]["average_confidence"] == 0.75

    def test_cleanup_old_proposals(self, interface, mock_services):
        """Test cleaning up old proposals."""
        mock_services["proposal_repo"].cleanup_old_proposals.return_value = 25

        message = {
            "type": MessageTypes.CLEANUP_OLD_PROPOSALS,
            "request_id": "test",
            "days": 60,
        }

        response = interface.process_message(message)

        assert response["type"] == MessageTypes.CLEANUP_COMPLETED
        assert response["cleaned_count"] == 25
        assert response["days"] == 60

        mock_services["proposal_repo"].cleanup_old_proposals.assert_called_once_with(60)

    def test_cleanup_old_proposals_default_days(self, interface, mock_services):
        """Test cleaning up old proposals with default days."""
        mock_services["proposal_repo"].cleanup_old_proposals.return_value = 10

        message = {"type": MessageTypes.CLEANUP_OLD_PROPOSALS, "request_id": "test"}

        response = interface.process_message(message)

        assert response["type"] == MessageTypes.CLEANUP_COMPLETED
        assert response["days"] == 30  # Default value

        mock_services["proposal_repo"].cleanup_old_proposals.assert_called_once_with(30)

    def test_message_with_auto_generated_request_id(self, interface, mock_services):
        """Test message processing with auto-generated request ID."""
        mock_services["proposal_repo"].get_statistics.return_value = {"total": 0}

        message = {
            "type": MessageTypes.GET_STATISTICS
            # No request_id provided
        }

        response = interface.process_message(message)

        assert response["type"] == MessageTypes.STATISTICS
        assert "request_id" in response
        assert response["request_id"] is not None
        assert len(response["request_id"]) > 0

    def test_processing_exception_handling(self, interface, mock_services):
        """Test exception handling during message processing."""
        mock_services["proposal_repo"].get_statistics.side_effect = Exception("Database error")

        message = {"type": MessageTypes.GET_STATISTICS, "request_id": "test"}

        response = interface.process_message(message)

        assert response["type"] == MessageTypes.ERROR
        assert response["error"]["code"] == "STATISTICS_ERROR"  # Specific handler catches it
        assert "Database error" in response["error"]["message"]
        assert response["request_id"] == "test"
