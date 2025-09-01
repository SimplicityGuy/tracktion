"""Unit tests for batch processor."""

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest

from services.analysis_service.src.file_rename_proposal.batch_processor import (
    BatchProcessingJob,
    BatchProcessor,
)
from services.analysis_service.src.file_rename_proposal.proposal_generator import (
    RenameProposal,
)


class TestBatchProcessingJob:
    """Test batch processing job functionality."""

    def test_job_initialization(self):
        """Test job initialization."""
        recording_ids = [uuid4(), uuid4(), uuid4()]
        options = {"max_workers": 8, "chunk_size": 50}

        job = BatchProcessingJob("test_job", recording_ids, options)

        assert job.job_id == "test_job"
        assert job.recording_ids == recording_ids
        assert job.options == options
        assert job.status == "pending"
        assert job.total_recordings == 3
        assert job.processed_recordings == 0
        assert job.successful_proposals == 0
        assert job.failed_recordings == 0
        assert job.errors == []
        assert job.proposal_ids == []
        assert job.started_at is None
        assert job.completed_at is None
        assert isinstance(job.created_at, datetime)


class TestBatchProcessor:
    """Test batch processor functionality."""

    @pytest.fixture
    def mock_services(self):
        """Create mock services."""
        return {
            "proposal_generator": Mock(),
            "conflict_detector": Mock(),
            "confidence_scorer": Mock(),
            "proposal_repo": Mock(),
            "recording_repo": Mock(),
        }

    @pytest.fixture
    def processor(self, mock_services):
        """Create batch processor with mock services."""
        return BatchProcessor(
            proposal_generator=mock_services["proposal_generator"],
            conflict_detector=mock_services["conflict_detector"],
            confidence_scorer=mock_services["confidence_scorer"],
            proposal_repo=mock_services["proposal_repo"],
            recording_repo=mock_services["recording_repo"],
        )

    def test_submit_batch_job(self, processor):
        """Test submitting a batch job."""
        recording_ids = [uuid4(), uuid4()]

        job = processor.submit_batch_job(recording_ids=recording_ids, job_id="test_job", max_workers=2, chunk_size=10)

        assert job.job_id == "test_job"
        assert job.recording_ids == recording_ids
        assert job.options["max_workers"] == 2
        assert job.options["chunk_size"] == 10
        assert job.status == "pending"
        assert "test_job" in processor.active_jobs

    def test_submit_batch_job_auto_id(self, processor):
        """Test submitting a batch job with auto-generated ID."""
        recording_ids = [uuid4()]

        job = processor.submit_batch_job(recording_ids=recording_ids)

        assert job.job_id.startswith("batch_")
        assert job.job_id in processor.active_jobs

    def test_get_job_status(self, processor):
        """Test getting job status."""
        recording_ids = [uuid4(), uuid4()]
        _ = processor.submit_batch_job(recording_ids, "test_job")

        status = processor.get_job_status("test_job")

        assert status is not None
        assert status["job_id"] == "test_job"
        assert status["status"] == "pending"
        assert status["total_recordings"] == 2
        assert status["processed_recordings"] == 0
        assert status["progress_percentage"] == 0

    def test_get_job_status_not_found(self, processor):
        """Test getting status of non-existent job."""
        status = processor.get_job_status("nonexistent")
        assert status is None

    def test_cancel_job(self, processor):
        """Test cancelling a job."""
        recording_ids = [uuid4()]
        job = processor.submit_batch_job(recording_ids, "test_job")

        result = processor.cancel_job("test_job")

        assert result is True
        assert job.status == "cancelled"
        assert job.completed_at is not None

    def test_cancel_job_not_found(self, processor):
        """Test cancelling non-existent job."""
        result = processor.cancel_job("nonexistent")
        assert result is False

    def test_cancel_completed_job(self, processor):
        """Test cancelling already completed job."""
        recording_ids = [uuid4()]
        job = processor.submit_batch_job(recording_ids, "test_job")
        job.status = "completed"

        result = processor.cancel_job("test_job")
        assert result is False

    def test_list_active_jobs(self, processor):
        """Test listing active jobs."""
        recording_ids1 = [uuid4()]
        recording_ids2 = [uuid4(), uuid4()]

        processor.submit_batch_job(recording_ids1, "job1")
        processor.submit_batch_job(recording_ids2, "job2")

        jobs = processor.list_active_jobs()

        assert len(jobs) == 2
        job_ids = {job["job_id"] for job in jobs}
        assert job_ids == {"job1", "job2"}

    def test_cleanup_completed_jobs(self, processor):
        """Test cleaning up old completed jobs."""
        # Create old completed job
        recording_ids = [uuid4()]
        old_job = processor.submit_batch_job(recording_ids, "old_job")
        old_job.status = "completed"
        old_job.completed_at = datetime.now(UTC) - timedelta(hours=25)

        # Create recent completed job
        recent_job = processor.submit_batch_job(recording_ids, "recent_job")
        recent_job.status = "completed"
        recent_job.completed_at = datetime.now(UTC) - timedelta(hours=1)

        # Create pending job
        processor.submit_batch_job(recording_ids, "pending_job")

        cleaned_count = processor.cleanup_completed_jobs(max_age_hours=24)

        assert cleaned_count == 1
        assert "old_job" not in processor.active_jobs
        assert "recent_job" in processor.active_jobs
        assert "pending_job" in processor.active_jobs

    @patch("os.path.exists")
    @patch("os.listdir")
    def test_collect_directory_contents(self, mock_listdir, mock_exists, processor, mock_services):
        """Test collecting directory contents."""
        # Setup recordings
        recording1 = Mock()
        recording1.file_path = "/music/album1/song1.mp3"
        recording2 = Mock()
        recording2.file_path = "/music/album1/song2.mp3"
        recording3 = Mock()
        recording3.file_path = "/music/album2/song3.mp3"

        mock_services["recording_repo"].get_by_id.side_effect = [
            recording1,
            recording2,
            recording3,
        ]

        # Setup directory listing
        mock_exists.return_value = True
        mock_listdir.side_effect = [
            ["song1.mp3", "song2.mp3", "cover.jpg"],  # /music/album1
            ["song3.mp3", "notes.txt"],  # /music/album2
        ]

        recording_ids = [uuid4(), uuid4(), uuid4()]
        result = processor._collect_directory_contents(recording_ids)

        assert len(result) == 2
        assert "/music/album1" in result
        assert "/music/album2" in result
        assert result["/music/album1"] == {"song1.mp3", "song2.mp3", "cover.jpg"}
        assert result["/music/album2"] == {"song3.mp3", "notes.txt"}

    def test_process_single_recording_success(self, processor, mock_services):
        """Test successful single recording processing."""
        recording_id = uuid4()
        recording = Mock()
        recording.file_name = "song.mp3"
        recording.file_path = "/music/song.mp3"

        mock_services["recording_repo"].get_by_id.return_value = recording
        mock_proposal_obj = RenameProposal(
            recording_id=recording_id,
            original_path="/music",
            original_filename="song.mp3",
            proposed_filename="Artist - Title.mp3",
            full_proposed_path="/music/Artist - Title.mp3",
            confidence_score=0.85,
            metadata_source="id3",
            pattern_used="{artist} - {title}",
        )
        mock_services["proposal_generator"].generate_proposal.return_value = mock_proposal_obj

        mock_services["conflict_detector"].detect_conflicts.return_value = {
            "conflicts": [],
            "warnings": [],
        }

        mock_services["confidence_scorer"].calculate_confidence.return_value = (
            0.95,
            {},
        )

        mock_proposal = Mock()
        mock_proposal.id = uuid4()
        mock_services["proposal_repo"].create.return_value = mock_proposal
        mock_services["proposal_repo"].get_pending_proposals.return_value = []

        job = BatchProcessingJob("test", [recording_id])
        job.options = {"auto_approve_threshold": 0.9}

        result = processor._process_single_recording(recording_id, job, {"/music": set()})

        assert result["success"] is True
        assert result["error"] is None
        assert result["proposal_id"] == mock_proposal.id

        # Verify proposal creation
        mock_services["proposal_repo"].create.assert_called_once()
        create_args = mock_services["proposal_repo"].create.call_args[1]
        assert create_args["recording_id"] == recording_id
        assert create_args["proposed_filename"] == "Artist - Title.mp3"
        assert create_args["status"] == "approved"  # High confidence

    def test_process_single_recording_with_conflicts(self, processor, mock_services):
        """Test single recording processing with conflicts."""
        recording_id = uuid4()
        recording = Mock()
        recording.file_name = "song.mp3"
        recording.file_path = "/music/song.mp3"

        mock_services["recording_repo"].get_by_id.return_value = recording
        mock_proposal_obj = RenameProposal(
            recording_id=recording_id,
            original_path="/music",
            original_filename="song.mp3",
            proposed_filename="Artist - Title.mp3",
            full_proposed_path="/music/Artist - Title.mp3",
            confidence_score=0.75,
            metadata_source="id3",
            pattern_used="{artist} - {title}",
        )
        mock_services["proposal_generator"].generate_proposal.return_value = mock_proposal_obj

        # Initial conflicts
        mock_services["conflict_detector"].detect_conflicts.side_effect = [
            {"conflicts": ["File already exists"], "warnings": []},
            {"conflicts": [], "warnings": []},  # After resolution
        ]

        # Conflict resolution
        mock_services["conflict_detector"].resolve_conflicts.return_value = "/music/Artist - Title_2.mp3"

        mock_services["confidence_scorer"].calculate_confidence.return_value = (0.8, {})

        mock_proposal = Mock()
        mock_proposal.id = uuid4()
        mock_services["proposal_repo"].create.return_value = mock_proposal
        mock_services["proposal_repo"].get_pending_proposals.return_value = []

        job = BatchProcessingJob("test", [recording_id])
        job.options = {
            "enable_conflict_resolution": True,
            "auto_approve_threshold": 0.9,
        }

        result = processor._process_single_recording(recording_id, job, {"/music": {"Artist - Title.mp3"}})

        assert result["success"] is True

        # Verify conflict resolution was attempted
        mock_services["conflict_detector"].resolve_conflicts.assert_called_once()

        # Verify final proposal used resolved path
        create_args = mock_services["proposal_repo"].create.call_args[1]
        assert create_args["full_proposed_path"] == "/music/Artist - Title_2.mp3"
        assert create_args["proposed_filename"] == "Artist - Title_2.mp3"
        assert create_args["status"] == "pending"  # Lower confidence

    def test_process_single_recording_unresolvable_conflicts(self, processor, mock_services):
        """Test single recording processing with unresolvable conflicts."""
        recording_id = uuid4()
        recording = Mock()
        recording.file_name = "song.mp3"
        recording.file_path = "/music/song.mp3"

        mock_services["recording_repo"].get_by_id.return_value = recording
        mock_proposal_obj = RenameProposal(
            recording_id=recording_id,
            original_path="/music",
            original_filename="song.mp3",
            proposed_filename="Artist - Title.mp3",
            full_proposed_path="/music/Artist - Title.mp3",
            confidence_score=0.80,
            metadata_source="id3",
            pattern_used="{artist} - {title}",
        )
        mock_services["proposal_generator"].generate_proposal.return_value = mock_proposal_obj

        mock_services["conflict_detector"].detect_conflicts.return_value = {
            "conflicts": ["Directory traversal detected"],
            "warnings": [],
        }

        # Cannot resolve conflicts
        mock_services["conflict_detector"].resolve_conflicts.return_value = None

        mock_services["confidence_scorer"].calculate_confidence.return_value = (0.8, {})

        mock_proposal = Mock()
        mock_proposal.id = uuid4()
        mock_services["proposal_repo"].create.return_value = mock_proposal
        mock_services["proposal_repo"].get_pending_proposals.return_value = []

        job = BatchProcessingJob("test", [recording_id])

        result = processor._process_single_recording(recording_id, job, {"/music": set()})

        assert result["success"] is True

        # Verify proposal was rejected due to unresolved conflicts
        create_args = mock_services["proposal_repo"].create.call_args[1]
        assert create_args["status"] == "rejected"
        assert create_args["conflicts"] == ["Directory traversal detected"]

    def test_process_single_recording_not_found(self, processor, mock_services):
        """Test processing when recording not found."""
        recording_id = uuid4()
        mock_services["recording_repo"].get_by_id.return_value = None

        job = BatchProcessingJob("test", [recording_id])

        result = processor._process_single_recording(recording_id, job, {})

        assert result["success"] is False
        assert "not found" in result["error"]
        assert result["proposal_id"] is None

    def test_process_single_recording_no_proposal(self, processor, mock_services):
        """Test processing when proposal generation fails."""
        recording_id = uuid4()
        recording = Mock()
        recording.file_name = "song.mp3"
        recording.file_path = "/music/song.mp3"

        mock_services["recording_repo"].get_by_id.return_value = recording
        # Simulate an exception during proposal generation
        mock_services["proposal_generator"].generate_proposal.side_effect = Exception("Generation failed")

        job = BatchProcessingJob("test", [recording_id])

        result = processor._process_single_recording(recording_id, job, {})

        assert result["success"] is False
        assert "Generation failed" in result["error"]
        assert result["proposal_id"] is None

    def test_update_job_progress(self, processor):
        """Test updating job progress."""
        job = BatchProcessingJob("test", [uuid4(), uuid4(), uuid4()])

        chunk_results = {
            "processed": 2,
            "successful": 1,
            "failed": 1,
            "proposal_ids": [uuid4()],
            "errors": ["Error message"],
        }

        processor._update_job_progress(job, chunk_results)

        assert job.processed_recordings == 2
        assert job.successful_proposals == 1
        assert job.failed_recordings == 1
        assert len(job.proposal_ids) == 1
        assert len(job.errors) == 1

    def test_process_batch_job_success(self, processor, mock_services):
        """Test successful batch job processing."""
        recording_ids = [uuid4(), uuid4()]
        job = processor.submit_batch_job(recording_ids, "test_job")

        # Mock the actual processing method to avoid threading complexity
        with patch.object(processor, "_process_recordings_in_batches") as mock_process:
            result_job = processor.process_batch_job("test_job")

        assert result_job.status == "completed"
        assert result_job.started_at is not None
        assert result_job.completed_at is not None
        mock_process.assert_called_once_with(job)

    def test_process_batch_job_not_found(self, processor):
        """Test processing non-existent job."""
        with pytest.raises(ValueError, match="Job nonexistent not found"):
            processor.process_batch_job("nonexistent")

    def test_process_batch_job_failure(self, processor):
        """Test batch job processing with failure."""
        recording_ids = [uuid4()]
        job = processor.submit_batch_job(recording_ids, "test_job")

        # Mock the processing method to raise an exception
        with patch.object(processor, "_process_recordings_in_batches") as mock_process:
            mock_process.side_effect = Exception("Processing failed")

            with pytest.raises(Exception, match="Processing failed"):
                processor.process_batch_job("test_job")

        assert job.status == "failed"
        assert "Job failed: Processing failed" in job.errors
