"""Unit tests for progress tracking system."""

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import pytest

from services.tracklist_service.src.progress.tracker import (
    BatchProgress,
    JobProgress,
    JobStatus,
    ProgressTracker,
)


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    redis_mock = Mock()
    redis_mock.get = Mock(return_value=None)
    redis_mock.set = Mock()
    redis_mock.delete = Mock()
    redis_mock.expire = Mock()
    redis_mock.hget = Mock(return_value=None)
    redis_mock.hset = Mock()
    redis_mock.hgetall = Mock(return_value={})
    redis_mock.hincrby = Mock()
    redis_mock.publish = Mock()
    return redis_mock


@pytest.fixture
def tracker(mock_redis):
    """Create ProgressTracker instance with mock Redis."""
    with patch("services.tracklist_service.src.progress.tracker.Redis", return_value=mock_redis):
        tracker = ProgressTracker()
        yield tracker


class TestJobStatus:
    """Test JobStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.PROCESSING.value == "processing"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.RETRYING.value == "retrying"
        assert JobStatus.CANCELLED.value == "cancelled"


class TestJobProgress:
    """Test JobProgress dataclass."""

    def test_initialization(self):
        """Test JobProgress initialization."""
        progress = JobProgress(
            job_id="job-123", batch_id="batch-456", url="http://example.com", status=JobStatus.PROCESSING
        )

        assert progress.job_id == "job-123"
        assert progress.batch_id == "batch-456"
        assert progress.url == "http://example.com"
        assert progress.status == JobStatus.PROCESSING
        assert progress.error is None
        assert progress.retry_count == 0

    def test_with_error(self):
        """Test JobProgress with error."""
        progress = JobProgress(
            job_id="job-123",
            batch_id="batch-456",
            url="http://example.com",
            status=JobStatus.FAILED,
            error="Connection timeout",
        )

        assert progress.status == JobStatus.FAILED
        assert progress.error == "Connection timeout"

    def test_to_dict(self):
        """Test converting JobProgress to dictionary."""
        now = datetime.now(UTC)
        progress = JobProgress(
            job_id="job-123",
            batch_id="batch-456",
            url="http://example.com",
            status=JobStatus.COMPLETED,
            started_at=now,
            completed_at=now + timedelta(seconds=10),
            processing_time=10.0,
        )

        data = progress.to_dict()
        assert data["job_id"] == "job-123"
        assert data["status"] == "completed"
        assert "started_at" in data
        assert "completed_at" in data


class TestBatchProgress:
    """Test BatchProgress dataclass."""

    def test_initialization(self):
        """Test BatchProgress initialization."""
        progress = BatchProgress(batch_id="batch-456", total_jobs=10)

        assert progress.batch_id == "batch-456"
        assert progress.total_jobs == 10
        assert progress.pending == 0
        assert progress.completed == 0
        assert progress.failed == 0

    def test_progress_percentage(self):
        """Test progress percentage calculation."""
        progress = BatchProgress(batch_id="batch-456", total_jobs=10, completed=3, failed=2, cancelled=1)

        # (3 + 2 + 1) / 10 * 100 = 60%
        assert progress.progress_percentage == 60.0

    def test_progress_percentage_empty(self):
        """Test progress percentage with no jobs."""
        progress = BatchProgress(batch_id="batch-456", total_jobs=0)

        assert progress.progress_percentage == 0.0

    def test_success_rate(self):
        """Test success rate calculation."""
        progress = BatchProgress(batch_id="batch-456", total_jobs=10, completed=7, failed=3)

        # 7 / (7 + 3) * 100 = 70%
        assert progress.success_rate == 70.0

    def test_to_dict(self):
        """Test converting BatchProgress to dictionary."""
        progress = BatchProgress(batch_id="batch-456", total_jobs=10, completed=5, start_time=datetime.now(UTC))

        data = progress.to_dict()
        assert data["batch_id"] == "batch-456"
        assert data["total_jobs"] == 10
        assert data["completed"] == 5
        assert "progress_percentage" in data
        assert "success_rate" in data
        assert "start_time" in data


class TestProgressTracker:
    """Test ProgressTracker class."""

    def test_initialization(self, mock_redis):
        """Test tracker initialization."""
        with patch("services.tracklist_service.src.progress.tracker.Redis", return_value=mock_redis):
            tracker = ProgressTracker()

        assert tracker.redis == mock_redis
        assert tracker.websocket_connections == {}
        assert tracker.job_progress == {}
        assert tracker.batch_progress == {}
        assert tracker.completion_callbacks == {}

    @pytest.mark.asyncio
    async def test_update_progress_new_job(self, tracker, mock_redis):
        """Test updating progress for new job."""
        await tracker.update_progress(
            batch_id="batch-123", job_id="job-1", status="processing", url="http://example.com"
        )

        # Check job progress updated
        assert "job-1" in tracker.job_progress
        job = tracker.job_progress["job-1"]
        assert job.status == JobStatus.PROCESSING
        assert job.batch_id == "batch-123"
        assert job.url == "http://example.com"

        # Check batch tracking
        assert "batch-123" in tracker.batch_jobs
        assert "job-1" in tracker.batch_jobs["batch-123"]

    @pytest.mark.asyncio
    async def test_update_progress_complete_job(self, tracker, mock_redis):
        """Test completing a job."""
        # First set to processing
        await tracker.update_progress(batch_id="batch-123", job_id="job-1", status="processing")

        # Then complete it
        result = {"data": "test"}
        await tracker.update_progress(batch_id="batch-123", job_id="job-1", status="completed", result=result)

        job = tracker.job_progress["job-1"]
        assert job.status == JobStatus.COMPLETED
        assert job.completed_at is not None
        assert job.result == result

    @pytest.mark.asyncio
    async def test_update_progress_fail_job(self, tracker, mock_redis):
        """Test failing a job."""
        await tracker.update_progress(batch_id="batch-123", job_id="job-1", status="failed", error="Network error")

        job = tracker.job_progress["job-1"]
        assert job.status == JobStatus.FAILED
        assert job.error == "Network error"

    @pytest.mark.asyncio
    async def test_calculate_eta_no_batch(self, tracker):
        """Test ETA calculation with no batch."""
        eta = await tracker.calculate_eta("batch-999")
        assert eta is None

    @pytest.mark.asyncio
    async def test_calculate_eta_with_batch(self, tracker):
        """Test ETA calculation with batch."""
        # Create batch progress
        tracker.batch_progress["batch-123"] = BatchProgress(
            batch_id="batch-123",
            total_jobs=5,
            completed=2,
            pending=3,
            start_time=datetime.now(UTC) - timedelta(seconds=20),
        )

        # Add some processing times
        tracker.processing_times = [10.0, 10.0]  # Average 10 seconds

        eta = await tracker.calculate_eta("batch-123")

        assert eta is not None
        # Should be approximately 30 seconds from now (3 pending * 10 seconds)
        expected = datetime.now(UTC) + timedelta(seconds=30)
        assert abs((eta - expected).total_seconds()) < 5

    @pytest.mark.asyncio
    async def test_broadcast_update(self, tracker):
        """Test broadcasting update to websockets."""
        mock_ws = AsyncMock()
        tracker.websocket_connections["batch-123"] = [mock_ws]

        update = {"type": "progress", "batch_id": "batch-123"}
        await tracker.broadcast_update("batch-123", update)

        # Check websocket received the update
        expected_message = json.dumps(update)
        mock_ws.send_text.assert_called_once_with(expected_message)

    @pytest.mark.asyncio
    async def test_broadcast_update_remove_failed_connections(self, tracker):
        """Test removing failed websocket connections."""
        mock_ws1 = AsyncMock()
        mock_ws1.send_text.side_effect = Exception("Connection closed")
        mock_ws2 = AsyncMock()
        tracker.websocket_connections["batch-123"] = [mock_ws1, mock_ws2]

        update = {"type": "progress"}
        await tracker.broadcast_update("batch-123", update)

        # Check that failed connection was removed
        assert mock_ws1 not in tracker.websocket_connections["batch-123"]
        assert mock_ws2 in tracker.websocket_connections["batch-123"]

    def test_add_websocket_connection(self, tracker):
        """Test adding websocket connection."""
        mock_ws = Mock()

        tracker.add_websocket("batch-123", mock_ws)

        assert "batch-123" in tracker.websocket_connections
        assert mock_ws in tracker.websocket_connections["batch-123"]

    def test_remove_websocket_connection(self, tracker):
        """Test removing websocket connection."""
        mock_ws = Mock()
        tracker.websocket_connections["batch-123"] = [mock_ws]

        tracker.remove_websocket("batch-123", mock_ws)

        # Check websocket was removed and if list is empty, key is removed
        if "batch-123" in tracker.websocket_connections:
            assert mock_ws not in tracker.websocket_connections["batch-123"]

    def test_register_completion_callback(self, tracker):
        """Test registering completion callback."""
        callback = Mock()

        tracker.register_completion_callback("batch-123", callback)

        assert "batch-123" in tracker.completion_callbacks
        assert callback in tracker.completion_callbacks["batch-123"]

    @pytest.mark.asyncio
    async def test_check_batch_completion(self, tracker):
        """Test batch completion check."""
        # Set up completed batch
        tracker.batch_progress["batch-123"] = BatchProgress(batch_id="batch-123", total_jobs=2, completed=2)

        # Register callback
        callback = AsyncMock()
        tracker.register_completion_callback("batch-123", callback)

        # Check completion
        await tracker._check_batch_completion("batch-123")

        # Callback should be triggered with batch_id and batch_progress
        callback.assert_called_once()
        call_args = callback.call_args[0]
        assert call_args[0] == "batch-123"
        assert isinstance(call_args[1], BatchProgress)
