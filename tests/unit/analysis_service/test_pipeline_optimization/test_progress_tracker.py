"""Unit tests for progress tracking system."""

import json
import time
import unittest
from unittest.mock import MagicMock, patch

import redis.exceptions

from services.analysis_service.src.progress_tracker import FileProgress, ProcessingStatus, ProgressTracker


class TestFileProgress(unittest.TestCase):
    """Test FileProgress dataclass."""

    def test_file_progress_creation(self) -> None:
        """Test creating FileProgress instance."""
        progress = FileProgress(
            file_path="/path/to/file.mp3",
            recording_id="rec123",
            correlation_id="corr123",
            status=ProcessingStatus.QUEUED,
            queued_at=time.time(),
        )

        self.assertEqual(progress.file_path, "/path/to/file.mp3")
        self.assertEqual(progress.recording_id, "rec123")
        self.assertEqual(progress.correlation_id, "corr123")
        self.assertEqual(progress.status, ProcessingStatus.QUEUED)
        self.assertIsNone(progress.started_at)
        self.assertIsNone(progress.completed_at)
        self.assertEqual(progress.retry_count, 0)
        self.assertEqual(progress.progress_percentage, 0.0)

    def test_to_dict_conversion(self) -> None:
        """Test conversion to dictionary."""
        current_time = time.time()
        progress = FileProgress(
            file_path="/path/to/file.mp3",
            recording_id="rec123",
            correlation_id="corr123",
            status=ProcessingStatus.IN_PROGRESS,
            queued_at=current_time,
            started_at=current_time + 1,
            progress_percentage=50.0,
            current_step="Analyzing BPM",
        )

        data = progress.to_dict()

        self.assertEqual(data["file_path"], "/path/to/file.mp3")
        self.assertEqual(data["status"], "in_progress")
        self.assertEqual(data["progress_percentage"], 50.0)
        self.assertEqual(data["current_step"], "Analyzing BPM")

    def test_from_dict_conversion(self) -> None:
        """Test creation from dictionary."""
        current_time = time.time()
        data = {
            "file_path": "/path/to/file.mp3",
            "recording_id": "rec123",
            "correlation_id": "corr123",
            "status": "completed",
            "queued_at": current_time,
            "started_at": current_time + 1,
            "completed_at": current_time + 10,
            "retry_count": 0,
            "error_message": None,
            "progress_percentage": 100.0,
            "current_step": "Completed",
        }

        progress = FileProgress.from_dict(data)

        self.assertEqual(progress.file_path, "/path/to/file.mp3")
        self.assertEqual(progress.status, ProcessingStatus.COMPLETED)
        self.assertEqual(progress.progress_percentage, 100.0)
        self.assertEqual(progress.completed_at, current_time + 10)


class TestProgressTracker(unittest.TestCase):
    """Test ProgressTracker class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Mock Redis client
        self.mock_redis = MagicMock()
        self.mock_redis.ping.return_value = True

        # Patch Redis constructor
        self.redis_patcher = patch("services.analysis_service.src.progress_tracker.redis.Redis")
        self.mock_redis_class = self.redis_patcher.start()
        self.mock_redis_class.return_value = self.mock_redis

        # Create tracker
        self.tracker = ProgressTracker(
            redis_host="localhost",
            redis_port=6379,
            key_prefix="test:progress",
            ttl_seconds=3600,
        )

    def tearDown(self) -> None:
        """Clean up after tests."""
        self.redis_patcher.stop()

    def test_initialization(self) -> None:
        """Test tracker initialization."""
        self.mock_redis_class.assert_called_once()
        self.mock_redis.ping.assert_called_once()
        self.assertEqual(self.tracker.key_prefix, "test:progress")
        self.assertEqual(self.tracker.ttl_seconds, 3600)

    def test_connection_error(self) -> None:
        """Test handling of Redis connection error."""
        self.mock_redis.ping.side_effect = redis.exceptions.ConnectionError("Connection failed")

        with self.assertRaises(redis.exceptions.ConnectionError):
            ProgressTracker(redis_host="localhost", redis_port=6379)

    def test_track_file_queued(self) -> None:
        """Test tracking a file being queued."""
        self.mock_redis.scard.return_value = 5

        correlation_id = self.tracker.track_file_queued(
            file_path="/path/to/file.mp3",
            recording_id="rec123",
            correlation_id="corr123",
        )

        self.assertEqual(correlation_id, "corr123")

        # Check Redis operations
        self.mock_redis.setex.assert_called_once()
        self.mock_redis.sadd.assert_called_with("test:progress:queue", "corr123")
        self.mock_redis.hincrby.assert_called()
        self.mock_redis.hset.assert_called()

    def test_track_file_queued_auto_correlation_id(self) -> None:
        """Test tracking with auto-generated correlation ID."""
        self.mock_redis.scard.return_value = 1

        correlation_id = self.tracker.track_file_queued(
            file_path="/path/to/file.mp3",
            recording_id="rec123",
        )

        self.assertIsNotNone(correlation_id)
        self.assertIsInstance(correlation_id, str)
        self.mock_redis.setex.assert_called_once()

    def test_track_file_started(self) -> None:
        """Test tracking file processing start."""
        # Mock existing progress data
        progress_data = {
            "file_path": "/path/to/file.mp3",
            "recording_id": "rec123",
            "correlation_id": "corr123",
            "status": "queued",
            "queued_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "retry_count": 0,
            "error_message": None,
            "progress_percentage": 0.0,
            "current_step": None,
        }
        self.mock_redis.get.return_value = json.dumps(progress_data)
        self.mock_redis.scard.return_value = 2

        self.tracker.track_file_started("corr123", "Analyzing BPM")

        # Check Redis operations
        self.mock_redis.get.assert_called_with("test:progress:file:corr123")
        self.mock_redis.setex.assert_called_once()
        self.mock_redis.srem.assert_called_with("test:progress:queue", "corr123")
        self.mock_redis.sadd.assert_called_with("test:progress:active", "corr123")

    def test_track_file_started_missing_data(self) -> None:
        """Test handling of missing progress data."""
        self.mock_redis.get.return_value = None

        # Should not raise error
        self.tracker.track_file_started("corr123")

        # Should not update Redis
        self.mock_redis.setex.assert_not_called()

    def test_update_progress(self) -> None:
        """Test updating file progress."""
        # Mock existing progress data
        progress_data = {
            "file_path": "/path/to/file.mp3",
            "recording_id": "rec123",
            "correlation_id": "corr123",
            "status": "in_progress",
            "queued_at": time.time(),
            "started_at": time.time(),
            "completed_at": None,
            "retry_count": 0,
            "error_message": None,
            "progress_percentage": 25.0,
            "current_step": "Analyzing BPM",
        }
        self.mock_redis.get.return_value = json.dumps(progress_data)

        self.tracker.update_progress("corr123", 50.0, "Detecting key")

        # Check Redis operations
        self.mock_redis.get.assert_called_with("test:progress:file:corr123")
        self.mock_redis.setex.assert_called_once()

        # Verify progress was updated
        call_args = self.mock_redis.setex.call_args[0]
        updated_data = json.loads(call_args[2])
        self.assertEqual(updated_data["progress_percentage"], 50.0)
        self.assertEqual(updated_data["current_step"], "Detecting key")

    def test_update_progress_clamping(self) -> None:
        """Test that progress percentage is clamped to 0-100."""
        progress_data = {
            "file_path": "/path/to/file.mp3",
            "recording_id": "rec123",
            "correlation_id": "corr123",
            "status": "in_progress",
            "queued_at": time.time(),
            "started_at": time.time(),
            "completed_at": None,
            "retry_count": 0,
            "error_message": None,
            "progress_percentage": 50.0,
            "current_step": None,
        }
        self.mock_redis.get.return_value = json.dumps(progress_data)

        # Test clamping above 100
        self.tracker.update_progress("corr123", 150.0)
        call_args = self.mock_redis.setex.call_args[0]
        updated_data = json.loads(call_args[2])
        self.assertEqual(updated_data["progress_percentage"], 100.0)

        # Test clamping below 0
        self.tracker.update_progress("corr123", -50.0)
        call_args = self.mock_redis.setex.call_args[0]
        updated_data = json.loads(call_args[2])
        self.assertEqual(updated_data["progress_percentage"], 0.0)

    def test_track_file_completed_success(self) -> None:
        """Test tracking successful file completion."""
        current_time = time.time()
        progress_data = {
            "file_path": "/path/to/file.mp3",
            "recording_id": "rec123",
            "correlation_id": "corr123",
            "status": "in_progress",
            "queued_at": current_time - 10,
            "started_at": current_time - 5,
            "completed_at": None,
            "retry_count": 0,
            "error_message": None,
            "progress_percentage": 75.0,
            "current_step": "Analyzing mood",
        }
        self.mock_redis.get.return_value = json.dumps(progress_data)

        self.tracker.track_file_completed("corr123", success=True)

        # Check Redis operations
        self.mock_redis.srem.assert_called_with("test:progress:active", "corr123")
        self.mock_redis.sadd.assert_called_with("test:progress:completed", "corr123")
        self.mock_redis.hincrby.assert_called()

        # Verify progress was updated
        call_args = self.mock_redis.setex.call_args[0]
        updated_data = json.loads(call_args[2])
        self.assertEqual(updated_data["status"], "completed")
        self.assertEqual(updated_data["progress_percentage"], 100.0)
        self.assertIsNotNone(updated_data["completed_at"])

    def test_track_file_completed_failure(self) -> None:
        """Test tracking failed file completion."""
        progress_data = {
            "file_path": "/path/to/file.mp3",
            "recording_id": "rec123",
            "correlation_id": "corr123",
            "status": "in_progress",
            "queued_at": time.time(),
            "started_at": time.time(),
            "completed_at": None,
            "retry_count": 0,
            "error_message": None,
            "progress_percentage": 50.0,
            "current_step": "Analyzing BPM",
        }
        self.mock_redis.get.return_value = json.dumps(progress_data)

        self.tracker.track_file_completed("corr123", success=False, error_message="BPM detection failed")

        # Check Redis operations
        self.mock_redis.srem.assert_called_with("test:progress:active", "corr123")
        self.mock_redis.sadd.assert_called_with("test:progress:failed", "corr123")

        # Verify progress was updated
        call_args = self.mock_redis.setex.call_args[0]
        updated_data = json.loads(call_args[2])
        self.assertEqual(updated_data["status"], "failed")
        self.assertEqual(updated_data["error_message"], "BPM detection failed")
        self.assertEqual(updated_data["progress_percentage"], 50.0)  # Not changed on failure

    def test_track_file_retry(self) -> None:
        """Test tracking file retry."""
        progress_data = {
            "file_path": "/path/to/file.mp3",
            "recording_id": "rec123",
            "correlation_id": "corr123",
            "status": "failed",
            "queued_at": time.time(),
            "started_at": time.time(),
            "completed_at": time.time(),
            "retry_count": 1,
            "error_message": "Temporary error",
            "progress_percentage": 50.0,
            "current_step": "Failed",
        }
        self.mock_redis.get.return_value = json.dumps(progress_data)

        self.tracker.track_file_retry("corr123")

        # Check Redis operations
        self.mock_redis.srem.assert_called_with("test:progress:failed", "corr123")
        self.mock_redis.sadd.assert_called_with("test:progress:queue", "corr123")

        # Verify retry count was incremented
        call_args = self.mock_redis.setex.call_args[0]
        updated_data = json.loads(call_args[2])
        self.assertEqual(updated_data["status"], "retrying")
        self.assertEqual(updated_data["retry_count"], 2)

    def test_get_progress(self) -> None:
        """Test getting progress for a file."""
        progress_data = {
            "file_path": "/path/to/file.mp3",
            "recording_id": "rec123",
            "correlation_id": "corr123",
            "status": "in_progress",
            "queued_at": time.time(),
            "started_at": time.time(),
            "completed_at": None,
            "retry_count": 0,
            "error_message": None,
            "progress_percentage": 75.0,
            "current_step": "Analyzing mood",
        }
        self.mock_redis.get.return_value = json.dumps(progress_data)

        progress = self.tracker.get_progress("corr123")

        self.assertIsNotNone(progress)
        assert progress is not None  # Type guard for mypy
        self.assertEqual(progress.file_path, "/path/to/file.mp3")
        self.assertEqual(progress.status, ProcessingStatus.IN_PROGRESS)
        self.assertEqual(progress.progress_percentage, 75.0)

    def test_get_progress_not_found(self) -> None:
        """Test getting progress for non-existent file."""
        self.mock_redis.get.return_value = None

        progress = self.tracker.get_progress("nonexistent")

        self.assertIsNone(progress)

    def test_get_queue_status(self) -> None:
        """Test getting queue status."""
        # Mock queue members
        self.mock_redis.smembers.side_effect = [
            {"corr1", "corr2"},  # queue
            {"corr3"},  # active
            {"corr4", "corr5"},  # completed
            {"corr6"},  # failed
        ]
        self.mock_redis.scard.side_effect = [2, 1, 2, 1]
        self.mock_redis.hgetall.return_value = {
            "total_queued": "10",
            "total_completed": "5",
            "total_failed": "2",
        }

        # Mock progress data for samples
        progress_data = {
            "file_path": "/path/to/file.mp3",
            "recording_id": "rec123",
            "correlation_id": "corr1",
            "status": "queued",
            "queued_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "retry_count": 0,
            "error_message": None,
            "progress_percentage": 0.0,
            "current_step": None,
        }
        self.mock_redis.get.return_value = json.dumps(progress_data)

        status = self.tracker.get_queue_status()

        self.assertIn("statistics", status)
        self.assertEqual(status["queue_depth"], 2)
        self.assertEqual(status["active_count"], 1)
        self.assertEqual(status["completed_count"], 2)
        self.assertEqual(status["failed_count"], 1)
        self.assertIn("recent_queued", status)
        self.assertIn("recent_active", status)

    def test_get_statistics(self) -> None:
        """Test getting processing statistics."""
        self.mock_redis.hgetall.return_value = {
            "total_queued": "100",
            "total_completed": "80",
            "total_failed": "10",
            "total_processing_time": "1600.5",
            "processed_file_count": "90",
        }

        stats = self.tracker.get_statistics()

        self.assertEqual(stats["total_queued"], 100)
        self.assertEqual(stats["total_completed"], 80)
        self.assertEqual(stats["total_failed"], 10)
        self.assertAlmostEqual(stats["average_processing_time"], 17.78, places=1)
        self.assertAlmostEqual(stats["success_rate"], 88.89, places=1)

    def test_clear_old_entries(self) -> None:
        """Test clearing old progress entries."""
        current_time = time.time()
        old_time = current_time - (25 * 3600)  # 25 hours ago

        # Mock all correlation IDs
        self.mock_redis.smembers.side_effect = [
            {"corr1"},  # queue
            {"corr2"},  # active
            {"corr3"},  # completed
            {"corr4"},  # failed
        ]

        # Mock progress data - some old, some new
        progress_data_old = {
            "file_path": "/old/file.mp3",
            "recording_id": "rec1",
            "correlation_id": "corr1",
            "status": "queued",
            "queued_at": old_time,
            "started_at": None,
            "completed_at": None,
            "retry_count": 0,
            "error_message": None,
            "progress_percentage": 0.0,
            "current_step": None,
        }

        progress_data_new = {
            "file_path": "/new/file.mp3",
            "recording_id": "rec2",
            "correlation_id": "corr2",
            "status": "in_progress",
            "queued_at": current_time - 3600,
            "started_at": current_time - 1800,
            "completed_at": None,
            "retry_count": 0,
            "error_message": None,
            "progress_percentage": 50.0,
            "current_step": None,
        }

        # Return old data for corr1 and corr3, new data for corr2, None for corr4
        self.mock_redis.get.side_effect = [
            json.dumps(progress_data_old),  # corr1
            json.dumps(progress_data_new),  # corr2
            json.dumps(progress_data_old),  # corr3
            None,  # corr4
        ]

        cleared_count = self.tracker.clear_old_entries(older_than_hours=24)

        self.assertEqual(cleared_count, 2)
        # Should delete old entries
        self.assertEqual(self.mock_redis.delete.call_count, 2)

    def test_reset_statistics(self) -> None:
        """Test resetting statistics."""
        self.tracker.reset_statistics()

        self.mock_redis.delete.assert_called_once_with("test:progress:stats")


if __name__ == "__main__":
    unittest.main()
