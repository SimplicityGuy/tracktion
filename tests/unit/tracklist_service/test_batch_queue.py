"""Unit tests for batch job queue management."""

import uuid
from datetime import UTC, datetime
from unittest.mock import Mock, call, patch

import pytest
from redis import Redis

from services.tracklist_service.src.queue.batch_queue import BatchJobQueue, Job, JobPriority


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    return Mock(spec=Redis)


@pytest.fixture
def mock_pika():
    """Create mock pika components."""
    with patch("services.tracklist_service.src.queue.batch_queue.pika") as mock:
        mock_connection = Mock()
        mock_channel = Mock()
        mock.BlockingConnection.return_value = mock_connection
        mock_connection.channel.return_value = mock_channel
        yield mock, mock_connection, mock_channel


@pytest.fixture
def batch_queue(mock_redis, mock_pika):
    """Create BatchJobQueue instance with mocks."""
    with patch("services.tracklist_service.src.queue.batch_queue.Redis") as mock_redis_cls:
        mock_redis_cls.return_value = mock_redis
        return BatchJobQueue()


class TestBatchJobQueue:
    """Test BatchJobQueue functionality."""

    def test_initialization(self, mock_pika):
        """Test queue initialization."""
        mock, mock_connection, mock_channel = mock_pika

        with patch("services.tracklist_service.src.queue.batch_queue.Redis"):
            BatchJobQueue()

            # Verify RabbitMQ setup
            mock.BlockingConnection.assert_called_once()
            mock_connection.channel.assert_called_once()

            # Verify queues declared
            assert mock_channel.queue_declare.call_count >= 3  # Priority queues

            # Check for DLQ declaration
            dlq_call = call(
                queue="batch_jobs_dlq",
                durable=True,
                arguments={"x-message-ttl": 604800000},
            )
            assert dlq_call in mock_channel.queue_declare.call_args_list

    def test_enqueue_batch(self, batch_queue, mock_redis):
        """Test batch enqueueing."""
        urls = ["http://example.com/1", "http://example.com/2"]
        user_id = "user123"

        # Mock Redis operations
        mock_redis.hset.return_value = True
        mock_redis.expire.return_value = True
        mock_redis.get.return_value = None  # No duplicates
        mock_redis.setex.return_value = True
        mock_redis.sadd.return_value = True

        batch_id = batch_queue.enqueue_batch(urls, "normal", user_id)

        # Verify batch ID returned
        assert batch_id
        assert isinstance(batch_id, str)

        # Verify batch metadata stored
        assert mock_redis.hset.called
        batch_key = f"batch:{batch_id}"
        assert any(batch_key in str(call) for call in mock_redis.hset.call_args_list)

        # Verify jobs published to queue
        assert batch_queue.channel.basic_publish.call_count == len(urls)

    def test_deduplicate_jobs(self, batch_queue, mock_redis):
        """Test job deduplication."""
        # Create duplicate jobs
        jobs = [
            Job(
                id=str(uuid.uuid4()),
                batch_id="batch1",
                url="http://example.com/track1",
                priority=JobPriority.NORMAL,
                user_id="user1",
                created_at=datetime.now(UTC),
            ),
            Job(
                id=str(uuid.uuid4()),
                batch_id="batch1",
                url="http://example.com/track1",  # Duplicate URL
                priority=JobPriority.NORMAL,
                user_id="user1",
                created_at=datetime.now(UTC),
            ),
            Job(
                id=str(uuid.uuid4()),
                batch_id="batch1",
                url="http://example.com/track2",
                priority=JobPriority.NORMAL,
                user_id="user1",
                created_at=datetime.now(UTC),
            ),
        ]

        # Mock Redis to indicate no existing jobs
        mock_redis.get.return_value = None
        mock_redis.hgetall.return_value = {}

        deduplicated = batch_queue.deduplicate_jobs(jobs)

        # Should remove one duplicate
        assert len(deduplicated) == 2
        unique_urls = {job.url for job in deduplicated}
        assert len(unique_urls) == 2

    def test_deduplicate_with_existing_jobs(self, batch_queue, mock_redis):
        """Test deduplication with existing active jobs."""
        jobs = [
            Job(
                id=str(uuid.uuid4()),
                batch_id="batch1",
                url="http://example.com/track1",
                priority=JobPriority.NORMAL,
                user_id="user1",
                created_at=datetime.now(UTC),
            ),
        ]

        # Mock existing active job
        mock_redis.get.return_value = "existing_job_id"
        mock_redis.hgetall.return_value = {"status": "processing"}

        deduplicated = batch_queue.deduplicate_jobs(jobs)

        # Should filter out job with active duplicate
        assert len(deduplicated) == 0

    def test_schedule_batch(self, batch_queue, mock_redis):
        """Test batch scheduling with cron expression."""
        urls = ["http://example.com/1"]
        cron_expr = "0 */6 * * *"  # Every 6 hours
        user_id = "user123"

        mock_redis.hset.return_value = True
        mock_redis.zadd.return_value = True

        schedule_id = batch_queue.schedule_batch(urls, cron_expr, user_id)

        # Verify schedule ID returned
        assert schedule_id
        assert isinstance(schedule_id, str)

        # Verify schedule stored
        assert mock_redis.hset.called
        schedule_key = f"schedule:{schedule_id}"
        assert any(schedule_key in str(call) for call in mock_redis.hset.call_args_list)

        # Verify added to sorted set
        assert mock_redis.zadd.called

    def test_schedule_batch_invalid_cron(self, batch_queue):
        """Test scheduling with invalid cron expression."""
        urls = ["http://example.com/1"]
        invalid_cron = "invalid cron"

        with pytest.raises(ValueError, match="Invalid cron expression"):
            batch_queue.schedule_batch(urls, invalid_cron)

    def test_get_batch_status(self, batch_queue, mock_redis):
        """Test retrieving batch status."""
        batch_id = "test_batch_id"

        # Mock batch metadata
        mock_redis.hgetall.return_value = {
            "batch_id": batch_id,
            "total_jobs": "3",
            "status": "processing",
        }

        # Mock job IDs
        mock_redis.smembers.return_value = {"job1", "job2", "job3"}

        # Mock job statuses
        def mock_job_data(key):
            if "job1" in key:
                return {"status": "completed"}
            if "job2" in key:
                return {"status": "processing"}
            if "job3" in key:
                return {"status": "pending"}
            return {}

        mock_redis.hgetall.side_effect = [
            mock_redis.hgetall.return_value,  # Batch metadata
            {"status": "completed"},  # job1
            {"status": "processing"},  # job2
            {"status": "pending"},  # job3
        ]

        status = batch_queue.get_batch_status(batch_id)

        assert status["batch_id"] == batch_id
        assert "jobs_status" in status
        assert status["jobs_status"]["completed"] == 1
        assert status["jobs_status"]["processing"] == 1
        assert status["jobs_status"]["pending"] == 1
        assert "progress_percentage" in status

    def test_get_batch_status_not_found(self, batch_queue, mock_redis):
        """Test status retrieval for non-existent batch."""
        mock_redis.hgetall.return_value = {}

        status = batch_queue.get_batch_status("non_existent")

        assert status == {"error": "Batch not found"}

    def test_cancel_batch(self, batch_queue, mock_redis):
        """Test batch cancellation."""
        batch_id = "test_batch_id"

        # Mock batch exists
        mock_redis.hgetall.return_value = {"batch_id": batch_id, "status": "processing"}
        mock_redis.smembers.return_value = {"job1", "job2"}
        mock_redis.hset.return_value = True

        # Mock job statuses
        def mock_job_data(key):
            if "job1" in key:
                return {"status": "pending"}
            if "job2" in key:
                return {"status": "completed"}
            return {}

        mock_redis.hgetall.side_effect = [
            mock_redis.hgetall.return_value,  # Batch metadata
            {"status": "pending"},  # job1
            {"status": "completed"},  # job2
        ]

        result = batch_queue.cancel_batch(batch_id)

        assert result is True

        # Verify batch status updated
        mock_redis.hset.assert_any_call(f"batch:{batch_id}", "status", "cancelled")

    def test_cancel_batch_not_found(self, batch_queue, mock_redis):
        """Test cancellation of non-existent batch."""
        mock_redis.hgetall.return_value = {}

        result = batch_queue.cancel_batch("non_existent")

        assert result is False

    def test_job_priority_enum(self):
        """Test JobPriority enum values."""
        assert JobPriority.IMMEDIATE.value == 0
        assert JobPriority.NORMAL.value == 5
        assert JobPriority.LOW.value == 10

    def test_close_connections(self, batch_queue, mock_redis):
        """Test connection cleanup."""
        batch_queue.connection.is_closed = False

        batch_queue.close()

        batch_queue.connection.close.assert_called_once()
        mock_redis.close.assert_called_once()
