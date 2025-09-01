"""Unit tests for batch API endpoints."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException

from services.tracklist_service.src.api.batch_endpoints import (
    BatchAction,
    BatchPriority,
    BatchRequest,
    BatchResponse,
    BatchScheduleRequest,
    BatchStatus,
    BatchTemplate,
    calculate_estimated_completion,
    control_batch,
    create_batch,
    get_batch_status,
    schedule_batch,
)


@pytest.fixture
def mock_batch_queue():
    """Mock BatchJobQueue."""
    with patch("services.tracklist_service.src.api.batch_endpoints.get_batch_queue") as mock:
        queue = Mock()
        mock.return_value = queue
        yield queue


class TestBatchEndpoints:
    """Test batch API endpoints."""

    @pytest.mark.asyncio
    async def test_create_batch(self, mock_batch_queue):
        """Test batch creation endpoint."""
        # Setup
        mock_batch_queue.enqueue_batch.return_value = "batch-123"
        mock_batch_queue.get_batch_status.return_value = {
            "status": "queued",
            "total_jobs": 2,
        }

        request = BatchRequest(
            urls=[
                "http://1001tracklists.com/track1",
                "http://1001tracklists.com/track2",
            ],
            priority=BatchPriority.NORMAL,
            user_id="user123",
        )

        background_tasks = Mock()

        # Execute
        response = await create_batch(request, background_tasks)

        # Verify
        assert isinstance(response, BatchResponse)
        assert response.batch_id == "batch-123"
        assert response.total_jobs == 2
        assert response.priority == "normal"
        assert response.status == "queued"

        mock_batch_queue.enqueue_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_batch_with_template(self, mock_batch_queue):
        """Test batch creation with template."""
        mock_batch_queue.enqueue_batch.return_value = "batch-456"
        mock_batch_queue.get_batch_status.return_value = {"status": "queued"}

        request = BatchRequest(
            urls=["http://1001tracklists.com/set1"],
            template=BatchTemplate.DJ_SET,
            priority=BatchPriority.IMMEDIATE,
        )

        background_tasks = Mock()
        response = await create_batch(request, background_tasks)

        assert response.batch_id == "batch-456"
        assert response.priority == "immediate"

    @pytest.mark.asyncio
    async def test_create_batch_invalid_domain(self):
        """Test batch creation with invalid domain."""
        with pytest.raises(ValueError, match="Unsupported domain"):
            BatchRequest(
                urls=["http://example.com/track1"],
                priority=BatchPriority.NORMAL,
            )

    @pytest.mark.asyncio
    async def test_create_batch_error(self, mock_batch_queue):
        """Test batch creation error handling."""
        mock_batch_queue.enqueue_batch.side_effect = Exception("Queue error")

        request = BatchRequest(
            urls=["http://1001tracklists.com/track1"],
        )

        background_tasks = Mock()

        with pytest.raises(HTTPException) as exc_info:
            await create_batch(request, background_tasks)

        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_get_batch_status_success(self, mock_batch_queue):
        """Test getting batch status."""
        mock_batch_queue.get_batch_status.return_value = {
            "batch_id": "batch-123",
            "status": "processing",
            "total_jobs": "5",
            "jobs_status": {
                "completed": 2,
                "processing": 1,
                "pending": 2,
            },
            "progress_percentage": 40.0,
            "created_at": "2025-01-01T00:00:00",
        }

        response = await get_batch_status("batch-123")

        assert isinstance(response, BatchStatus)
        assert response.batch_id == "batch-123"
        assert response.status == "processing"
        assert response.total_jobs == 5
        assert response.progress_percentage == 40.0

    @pytest.mark.asyncio
    async def test_get_batch_status_not_found(self, mock_batch_queue):
        """Test getting status for non-existent batch."""
        mock_batch_queue.get_batch_status.return_value = {"error": "Batch not found"}

        with pytest.raises(HTTPException) as exc_info:
            await get_batch_status("invalid-batch")

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_control_batch_cancel(self, mock_batch_queue):
        """Test cancelling a batch."""
        mock_batch_queue.cancel_batch.return_value = True

        response = await control_batch("batch-123", BatchAction.CANCEL)

        assert response["status"] == "success"
        assert "cancelled" in response["message"]
        mock_batch_queue.cancel_batch.assert_called_once_with("batch-123")

    @pytest.mark.asyncio
    async def test_control_batch_cancel_not_found(self, mock_batch_queue):
        """Test cancelling non-existent batch."""
        mock_batch_queue.cancel_batch.return_value = False

        with pytest.raises(HTTPException) as exc_info:
            await control_batch("invalid-batch", BatchAction.CANCEL)

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_control_batch_pause(self, mock_batch_queue):
        """Test pausing a batch."""
        response = await control_batch("batch-123", BatchAction.PAUSE)

        assert response["status"] == "success"
        assert "paused" in response["message"]

    @pytest.mark.asyncio
    async def test_control_batch_resume(self, mock_batch_queue):
        """Test resuming a batch."""
        response = await control_batch("batch-123", BatchAction.RESUME)

        assert response["status"] == "success"
        assert "resumed" in response["message"]

    @pytest.mark.asyncio
    async def test_schedule_batch(self, mock_batch_queue):
        """Test scheduling a batch."""
        mock_batch_queue.schedule_batch.return_value = "schedule-123"

        request = BatchScheduleRequest(
            urls=["http://1001tracklists.com/track1"],
            cron_expression="0 */6 * * *",
            user_id="user123",
        )

        response = await schedule_batch(request)

        assert response["schedule_id"] == "schedule-123"
        assert "scheduled successfully" in response["message"]
        assert response["cron"] == "0 */6 * * *"

    @pytest.mark.asyncio
    async def test_schedule_batch_invalid_cron(self, mock_batch_queue):
        """Test scheduling with invalid cron expression."""
        mock_batch_queue.schedule_batch.side_effect = ValueError("Invalid cron")

        request = BatchScheduleRequest(
            urls=["http://1001tracklists.com/track1"],
            cron_expression="invalid",
        )

        with pytest.raises(HTTPException) as exc_info:
            await schedule_batch(request)

        assert exc_info.value.status_code == 400

    def test_calculate_estimated_completion(self):
        """Test estimated completion calculation."""
        # Test immediate priority
        result = calculate_estimated_completion(10, BatchPriority.IMMEDIATE)
        assert isinstance(result, datetime)

        # Test normal priority
        result = calculate_estimated_completion(10, BatchPriority.NORMAL)
        assert isinstance(result, datetime)

        # Test low priority
        result = calculate_estimated_completion(10, BatchPriority.LOW)
        assert isinstance(result, datetime)

    def test_batch_priority_enum(self):
        """Test BatchPriority enum values."""
        assert BatchPriority.IMMEDIATE.value == "immediate"
        assert BatchPriority.NORMAL.value == "normal"
        assert BatchPriority.LOW.value == "low"

    def test_batch_template_enum(self):
        """Test BatchTemplate enum values."""
        assert BatchTemplate.DJ_SET.value == "dj_set"
        assert BatchTemplate.FESTIVAL.value == "festival"
        assert BatchTemplate.PODCAST.value == "podcast"
        assert BatchTemplate.COMPILATION.value == "compilation"

    def test_batch_action_enum(self):
        """Test BatchAction enum values."""
        assert BatchAction.PAUSE.value == "pause"
        assert BatchAction.RESUME.value == "resume"
        assert BatchAction.CANCEL.value == "cancel"
