"""Batch processing API endpoints for tracklist service."""

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field, HttpUrl, field_validator

from services.tracklist_service.src.queue.batch_queue import BatchJobQueue

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["batch"])


class BatchPriority(str, Enum):
    """Batch processing priority levels."""

    IMMEDIATE = "immediate"
    NORMAL = "normal"
    LOW = "low"


class BatchTemplate(str, Enum):
    """Predefined batch templates."""

    DJ_SET = "dj_set"
    FESTIVAL = "festival"
    PODCAST = "podcast"
    COMPILATION = "compilation"


class BatchRequest(BaseModel):
    """Request model for batch processing."""

    urls: list[HttpUrl] = Field(..., min_length=1, max_length=1000)
    priority: BatchPriority = BatchPriority.NORMAL
    user_id: str | None = None
    template: BatchTemplate | None = None
    options: dict[str, Any] | None = None

    @field_validator("urls")
    def validate_urls(self, v: list[HttpUrl]) -> list[HttpUrl]:
        """Validate URLs are from supported domains."""
        supported_domains = ["1001tracklists.com", "www.1001tracklists.com"]
        for url in v:
            if url.host not in supported_domains:
                raise ValueError(f"Unsupported domain: {url.host}")
        return v


class BatchResponse(BaseModel):
    """Response model for batch submission."""

    batch_id: str
    total_jobs: int
    priority: str
    status: str
    estimated_completion: datetime | None = None
    message: str = "Batch successfully queued"


class BatchStatus(BaseModel):
    """Status model for batch jobs."""

    batch_id: str
    status: str
    total_jobs: int
    jobs_status: dict[str, int]
    progress_percentage: float
    created_at: str | None = None
    updated_at: str | None = None
    error: str | None = None


class BatchAction(str, Enum):
    """Batch action types."""

    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"


class BatchScheduleRequest(BaseModel):
    """Request model for scheduling batch jobs."""

    urls: list[HttpUrl] = Field(..., min_length=1, max_length=1000)
    cron_expression: str = Field(..., description="Cron expression for scheduling")
    user_id: str | None = None
    name: str | None = None


# Module-level queue instance (would be dependency injected in production)
_batch_queue: BatchJobQueue | None = None


def get_batch_queue() -> BatchJobQueue:
    """Get or create batch queue instance."""
    global _batch_queue  # noqa: PLW0603 - Module-level singleton pattern for API endpoints
    if _batch_queue is None:
        _batch_queue = BatchJobQueue()
    return _batch_queue


@router.post("/batch", response_model=BatchResponse)
async def create_batch(
    request: BatchRequest,
    background_tasks: BackgroundTasks,
) -> BatchResponse:
    """Submit multiple URLs for batch processing.

    Args:
        request: Batch processing request
        background_tasks: Background task manager

    Returns:
        Batch response with tracking information
    """
    try:
        queue = get_batch_queue()

        # Convert HttpUrl objects to strings
        urls = [str(url) for url in request.urls]

        # Apply template if specified
        if request.template:
            urls = apply_template(urls, request.template, request.options)

        # Enqueue batch
        batch_id = queue.enqueue_batch(
            urls=urls,
            priority=request.priority.value,
            user_id=request.user_id or "anonymous",
        )

        # Calculate estimated completion
        estimated_time = calculate_estimated_completion(len(urls), request.priority)

        # Get initial status
        status = queue.get_batch_status(batch_id)

        return BatchResponse(
            batch_id=batch_id,
            total_jobs=len(urls),
            priority=request.priority.value,
            status=status.get("status", "queued"),
            estimated_completion=estimated_time,
        )

    except Exception as e:
        logger.error(f"Failed to create batch: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/batch/{batch_id}/status", response_model=BatchStatus)
async def get_batch_status(batch_id: str) -> BatchStatus:
    """Get detailed status of a batch job.

    Args:
        batch_id: Batch identifier

    Returns:
        Detailed batch status
    """
    try:
        queue = get_batch_queue()
        status = queue.get_batch_status(batch_id)

        if "error" in status:
            raise HTTPException(status_code=404, detail=status["error"])

        return BatchStatus(
            batch_id=batch_id,
            status=status.get("status", "unknown"),
            total_jobs=int(status.get("total_jobs", 0)),
            jobs_status=status.get("jobs_status", {}),
            progress_percentage=status.get("progress_percentage", 0.0),
            created_at=status.get("created_at"),
            updated_at=datetime.now(UTC).isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get batch status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/batch/{batch_id}/{action}")
async def control_batch(batch_id: str, action: BatchAction) -> dict[str, Any]:
    """Control a running batch job (pause/resume/cancel).

    Args:
        batch_id: Batch identifier
        action: Action to perform

    Returns:
        Action result
    """
    try:
        queue = get_batch_queue()

        if action == BatchAction.CANCEL:
            success = queue.cancel_batch(batch_id)
            if not success:
                raise HTTPException(status_code=404, detail="Batch not found")
            return {"status": "success", "message": f"Batch {batch_id} cancelled"}

        if action == BatchAction.PAUSE:
            # Implementation would pause job processing
            return {"status": "success", "message": f"Batch {batch_id} paused"}

        if action == BatchAction.RESUME:
            # Implementation would resume job processing
            return {"status": "success", "message": f"Batch {batch_id} resumed"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to {action} batch: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/batch/{batch_id}/cancel")
async def cancel_batch(batch_id: str) -> dict[str, Any]:
    """Cancel a running batch job.

    Args:
        batch_id: Batch identifier

    Returns:
        Cancellation result
    """
    result = await control_batch(batch_id, BatchAction.CANCEL)
    return dict(result)


@router.post("/batch/schedule", response_model=dict[str, str])
async def schedule_batch(request: BatchScheduleRequest) -> dict[str, str]:
    """Schedule a batch for recurring execution.

    Args:
        request: Batch scheduling request

    Returns:
        Schedule information
    """
    try:
        queue = get_batch_queue()

        # Convert URLs to strings
        urls = [str(url) for url in request.urls]

        # Schedule batch
        schedule_id = queue.schedule_batch(
            urls=urls,
            cron_expression=request.cron_expression,
            user_id=request.user_id or "anonymous",
        )

        return {
            "schedule_id": schedule_id,
            "message": "Batch scheduled successfully",
            "cron": request.cron_expression,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to schedule batch: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.websocket("/batch/{batch_id}/progress")
async def progress_websocket(websocket: WebSocket, batch_id: str) -> None:
    """WebSocket endpoint for real-time batch progress updates.

    Args:
        websocket: WebSocket connection
        batch_id: Batch identifier
    """
    await websocket.accept()

    try:
        queue = get_batch_queue()

        # Verify batch exists
        status = queue.get_batch_status(batch_id)
        if "error" in status:
            await websocket.send_json({"error": "Batch not found"})
            await websocket.close()
            return

        # Send updates every second
        while True:
            status = queue.get_batch_status(batch_id)

            # Send progress update
            await websocket.send_json(
                {
                    "batch_id": batch_id,
                    "status": status.get("status"),
                    "progress": status.get("progress_percentage", 0),
                    "jobs_status": status.get("jobs_status", {}),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

            # Check if batch is complete
            if status.get("status") in ["completed", "failed", "cancelled"]:
                break

            await asyncio.sleep(1)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for batch {batch_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()


def apply_template(
    urls: list[str],
    template: BatchTemplate,
    options: dict[str, Any] | None = None,
) -> list[str]:
    """Apply a template to batch URLs.

    Args:
        urls: List of URLs
        template: Template to apply
        options: Template options

    Returns:
        Modified URL list
    """
    # Template logic would go here
    # For now, just return original URLs
    return urls


def calculate_estimated_completion(
    job_count: int,
    priority: BatchPriority,
) -> datetime:
    """Calculate estimated completion time.

    Args:
        job_count: Number of jobs
        priority: Job priority

    Returns:
        Estimated completion datetime
    """
    # Estimate 6 seconds per job for normal priority
    seconds_per_job = {
        BatchPriority.IMMEDIATE: 3,
        BatchPriority.NORMAL: 6,
        BatchPriority.LOW: 10,
    }

    total_seconds = job_count * seconds_per_job[priority]

    return datetime.now(UTC).replace(microsecond=0) + timedelta(seconds=total_seconds)
