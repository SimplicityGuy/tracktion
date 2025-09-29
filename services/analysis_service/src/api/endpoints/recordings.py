"""Recording management endpoints for Analysis Service."""

from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from services.analysis_service.src.api_message_publisher import APIMessagePublisher
from services.analysis_service.src.repositories import AsyncMetadataRepository, AsyncRecordingRepository
from services.analysis_service.src.structured_logging import get_logger
from shared.core_types.src.async_database import AsyncDatabaseManager

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/recordings", tags=["recordings"])

# Initialize dependencies - in production these would be dependency injected
db_manager = AsyncDatabaseManager()
recording_repository = AsyncRecordingRepository(db_manager)
message_publisher = APIMessagePublisher("amqp://localhost:5672")


class RecordingRequest(BaseModel):
    """Request model for recording analysis."""

    file_path: str
    priority: int | None = 5
    metadata: dict[str, Any] | None = {}


class RecordingResponse(BaseModel):
    """Response model for recording."""

    id: UUID
    file_path: str
    status: str
    priority: int
    metadata: dict[str, Any]


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def submit_recording(request: RecordingRequest) -> dict[str, Any]:
    """Submit a recording for analysis.

    Args:
        request: Recording submission request

    Returns:
        Submission confirmation with recording ID
    """
    try:
        # Create recording in database
        file_name = Path(request.file_path).name
        recording = await recording_repository.create(
            file_path=request.file_path,
            file_name=file_name,
        )

        # Validate recording was created successfully
        if not recording or not recording.id:
            raise HTTPException(status_code=500, detail="Failed to create recording")

        # Submit to message queue for processing
        correlation_id = await message_publisher.publish_analysis_request(
            recording_id=recording.id,
            file_path=request.file_path,
            analysis_types=["bpm", "key", "mood"],
            priority=request.priority or 5,
            metadata=request.metadata,
        )

        logger.info(
            "Recording submitted for analysis",
            extra={
                "recording_id": str(recording.id),
                "file_path": request.file_path,
                "priority": request.priority,
                "correlation_id": correlation_id,
            },
        )

        return {
            "id": str(recording.id),
            "status": "queued",
            "message": "Recording submitted for analysis",
            "correlation_id": correlation_id,
        }
    except Exception as e:
        logger.error(f"Failed to submit recording: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to submit recording for analysis: {e!s}") from e


@router.get("/{recording_id}")
async def get_recording_status(recording_id: UUID) -> RecordingResponse:
    """Get status of a recording analysis.

    Args:
        recording_id: UUID of the recording

    Returns:
        Recording status and metadata
    """
    try:
        recording = await recording_repository.get_by_id(recording_id)
        if not recording:
            raise HTTPException(status_code=404, detail=f"Recording {recording_id} not found")

        # Validate recording fields
        if not recording.id or not recording.file_path:
            raise HTTPException(status_code=500, detail="Recording data is incomplete")

        # Get metadata from database
        metadata_repo = AsyncMetadataRepository(recording_repository.db)
        metadata_list = await metadata_repo.get_by_recording_id(recording_id)
        metadata = {item.key: item.value for item in metadata_list}

        return RecordingResponse(
            id=recording.id,
            file_path=recording.file_path,
            status=recording.processing_status or "pending",
            priority=5,  # Default priority since not stored in Recording model
            metadata=metadata,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get recording status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve recording status: {e!s}") from e


@router.get("")
async def list_recordings(
    status: str | None = Query(None, description="Filter by status"),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> list[RecordingResponse]:
    """List recordings with optional filtering.

    Args:
        status: Optional status filter
        limit: Maximum number of results
        offset: Pagination offset

    Returns:
        List of recordings
    """
    try:
        page = (offset // limit) + 1
        recordings, _total_count = await recording_repository.list_paginated(
            page=page,
            limit=limit,
            status_filter=status,
        )

        # Convert to response format with validation
        validated_recordings: list[RecordingResponse] = []
        for recording in recordings:
            if not recording.id or not recording.file_path:
                continue  # Skip invalid recordings
            validated_recordings.append(
                RecordingResponse(
                    id=recording.id,
                    file_path=recording.file_path,
                    status=recording.processing_status or "pending",
                    priority=5,  # Default priority since not stored in Recording model
                    metadata={"file_size": recording.file_size} if recording.file_size else {},
                )
            )
        return validated_recordings
    except Exception as e:
        logger.error(f"Failed to list recordings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve recordings: {e!s}") from e


@router.delete("/{recording_id}")
async def cancel_recording(recording_id: UUID) -> dict[str, str]:
    """Cancel a recording analysis.

    Args:
        recording_id: UUID of the recording to cancel

    Returns:
        Cancellation confirmation
    """
    try:
        # Verify recording exists
        recording = await recording_repository.get_by_id(recording_id)
        if not recording:
            raise HTTPException(status_code=404, detail=f"Recording {recording_id} not found")

        # Update status to cancelled in database
        await recording_repository.update_status(recording_id, "cancelled")

        # Send cancellation message to queue
        correlation_id = await message_publisher.cancel_processing(recording_id)

        logger.info(
            "Cancelling recording analysis",
            extra={
                "recording_id": str(recording_id),
                "correlation_id": correlation_id,
            },
        )

        return {
            "id": str(recording_id),
            "status": "cancelled",
            "message": "Recording analysis cancelled",
            "correlation_id": correlation_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel recording: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel recording analysis: {e!s}") from e
