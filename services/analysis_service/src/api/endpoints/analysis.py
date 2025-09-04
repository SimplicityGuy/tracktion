"""Analysis endpoints for Analysis Service."""

import os
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from services.analysis_service.src.api_message_publisher import APIMessagePublisher
from services.analysis_service.src.repositories import (
    AsyncAnalysisResultRepository,
    AsyncRecordingRepository,
)
from services.analysis_service.src.structured_logging import get_logger
from shared.core_types.src.async_database import AsyncDatabaseManager

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/analysis", tags=["analysis"])

# Initialize database and message queue components
db_manager = AsyncDatabaseManager()
message_publisher = APIMessagePublisher(rabbitmq_url=os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/"))
recording_repo = AsyncRecordingRepository(db_manager)
analysis_repo = AsyncAnalysisResultRepository(db_manager)


class AnalysisRequest(BaseModel):
    """Request model for analysis."""

    recording_id: UUID
    analysis_types: list[str] = ["bpm", "key", "mood", "energy"]
    priority: int = 5


class AnalysisResult(BaseModel):
    """Model for analysis result."""

    type: str
    value: Any
    confidence: float
    metadata: dict[str, Any] = {}


class AnalysisResponse(BaseModel):
    """Response model for analysis."""

    recording_id: UUID
    status: str
    progress: float
    results: list[AnalysisResult]
    started_at: str | None
    completed_at: str | None


@router.post("")
async def start_analysis(request: AnalysisRequest) -> dict[str, Any]:
    """Start analysis for a recording.

    Args:
        request: Analysis request

    Returns:
        Analysis task confirmation
    """
    # Verify recording exists
    recording = await recording_repo.get_by_id(request.recording_id)
    if not recording:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Recording not found: {request.recording_id}"
        )

    # Verify file path and existence
    if not recording.file_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Recording {request.recording_id} has no file path"
        )

    if not Path(recording.file_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Audio file not found: {recording.file_path}"
        )

    # Submit analysis request to message queue
    correlation_id = await message_publisher.publish_analysis_request(
        recording_id=request.recording_id,
        file_path=recording.file_path,
        analysis_types=request.analysis_types,
        priority=request.priority,
    )

    # Update recording status to processing
    await recording_repo.update_status(request.recording_id, "processing")

    logger.info(
        "Analysis started",
        extra={
            "recording_id": str(request.recording_id),
            "types": request.analysis_types,
            "priority": request.priority,
            "correlation_id": correlation_id,
        },
    )

    return {
        "task_id": correlation_id,
        "recording_id": str(request.recording_id),
        "status": "queued",
        "message": "Analysis started",
        "analysis_types": request.analysis_types,
        "correlation_id": correlation_id,
    }


@router.get("/{recording_id}")
async def get_analysis_status(recording_id: UUID) -> AnalysisResponse:
    """Get analysis status and results for a recording.

    Args:
        recording_id: UUID of the recording

    Returns:
        Analysis status and results
    """
    # Verify recording exists
    recording = await recording_repo.get_by_id(recording_id)
    if not recording:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Recording not found: {recording_id}")

    # Get analysis results from database
    analysis_results = await analysis_repo.get_by_recording_id(recording_id)

    # Convert analysis results to response format
    result_items: list[AnalysisResult] = [
        AnalysisResult(
            type=analysis.analysis_type or "unknown",
            value=analysis.result_data,
            confidence=analysis.confidence_score or 0.0,
            metadata={
                "processing_time_ms": analysis.processing_time_ms,
                "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
            },
        )
        for analysis in analysis_results
        if analysis.result_data and analysis.status == "completed"
    ]

    # Determine overall status and progress
    if not analysis_results:
        status_str = "pending"
        progress = 0.0
    elif any(r.status == "failed" for r in analysis_results):
        status_str = "failed"
        progress = 1.0
    elif any(r.status == "processing" for r in analysis_results):
        status_str = "processing"
        completed_count = sum(1 for r in analysis_results if r.status == "completed")
        progress = completed_count / len(analysis_results)
    elif all(r.status == "completed" for r in analysis_results):
        status_str = "completed"
        progress = 1.0
    else:
        status_str = "processing"
        completed_count = sum(1 for r in analysis_results if r.status == "completed")
        progress = completed_count / len(analysis_results)

    # Get timestamps from recording
    started_at = recording.created_at.isoformat() if recording.created_at else None
    completed_at = None
    if status_str == "completed" and recording.updated_at:
        completed_at = recording.updated_at.isoformat()

    return AnalysisResponse(
        recording_id=recording_id,
        status=status_str,
        progress=progress,
        results=result_items,
        started_at=started_at,
        completed_at=completed_at,
    )


@router.get("/{recording_id}/bpm")
async def get_bpm_analysis(recording_id: UUID) -> dict[str, Any]:
    """Get BPM analysis for a recording.

    Args:
        recording_id: UUID of the recording

    Returns:
        BPM analysis results
    """
    # Verify recording exists
    recording = await recording_repo.get_by_id(recording_id)
    if not recording:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Recording not found: {recording_id}")

    # Get BPM analysis result from database
    bpm_result = await analysis_repo.get_by_recording_and_type(recording_id, "bpm")

    if not bpm_result or bpm_result.status != "completed":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="BPM analysis not found")

    return {
        "recording_id": str(recording_id),
        "analysis_type": "bpm",
        "result": bpm_result.result_data,
        "confidence": bpm_result.confidence_score,
        "status": bpm_result.status,
        "created_at": bpm_result.created_at.isoformat() if bpm_result.created_at else None,
        "processing_time_ms": bpm_result.processing_time_ms,
    }


@router.get("/{recording_id}/key")
async def get_key_analysis(recording_id: UUID) -> dict[str, Any]:
    """Get key detection analysis for a recording.

    Args:
        recording_id: UUID of the recording

    Returns:
        Key analysis results
    """
    # Verify recording exists
    recording = await recording_repo.get_by_id(recording_id)
    if not recording:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Recording not found: {recording_id}")

    # Get key analysis result from database
    key_result = await analysis_repo.get_by_recording_and_type(recording_id, "key")

    if not key_result or key_result.status != "completed":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Key analysis not found")

    return {
        "recording_id": str(recording_id),
        "analysis_type": "key",
        "result": key_result.result_data,
        "confidence": key_result.confidence_score,
        "status": key_result.status,
        "created_at": key_result.created_at.isoformat() if key_result.created_at else None,
        "processing_time_ms": key_result.processing_time_ms,
    }


@router.get("/{recording_id}/mood")
async def get_mood_analysis(recording_id: UUID) -> dict[str, Any]:
    """Get mood analysis for a recording.

    Args:
        recording_id: UUID of the recording

    Returns:
        Mood analysis results
    """
    # Verify recording exists
    recording = await recording_repo.get_by_id(recording_id)
    if not recording:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Recording not found: {recording_id}")

    # Get mood analysis result from database
    mood_result = await analysis_repo.get_by_recording_and_type(recording_id, "mood")

    if not mood_result or mood_result.status != "completed":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Mood analysis not found")

    return {
        "recording_id": str(recording_id),
        "analysis_type": "mood",
        "result": mood_result.result_data,
        "confidence": mood_result.confidence_score,
        "status": mood_result.status,
        "created_at": mood_result.created_at.isoformat() if mood_result.created_at else None,
        "processing_time_ms": mood_result.processing_time_ms,
    }


@router.post("/{recording_id}/waveform")
async def generate_waveform(
    recording_id: UUID,
    width: int = Query(1920, description="Waveform image width"),
    height: int = Query(256, description="Waveform image height"),
    color: str = Query("#00ff00", description="Waveform color"),
) -> dict[str, Any]:
    """Generate waveform visualization for a recording.

    Args:
        recording_id: UUID of the recording
        width: Image width in pixels
        height: Image height in pixels
        color: Waveform color (hex)

    Returns:
        Waveform generation task confirmation
    """
    # Verify recording exists
    recording = await recording_repo.get_by_id(recording_id)
    if not recording:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Recording not found: {recording_id}")

    # Verify file path and existence
    if not recording.file_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Recording {recording_id} has no file path"
        )

    if not Path(recording.file_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Audio file not found: {recording.file_path}"
        )

    # Submit waveform generation request to message queue
    correlation_id = await message_publisher.publish_analysis_request(
        recording_id=recording_id,
        file_path=recording.file_path,
        analysis_types=["waveform"],
        priority=6,
        metadata={"width": width, "height": height, "color": color},
    )

    logger.info(
        "Waveform generation started",
        extra={
            "recording_id": str(recording_id),
            "width": width,
            "height": height,
            "color": color,
            "correlation_id": correlation_id,
        },
    )

    return {
        "task_id": correlation_id,
        "recording_id": str(recording_id),
        "status": "generating",
        "message": "Waveform generation started",
        "parameters": {"width": width, "height": height, "color": color},
        "correlation_id": correlation_id,
    }


@router.post("/{recording_id}/spectrogram")
async def generate_spectrogram(
    recording_id: UUID,
    fft_size: int = Query(2048, description="FFT window size"),
    hop_size: int = Query(512, description="Hop size"),
    color_map: str = Query("viridis", description="Color map"),
) -> dict[str, Any]:
    """Generate spectrogram for a recording.

    Args:
        recording_id: UUID of the recording
        fft_size: FFT window size
        hop_size: Hop size between windows
        color_map: Color map for visualization

    Returns:
        Spectrogram generation task confirmation
    """
    # Verify recording exists
    recording = await recording_repo.get_by_id(recording_id)
    if not recording:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Recording not found: {recording_id}")

    # Verify file path and existence
    if not recording.file_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Recording {recording_id} has no file path"
        )

    if not Path(recording.file_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Audio file not found: {recording.file_path}"
        )

    # Submit spectrogram generation request to message queue
    correlation_id = await message_publisher.publish_analysis_request(
        recording_id=recording_id,
        file_path=recording.file_path,
        analysis_types=["spectrogram"],
        priority=5,
        metadata={
            "fft_size": fft_size,
            "hop_size": hop_size,
            "color_map": color_map,
        },
    )

    logger.info(
        "Spectrogram generation started",
        extra={
            "recording_id": str(recording_id),
            "fft_size": fft_size,
            "hop_size": hop_size,
            "color_map": color_map,
            "correlation_id": correlation_id,
        },
    )

    return {
        "task_id": correlation_id,
        "recording_id": str(recording_id),
        "status": "generating",
        "message": "Spectrogram generation started",
        "parameters": {
            "fft_size": fft_size,
            "hop_size": hop_size,
            "color_map": color_map,
        },
        "correlation_id": correlation_id,
    }
