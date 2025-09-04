"""Tracklist management endpoints for Analysis Service."""

import os
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from services.analysis_service.src.api_message_publisher import APIMessagePublisher
from services.analysis_service.src.repositories import (
    AsyncRecordingRepository,
    AsyncTracklistRepository,
)
from services.analysis_service.src.structured_logging import get_logger
from shared.core_types.src.async_database import AsyncDatabaseManager

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/tracklist", tags=["tracklist"])

# Initialize database and message queue components
db_manager = AsyncDatabaseManager()
message_publisher = APIMessagePublisher(rabbitmq_url=os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/"))
recording_repo = AsyncRecordingRepository(db_manager)
tracklist_repo = AsyncTracklistRepository(db_manager)


class TrackInfo(BaseModel):
    """Model for track information."""

    index: int
    title: str
    artist: str | None
    start_time: float
    end_time: float
    duration: float
    file_path: str | None


class TracklistResponse(BaseModel):
    """Response model for tracklist."""

    recording_id: UUID
    format: str
    total_tracks: int
    total_duration: float
    tracks: list[TrackInfo]


class CueSheetRequest(BaseModel):
    """Request model for CUE sheet processing."""

    cue_content: str
    audio_file_path: str
    validate_cue: bool = True


@router.get("/{recording_id}")
async def get_tracklist(recording_id: UUID) -> TracklistResponse:
    """Get tracklist for a recording.

    Args:
        recording_id: UUID of the recording

    Returns:
        Tracklist information
    """
    # Verify recording exists
    recording = await recording_repo.get_by_id(recording_id)
    if not recording:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Recording not found: {recording_id}")

    # Get tracklist from database
    tracklist = await tracklist_repo.get_by_recording_id(recording_id)

    if not tracklist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Tracklist not found for recording: {recording_id}"
        )

    # Convert database tracks to API response format
    tracks: list[TrackInfo] = []
    if tracklist.tracks:
        for i, track_data in enumerate(tracklist.tracks, 1):
            # Create TrackInfo from the JSONB track data
            track = TrackInfo(
                index=track_data.get("index", i),
                title=track_data.get("title", f"Track {i}"),
                artist=track_data.get("artist"),
                start_time=track_data.get("start_time", 0.0),
                end_time=track_data.get("end_time", 0.0),
                duration=track_data.get("duration", 0.0),
                file_path=track_data.get("file_path"),
            )
            tracks.append(track)

    # Determine format based on source
    format_type = "cue" if tracklist.cue_file_path else (tracklist.source or "unknown")

    return TracklistResponse(
        recording_id=recording_id,
        format=format_type,
        total_tracks=len(tracks),
        total_duration=sum(t.duration for t in tracks),
        tracks=tracks,
    )


@router.post("/{recording_id}/detect")
async def detect_tracks(
    recording_id: UUID,
    min_duration: float = Query(30.0, description="Minimum track duration in seconds"),
    sensitivity: float = Query(0.5, ge=0.0, le=1.0, description="Detection sensitivity"),
) -> dict[str, Any]:
    """Detect tracks in a recording using silence detection.

    Args:
        recording_id: UUID of the recording
        min_duration: Minimum track duration
        sensitivity: Detection sensitivity (0-1)

    Returns:
        Detection task confirmation
    """
    # Verify recording exists
    recording = await recording_repo.get_by_id(recording_id)
    if not recording:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Recording not found: {recording_id}")

    # Verify file exists
    if not recording.file_path or not Path(recording.file_path or "").exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Audio file not found: {recording.file_path}"
        )

    # Submit track detection request to message queue
    correlation_id = await message_publisher.publish_tracklist_generation(
        recording_id=recording_id,
        source_hint="silence_detection",
        priority=6,
    )

    logger.info(
        "Track detection started",
        extra={
            "recording_id": str(recording_id),
            "min_duration": min_duration,
            "sensitivity": sensitivity,
            "correlation_id": correlation_id,
        },
    )

    return {
        "id": str(recording_id),
        "status": "detecting",
        "message": "Track detection started",
        "parameters": {"min_duration": min_duration, "sensitivity": sensitivity},
        "correlation_id": correlation_id,
    }


@router.post("/{recording_id}/split")
async def split_tracks(
    recording_id: UUID,
    output_format: str = Query("flac", description="Output format for split tracks"),
) -> dict[str, Any]:
    """Split recording into individual track files.

    Args:
        recording_id: UUID of the recording
        output_format: Format for split track files

    Returns:
        Split task confirmation
    """
    # Verify recording exists
    recording = await recording_repo.get_by_id(recording_id)
    if not recording:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Recording not found: {recording_id}")

    # Verify file exists
    if not recording.file_path or not Path(recording.file_path or "").exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Audio file not found: {recording.file_path}"
        )

    # Verify tracklist exists
    tracklist = await tracklist_repo.get_by_recording_id(recording_id)
    if not tracklist or not tracklist.tracks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No tracklist found - run track detection first"
        )

    # Submit track splitting request to message queue
    correlation_id = await message_publisher.publish_analysis_request(
        recording_id=recording_id,
        file_path=recording.file_path,
        analysis_types=["track_splitting"],
        priority=7,  # High priority for splitting
        metadata={"output_format": output_format, "tracklist_id": str(tracklist.id)},
    )

    logger.info(
        "Track splitting started",
        extra={
            "recording_id": str(recording_id),
            "output_format": output_format,
            "correlation_id": correlation_id,
        },
    )

    return {
        "id": str(recording_id),
        "status": "splitting",
        "message": "Track splitting started",
        "output_format": output_format,
        "correlation_id": correlation_id,
    }


@router.post("/parse-cue")
async def parse_cue_sheet(request: CueSheetRequest) -> dict[str, Any]:
    """Parse a CUE sheet and extract tracklist.

    Args:
        request: CUE sheet parsing request

    Returns:
        Parsed tracklist information
    """
    # Verify audio file exists
    if not Path(request.audio_file_path).exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Audio file not found: {request.audio_file_path}"
        )

    # Check if recording exists for this file
    recording = await recording_repo.get_by_file_path(request.audio_file_path)
    if not recording:
        # Create recording if it doesn't exist
        audio_file_path = Path(request.audio_file_path)
        file_name = audio_file_path.name
        file_size = audio_file_path.stat().st_size
        recording = await recording_repo.create(
            file_path=request.audio_file_path,
            file_name=file_name,
            file_size=file_size,
        )

    logger.info(
        "Parsing CUE sheet",
        extra={
            "recording_id": str(recording.id),
            "audio_file": request.audio_file_path,
            "validate": request.validate_cue,
        },
    )

    # Submit CUE parsing request to message queue
    if not recording.id:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Recording ID is None")

    correlation_id = await message_publisher.publish_analysis_request(
        recording_id=recording.id,
        file_path=request.audio_file_path,
        analysis_types=["cue_parsing"],
        priority=7,
        metadata={
            "cue_content": request.cue_content,
            "validate_cue": request.validate_cue,
        },
    )

    # For immediate response, we could parse the CUE content here
    # But for consistency, we'll return the task status
    return {
        "status": "parsing",
        "format": "cue",
        "recording_id": str(recording.id),
        "audio_file": request.audio_file_path,
        "message": "CUE sheet parsing started",
        "correlation_id": correlation_id,
    }


@router.put("/{recording_id}/tracks")
async def update_tracklist(recording_id: UUID, tracks: list[TrackInfo]) -> dict[str, str]:
    """Update tracklist for a recording.

    Args:
        recording_id: UUID of the recording
        tracks: Updated track information

    Returns:
        Update confirmation
    """
    # Verify recording exists
    recording = await recording_repo.get_by_id(recording_id)
    if not recording:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Recording not found: {recording_id}")

    # Validate track data
    if not tracks:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Track list cannot be empty")

    # Convert TrackInfo objects to dict format for JSONB storage
    track_dicts = []
    for track in tracks:
        track_dict = {
            "index": track.index,
            "title": track.title,
            "artist": track.artist,
            "start_time": track.start_time,
            "end_time": track.end_time,
            "duration": track.duration,
            "file_path": track.file_path,
        }
        track_dicts.append(track_dict)

    # Check if tracklist exists
    existing_tracklist = await tracklist_repo.get_by_recording_id(recording_id)

    if existing_tracklist:
        # Update existing tracklist
        await tracklist_repo.update_tracks(recording_id, track_dicts)
    else:
        # Create new tracklist
        await tracklist_repo.create(
            recording_id=recording_id,
            source="manual",
            tracks=track_dicts,
        )

    logger.info(
        "Tracklist updated",
        extra={
            "recording_id": str(recording_id),
            "track_count": len(tracks),
            "action": "updated" if existing_tracklist else "created",
        },
    )

    return {
        "id": str(recording_id),
        "status": "updated",
        "message": f"Tracklist updated with {len(tracks)} tracks",
    }
