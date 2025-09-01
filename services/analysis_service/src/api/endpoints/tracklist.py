"""Tracklist management endpoints for Analysis Service."""

from typing import Any
from uuid import UUID

from fastapi import APIRouter, Query
from pydantic import BaseModel

from services.analysis_service.src.structured_logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/tracklist", tags=["tracklist"])


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
    # In real implementation, fetch from database
    tracks = [
        TrackInfo(
            index=1,
            title="Track 1",
            artist="Artist 1",
            start_time=0.0,
            end_time=300.5,
            duration=300.5,
            file_path=None,
        ),
        TrackInfo(
            index=2,
            title="Track 2",
            artist="Artist 2",
            start_time=300.5,
            end_time=600.0,
            duration=299.5,
            file_path=None,
        ),
    ]

    return TracklistResponse(
        recording_id=recording_id,
        format="cue",
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
    logger.info(
        "Starting track detection",
        extra={
            "recording_id": str(recording_id),
            "min_duration": min_duration,
            "sensitivity": sensitivity,
        },
    )

    # In real implementation, send to processing queue
    return {
        "id": str(recording_id),
        "status": "detecting",
        "message": "Track detection started",
        "parameters": {"min_duration": min_duration, "sensitivity": sensitivity},
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
    logger.info(
        "Starting track splitting",
        extra={"recording_id": str(recording_id), "output_format": output_format},
    )

    # In real implementation, send to processing queue
    return {
        "id": str(recording_id),
        "status": "splitting",
        "message": "Track splitting started",
        "output_format": output_format,
    }


@router.post("/parse-cue")
async def parse_cue_sheet(request: CueSheetRequest) -> dict[str, Any]:
    """Parse a CUE sheet and extract tracklist.

    Args:
        request: CUE sheet parsing request

    Returns:
        Parsed tracklist information
    """
    logger.info(
        "Parsing CUE sheet",
        extra={"audio_file": request.audio_file_path, "validate": request.validate_cue},
    )

    # In real implementation, use CUE parser
    # For now, return mock data
    tracks = [
        {
            "index": 1,
            "title": "Parsed Track 1",
            "artist": "Artist",
            "start_time": 0.0,
            "end_time": 240.0,
            "duration": 240.0,
        }
    ]

    return {
        "status": "parsed",
        "format": "cue",
        "audio_file": request.audio_file_path,
        "total_tracks": len(tracks),
        "tracks": tracks,
        "validation": {"valid": True, "warnings": []} if request.validate_cue else None,
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
    logger.info(
        "Updating tracklist",
        extra={"recording_id": str(recording_id), "track_count": len(tracks)},
    )

    # In real implementation, update in database
    return {
        "id": str(recording_id),
        "status": "updated",
        "message": f"Tracklist updated with {len(tracks)} tracks",
    }
