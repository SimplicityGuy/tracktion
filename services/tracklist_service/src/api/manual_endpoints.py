"""API endpoints for manual tracklist creation and management.

This module provides REST endpoints for creating and managing
manual tracklists including track CRUD operations and draft management.
"""

from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from services.tracklist_service.src.models.tracklist import TrackEntry, Tracklist
from services.tracklist_service.src.services.draft_service import DraftService
from shared.core_types.src.database import get_db_session


router = APIRouter(prefix="/api/v1/tracklists", tags=["manual-tracklist"])


class CreateManualTracklistRequest(BaseModel):
    """Request model for creating manual tracklist."""

    audio_file_id: UUID = Field(description="ID of the audio file")
    tracks: Optional[List[TrackEntry]] = Field(None, description="Initial list of tracks")
    is_draft: bool = Field(default=True, description="Whether to save as draft")


class UpdateTracklistRequest(BaseModel):
    """Request model for updating a tracklist."""

    tracks: List[TrackEntry] = Field(description="Updated list of tracks")
    is_draft: bool = Field(default=True, description="Whether to keep as draft")


class AddTrackRequest(BaseModel):
    """Request model for adding a track."""

    position: int = Field(ge=1, description="Track position")
    artist: str = Field(description="Artist name(s)")
    title: str = Field(description="Track title")
    start_time: str = Field(description="Start time in HH:MM:SS format")
    end_time: Optional[str] = Field(None, description="End time in HH:MM:SS format")
    remix: Optional[str] = Field(None, description="Remix or edit information")
    label: Optional[str] = Field(None, description="Record label")


class UpdateTrackRequest(BaseModel):
    """Request model for updating a track."""

    artist: Optional[str] = Field(None, description="Artist name(s)")
    title: Optional[str] = Field(None, description="Track title")
    start_time: Optional[str] = Field(None, description="Start time in HH:MM:SS format")
    end_time: Optional[str] = Field(None, description="End time in HH:MM:SS format")
    remix: Optional[str] = Field(None, description="Remix or edit information")
    label: Optional[str] = Field(None, description="Record label")


class UpdateTrackTimingRequest(BaseModel):
    """Request model for updating track timing."""

    start_time: str = Field(description="Start time in HH:MM:SS format")
    end_time: Optional[str] = Field(None, description="End time in HH:MM:SS format")


def parse_time_string(time_str: str) -> int:
    """Parse HH:MM:SS or MM:SS time string to seconds.

    Args:
        time_str: Time string in HH:MM:SS or MM:SS format.

    Returns:
        Total seconds.

    Raises:
        ValueError: If time string format is invalid.
    """
    parts = time_str.split(":")
    if len(parts) == 2:
        # MM:SS format
        minutes, seconds = map(int, parts)
        return minutes * 60 + seconds
    elif len(parts) == 3:
        # HH:MM:SS format
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f"Invalid time format: {time_str}")


@router.post("/manual", response_model=Tracklist, status_code=status.HTTP_201_CREATED)
def create_manual_tracklist(
    request: CreateManualTracklistRequest,
    db: Session = Depends(get_db_session),
) -> Tracklist:
    """Create a new manual tracklist.

    Args:
        request: Create tracklist request.
        db: Database session.

    Returns:
        Created tracklist.
    """
    draft_service = DraftService(db)

    # Convert tracks if provided
    tracks = request.tracks or []

    # Create as draft or final
    if request.is_draft:
        tracklist = draft_service.create_draft(request.audio_file_id, tracks)
    else:
        # Create as draft first, then publish
        draft = draft_service.create_draft(request.audio_file_id, tracks)
        tracklist = draft_service.publish_draft(draft.id)

    return tracklist


@router.put("/{tracklist_id}", response_model=Tracklist)
def update_tracklist(
    tracklist_id: UUID,
    request: UpdateTracklistRequest,
    db: Session = Depends(get_db_session),
) -> Tracklist:
    """Update an existing tracklist.

    Args:
        tracklist_id: ID of the tracklist to update.
        request: Update request.
        db: Database session.

    Returns:
        Updated tracklist.

    Raises:
        HTTPException: If tracklist not found or not editable.
    """
    draft_service = DraftService(db)

    try:
        # Update the draft
        updated = draft_service.save_draft(tracklist_id, request.tracks)

        # Publish if requested
        if not request.is_draft:
            updated = draft_service.publish_draft(updated.id)

        return updated
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post("/{tracklist_id}/tracks", response_model=TrackEntry, status_code=status.HTTP_201_CREATED)
def add_track(
    tracklist_id: UUID,
    request: AddTrackRequest,
    db: Session = Depends(get_db_session),
) -> TrackEntry:
    """Add a track to a tracklist.

    Args:
        tracklist_id: ID of the tracklist.
        request: Track to add.
        db: Database session.

    Returns:
        Added track.

    Raises:
        HTTPException: If tracklist not found or position conflict.
    """
    draft_service = DraftService(db)

    # Get the draft
    draft = draft_service.get_draft(tracklist_id)
    if not draft:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Draft tracklist {tracklist_id} not found",
        )

    # Check for position conflict
    existing_positions = {track.position for track in draft.tracks}
    if request.position in existing_positions:
        # Shift positions to make room
        for track in draft.tracks:
            if track.position >= request.position:
                track.position += 1

    # Create new track
    from datetime import timedelta

    new_track = TrackEntry(
        position=request.position,
        artist=request.artist,
        title=request.title,
        start_time=timedelta(seconds=parse_time_string(request.start_time)),
        end_time=timedelta(seconds=parse_time_string(request.end_time)) if request.end_time else None,
        remix=request.remix,
        label=request.label,
        is_manual_entry=True,
    )

    # Add track and sort by position
    draft.tracks.append(new_track)
    draft.tracks.sort(key=lambda t: t.position)

    # Save updated draft
    draft_service.save_draft(tracklist_id, draft.tracks, auto_version=False)

    return new_track


@router.put("/{tracklist_id}/tracks/{position}", response_model=TrackEntry)
def update_track(
    tracklist_id: UUID,
    position: int,
    request: UpdateTrackRequest,
    db: Session = Depends(get_db_session),
) -> TrackEntry:
    """Update a track in a tracklist.

    Args:
        tracklist_id: ID of the tracklist.
        position: Position of the track to update.
        request: Update data.
        db: Database session.

    Returns:
        Updated track.

    Raises:
        HTTPException: If tracklist or track not found.
    """
    draft_service = DraftService(db)

    # Get the draft
    draft = draft_service.get_draft(tracklist_id)
    if not draft:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Draft tracklist {tracklist_id} not found",
        )

    # Find the track
    track = None
    for t in draft.tracks:
        if t.position == position:
            track = t
            break

    if not track:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Track at position {position} not found",
        )

    # Update track fields
    from datetime import timedelta

    if request.artist is not None:
        track.artist = request.artist
    if request.title is not None:
        track.title = request.title
    if request.start_time is not None:
        track.start_time = timedelta(seconds=parse_time_string(request.start_time))
    if request.end_time is not None:
        track.end_time = timedelta(seconds=parse_time_string(request.end_time))
    if request.remix is not None:
        track.remix = request.remix
    if request.label is not None:
        track.label = request.label

    # Save updated draft
    draft_service.save_draft(tracklist_id, draft.tracks, auto_version=False)

    return track


@router.delete("/{tracklist_id}/tracks/{position}", status_code=status.HTTP_204_NO_CONTENT)
def delete_track(
    tracklist_id: UUID,
    position: int,
    db: Session = Depends(get_db_session),
) -> None:
    """Delete a track from a tracklist.

    Args:
        tracklist_id: ID of the tracklist.
        position: Position of the track to delete.
        db: Database session.

    Raises:
        HTTPException: If tracklist or track not found.
    """
    draft_service = DraftService(db)

    # Get the draft
    draft = draft_service.get_draft(tracklist_id)
    if not draft:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Draft tracklist {tracklist_id} not found",
        )

    # Find and remove the track
    track_found = False
    new_tracks = []
    for track in draft.tracks:
        if track.position == position:
            track_found = True
            continue
        # Adjust positions for tracks after the deleted one
        if track.position > position:
            track.position -= 1
        new_tracks.append(track)

    if not track_found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Track at position {position} not found",
        )

    # Save updated draft
    draft_service.save_draft(tracklist_id, new_tracks, auto_version=False)


@router.put("/{tracklist_id}/tracks/{position}/timing", response_model=TrackEntry)
def update_track_timing(
    tracklist_id: UUID,
    position: int,
    request: UpdateTrackTimingRequest,
    db: Session = Depends(get_db_session),
) -> TrackEntry:
    """Update timing for a track.

    Args:
        tracklist_id: ID of the tracklist.
        position: Position of the track.
        request: Timing update.
        db: Database session.

    Returns:
        Updated track.

    Raises:
        HTTPException: If tracklist or track not found.
    """
    draft_service = DraftService(db)

    # Get the draft
    draft = draft_service.get_draft(tracklist_id)
    if not draft:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Draft tracklist {tracklist_id} not found",
        )

    # Find the track
    track = None
    for t in draft.tracks:
        if t.position == position:
            track = t
            break

    if not track:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Track at position {position} not found",
        )

    # Update timing
    from datetime import timedelta

    track.start_time = timedelta(seconds=parse_time_string(request.start_time))
    if request.end_time:
        track.end_time = timedelta(seconds=parse_time_string(request.end_time))

    # Validate no overlaps
    for other in draft.tracks:
        if other.position == position:
            continue

        # Calculate end times (default to start + 1 second if no end time)
        from datetime import timedelta as td

        other_end = other.end_time if other.end_time else (other.start_time + td(seconds=1))
        track_end = track.end_time if track.end_time else (track.start_time + td(seconds=1))

        # Check for overlap: start1 < end2 AND start2 < end1
        if track.start_time < other_end and other.start_time < track_end:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Timing conflict with track at position {other.position}",
            )

    # Save updated draft
    draft_service.save_draft(tracklist_id, draft.tracks, auto_version=False)

    return track


@router.get("/{audio_file_id}/drafts", response_model=List[Tracklist])
def list_drafts(
    audio_file_id: UUID,
    include_versions: bool = False,
    db: Session = Depends(get_db_session),
) -> List[Tracklist]:
    """List all draft tracklists for an audio file.

    Args:
        audio_file_id: ID of the audio file.
        include_versions: Whether to include all versions.
        db: Database session.

    Returns:
        List of draft tracklists.
    """
    draft_service = DraftService(db)
    return draft_service.list_drafts(audio_file_id, include_versions)


@router.post("/{tracklist_id}/publish", response_model=Tracklist)
def publish_draft(
    tracklist_id: UUID,
    db: Session = Depends(get_db_session),
) -> Tracklist:
    """Publish a draft as final version.

    Args:
        tracklist_id: ID of the draft to publish.
        db: Database session.

    Returns:
        Published tracklist.

    Raises:
        HTTPException: If draft not found or already published.
    """
    draft_service = DraftService(db)

    try:
        published = draft_service.publish_draft(tracklist_id)

        # TODO: Trigger CUE generation via RabbitMQ message

        return published
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
