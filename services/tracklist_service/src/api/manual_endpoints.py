"""API endpoints for manual tracklist creation and management.

This module provides REST endpoints for creating and managing
manual tracklists including track CRUD operations and draft management.
"""

from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from services.tracklist_service.src.models.tracklist import TrackEntry, Tracklist
from services.tracklist_service.src.services.draft_service import DraftService
from services.tracklist_service.src.services.catalog_search_service import CatalogSearchService
from services.tracklist_service.src.services.timing_service import TimingService
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


class CatalogSearchRequest(BaseModel):
    """Request model for catalog search."""

    query: Optional[str] = Field(None, description="General search query")
    artist: Optional[str] = Field(None, description="Artist name to search for")
    title: Optional[str] = Field(None, description="Track title to search for")
    limit: int = Field(10, ge=1, le=50, description="Maximum results to return")


class CatalogSearchResult(BaseModel):
    """Catalog search result."""

    catalog_track_id: UUID = Field(description="Recording ID in catalog")
    artist: Optional[str] = Field(None, description="Artist name from metadata")
    title: Optional[str] = Field(None, description="Track title from metadata")
    album: Optional[str] = Field(None, description="Album name from metadata")
    genre: Optional[str] = Field(None, description="Genre from metadata")
    bpm: Optional[float] = Field(None, description="BPM from metadata")
    key: Optional[str] = Field(None, description="Musical key from metadata")
    confidence: float = Field(description="Match confidence score")


class CatalogMatchRequest(BaseModel):
    """Request to match tracks to catalog."""

    tracks: List[TrackEntry] = Field(description="Tracks to match")
    threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum confidence threshold")


@router.get("/catalog/search", response_model=List[CatalogSearchResult])
def search_catalog(
    query: Optional[str] = None,
    artist: Optional[str] = None,
    title: Optional[str] = None,
    limit: int = 10,
    db: Session = Depends(get_db_session),
) -> List[CatalogSearchResult]:
    """Search the catalog for tracks.

    Args:
        query: General search query.
        artist: Artist name to search for.
        title: Track title to search for.
        limit: Maximum results to return.
        db: Database session.

    Returns:
        List of catalog search results.
    """
    if not any([query, artist, title]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one search parameter (query, artist, title) must be provided",
        )

    catalog_service = CatalogSearchService(db)
    results = catalog_service.search_catalog(
        query=query,
        artist=artist,
        title=title,
        limit=limit,
    )

    search_results = []
    for recording, confidence in results:
        # Get metadata for the recording
        metadata = catalog_service.get_catalog_track_metadata(recording.id)

        result = CatalogSearchResult(
            catalog_track_id=recording.id,
            artist=metadata.get("artist"),
            title=metadata.get("title"),
            album=metadata.get("album"),
            genre=metadata.get("genre"),
            bpm=float(metadata["bpm"]) if "bpm" in metadata else None,
            key=metadata.get("key"),
            confidence=confidence,
        )
        search_results.append(result)

    return search_results


@router.post("/catalog/match", response_model=List[TrackEntry])
def match_tracks_to_catalog(
    request: CatalogMatchRequest,
    db: Session = Depends(get_db_session),
) -> List[TrackEntry]:
    """Match multiple tracks to catalog entries.

    Args:
        request: Match request with tracks and threshold.
        db: Database session.

    Returns:
        List of tracks with catalog_track_id populated.
    """
    catalog_service = CatalogSearchService(db)
    matched_tracks = catalog_service.fuzzy_match_tracks(
        tracks=request.tracks,
        threshold=request.threshold,
    )

    return matched_tracks


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


class BulkTrackUpdateRequest(BaseModel):
    """Request for bulk track updates."""

    tracks: List[TrackEntry] = Field(description="Updated tracks")


class TrackReorderRequest(BaseModel):
    """Request for reordering tracks."""

    from_position: int = Field(ge=1, description="Position to move from")
    to_position: int = Field(ge=1, description="Position to move to")


class TimingSuggestionsRequest(BaseModel):
    """Request for timing suggestions."""

    target_duration: Optional[str] = Field(None, description="Target duration (HH:MM:SS)")


@router.put("/{tracklist_id}/tracks/bulk", response_model=List[TrackEntry])
def bulk_update_tracks(
    tracklist_id: UUID,
    request: BulkTrackUpdateRequest,
    db: Session = Depends(get_db_session),
) -> List[TrackEntry]:
    """Bulk update multiple tracks.

    Args:
        tracklist_id: ID of the tracklist.
        request: Bulk update request.
        db: Database session.

    Returns:
        List of updated tracks.

    Raises:
        HTTPException: If tracklist not found or validation fails.
    """
    draft_service = DraftService(db)

    # Get the draft
    draft = draft_service.get_draft(tracklist_id)
    if not draft:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tracklist {tracklist_id} not found",
        )

    # Validate positions
    positions = [t.position for t in request.tracks]
    if len(positions) != len(set(positions)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Duplicate positions in track list",
        )

    # Replace all tracks
    draft_service.save_draft(tracklist_id, request.tracks, auto_version=False)

    return request.tracks


@router.post("/{tracklist_id}/tracks/reorder", response_model=List[TrackEntry])
def reorder_track(
    tracklist_id: UUID,
    request: TrackReorderRequest,
    db: Session = Depends(get_db_session),
) -> List[TrackEntry]:
    """Reorder a track within the tracklist.

    Args:
        tracklist_id: ID of the tracklist.
        request: Reorder request.
        db: Database session.

    Returns:
        List of reordered tracks.

    Raises:
        HTTPException: If tracklist or track not found.
    """
    draft_service = DraftService(db)

    # Get the draft
    draft = draft_service.get_draft(tracklist_id)
    if not draft:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tracklist {tracklist_id} not found",
        )

    # Find the track to move
    track_to_move = None
    for track in draft.tracks:
        if track.position == request.from_position:
            track_to_move = track
            break

    if not track_to_move:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Track at position {request.from_position} not found",
        )

    # Remove from current position
    draft.tracks.remove(track_to_move)

    # Reorder remaining tracks
    for track in draft.tracks:
        if request.from_position < request.to_position:
            # Moving down: shift tracks up
            if request.from_position < track.position <= request.to_position:
                track.position -= 1
        else:
            # Moving up: shift tracks down
            if request.to_position <= track.position < request.from_position:
                track.position += 1

    # Insert at new position
    track_to_move.position = request.to_position
    draft.tracks.append(track_to_move)

    # Normalize positions
    timing_service = TimingService()
    normalized_tracks = timing_service.normalize_track_positions(draft.tracks)

    # Save the reordered tracks
    draft_service.save_draft(tracklist_id, normalized_tracks, auto_version=False)

    return normalized_tracks


@router.post("/{tracklist_id}/tracks/auto-calculate-end-times", response_model=List[TrackEntry])
def auto_calculate_end_times(
    tracklist_id: UUID,
    audio_duration: Optional[str] = None,
    db: Session = Depends(get_db_session),
) -> List[TrackEntry]:
    """Auto-calculate end times for all tracks.

    Args:
        tracklist_id: ID of the tracklist.
        audio_duration: Total audio duration (HH:MM:SS format).
        db: Database session.

    Returns:
        List of tracks with calculated end times.

    Raises:
        HTTPException: If tracklist not found.
    """
    draft_service = DraftService(db)

    # Get the draft
    draft = draft_service.get_draft(tracklist_id)
    if not draft:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tracklist {tracklist_id} not found",
        )

    # Parse audio duration if provided
    audio_duration_td = None
    if audio_duration:
        from datetime import timedelta

        audio_duration_td = timedelta(seconds=parse_time_string(audio_duration))

    # Calculate end times
    timing_service = TimingService()
    updated_tracks = timing_service.auto_calculate_end_times(
        draft.tracks,
        audio_duration=audio_duration_td,
    )

    # Save the updated tracks
    draft_service.save_draft(tracklist_id, updated_tracks, auto_version=False)

    return updated_tracks


@router.get("/{tracklist_id}/tracks/timing-suggestions", response_model=List[Dict[str, Any]])
def get_timing_suggestions(
    tracklist_id: UUID,
    target_duration: Optional[str] = None,
    db: Session = Depends(get_db_session),
) -> List[Dict[str, Any]]:
    """Get timing adjustment suggestions.

    Args:
        tracklist_id: ID of the tracklist.
        target_duration: Target total duration (HH:MM:SS format).
        db: Database session.

    Returns:
        List of timing suggestions.

    Raises:
        HTTPException: If tracklist not found.
    """
    draft_service = DraftService(db)

    # Get the draft
    draft = draft_service.get_draft(tracklist_id)
    if not draft:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tracklist {tracklist_id} not found",
        )

    # Parse target duration if provided
    target_duration_td = None
    if target_duration:
        from datetime import timedelta

        target_duration_td = timedelta(seconds=parse_time_string(target_duration))

    # Get timing suggestions
    timing_service = TimingService()
    suggestions = timing_service.suggest_timing_adjustments(
        draft.tracks,
        target_duration=target_duration_td,
    )

    return suggestions


@router.post("/{tracklist_id}/tracks/validate-timing", response_model=Dict[str, Any])
def validate_timing(
    tracklist_id: UUID,
    audio_duration: Optional[str] = None,
    allow_gaps: bool = True,
    db: Session = Depends(get_db_session),
) -> Dict[str, Any]:
    """Validate track timing consistency.

    Args:
        tracklist_id: ID of the tracklist.
        audio_duration: Total audio duration (HH:MM:SS format).
        allow_gaps: Whether to allow gaps between tracks.
        db: Database session.

    Returns:
        Validation result with issues if any.

    Raises:
        HTTPException: If tracklist not found.
    """
    draft_service = DraftService(db)

    # Get the draft
    draft = draft_service.get_draft(tracklist_id)
    if not draft:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tracklist {tracklist_id} not found",
        )

    # Parse audio duration if provided
    audio_duration_td = None
    if audio_duration:
        from datetime import timedelta

        audio_duration_td = timedelta(seconds=parse_time_string(audio_duration))

    # Validate timing
    timing_service = TimingService()

    # Use the existing validation method for audio duration checks
    if audio_duration_td:
        is_valid, issues = timing_service.validate_timing_consistency(
            draft.tracks,
            audio_duration_td,
        )
    else:
        # Basic validation without audio duration
        is_valid, issues = True, []

        # Check for overlaps
        for track in draft.tracks:
            conflicts = timing_service.detect_timing_conflicts(track, draft.tracks)
            for conflict in conflicts:
                is_valid = False
                issues.append(
                    f"Track {conflict['track_position']} overlaps with track "
                    f"{conflict['conflicting_position']} by {conflict['overlap_duration']:.1f} seconds"
                )

    return {
        "is_valid": is_valid,
        "issues": issues,
        "track_count": len(draft.tracks),
    }


@router.post("/{tracklist_id}/tracks/match-to-catalog", response_model=List[TrackEntry])
def match_all_tracks_to_catalog(
    tracklist_id: UUID,
    threshold: float = 0.7,
    db: Session = Depends(get_db_session),
) -> List[TrackEntry]:
    """Match all tracks in tracklist to catalog.

    Args:
        tracklist_id: ID of the tracklist.
        threshold: Minimum confidence threshold.
        db: Database session.

    Returns:
        List of tracks with catalog matches.

    Raises:
        HTTPException: If tracklist not found.
    """
    draft_service = DraftService(db)

    # Get the draft
    draft = draft_service.get_draft(tracklist_id)
    if not draft:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tracklist {tracklist_id} not found",
        )

    # Match tracks to catalog
    catalog_service = CatalogSearchService(db)
    matched_tracks = catalog_service.fuzzy_match_tracks(
        draft.tracks,
        threshold=threshold,
    )

    # Save the updated tracks
    draft_service.save_draft(tracklist_id, matched_tracks, auto_version=False)

    return matched_tracks
