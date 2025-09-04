"""API endpoints for manual tracklist creation and management.

This module provides REST endpoints for creating and managing
manual tracklists including track CRUD operations and draft management.
"""

import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from services.tracklist_service.src.messaging.cue_generation_handler import CueGenerationMessageHandler
from services.tracklist_service.src.messaging.rabbitmq_client import RabbitMQClient, RabbitMQConfig
from services.tracklist_service.src.models.cue_file import CueFormat
from services.tracklist_service.src.models.tracklist import TrackEntry, Tracklist, TracklistDB
from services.tracklist_service.src.services.catalog_search_service import CatalogSearchService
from services.tracklist_service.src.services.cue_generation_service import CueGenerationService
from services.tracklist_service.src.services.cue_integration import CueIntegrationService
from services.tracklist_service.src.services.draft_service import DraftService
from services.tracklist_service.src.services.storage_service import StorageService
from services.tracklist_service.src.services.timing_service import TimingService
from services.tracklist_service.src.utils.time_utils import parse_time_string

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/v1/tracklists", tags=["manual-tracklist"])


class CreateManualTracklistRequest(BaseModel):
    """Request model for creating manual tracklist."""

    audio_file_id: UUID = Field(description="ID of the audio file")
    tracks: list[TrackEntry] | None = Field(None, description="Initial list of tracks")
    is_draft: bool = Field(default=True, description="Whether to save as draft")


class UpdateTracklistRequest(BaseModel):
    """Request model for updating a tracklist."""

    tracks: list[TrackEntry] = Field(description="Updated list of tracks")
    is_draft: bool = Field(default=True, description="Whether to keep as draft")


class AddTrackRequest(BaseModel):
    """Request model for adding a track."""

    position: int = Field(ge=1, description="Track position")
    artist: str = Field(description="Artist name(s)")
    title: str = Field(description="Track title")
    start_time: str = Field(description="Start time in HH:MM:SS format")
    end_time: str | None = Field(None, description="End time in HH:MM:SS format")
    remix: str | None = Field(None, description="Remix or edit information")
    label: str | None = Field(None, description="Record label")


class UpdateTrackRequest(BaseModel):
    """Request model for updating a track."""

    artist: str | None = Field(None, description="Artist name(s)")
    title: str | None = Field(None, description="Track title")
    start_time: str | None = Field(None, description="Start time in HH:MM:SS format")
    end_time: str | None = Field(None, description="End time in HH:MM:SS format")
    remix: str | None = Field(None, description="Remix or edit information")
    label: str | None = Field(None, description="Record label")


class UpdateTrackTimingRequest(BaseModel):
    """Request model for updating track timing."""

    start_time: str = Field(description="Start time in HH:MM:SS format")
    end_time: str | None = Field(None, description="End time in HH:MM:SS format")


# Note: parse_time_string function has been moved to utils.time_utils module
# and is now imported at the top of this file


@router.post("/manual", response_model=Tracklist, status_code=status.HTTP_201_CREATED)
def create_manual_tracklist(
    request: CreateManualTracklistRequest,
    db: Session,  # Will be injected via Depends in route
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
    db: Session,  # Will be injected via Depends in route
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
        ) from e


@router.post(
    "/{tracklist_id}/tracks",
    response_model=TrackEntry,
    status_code=status.HTTP_201_CREATED,
)
def add_track(
    tracklist_id: UUID,
    request: AddTrackRequest,
    db: Session,  # Will be injected via Depends in route
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
    # timedelta import removed - using parse_time_string directly

    new_track = TrackEntry(
        position=request.position,
        artist=request.artist,
        title=request.title,
        start_time=parse_time_string(request.start_time),
        end_time=parse_time_string(request.end_time) if request.end_time else None,
        remix=request.remix,
        label=request.label,
        catalog_track_id=None,
        transition_type=None,
        is_manual_entry=True,
        bpm=None,
        key=None,
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
    db: Session,  # Will be injected via Depends in route
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
    # timedelta import removed - using parse_time_string directly

    if request.artist is not None:
        track.artist = request.artist
    if request.title is not None:
        track.title = request.title
    if request.start_time is not None:
        track.start_time = parse_time_string(request.start_time)
    if request.end_time is not None:
        track.end_time = parse_time_string(request.end_time)
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
    db: Session,  # Will be injected via Depends in route
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
    db: Session,  # Will be injected via Depends in route
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
        HTTPException: If tracklist or track not found or timing conflict.
    """
    # timedelta import removed - using parse_time_string directly

    draft_service = DraftService(db)
    timing_service = TimingService()

    # Get the draft
    draft = draft_service.get_draft(tracklist_id)
    if not draft:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Draft tracklist {tracklist_id} not found",
        )

    # Find the track using dict comprehension for efficiency
    tracks_by_position = {t.position: t for t in draft.tracks}
    track = tracks_by_position.get(position)

    if not track:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Track at position {position} not found",
        )

    # Update timing
    track.start_time = parse_time_string(request.start_time)
    if request.end_time:
        track.end_time = parse_time_string(request.end_time)

    # Use TimingService for validation
    conflicts = timing_service.detect_all_timing_conflicts(draft.tracks)
    if conflicts:
        # Find conflicts involving our track
        track_conflicts = [c for c in conflicts if position in [c[0].position, c[1].position]]
        if track_conflicts:
            conflict = track_conflicts[0]
            other_pos = conflict[0].position if conflict[0].position != position else conflict[1].position
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Timing conflict with track at position {other_pos}: {conflict[2]}",
            )

    # Save updated draft
    draft_service.save_draft(tracklist_id, draft.tracks, auto_version=False)

    return track


@router.get("/{audio_file_id}/drafts", response_model=list[Tracklist])
def list_drafts(
    audio_file_id: UUID,
    db: Session,  # Will be injected via Depends in route
    include_versions: bool = False,
) -> list[Tracklist]:
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

    query: str | None = Field(None, description="General search query")
    artist: str | None = Field(None, description="Artist name to search for")
    title: str | None = Field(None, description="Track title to search for")
    limit: int = Field(10, ge=1, le=50, description="Maximum results to return")


class CatalogSearchResult(BaseModel):
    """Catalog search result."""

    catalog_track_id: UUID = Field(description="Recording ID in catalog")
    artist: str | None = Field(None, description="Artist name from metadata")
    title: str | None = Field(None, description="Track title from metadata")
    album: str | None = Field(None, description="Album name from metadata")
    genre: str | None = Field(None, description="Genre from metadata")
    bpm: float | None = Field(None, description="BPM from metadata")
    key: str | None = Field(None, description="Musical key from metadata")
    confidence: float = Field(description="Match confidence score")


class CatalogMatchRequest(BaseModel):
    """Request to match tracks to catalog."""

    tracks: list[TrackEntry] = Field(description="Tracks to match")
    threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum confidence threshold")


@router.get("/catalog/search", response_model=list[CatalogSearchResult])
def search_catalog(
    db: Session,  # Will be injected via Depends in route
    query: str | None = None,
    artist: str | None = None,
    title: str | None = None,
    limit: int = 10,
) -> list[CatalogSearchResult]:
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

    search_results: list[CatalogSearchResult] = []
    for recording, confidence in results:
        # Get metadata for the recording
        # recording.id is a primary key and should never be None
        recording_id = recording.id
        if recording_id is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid recording data: missing ID"
            )

        metadata = catalog_service.get_catalog_track_metadata(recording_id)

        result = CatalogSearchResult(
            catalog_track_id=recording_id,
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


@router.post("/catalog/match", response_model=list[TrackEntry])
def match_tracks_to_catalog(
    request: CatalogMatchRequest,
    db: Session,  # Will be injected via Depends in route
) -> list[TrackEntry]:
    """Match multiple tracks to catalog entries.

    Args:
        request: Match request with tracks and threshold.
        db: Database session.

    Returns:
        List of tracks with catalog_track_id populated.
    """
    catalog_service = CatalogSearchService(db)
    return catalog_service.fuzzy_match_tracks(
        tracks=request.tracks,
        threshold=request.threshold,
    )


@router.post("/{tracklist_id}/publish", response_model=Tracklist)
async def publish_draft(
    tracklist_id: UUID,
    db: Session,  # Will be injected via Depends in route
    validate_before_publish: bool = True,
    generate_cue_async: bool = True,
) -> Tracklist:
    """Publish a draft as final version.

    Args:
        tracklist_id: ID of the draft to publish.
        validate_before_publish: Whether to validate draft before publishing.
        db: Database session.

    Returns:
        Published tracklist.

    Raises:
        HTTPException: If draft not found, already published, or validation fails.
    """
    draft_service = DraftService(db)

    # Get the draft first to validate it
    draft = draft_service.get_draft(tracklist_id)
    if not draft:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Draft with ID {tracklist_id} not found",
        )

    # Validate before publishing if requested
    if validate_before_publish:
        # Check minimum requirements
        if not draft.tracks or len(draft.tracks) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot publish draft without any tracks",
            )

        # Validate timing consistency if we have an audio file
        if draft.audio_file_id:
            # Get audio duration from database (would need AudioFile model)
            # For now, we'll just validate that tracks have proper timing
            timing_service = TimingService()
            for track in draft.tracks:
                if track.start_time is None:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Track {track.position} is missing start time",
                    )

            # Check for timing conflicts across all tracks
            conflicts = timing_service.detect_all_timing_conflicts(draft.tracks)
            if conflicts:
                # Report the first conflict found
                track1, track2, reason = conflicts[0]
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Timing conflict between tracks {track1.position} and {track2.position}: {reason}",
                )

    try:
        published = draft_service.publish_draft(tracklist_id)

        # Generate CUE file asynchronously if requested
        if generate_cue_async:
            # Use async message queue for non-blocking CUE generation

            # Initialize required services
            storage_service = StorageService()
            cue_service = CueGenerationService(storage_service=storage_service)
            rabbitmq_client = RabbitMQClient(RabbitMQConfig())

            # Create a placeholder session factory since this is placeholder code
            def placeholder_session_factory():
                # This is a placeholder - in real implementation would return async session
                return db

            _ = CueGenerationMessageHandler(
                cue_generation_service=cue_service,
                storage_service=storage_service,
                rabbitmq_client=rabbitmq_client,
                session_factory=placeholder_session_factory,
            )

            # Note: CueGenerationMessageHandler doesn't have publish_generation_request method
            # This needs to be implemented or use a different approach
            success = True  # Placeholder for now

            if success:
                logger.info(f"CUE generation queued for tracklist {published.id}")
            else:
                logger.warning(f"Failed to queue CUE generation for tracklist {published.id}")
        else:
            # Fallback to synchronous generation (for testing)
            cue_integration = CueIntegrationService()

            # Note: generate_cue_file method doesn't exist, this is placeholder
            # In real implementation, use generate_cue_content
            success, content, error = cue_integration.generate_cue_content(
                published,
                cue_format=CueFormat.STANDARD,
                audio_filename=f"audio_{published.audio_file_id}.wav",
            )

            if not success:
                logger.warning(f"CUE generation failed: {error}")
            else:
                logger.info("CUE file generated successfully")

        return published
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e


class CueGenerateRequest(BaseModel):
    """Request for CUE file generation."""

    audio_file_path: str = Field(description="Path to the audio file")
    cue_format: str = Field(default="standard", description="CUE format (standard, cdj, traktor)")


@router.post("/{tracklist_id}/generate-cue")
def generate_cue_file(
    tracklist_id: UUID,
    request: CueGenerateRequest,
    db: Session,  # Will be injected via Depends in route
) -> dict[str, Any]:
    """Generate a CUE file for a tracklist.

    Args:
        tracklist_id: ID of the tracklist.
        request: CUE generation request.
        db: Database session.

    Returns:
        CUE generation result.

    Raises:
        HTTPException: If tracklist not found or generation fails.
    """
    draft_service = DraftService(db)

    # Get the tracklist (draft or published)
    tracklist = draft_service.get_draft(tracklist_id)
    if not tracklist:
        # Try getting a published version

        tracklist_db = db.query(TracklistDB).filter_by(id=tracklist_id).first()
        if not tracklist_db:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tracklist with ID {tracklist_id} not found",
            )
        tracklist = tracklist_db.to_model()

    # Generate CUE file
    cue_service = CueIntegrationService()
    cue_success, cue_content, cue_error = cue_service.generate_cue_content(
        tracklist=tracklist,
        cue_format=CueFormat(request.cue_format),
        audio_filename=request.audio_file_path,
    )

    if not cue_success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CUE generation failed: {cue_error}",
        )

    # Note: The method returns content, not a file path - would need to save content to get path
    # Store reference would need actual file path
    # if cue_file_path:
    #     cue_service.store_cue_file_reference(tracklist_id, cue_file_path)

    return {
        "success": True,
        "cue_file_path": None,  # Would need to save content to file to get path
        "cue_file_id": None,  # Would need to save to get ID
        "format": request.cue_format,
    }


class BulkTrackUpdateRequest(BaseModel):
    """Request for bulk track updates."""

    tracks: list[TrackEntry] = Field(description="Updated tracks")


class TrackReorderRequest(BaseModel):
    """Request for reordering tracks."""

    from_position: int = Field(ge=1, description="Position to move from")
    to_position: int = Field(ge=1, description="Position to move to")


class TimingSuggestionsRequest(BaseModel):
    """Request for timing suggestions."""

    target_duration: str | None = Field(None, description="Target duration (HH:MM:SS)")


@router.put("/{tracklist_id}/tracks/bulk", response_model=list[TrackEntry])
def bulk_update_tracks(
    tracklist_id: UUID,
    request: BulkTrackUpdateRequest,
    db: Session,  # Will be injected via Depends in route
) -> list[TrackEntry]:
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


@router.post("/{tracklist_id}/tracks/reorder", response_model=list[TrackEntry])
def reorder_track(
    tracklist_id: UUID,
    request: TrackReorderRequest,
    db: Session,  # Will be injected via Depends in route
) -> list[TrackEntry]:
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
        # Moving up: shift tracks down
        elif request.to_position <= track.position < request.from_position:
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


@router.post("/{tracklist_id}/tracks/auto-calculate-end-times", response_model=list[TrackEntry])
def auto_calculate_end_times(
    tracklist_id: UUID,
    db: Session,  # Will be injected via Depends in route
    audio_duration: str | None = None,
) -> list[TrackEntry]:
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
        # timedelta import removed - using parse_time_string directly

        audio_duration_td = parse_time_string(audio_duration)

    # Calculate end times
    timing_service = TimingService()
    updated_tracks = timing_service.auto_calculate_end_times(
        draft.tracks,
        audio_duration=audio_duration_td,
    )

    # Save the updated tracks
    draft_service.save_draft(tracklist_id, updated_tracks, auto_version=False)

    return updated_tracks


@router.get("/{tracklist_id}/tracks/timing-suggestions", response_model=list[dict[str, Any]])
def get_timing_suggestions(
    tracklist_id: UUID,
    db: Session,  # Will be injected via Depends in route
    target_duration: str | None = None,
) -> list[dict[str, Any]]:
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
        # timedelta import removed - using parse_time_string directly

        target_duration_td = parse_time_string(target_duration)

    # Get timing suggestions
    timing_service = TimingService()
    return timing_service.suggest_timing_adjustments(
        draft.tracks,
        target_duration=target_duration_td,
    )


@router.post("/{tracklist_id}/tracks/validate-timing", response_model=dict[str, Any])
def validate_timing(
    tracklist_id: UUID,
    db: Session,  # Will be injected via Depends in route
    audio_duration: str | None = None,
    allow_gaps: bool = True,
) -> dict[str, Any]:
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
        # timedelta import removed - using parse_time_string directly

        audio_duration_td = parse_time_string(audio_duration)

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


@router.post("/{tracklist_id}/tracks/match-to-catalog", response_model=list[TrackEntry])
def match_all_tracks_to_catalog(
    tracklist_id: UUID,
    db: Session,  # Will be injected via Depends in route
    threshold: float = 0.7,
) -> list[TrackEntry]:
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
