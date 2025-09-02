"""API endpoints for recordings."""

from collections.abc import Sequence
from typing import cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import get_db_session
from src.repositories import MetadataRepository, RecordingRepository, TracklistRepository

from .schemas import MetadataResponse, RecordingDetailResponse, RecordingResponse, SearchRequest, TracklistResponse

router = APIRouter(prefix="/recordings", tags=["recordings"])

# Module-level dependency to avoid B008 ruff error
SessionDep = Depends(get_db_session)


@router.get("", response_model=list[RecordingResponse])
async def list_recordings(
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    session: AsyncSession = SessionDep,
) -> Sequence[RecordingResponse]:
    """List all recordings with pagination."""
    repo = RecordingRepository(session)
    recordings = await repo.get_all(limit=limit, offset=offset)
    return cast("Sequence[RecordingResponse]", recordings)


@router.get("/{recording_id}", response_model=RecordingDetailResponse)
async def get_recording(
    recording_id: UUID,
    session: AsyncSession = SessionDep,
) -> RecordingDetailResponse:
    """Get a specific recording with all its relations."""
    repo = RecordingRepository(session)
    recording = await repo.get_with_all_relations(recording_id)

    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    return cast("RecordingDetailResponse", recording)


@router.get("/{recording_id}/metadata", response_model=list[MetadataResponse])
async def get_recording_metadata(
    recording_id: UUID,
    session: AsyncSession = SessionDep,
) -> Sequence[MetadataResponse]:
    """Get metadata for a specific recording."""
    # First check if recording exists
    recording_repo = RecordingRepository(session)
    recording = await recording_repo.get_by_id(recording_id)
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    # Get metadata
    metadata_repo = MetadataRepository(session)
    metadata = await metadata_repo.get_by_recording_id(recording_id)
    return cast("Sequence[MetadataResponse]", metadata)


@router.get("/{recording_id}/tracklist", response_model=list[TracklistResponse])
async def get_recording_tracklists(
    recording_id: UUID,
    session: AsyncSession = SessionDep,
) -> Sequence[TracklistResponse]:
    """Get tracklists for a specific recording."""
    # First check if recording exists
    recording_repo = RecordingRepository(session)
    recording = await recording_repo.get_by_id(recording_id)
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    # Get tracklists
    tracklist_repo = TracklistRepository(session)
    tracklists = await tracklist_repo.get_by_recording_id(recording_id)
    return cast("Sequence[TracklistResponse]", tracklists)


@router.post("/search", response_model=list[RecordingResponse])
async def search_recordings(
    search: SearchRequest,
    session: AsyncSession = SessionDep,
) -> Sequence[RecordingResponse]:
    """Search for recordings based on various criteria."""
    repo = RecordingRepository(session)

    if search.field == "file_name":
        results = await repo.search_by_file_name(search.query, limit=search.limit)
        return cast("Sequence[RecordingResponse]", results)
    if search.field == "file_path":
        # Search by file path pattern
        recording = await repo.get_by_file_path(search.query)
        return cast("Sequence[RecordingResponse]", [recording] if recording else [])
    if search.field == "sha256_hash":
        recording = await repo.get_by_sha256_hash(search.query)
        return cast("Sequence[RecordingResponse]", [recording] if recording else [])
    if search.field == "xxh128_hash":
        recording = await repo.get_by_xxh128_hash(search.query)
        return cast("Sequence[RecordingResponse]", [recording] if recording else [])
    raise HTTPException(status_code=400, detail=f"Invalid search field: {search.field}")


@router.get("/by-path/{file_path:path}", response_model=RecordingResponse | None)
async def get_recording_by_path(
    file_path: str,
    session: AsyncSession = SessionDep,
) -> RecordingResponse | None:
    """Get a recording by its file path."""
    repo = RecordingRepository(session)
    recording = await repo.get_by_file_path(file_path)

    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")

    return cast("RecordingResponse", recording)
