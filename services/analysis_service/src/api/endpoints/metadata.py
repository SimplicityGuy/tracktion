"""Metadata management endpoints for Analysis Service."""

import os
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from services.analysis_service.src.api_message_publisher import APIMessagePublisher
from services.analysis_service.src.repositories import AsyncMetadataRepository, AsyncRecordingRepository
from services.analysis_service.src.structured_logging import get_logger
from shared.core_types.src.async_database import AsyncDatabaseManager

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/metadata", tags=["metadata"])

# Initialize database and message queue components
db_manager = AsyncDatabaseManager()
message_publisher = APIMessagePublisher(rabbitmq_url=os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/"))
recording_repo = AsyncRecordingRepository(db_manager)
metadata_repo = AsyncMetadataRepository(db_manager)


class MetadataUpdate(BaseModel):
    """Request model for metadata update."""

    title: str | None = None
    artist: str | None = None
    album: str | None = None
    genre: str | None = None
    year: int | None = None
    track_number: int | None = None
    custom_fields: dict[str, Any] | None = {}


class MetadataResponse(BaseModel):
    """Response model for metadata."""

    recording_id: UUID
    title: str
    artist: str
    album: str | None
    genre: str | None
    year: int | None
    duration: float
    format: str
    bitrate: int
    sample_rate: int
    channels: int
    custom_fields: dict[str, Any]


@router.get("/{recording_id}")
async def get_metadata(recording_id: UUID) -> MetadataResponse:
    """Get metadata for a recording.

    Args:
        recording_id: UUID of the recording

    Returns:
        Recording metadata
    """
    # Verify recording exists
    recording = await recording_repo.get_by_id(recording_id)
    if not recording:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Recording not found: {recording_id}")

    # Get metadata from database
    metadata_items = await metadata_repo.get_by_recording_id(recording_id)
    # Ensure key and value are strings to match dict[str, str] expectation
    metadata_dict: dict[str, str] = {
        str(item.key): str(item.value) for item in metadata_items if item.key is not None and item.value is not None
    }

    # Extract standard metadata fields with defaults
    def get_metadata_value(key: str, default: Any = None, convert_type: type = str) -> Any:
        value = metadata_dict.get(key, str(default) if default is not None else None)
        if value is not None and convert_type is not str:
            try:
                return convert_type(value)
            except (ValueError, TypeError):
                return default
        return value

    # Build custom fields (non-standard metadata)
    standard_fields = {
        "title",
        "artist",
        "album",
        "genre",
        "year",
        "duration",
        "format",
        "bitrate",
        "sample_rate",
        "channels",
    }
    custom_fields: dict[str, Any] = {k: v for k, v in metadata_dict.items() if k not in standard_fields}

    return MetadataResponse(
        recording_id=recording_id,
        title=get_metadata_value("title", "Unknown Title"),
        artist=get_metadata_value("artist", "Unknown Artist"),
        album=get_metadata_value("album"),
        genre=get_metadata_value("genre"),
        year=get_metadata_value("year", convert_type=int),
        duration=get_metadata_value("duration", 0.0, float),
        format=get_metadata_value("format", "unknown"),
        bitrate=get_metadata_value("bitrate", 0, int),
        sample_rate=get_metadata_value("sample_rate", 0, int),
        channels=get_metadata_value("channels", 0, int),
        custom_fields=custom_fields,
    )


@router.put("/{recording_id}")
async def update_metadata(recording_id: UUID, metadata: MetadataUpdate) -> dict[str, str]:
    """Update metadata for a recording.

    Args:
        recording_id: UUID of the recording
        metadata: Metadata updates

    Returns:
        Update confirmation
    """
    # Verify recording exists
    recording = await recording_repo.get_by_id(recording_id)
    if not recording:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Recording not found: {recording_id}")

    # Get updates to apply
    updates = metadata.model_dump(exclude_none=True)

    # Update each metadata field in database
    updated_count = 0
    for key, value in updates.items():
        if key == "custom_fields" and isinstance(value, dict):
            # Handle custom fields as separate metadata items
            for custom_key, custom_value in value.items():
                await metadata_repo.update_by_key(recording_id=recording_id, key=custom_key, value=str(custom_value))
                updated_count += 1
        else:
            # Handle standard metadata fields
            await metadata_repo.update_by_key(recording_id=recording_id, key=key, value=str(value))
            updated_count += 1

    logger.info(
        "Metadata updated",
        extra={
            "recording_id": str(recording_id),
            "updates": updates,
            "fields_updated": updated_count,
        },
    )

    return {
        "id": str(recording_id),
        "status": "updated",
        "message": f"Metadata updated successfully ({updated_count} fields)",
    }


@router.post("/{recording_id}/extract")
async def extract_metadata(recording_id: UUID) -> dict[str, str]:
    """Trigger metadata extraction for a recording.

    Args:
        recording_id: UUID of the recording

    Returns:
        Extraction task confirmation
    """
    # Verify recording exists
    recording = await recording_repo.get_by_id(recording_id)
    if not recording:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Recording not found: {recording_id}")

    # Submit extraction request to message queue
    correlation_id = await message_publisher.publish_metadata_extraction(
        recording_id=recording_id,
        extraction_types=["id3_tags", "audio_analysis"],
        priority=6,  # Higher priority for metadata extraction
    )

    logger.info(
        "Metadata extraction started",
        extra={
            "recording_id": str(recording_id),
            "correlation_id": correlation_id,
        },
    )

    return {
        "id": str(recording_id),
        "status": "extracting",
        "message": "Metadata extraction started",
        "correlation_id": correlation_id,
    }


@router.post("/{recording_id}/enrich")
async def enrich_metadata(recording_id: UUID) -> dict[str, str]:
    """Enrich metadata using external sources.

    Args:
        recording_id: UUID of the recording

    Returns:
        Enrichment task confirmation
    """
    # Verify recording exists
    recording = await recording_repo.get_by_id(recording_id)
    if not recording:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Recording not found: {recording_id}")

    # Submit enrichment request to message queue
    correlation_id = await message_publisher.publish_metadata_extraction(
        recording_id=recording_id,
        extraction_types=["external_enrichment", "musicbrainz", "lastfm"],
        priority=4,  # Lower priority for enrichment
    )

    logger.info(
        "Metadata enrichment started",
        extra={
            "recording_id": str(recording_id),
            "correlation_id": correlation_id,
        },
    )

    return {
        "id": str(recording_id),
        "status": "enriching",
        "message": "Metadata enrichment started",
        "correlation_id": correlation_id,
    }
