"""Pydantic schemas for API requests and responses."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class RecordingBase(BaseModel):
    """Base schema for Recording."""

    file_path: str
    file_name: str
    sha256_hash: str | None = None
    xxh128_hash: str | None = None


class RecordingCreate(RecordingBase):
    """Schema for creating a Recording."""


class RecordingUpdate(BaseModel):
    """Schema for updating a Recording."""

    file_path: str | None = None
    file_name: str | None = None
    sha256_hash: str | None = None
    xxh128_hash: str | None = None


class RecordingResponse(RecordingBase):
    """Schema for Recording response."""

    id: UUID
    created_at: datetime

    class Config:
        """Pydantic configuration."""

        from_attributes = True


class MetadataBase(BaseModel):
    """Base schema for Metadata."""

    key: str
    value: str


class MetadataCreate(MetadataBase):
    """Schema for creating Metadata."""


class MetadataResponse(MetadataBase):
    """Schema for Metadata response."""

    id: UUID
    recording_id: UUID

    class Config:
        """Pydantic configuration."""

        from_attributes = True


class TracklistBase(BaseModel):
    """Base schema for Tracklist."""

    source: str
    cue_file_path: str | None = None
    tracks: list[dict[str, Any]] = Field(default_factory=list)


class TracklistCreate(TracklistBase):
    """Schema for creating a Tracklist."""


class TracklistResponse(TracklistBase):
    """Schema for Tracklist response."""

    id: UUID
    recording_id: UUID

    class Config:
        """Pydantic configuration."""

        from_attributes = True


class RecordingDetailResponse(RecordingResponse):
    """Schema for detailed Recording response with relations."""

    metadata: list[MetadataResponse] = Field(default_factory=list)
    tracklists: list[TracklistResponse] = Field(default_factory=list)


class SearchRequest(BaseModel):
    """Schema for search requests."""

    query: str
    field: str = "file_name"  # file_name, file_path, metadata_key, metadata_value
    limit: int = Field(default=100, le=1000)
    offset: int = Field(default=0, ge=0)


class PaginationParams(BaseModel):
    """Schema for pagination parameters."""

    limit: int = Field(default=100, le=1000)
    offset: int = Field(default=0, ge=0)


class HealthResponse(BaseModel):
    """Schema for health check response."""

    status: str
    service: str
    version: str
    timestamp: datetime
