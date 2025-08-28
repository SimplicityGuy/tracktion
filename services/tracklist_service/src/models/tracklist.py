"""
Tracklist models for tracklist management and import functionality.

This module defines the core data models for managing tracklists,
including import from 1001tracklists and CUE file generation.
"""

from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, JSON, Text, Boolean, Integer
from sqlalchemy.dialects.postgresql import UUID as PostgresUUID
from sqlalchemy.orm import DeclarativeBase
from typing import TYPE_CHECKING


# Create Base class with proper typing
class Base(DeclarativeBase):
    pass


if TYPE_CHECKING:
    pass


class TrackEntry(BaseModel):
    """Individual track entry in a tracklist."""

    position: int = Field(description="Track position in the tracklist")
    start_time: timedelta = Field(description="Start time of the track")
    end_time: Optional[timedelta] = Field(None, description="End time of the track")
    artist: str = Field(description="Artist name(s)")
    title: str = Field(description="Track title")
    remix: Optional[str] = Field(None, description="Remix or edit information")
    label: Optional[str] = Field(None, description="Record label")
    catalog_track_id: Optional[UUID] = Field(None, description="Link to catalog track")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score")
    transition_type: Optional[str] = Field(None, description="Type of transition to next track")
    is_manual_entry: bool = Field(default=False, description="Flag for manually entered tracks")

    @field_validator("position")
    @classmethod
    def validate_position(cls, v: int) -> int:
        """Validate position is positive."""
        if v < 1:
            raise ValueError("Position must be positive")
        return v

    @field_validator("artist", "title")
    @classmethod
    def validate_not_empty(cls, v: str) -> str:
        """Validate string fields are not empty."""
        if not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON storage."""
        return {
            "position": self.position,
            "start_time": self.start_time.total_seconds(),
            "end_time": self.end_time.total_seconds() if self.end_time else None,
            "artist": self.artist,
            "title": self.title,
            "remix": self.remix,
            "label": self.label,
            "catalog_track_id": str(self.catalog_track_id) if self.catalog_track_id else None,
            "confidence": self.confidence,
            "transition_type": self.transition_type,
            "is_manual_entry": self.is_manual_entry,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TrackEntry":
        """Create from dictionary."""
        return cls(
            position=data["position"],
            start_time=timedelta(seconds=data["start_time"]),
            end_time=timedelta(seconds=data["end_time"]) if data.get("end_time") else None,
            artist=data["artist"],
            title=data["title"],
            remix=data.get("remix"),
            label=data.get("label"),
            catalog_track_id=UUID(data["catalog_track_id"]) if data.get("catalog_track_id") else None,
            confidence=data.get("confidence", 1.0),
            transition_type=data.get("transition_type"),
            is_manual_entry=data.get("is_manual_entry", False),
        )


class Tracklist(BaseModel):
    """Complete tracklist model."""

    id: UUID = Field(default_factory=uuid4, description="Unique identifier")
    audio_file_id: UUID = Field(description="Links to Recording/audio file")
    source: str = Field(description="Source of tracklist: manual, 1001tracklists, auto-detected")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    tracks: List[TrackEntry] = Field(default_factory=list, description="List of tracks")
    cue_file_id: Optional[UUID] = Field(None, description="Associated CUE file ID")
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0, description="Overall confidence score")
    draft_version: Optional[int] = Field(None, description="Version number for drafts")
    is_draft: bool = Field(default=False, description="Flag for draft status")
    parent_tracklist_id: Optional[UUID] = Field(None, description="For versioning")

    model_config = {"json_encoders": {UUID: str}}

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        """Validate source is one of the allowed values."""
        allowed = ["manual", "1001tracklists", "auto-detected"]
        if v not in allowed:
            raise ValueError(f"Source must be one of: {', '.join(allowed)}")
        return v

    @field_validator("tracks")
    @classmethod
    def validate_tracks_order(cls, v: List[TrackEntry]) -> List[TrackEntry]:
        """Validate tracks are in sequential order."""
        if not v:
            return v

        positions = [track.position for track in v]
        if positions != sorted(positions):
            raise ValueError("Tracks must be in sequential order by position")

        # Check for gaps
        expected = list(range(1, len(v) + 1))
        if positions != expected:
            # Allow some gaps but not too many
            max_pos = max(positions)
            if max_pos > len(v) * 2:
                raise ValueError("Too many gaps in track positions")

        return v


class TracklistDB(Base):
    """SQLAlchemy model for tracklist storage."""

    __tablename__ = "tracklists"

    id = Column(PostgresUUID(as_uuid=True), primary_key=True, default=uuid4)
    audio_file_id = Column(PostgresUUID(as_uuid=True), nullable=False)  # Foreign key to recordings.id
    source = Column(String(50), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    tracks = Column(JSON, nullable=False, default=list)
    cue_file_path = Column(Text, nullable=True)
    cue_file_id = Column(PostgresUUID(as_uuid=True), nullable=True)
    confidence_score = Column(Float, default=1.0, nullable=False)
    draft_version = Column(Integer, nullable=True)
    is_draft = Column(Boolean, default=False, nullable=False)
    parent_tracklist_id = Column(PostgresUUID(as_uuid=True), ForeignKey("tracklists.id"), nullable=True)

    # Relationships
    # Note: Recording model is in a different service, so relationship is commented out
    # recording = relationship("Recording", backref="tracklists")

    def to_model(self) -> Tracklist:
        """Convert to Pydantic model."""
        # Note: When this method is called on a loaded instance, these are values not Columns
        tracks_data: list = self.tracks if isinstance(self.tracks, list) else []
        tracks_list = [TrackEntry.from_dict(t) for t in tracks_data]

        return Tracklist(
            id=self.id,  # These are instance values, not Column objects
            audio_file_id=self.audio_file_id,
            source=self.source,
            created_at=self.created_at,
            updated_at=self.updated_at,
            tracks=tracks_list,
            cue_file_id=self.cue_file_id,
            confidence_score=self.confidence_score,
            draft_version=self.draft_version,
            is_draft=self.is_draft,
            parent_tracklist_id=self.parent_tracklist_id,
        )

    @classmethod
    def from_model(cls, model: Tracklist) -> "TracklistDB":
        """Create from Pydantic model."""
        return cls(
            id=model.id,
            audio_file_id=model.audio_file_id,
            source=model.source,
            created_at=model.created_at,
            updated_at=model.updated_at,
            tracks=[track.to_dict() for track in model.tracks],
            cue_file_id=model.cue_file_id,
            confidence_score=model.confidence_score,
            draft_version=model.draft_version,
            is_draft=model.is_draft,
            parent_tracklist_id=model.parent_tracklist_id,
        )


class ImportTracklistRequest(BaseModel):
    """Request model for importing tracklist from 1001tracklists."""

    url: str = Field(description="1001tracklists URL")
    audio_file_id: UUID = Field(description="ID of the audio file to match")
    force_refresh: bool = Field(default=False, description="Force re-fetch even if cached")
    cue_format: str = Field(default="standard", description="CUE format to generate")

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL is from 1001tracklists."""
        valid_prefixes = [
            "http://1001tracklists.com",
            "https://1001tracklists.com",
            "http://www.1001tracklists.com",
            "https://www.1001tracklists.com",
        ]
        if not any(v.startswith(prefix) for prefix in valid_prefixes):
            raise ValueError("URL must be from 1001tracklists.com")
        return v


class ImportTracklistResponse(BaseModel):
    """Response model for tracklist import."""

    success: bool = Field(description="Whether import was successful")
    tracklist: Optional[Tracklist] = Field(None, description="Imported tracklist")
    cue_file_path: Optional[str] = Field(None, description="Generated CUE file path")
    error: Optional[str] = Field(None, description="Error message if failed")
    cached: bool = Field(default=False, description="Whether data was from cache")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")
    message: Optional[str] = Field(None, description="Additional message")

    model_config = {"json_encoders": {UUID: str}}
