"""
Tracklist data models for 1001tracklists.com integration.

Provides comprehensive models for tracklist data including tracks,
cue points, transitions, and complete tracklist information.
"""

from datetime import date as date_type, datetime, timezone
from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, model_validator, ValidationInfo


class TransitionType(str, Enum):
    """Types of transitions between tracks."""

    CUT = "cut"
    FADE = "fade"
    BLEND = "blend"
    CROSSFADE = "crossfade"
    ECHO = "echo"
    FILTER = "filter"
    MASHUP = "mashup"
    UNKNOWN = "unknown"


class CuePoint(BaseModel):
    """Cue point information for a track."""

    track_number: int = Field(ge=1, description="Track number in the tracklist")
    timestamp_ms: int = Field(ge=0, description="Timestamp in milliseconds")
    formatted_time: str = Field(description="Formatted timestamp (HH:MM:SS or MM:SS)")

    @field_validator("formatted_time")
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        """Validate time format is HH:MM:SS or MM:SS."""
        parts = v.split(":")
        if len(parts) not in [2, 3]:
            raise ValueError("Time must be in MM:SS or HH:MM:SS format")

        try:
            # Validate each part is a valid integer
            for part in parts:
                int(part)
        except ValueError:
            raise ValueError("Invalid time format, must contain only numbers")

        return v


class Track(BaseModel):
    """Individual track information."""

    number: int = Field(ge=1, description="Track number in the set")
    timestamp: Optional[CuePoint] = Field(None, description="Cue point for this track")
    artist: str = Field(description="Artist name(s)")
    title: str = Field(description="Track title")
    remix: Optional[str] = Field(None, description="Remix or edit information")
    label: Optional[str] = Field(None, description="Record label")
    is_id: bool = Field(default=False, description="Whether this is an ID/unknown track")
    bpm: Optional[float] = Field(None, ge=60, le=200, description="BPM if available")
    key: Optional[str] = Field(None, description="Musical key if available")
    genre: Optional[str] = Field(None, description="Genre classification")
    notes: Optional[str] = Field(None, description="Additional notes or comments")

    @field_validator("artist")
    @classmethod
    def validate_artist(cls, v: str) -> str:
        """Clean and validate artist name."""
        cleaned = v.strip()
        if not cleaned and not v.startswith("ID"):
            raise ValueError("Artist name cannot be empty")
        return cleaned

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Clean and validate track title."""
        cleaned = v.strip()
        if not cleaned and not v.startswith("ID"):
            raise ValueError("Track title cannot be empty")
        return cleaned


class Transition(BaseModel):
    """Transition information between tracks."""

    from_track: int = Field(ge=1, description="Source track number")
    to_track: int = Field(ge=1, description="Destination track number")
    transition_type: TransitionType = Field(default=TransitionType.UNKNOWN, description="Type of transition")
    timestamp_ms: Optional[int] = Field(None, ge=0, description="Timestamp of transition in milliseconds")
    duration_ms: Optional[int] = Field(None, ge=0, description="Duration of transition in milliseconds")
    notes: Optional[str] = Field(None, description="Additional transition notes")

    @field_validator("to_track")
    @classmethod
    def validate_track_order(cls, v: int, info: ValidationInfo) -> int:
        """Validate that to_track is after from_track."""
        if info.data and "from_track" in info.data:
            from_track = info.data["from_track"]
            if v <= from_track:
                raise ValueError("to_track must be greater than from_track")
        return v


class TracklistMetadata(BaseModel):
    """Additional metadata for a tracklist."""

    recording_type: Optional[str] = Field(
        None, description="Type of recording (e.g., 'DJ Set', 'Live Set', 'Radio Show')"
    )
    duration_minutes: Optional[int] = Field(None, ge=1, description="Total duration in minutes")
    play_count: Optional[int] = Field(None, ge=0, description="Number of plays on 1001tracklists")
    favorite_count: Optional[int] = Field(None, ge=0, description="Number of favorites on 1001tracklists")
    comment_count: Optional[int] = Field(None, ge=0, description="Number of comments on 1001tracklists")
    download_url: Optional[str] = Field(None, description="Download URL if available")
    stream_url: Optional[str] = Field(None, description="Stream URL if available")
    soundcloud_url: Optional[str] = Field(None, description="SoundCloud URL if available")
    mixcloud_url: Optional[str] = Field(None, description="Mixcloud URL if available")
    youtube_url: Optional[str] = Field(None, description="YouTube URL if available")
    tags: List[str] = Field(default_factory=list, description="Genre/style tags")


class Tracklist(BaseModel):
    """Complete tracklist information."""

    id: UUID = Field(default_factory=uuid4, description="Unique identifier")
    url: str = Field(description="Source URL from 1001tracklists.com")
    dj_name: str = Field(description="DJ or artist name")
    event_name: Optional[str] = Field(None, description="Event or festival name")
    venue: Optional[str] = Field(None, description="Venue name")
    date: Optional[date_type] = Field(None, description="Date of the set")
    tracks: List[Track] = Field(default_factory=list, description="List of tracks in order")
    transitions: List[Transition] = Field(default_factory=list, description="Transition information between tracks")
    metadata: Optional[TracklistMetadata] = Field(None, description="Additional metadata")
    scraped_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="When the data was scraped"
    )
    source_html_hash: Optional[str] = Field(None, description="Hash of source HTML for change detection")

    @field_validator("tracks")
    @classmethod
    def validate_track_order(cls, v: List[Track]) -> List[Track]:
        """Validate tracks are in sequential order."""
        if not v:
            return v

        # Check for sequential track numbers
        expected_numbers = set(range(1, len(v) + 1))
        actual_numbers = {track.number for track in v}

        if expected_numbers != actual_numbers:
            # Allow for some missing tracks (guest mixes, etc.)
            max_track = max(actual_numbers) if actual_numbers else 0
            if max_track > len(v) * 2:  # Too many gaps
                raise ValueError("Track numbers have too many gaps")

        return v

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL is from 1001tracklists.com."""
        if not v.startswith(
            (
                "http://1001tracklists.com",
                "https://1001tracklists.com",
                "http://www.1001tracklists.com",
                "https://www.1001tracklists.com",
            )
        ):
            raise ValueError("URL must be from 1001tracklists.com")
        return v


class TracklistRequest(BaseModel):
    """Request model for tracklist retrieval."""

    url: Optional[str] = Field(None, description="Direct URL to tracklist")
    tracklist_id: Optional[str] = Field(None, description="Tracklist ID")
    force_refresh: bool = Field(default=False, description="Force re-scraping even if cached")
    include_transitions: bool = Field(default=True, description="Include transition information")
    correlation_id: UUID = Field(default_factory=uuid4, description="Request tracking ID")

    @model_validator(mode="after")
    def validate_request(self) -> "TracklistRequest":
        """Validate that either URL or tracklist_id is provided."""
        if self.url is None and self.tracklist_id is None:
            raise ValueError("Either url or tracklist_id must be provided")
        return self


class TracklistResponse(BaseModel):
    """Response model for tracklist retrieval."""

    success: bool = Field(description="Whether retrieval was successful")
    tracklist: Optional[Tracklist] = Field(None, description="Retrieved tracklist data")
    error: Optional[str] = Field(None, description="Error message if failed")
    cached: bool = Field(default=False, description="Whether data was from cache")
    processing_time_ms: Optional[int] = Field(None, description="Processing time in milliseconds")
    correlation_id: UUID = Field(description="Request tracking ID")
