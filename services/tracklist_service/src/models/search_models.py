"""
Search data models for 1001tracklists.com integration.

Provides Pydantic models for search requests, results, and responses
with proper validation and serialization.
"""

from datetime import date as date_type, datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator, ValidationInfo


class SearchType(str, Enum):
    """Enumeration of supported search types."""

    DJ = "dj"
    EVENT = "event"
    TRACK = "track"


class PaginationInfo(BaseModel):
    """Pagination information for search results."""

    page: int = Field(ge=1, description="Current page number (1-based)")
    limit: int = Field(ge=1, le=100, description="Number of items per page")
    total_pages: int = Field(ge=0, description="Total number of pages")
    total_items: int = Field(ge=0, description="Total number of items")
    has_next: bool = Field(description="Whether there is a next page")
    has_previous: bool = Field(description="Whether there is a previous page")


class SearchRequest(BaseModel):
    """Request model for search operations."""

    query: str = Field(min_length=1, max_length=255, description="Search query string")
    search_type: SearchType = Field(default=SearchType.DJ, description="Type of search to perform")
    start_date: Optional[date_type] = Field(None, description="Start date for filtering results")
    end_date: Optional[date_type] = Field(None, description="End date for filtering results")
    page: int = Field(default=1, ge=1, description="Page number for pagination (1-based)")
    limit: int = Field(default=20, ge=1, le=100, description="Number of results per page")
    correlation_id: UUID = Field(default_factory=uuid4, description="Unique identifier for request tracking")

    @field_validator("end_date")
    @classmethod
    def validate_date_range(cls, v: Optional[date_type], info: ValidationInfo) -> Optional[date_type]:
        """Validate that end_date is after start_date."""
        if v is not None and info.data and "start_date" in info.data:
            start_date = info.data["start_date"]
            if start_date is not None and v < start_date:
                raise ValueError("end_date must be after start_date")
        return v

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate and clean the search query."""
        # Strip whitespace and ensure non-empty
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query cannot be empty or only whitespace")
        return cleaned


class SearchResult(BaseModel):
    """Individual search result item."""

    dj_name: str = Field(description="Name of the DJ or artist")
    event_name: Optional[str] = Field(None, description="Name of the event or festival")
    date: Optional[date_type] = Field(None, description="Date of the set")
    venue: Optional[str] = Field(None, description="Venue where the set was performed")
    set_type: Optional[str] = Field(None, description="Type of set (e.g., 'DJ Set', 'Live Set')")
    url: str = Field(description="URL to the tracklist on 1001tracklists.com")
    duration: Optional[str] = Field(None, description="Duration of the set")
    track_count: Optional[int] = Field(None, ge=0, description="Number of tracks in the set")
    genre: Optional[str] = Field(None, description="Primary genre of the set")
    description: Optional[str] = Field(None, description="Additional description or notes")

    # Metadata for caching and tracking
    scraped_at: datetime = Field(default_factory=datetime.utcnow, description="When this data was scraped")
    source_url: str = Field(description="Source URL that was scraped")

    @field_validator("url", "source_url")
    @classmethod
    def validate_urls(cls, v: str) -> str:
        """Validate that URLs are properly formatted."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("URL must start with http:// or https://")
        return v


class SearchResponse(BaseModel):
    """Response model for search operations."""

    results: List[SearchResult] = Field(description="List of search results")
    pagination: PaginationInfo = Field(description="Pagination information")
    query_info: Dict[str, Any] = Field(description="Information about the query performed")
    cache_hit: bool = Field(description="Whether this response was served from cache")
    response_time_ms: float = Field(ge=0, description="Response time in milliseconds")
    correlation_id: UUID = Field(description="Correlation ID from the request")

    @field_validator("results")
    @classmethod
    def validate_results_count(cls, v: List[SearchResult], info: ValidationInfo) -> List[SearchResult]:
        """Validate that results count matches pagination info."""
        if info.data and "pagination" in info.data:
            pagination = info.data["pagination"]
            expected_max = pagination.limit
            if len(v) > expected_max:
                raise ValueError(f"Results count ({len(v)}) exceeds page limit ({expected_max})")
        return v


class SearchError(BaseModel):
    """Error model for search operations."""

    error_code: str = Field(description="Machine-readable error code")
    error_message: str = Field(description="Human-readable error message")
    correlation_id: UUID = Field(description="Correlation ID from the request")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    retry_after: Optional[int] = Field(None, ge=0, description="Seconds to wait before retrying")


# Message queue models


class SearchRequestMessage(BaseModel):
    """Message model for search requests via message queue."""

    request: SearchRequest = Field(description="The search request")
    reply_to: Optional[str] = Field(None, description="Queue to send the response to")
    timeout_seconds: int = Field(default=30, ge=1, description="Request timeout in seconds")


class SearchResponseMessage(BaseModel):
    """Message model for search responses via message queue."""

    success: bool = Field(description="Whether the search was successful")
    response: Optional[SearchResponse] = Field(None, description="Search response if successful")
    error: Optional[SearchError] = Field(None, description="Error details if unsuccessful")
    processing_time_ms: float = Field(ge=0, description="Total processing time in milliseconds")


# Cache models


class CacheKey(BaseModel):
    """Model for generating consistent cache keys."""

    search_type: SearchType
    query: str
    start_date: Optional[date_type] = None
    end_date: Optional[date_type] = None
    page: int
    limit: int

    def generate_key(self, prefix: str = "search") -> str:
        """Generate a cache key string."""
        components = [
            prefix,
            self.search_type.value,
            self.query.lower().replace(" ", "_"),
            f"page_{self.page}",
            f"limit_{self.limit}",
        ]

        if self.start_date:
            components.append(f"start_{self.start_date.isoformat()}")
        if self.end_date:
            components.append(f"end_{self.end_date.isoformat()}")

        return ":".join(components)


class CachedSearchResponse(BaseModel):
    """Model for cached search responses."""

    response: SearchResponse = Field(description="The cached search response")
    cached_at: datetime = Field(default_factory=datetime.utcnow, description="When this was cached")
    expires_at: datetime = Field(description="When this cache entry expires")
    cache_version: str = Field(default="1.0", description="Cache format version")
