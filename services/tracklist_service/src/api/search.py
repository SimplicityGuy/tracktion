"""
Search API endpoints for the tracklist service.

Provides REST endpoints for searching 1001tracklists.com.
"""

import logging
import time
from datetime import date
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from services.tracklist_service.src.cache.redis_cache import get_cache
from services.tracklist_service.src.config import get_config
from services.tracklist_service.src.models.search_models import SearchError, SearchRequest, SearchResponse, SearchType
from services.tracklist_service.src.scraper.search_scraper import SearchScraper

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/search", tags=["search"])


def get_scraper() -> SearchScraper:
    """Dependency to get a scraper instance.

    Returns:
        SearchScraper instance
    """
    return SearchScraper()


@router.get("/", response_model=SearchResponse)
async def search_tracklists(
    query: str,
    scraper: SearchScraper,  # Will be injected via Depends in route
    search_type: SearchType = SearchType.DJ,
    page: int = 1,
    limit: int = 20,
    start_date: str | None = None,
    end_date: str | None = None,
) -> SearchResponse:
    """Search for tracklists on 1001tracklists.com.

    Args:
        query: Search query string
        search_type: Type of search (DJ, EVENT, TRACK)
        page: Page number for pagination
        limit: Number of results per page
        start_date: Optional start date filter
        end_date: Optional end date filter
        scraper: Scraper instance (injected)

    Returns:
        Search response with results and pagination

    Raises:
        HTTPException: If search fails or validation errors occur
    """
    start_time = time.time()
    correlation_id = uuid4()

    try:
        # Parse dates if provided

        parsed_start_date = None
        parsed_end_date = None

        if start_date:
            try:
                parsed_start_date = date.fromisoformat(start_date)
            except ValueError as e:
                raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD") from e

        if end_date:
            try:
                parsed_end_date = date.fromisoformat(end_date)
            except ValueError as e:
                raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD") from e

        # Create search request
        request = SearchRequest(
            query=query,
            search_type=search_type,
            page=page,
            limit=limit,
            start_date=parsed_start_date,
            end_date=parsed_end_date,
            correlation_id=correlation_id,
        )

        # Check cache first
        cache = get_cache()
        cached_response = cache.get_cached_response(request)

        if cached_response:
            # Update response time for cached response
            cached_response.response_time_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Cache hit: correlation_id={correlation_id}, "
                f"query='{query}', time={cached_response.response_time_ms:.2f}ms"
            )
            return cached_response

        # Check if this search failed recently
        recent_error = cache.is_search_failed_recently(request)
        if recent_error:
            logger.warning(f"Recent failed search: correlation_id={correlation_id}, query='{query}'")
            raise HTTPException(status_code=503, detail=f"Search temporarily unavailable: {recent_error}")

        # Execute search
        response = scraper.search(request)

        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        response.response_time_ms = response_time_ms

        # Cache successful response
        cache.cache_response(request, response)

        logger.info(
            f"Search completed: correlation_id={correlation_id}, "
            f"query='{query}', results={len(response.results)}, "
            f"time={response_time_ms:.2f}ms"
        )

        return response

    except ValidationError as e:
        logger.error(f"Validation error: correlation_id={correlation_id}, error={e}")
        raise HTTPException(status_code=400, detail=str(e)) from e

    except HTTPException:
        # Re-raise HTTP exceptions (like date validation errors)
        raise

    except Exception as e:
        logger.error(f"Search failed: correlation_id={correlation_id}, error={e}")

        # Cache the failed search to prevent hammering (only if request was created)
        if "request" in locals():
            cache = get_cache()
            cache.cache_failed_search(request, str(e))

        # Return error response
        error = SearchError(
            error_code="SEARCH_FAILED",
            error_message=str(e),
            correlation_id=correlation_id,
            details={"query": query, "type": search_type.value},
            retry_after=30,
        )

        raise HTTPException(status_code=500, detail=error.model_dump(mode="json")) from e


@router.get("/dj/{dj_slug}", response_model=SearchResponse)
async def get_dj_tracklists(
    dj_slug: str,
    scraper: SearchScraper,  # Will be injected via Depends in route
    page: int = 1,
    limit: int = 20,
) -> SearchResponse:
    """Get tracklists for a specific DJ.

    Args:
        dj_slug: DJ identifier/slug
        page: Page number for pagination
        limit: Number of results per page
        scraper: Scraper instance (injected)

    Returns:
        Search response with DJ's tracklists

    Raises:
        HTTPException: If request fails
    """
    start_time = time.time()

    try:
        # Get DJ tracklists
        response = scraper.get_dj_tracklists(dj_slug, page, limit)

        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        response.response_time_ms = response_time_ms

        logger.info(
            f"DJ tracklists retrieved: dj_slug={dj_slug}, "
            f"results={len(response.results)}, time={response_time_ms:.2f}ms"
        )

        return response

    except Exception as e:
        logger.error(f"Failed to get DJ tracklists: dj_slug={dj_slug}, error={e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve tracklists for DJ: {dj_slug}") from e


@router.get("/event/{event_slug}", response_model=SearchResponse)
async def get_event_tracklists(
    event_slug: str,
    scraper: SearchScraper,  # Will be injected via Depends in route
    page: int = 1,
    limit: int = 20,
) -> SearchResponse:
    """Get tracklists for a specific event.

    Args:
        event_slug: Event identifier/slug
        page: Page number for pagination
        limit: Number of results per page
        scraper: Scraper instance (injected)

    Returns:
        Search response with event's tracklists

    Raises:
        HTTPException: If request fails
    """
    start_time = time.time()

    try:
        # Get event tracklists
        response = scraper.get_event_tracklists(event_slug, page, limit)

        # Calculate response time
        response_time_ms = (time.time() - start_time) * 1000
        response.response_time_ms = response_time_ms

        logger.info(
            f"Event tracklists retrieved: event_slug={event_slug}, "
            f"results={len(response.results)}, time={response_time_ms:.2f}ms"
        )

        return response

    except Exception as e:
        logger.error(f"Failed to get event tracklists: event_slug={event_slug}, error={e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve tracklists for event: {event_slug}") from e


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint.

    Returns:
        Health status information
    """
    config = get_config()

    return {
        "status": "healthy",
        "service": "tracklist-service",
        "version": "0.1.0",
        "cache_enabled": config.cache.enabled,
        "scraping_enabled": True,
    }
