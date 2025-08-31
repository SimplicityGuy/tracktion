"""
Search API endpoints for 1001tracklists.com integration.

Provides REST API endpoints for searching DJ sets and tracklists.
"""

import logging
import time
from typing import List, Optional, Dict, Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..services.import_service import ImportService
from ..cache.redis_cache import RedisCache

logger = logging.getLogger(__name__)

# Create router for search endpoints
router = APIRouter(prefix="/api/v1/tracklists", tags=["search"])

# Initialize services
import_service = ImportService()
cache = RedisCache()


class SearchResult(BaseModel):
    """Individual search result from 1001tracklists."""

    id: str = Field(description="Tracklist ID from 1001tracklists")
    url: str = Field(description="Full URL to the tracklist")
    title: str = Field(description="Tracklist title")
    dj_name: str = Field(description="DJ name")
    date: Optional[str] = Field(None, description="Event date if available")
    event_name: Optional[str] = Field(None, description="Event name if available")
    track_count: Optional[int] = Field(None, description="Number of tracks")
    duration: Optional[str] = Field(None, description="Duration if available")
    genre: Optional[str] = Field(None, description="Genre information")
    confidence: float = Field(default=1.0, description="Search result confidence score")


class SearchResponse(BaseModel):
    """Response model for tracklist search."""

    success: bool = Field(description="Whether search was successful")
    results: List[SearchResult] = Field(default_factory=list, description="Search results")
    total_count: int = Field(description="Total number of results found")
    page: int = Field(description="Current page number")
    page_size: int = Field(description="Results per page")
    has_more: bool = Field(description="Whether there are more results available")
    error: Optional[str] = Field(None, description="Error message if failed")
    cached: bool = Field(default=False, description="Whether results were from cache")
    processing_time_ms: int = Field(description="Processing time in milliseconds")
    correlation_id: str = Field(description="Request correlation ID")


@router.get("/search/1001tracklists", response_model=SearchResponse)
async def search_1001tracklists(
    query: Optional[str] = Query(None, description="General search query"),
    artist: Optional[str] = Query(None, description="DJ/Artist name to search for"),
    title: Optional[str] = Query(None, description="Tracklist title to search for"),
    genre: Optional[str] = Query(None, description="Genre to filter by"),
    date_from: Optional[str] = Query(None, description="Start date filter (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="End date filter (YYYY-MM-DD)"),
    page: int = Query(1, ge=1, description="Page number (starts from 1)"),
    page_size: int = Query(20, ge=1, le=100, description="Number of results per page"),
    force_refresh: bool = Query(False, description="Force re-search even if cached"),
) -> SearchResponse:
    """
    Search for tracklists on 1001tracklists.com.

    Args:
        query: General search query (searches title, DJ, event)
        artist: Specific DJ/artist name to search for
        title: Specific tracklist title to search for
        genre: Genre to filter by
        date_from: Start date for date range filter
        date_to: End date for date range filter
        page: Page number for pagination
        page_size: Number of results per page
        force_refresh: Force re-search even if cached

    Returns:
        SearchResponse with paginated search results
    """
    start_time = time.time()
    correlation_id = str(uuid4())

    try:
        # Validate search parameters
        if not any([query, artist, title]):
            raise HTTPException(
                status_code=400, detail="At least one search parameter (query, artist, or title) must be provided"
            )

        # Build search parameters
        search_params = {
            "query": query,
            "artist": artist,
            "title": title,
            "genre": genre,
            "date_from": date_from,
            "date_to": date_to,
            "page": page,
            "page_size": page_size,
        }

        # Remove None values
        search_params = {k: v for k, v in search_params.items() if v is not None}

        # Check cache first unless force refresh
        cache_key = f"search:1001:{hash(str(sorted(search_params.items())))}"
        cached_results = None

        if not force_refresh:
            try:
                cached_data = await cache.get(cache_key)
                if cached_data:
                    import json

                    try:
                        cached_results = json.loads(cached_data)
                        logger.info(f"Using cached search results for query: {search_params}")
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse cached data as JSON")
                        cached_results = None
            except Exception as e:
                logger.warning(f"Cache lookup failed: {e}")

        if cached_results:
            processing_time = int((time.time() - start_time) * 1000)
            # cached_results should already be a dict from cache
            if isinstance(cached_results, dict):
                # Convert legacy cache format to new SearchResponse format
                if "results" in cached_results and "pagination" not in cached_results:
                    # Create proper pagination structure
                    page = cached_results.get("page", 1)
                    page_size = cached_results.get("page_size", 20)
                    total_results = cached_results.get("total_count", 0)
                    has_more = cached_results.get("has_more", False)

                    # pagination_info not used in current implementation

                    formatted_results = {
                        "success": True,
                        "results": cached_results.get("results", []),
                        "total_count": total_results,
                        "page": page,
                        "page_size": page_size,
                        "has_more": has_more,
                        "error": None,
                        "cached": True,
                        "processing_time_ms": processing_time,
                        "correlation_id": correlation_id,
                    }
                    return SearchResponse(**formatted_results)
                else:
                    # Assume it's already in correct format, just update metadata
                    cached_results.update(
                        {"cached": True, "processing_time_ms": processing_time, "correlation_id": correlation_id}
                    )
                    return SearchResponse(**cached_results)
            else:
                # If it's not a dict, skip using cache
                logger.warning(f"Cached data is not in expected format: {type(cached_results)}")
                cached_results = None

        # Perform search using scraper service
        logger.info(f"Performing 1001tracklists search with params: {search_params}")

        # This would integrate with the actual scraper service
        # For now, we'll simulate the search results
        search_results = await _perform_search(search_params)

        # Calculate pagination info
        total_results = len(search_results)
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        paginated_results = search_results[start_index:end_index]

        has_more = end_index < total_results

        # pagination_info not used in current implementation

        response_data = {
            "success": True,
            "results": paginated_results,
            "total_count": total_results,
            "page": page,
            "page_size": page_size,
            "has_more": has_more,
            "error": None,
            "cached": False,
            "processing_time_ms": int((time.time() - start_time) * 1000),
            "correlation_id": correlation_id,
        }

        # Cache the results for future requests
        try:
            import json

            cache_value = json.dumps(response_data)
            await cache.set(cache_key, cache_value, ttl=1800)  # Cache for 30 minutes
        except Exception as e:
            logger.warning(f"Failed to cache search results: {e}")

        return SearchResponse(**response_data)  # type: ignore[arg-type]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        processing_time = int((time.time() - start_time) * 1000)

        # error_pagination_info not used in current implementation

        return SearchResponse(
            success=False,
            results=[],
            total_count=0,
            page=page,
            page_size=page_size,
            has_more=False,
            error=f"Search failed: {str(e)}",
            cached=False,
            processing_time_ms=processing_time,
            correlation_id=correlation_id,
        )


async def _perform_search(search_params: Dict[str, Any]) -> List[SearchResult]:
    """
    Perform the actual search using the scraper service.

    This is a placeholder implementation that would integrate with
    the actual 1001tracklists scraper when available.

    Args:
        search_params: Search parameters

    Returns:
        List of SearchResult objects
    """
    # This would use the scraper service to perform actual search
    # For now, return mock results to demonstrate the API structure

    mock_results = [
        SearchResult(
            id="12345",
            url="https://1001tracklists.com/tracklist/12345/example-set",
            title="Amazing Trance Set",
            dj_name="DJ Example",
            date="2024-01-15",
            event_name="Winter Festival 2024",
            track_count=25,
            duration="1:30:00",
            genre="Trance",
            confidence=0.95,
        ),
        SearchResult(
            id="12346",
            url="https://1001tracklists.com/tracklist/12346/another-set",
            title="Progressive House Journey",
            dj_name="DJ Example",
            date="2024-01-10",
            event_name="Club Night",
            track_count=20,
            duration="1:00:00",
            genre="Progressive House",
            confidence=0.88,
        ),
    ]

    # Filter results based on search parameters
    filtered_results = []
    for result in mock_results:
        matches = True

        # Simple text matching for demonstration
        if search_params.get("query"):
            query_lower = search_params["query"].lower()
            if not any(
                [
                    query_lower in result.title.lower(),
                    query_lower in result.dj_name.lower(),
                    query_lower in (result.event_name or "").lower(),
                ]
            ):
                matches = False

        if search_params.get("artist"):
            if search_params["artist"].lower() not in result.dj_name.lower():
                matches = False

        if search_params.get("title"):
            if search_params["title"].lower() not in result.title.lower():
                matches = False

        if search_params.get("genre"):
            if search_params["genre"].lower() != (result.genre or "").lower():
                matches = False

        if matches:
            filtered_results.append(result)

    return filtered_results


@router.get("/search/health")
async def search_health_check() -> JSONResponse:
    """
    Health check endpoint for the search API.

    Returns:
        JSON response with search service health status
    """
    health_status: Dict[str, Any] = {
        "service": "tracklist_search_api",
        "status": "healthy",
        "timestamp": time.time(),
        "components": {},
    }

    # Check import service (used for search functionality)
    try:
        ImportService()
        health_status["components"]["import_service"] = "healthy"
    except Exception as e:
        health_status["components"]["import_service"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"

    # Check cache connection
    try:
        await cache.ping()
        health_status["components"]["cache"] = "healthy"
    except Exception as e:
        health_status["components"]["cache"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"

    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)
