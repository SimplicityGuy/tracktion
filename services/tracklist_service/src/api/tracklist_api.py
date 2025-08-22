"""
Tracklist retrieval API endpoints.

Provides REST endpoints for retrieving tracklist data from 1001tracklists.com
with caching and async processing support.
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse

from ..models.tracklist_models import (
    Tracklist,
    TracklistRequest,
    TracklistResponse,
)
from ..scraper.tracklist_scraper import TracklistScraper
from ..cache.redis_cache import RedisCache
from ..messaging.simple_handler import MessageHandler

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["tracklist"])

# Initialize components
scraper = TracklistScraper()
cache = RedisCache()
message_handler = MessageHandler()


def generate_cache_key(url: str) -> str:
    """Generate a cache key from tracklist URL.

    Uses SHA256 for better security and collision resistance.
    """
    return f"tracklist:{hashlib.sha256(url.encode()).hexdigest()}"


async def process_tracklist_async(request: TracklistRequest) -> None:
    """
    Process tracklist retrieval asynchronously via message queue.

    Args:
        request: Tracklist request with URL or ID
    """
    message = {
        "type": "tracklist_retrieval",
        "request": request.model_dump(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    await message_handler.publish("tracklist.retrieval", message)


@router.get("/tracklist/{tracklist_id}", response_model=TracklistResponse)
async def get_tracklist_by_id(
    tracklist_id: str,
    force_refresh: bool = Query(False, description="Force re-scraping even if cached"),
    include_transitions: bool = Query(True, description="Include transition information"),
) -> TracklistResponse:
    """
    Retrieve a tracklist by its ID.

    Args:
        tracklist_id: Unique identifier for the tracklist
        force_refresh: Force re-scraping even if cached
        include_transitions: Include transition information in response

    Returns:
        TracklistResponse with tracklist data or error
    """
    start_time = time.time()
    correlation_id = uuid4()

    try:
        # For now, we'll need the full URL - in production this would
        # be resolved from a database mapping
        # This is a placeholder implementation
        raise HTTPException(
            status_code=501,
            detail="Tracklist ID resolution not yet implemented. Please use POST /api/v1/tracklist with URL.",
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        # Log unexpected errors for debugging
        logger.error(f"Unexpected error in get_tracklist_by_id: {e}", exc_info=True)

        processing_time = int((time.time() - start_time) * 1000)
        return TracklistResponse(
            success=False,
            error=f"Internal server error: {str(e)}",
            processing_time_ms=processing_time,
            correlation_id=correlation_id,
        )


@router.post("/tracklist", response_model=TracklistResponse)
async def retrieve_tracklist(
    request: TracklistRequest,
    background_tasks: BackgroundTasks,
    async_processing: bool = Query(False, description="Process asynchronously"),
) -> TracklistResponse:
    """
    Retrieve a tracklist from a URL.

    Args:
        request: TracklistRequest with URL or tracklist_id
        background_tasks: FastAPI background tasks for async processing
        async_processing: Whether to process asynchronously

    Returns:
        TracklistResponse with tracklist data or job status
    """
    start_time = time.time()

    try:
        # Validate request has URL (ID resolution not yet implemented)
        if not request.url:
            raise HTTPException(
                status_code=400,
                detail="URL is required. Tracklist ID resolution not yet implemented.",
            )

        # Handle async processing
        if async_processing:
            # Queue for async processing
            background_tasks.add_task(process_tracklist_async, request)

            return TracklistResponse(
                success=True,
                error=None,
                cached=False,
                processing_time_ms=int((time.time() - start_time) * 1000),
                correlation_id=request.correlation_id,
                tracklist=None,  # Will be available via separate status endpoint
            )

        # Check cache unless force refresh
        cache_key = generate_cache_key(request.url)
        if not request.force_refresh:
            cached_data = await cache.get(cache_key)
            if cached_data:
                # Deserialize cached tracklist
                tracklist_dict = json.loads(cached_data)
                tracklist = Tracklist(**tracklist_dict)

                # Filter transitions if not requested
                if not request.include_transitions:
                    tracklist.transitions = []

                processing_time = int((time.time() - start_time) * 1000)
                return TracklistResponse(
                    success=True,
                    tracklist=tracklist,
                    cached=True,
                    processing_time_ms=processing_time,
                    correlation_id=request.correlation_id,
                )

        # Scrape the tracklist
        tracklist = scraper.scrape_tracklist(request.url)

        # Filter transitions if not requested
        if not request.include_transitions:
            tracklist.transitions = []

        # Cache the result (async)
        tracklist_json = tracklist.model_dump_json()
        background_tasks.add_task(
            cache.set,
            cache_key,
            tracklist_json,
            ttl=7 * 24 * 60 * 60,  # 7 days
        )

        # Publish to message queue for downstream processing
        background_tasks.add_task(
            message_handler.publish,
            "tracklist.scraped",
            {
                "tracklist_id": str(tracklist.id),
                "url": tracklist.url,
                "dj_name": tracklist.dj_name,
                "track_count": len(tracklist.tracks),
                "scraped_at": tracklist.scraped_at.isoformat(),
            },
        )

        processing_time = int((time.time() - start_time) * 1000)
        return TracklistResponse(
            success=True,
            tracklist=tracklist,
            cached=False,
            processing_time_ms=processing_time,
            correlation_id=request.correlation_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        processing_time = int((time.time() - start_time) * 1000)
        return TracklistResponse(
            success=False,
            error=f"Failed to retrieve tracklist: {str(e)}",
            processing_time_ms=processing_time,
            correlation_id=request.correlation_id,
        )


@router.get("/tracklist/status/{correlation_id}")
async def get_tracklist_status(correlation_id: UUID) -> JSONResponse:
    """
    Get the status of an async tracklist retrieval job.

    Args:
        correlation_id: Job correlation ID from async request

    Returns:
        JSON response with job status and result if available
    """
    # Check job status in cache
    status_key = f"job:status:{correlation_id}"
    status = await cache.get(status_key)

    if not status:
        raise HTTPException(
            status_code=404,
            detail=f"No job found with correlation ID: {correlation_id}",
        )

    status_data = json.loads(status)

    # If completed, include the tracklist
    if status_data.get("status") == "completed":
        result_key = f"job:result:{correlation_id}"
        result = await cache.get(result_key)
        if result:
            status_data["tracklist"] = json.loads(result)

    return JSONResponse(content=status_data)


@router.delete("/tracklist/cache")
async def clear_tracklist_cache(
    url: Optional[str] = Query(None, description="Specific URL to clear from cache"),
) -> JSONResponse:
    """
    Clear tracklist cache.

    Args:
        url: Optional specific URL to clear, otherwise clears all tracklist cache

    Returns:
        JSON response with number of entries cleared
    """
    try:
        if url:
            # Clear specific URL
            cache_key = generate_cache_key(url)
            deleted = await cache.delete(cache_key)
            return JSONResponse(
                content={
                    "success": True,
                    "message": f"Cleared cache for URL: {url}",
                    "entries_cleared": deleted,
                }
            )
        else:
            # Clear all tracklist cache entries
            # This would need implementation in RedisCache
            return JSONResponse(
                content={
                    "success": False,
                    "message": "Bulk cache clearing not yet implemented",
                }
            )

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": f"Failed to clear cache: {str(e)}",
            },
        )


@router.get("/health")
async def health_check() -> JSONResponse:
    """
    Health check endpoint for the tracklist API.

    Returns:
        JSON response with service health status
    """
    health_status: Dict[str, Any] = {
        "service": "tracklist_api",
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {},
    }

    # Check cache connection
    try:
        await cache.ping()
        health_status["components"]["cache"] = "healthy"
    except Exception as e:
        health_status["components"]["cache"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"

    # Check message queue connection
    try:
        await message_handler.ping()
        health_status["components"]["message_queue"] = "healthy"
    except Exception as e:
        health_status["components"]["message_queue"] = f"unhealthy: {str(e)}"
        health_status["status"] = "degraded"

    # Check scraper functionality
    try:
        # Basic check that scraper can be instantiated
        TracklistScraper()
        health_status["components"]["scraper"] = "healthy"
    except Exception as e:
        health_status["components"]["scraper"] = f"unhealthy: {str(e)}"
        health_status["status"] = "unhealthy"

    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)
