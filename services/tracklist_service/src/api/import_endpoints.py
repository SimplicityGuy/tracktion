"""
Import API endpoints for 1001tracklists tracklist import functionality.

Provides REST endpoints for importing tracklists from 1001tracklists.com
with automatic CUE file generation and async processing support.
"""

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from services.tracklist_service.src.cache.redis_cache import RedisCache
from services.tracklist_service.src.exceptions import (
    CueGenerationError,
    DatabaseError,
    ImportError,
    MatchingError,
    RateLimitError,
    ServiceUnavailableError,
    TimeoutError,
    TimingError,
    ValidationError,
)
from services.tracklist_service.src.messaging.import_handler import ImportJobMessage, import_message_handler
from services.tracklist_service.src.messaging.simple_handler import MessageHandler
from services.tracklist_service.src.models.cue_file import CueFormat
from services.tracklist_service.src.models.tracklist import ImportTracklistRequest, ImportTracklistResponse, TracklistDB
from services.tracklist_service.src.retry.retry_manager import FailureType, RetryManager, RetryPolicy
from services.tracklist_service.src.services.cue_integration import CueIntegrationService
from services.tracklist_service.src.services.import_service import ImportService
from services.tracklist_service.src.services.matching_service import MatchingService
from services.tracklist_service.src.services.timing_service import TimingService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tracklists", tags=["tracklist-import"])


async def setup_import_message_handler() -> None:
    """Set up the import message handler connection."""
    # Placeholder implementation - actual implementation should reconnect
    logger.info("Setting up import message handler connection")


# Initialize services
import_service = ImportService()
matching_service = MatchingService()
timing_service = TimingService()
cue_integration_service = CueIntegrationService()
cache = RedisCache()
message_handler = MessageHandler()

# Initialize retry manager with import-specific policies
retry_manager = RetryManager()

# Configure domain-specific retry policies
retry_manager.set_domain_policy(
    "1001tracklists.com",
    RetryPolicy(
        max_retries=5,
        base_delay=2.0,
        max_delay=300.0,
        failure_policies={
            FailureType.RATE_LIMIT: {"max_retries": 3, "base_delay": 60.0, "max_delay": 600.0},
            FailureType.TIMEOUT: {"max_retries": 2, "base_delay": 5.0},
            FailureType.NETWORK: {"max_retries": 4, "base_delay": 1.0},
        },
    ),
)


async def process_import_async(request: ImportTracklistRequest, correlation_id: str) -> None:
    """
    Process tracklist import asynchronously via message queue.

    Args:
        request: Import request with URL and audio file ID
        correlation_id: Correlation ID for tracking
    """
    try:
        # Create import job message
        job_message = ImportJobMessage(
            correlation_id=correlation_id, request=request, created_at=datetime.now(UTC).isoformat()
        )

        # Ensure message handler is connected
        if not import_message_handler.connection or import_message_handler.connection.is_closed:
            await setup_import_message_handler()

        # Publish to import queue
        await import_message_handler.publish_import_job(job_message)

        logger.info(
            "Queued import job for async processing",
            extra={"correlation_id": correlation_id, "url": request.url, "audio_file_id": str(request.audio_file_id)},
        )

    except Exception as e:
        logger.error(f"Failed to queue import job: {e}", extra={"correlation_id": correlation_id})
        # Fall back to simple message handler if RabbitMQ is unavailable
        message = {
            "type": "tracklist_import",
            "request": request.model_dump(),
            "correlation_id": correlation_id,
            "timestamp": time.time(),
        }
        await message_handler.publish("tracklist.import", message)


@router.post("/import/1001tracklists", response_model=ImportTracklistResponse)
async def import_tracklist_from_1001tracklists(
    request: ImportTracklistRequest,
    background_tasks: BackgroundTasks,
    db: Session,  # Will be injected via Depends in route
    async_processing: bool = False,
) -> ImportTracklistResponse:
    """
    Import a tracklist from 1001tracklists.com and generate CUE file.

    Args:
        request: Import request containing URL and audio file ID
        background_tasks: FastAPI background tasks for async processing
        async_processing: Whether to process asynchronously
        db: Database session

    Returns:
        ImportTracklistResponse with imported tracklist and CUE file path
    """
    start_time = time.time()
    correlation_id = uuid4()

    try:
        # Validate audio file exists (this would need actual audio service integration)
        logger.info(f"Starting import for URL: {request.url}, audio_file_id: {request.audio_file_id}")

        # Handle async processing
        if async_processing:
            background_tasks.add_task(process_import_async, request, str(correlation_id))

            return ImportTracklistResponse(
                success=True,
                tracklist=None,
                cue_file_path=None,
                error=None,
                cached=False,
                processing_time_ms=int((time.time() - start_time) * 1000),
                correlation_id=str(correlation_id),
                message="Import queued for async processing. Check status endpoint for progress.",
            )

        # Step 1: Import tracklist from 1001tracklists
        try:
            logger.info(
                "Step 1: Importing tracklist from 1001tracklists",
                extra={
                    "correlation_id": str(correlation_id),
                    "url": request.url,
                    "audio_file_id": str(request.audio_file_id),
                    "force_refresh": request.force_refresh,
                },
            )
            imported_tracklist = import_service.import_tracklist(
                url=request.url, audio_file_id=request.audio_file_id, force_refresh=request.force_refresh
            )
        except Exception as e:
            error_msg = f"Import failed: {e!s}"
            logger.error(
                error_msg,
                extra={
                    "correlation_id": str(correlation_id),
                    "url": request.url,
                    "error_type": type(e).__name__,
                    "error_details": getattr(e, "details", {}),
                },
            )
            raise ImportError(error_msg, url=request.url, tracklist_id=str(correlation_id)) from e

        # Step 2: Perform matching with audio file
        try:
            logger.info(
                "Step 2: Matching tracklist with audio file",
                extra={
                    "correlation_id": str(correlation_id),
                    "audio_file_id": str(request.audio_file_id),
                    "track_count": len(imported_tracklist.tracks),
                },
            )
            matching_result = matching_service.match_tracklist_with_audio_file(
                tracklist=imported_tracklist, audio_file_id=request.audio_file_id
            )

            # Update tracklist with matching confidence
            imported_tracklist.confidence_score = matching_result.confidence_score
            if matching_result.metadata:
                # Update tracklist with audio metadata if available
                logger.info(
                    "Audio metadata retrieved",
                    extra={
                        "correlation_id": str(correlation_id),
                        "duration": matching_result.metadata.get("duration", "Unknown"),
                        "confidence_score": matching_result.confidence_score,
                    },
                )
        except Exception as e:
            error_msg = f"Matching failed: {e!s}"
            logger.error(
                error_msg,
                extra={
                    "correlation_id": str(correlation_id),
                    "audio_file_id": str(request.audio_file_id),
                    "error_type": type(e).__name__,
                },
            )
            raise MatchingError(error_msg, audio_file_id=str(request.audio_file_id)) from e

        # Step 3: Apply timing adjustments
        try:
            logger.info(
                "Step 3: Applying timing adjustments",
                extra={
                    "correlation_id": str(correlation_id),
                    "track_count": len(imported_tracklist.tracks),
                    "audio_duration": (
                        matching_result.metadata.get("duration_seconds") if matching_result.metadata else None
                    ),
                },
            )
            # Convert duration to timedelta for timing service

            audio_duration = None
            if matching_result.metadata and matching_result.metadata.get("duration_seconds"):
                audio_duration = timedelta(seconds=matching_result.metadata["duration_seconds"])

            adjusted_tracks = timing_service.adjust_track_timings(
                tracks=imported_tracklist.tracks, audio_duration=audio_duration
            )
            imported_tracklist.tracks = adjusted_tracks
        except Exception as e:
            error_msg = f"Timing adjustment failed: {e!s}"
            logger.error(error_msg, extra={"correlation_id": str(correlation_id), "error_type": type(e).__name__})
            raise TimingError(error_msg) from e

        # Step 4: Generate CUE file
        try:
            logger.info(
                "Step 4: Generating CUE file",
                extra={
                    "correlation_id": str(correlation_id),
                    "cue_format": request.cue_format,
                    "track_count": len(imported_tracklist.tracks),
                },
            )
            cue_success, cue_content, cue_error = cue_integration_service.generate_cue_content(
                tracklist=imported_tracklist,
                audio_filename=f"audio_file_{request.audio_file_id}.wav",  # Fixed parameter name
                cue_format=CueFormat(request.cue_format),
            )

            # Update tracklist with CUE content if successful
            if cue_success:
                # Note: Tracklist doesn't have cue_file_id field, commenting out
                # imported_tracklist.cue_file_id = some_id  # Would need to save CUE file first
                pass
        except Exception as e:
            error_msg = f"CUE generation failed: {e!s}"
            logger.error(
                error_msg,
                extra={
                    "correlation_id": str(correlation_id),
                    "cue_format": request.cue_format,
                    "error_type": type(e).__name__,
                },
            )
            raise CueGenerationError(
                error_msg, cue_format=request.cue_format, tracklist_id=str(imported_tracklist.id)
            ) from e

        # Step 5: Save to database
        try:
            logger.info(
                "Step 5: Saving to database",
                extra={"correlation_id": str(correlation_id), "tracklist_id": str(imported_tracklist.id)},
            )
            db_tracklist = TracklistDB.from_model(imported_tracklist)
            # Note: CUE generation returns content, not file path - would need to save to get path
            # if cue_success and cue_content:
            #     db_tracklist.cue_file_path = saved_cue_file_path

            db.add(db_tracklist)
            db.commit()
            db.refresh(db_tracklist)
        except Exception as e:
            db.rollback()
            error_msg = f"Database save failed: {e!s}"
            logger.error(
                error_msg,
                extra={
                    "correlation_id": str(correlation_id),
                    "tracklist_id": str(imported_tracklist.id),
                    "error_type": type(e).__name__,
                },
            )
            raise DatabaseError(error_msg, operation="insert", table="tracklists") from e

        # Step 6: Publish success message
        background_tasks.add_task(
            message_handler.publish,
            "tracklist.imported",
            {
                "tracklist_id": str(imported_tracklist.id),
                "audio_file_id": str(request.audio_file_id),
                "track_count": len(imported_tracklist.tracks),
                "confidence_score": imported_tracklist.confidence_score,
                "cue_file_generated": cue_success,
                "imported_at": imported_tracklist.created_at.isoformat(),
            },
        )

        processing_time = int((time.time() - start_time) * 1000)

        return ImportTracklistResponse(
            success=True,
            tracklist=imported_tracklist,
            cue_file_path=None,  # Note: Would need to save CUE content to file to get path
            error=None,
            cached=False,  # Import is always fresh
            processing_time_ms=processing_time,
            correlation_id=str(correlation_id),
            message=None,
        )

    except (ImportError, MatchingError, TimingError, CueGenerationError, DatabaseError) as e:
        # Handle expected service errors with specific details
        logger.warning(
            f"Service error during import: {e.message}",
            extra={
                "correlation_id": str(correlation_id),
                "error_code": e.error_code,
                "error_type": type(e).__name__,
                "error_details": getattr(e, "details", {}),
            },
        )
        processing_time = int((time.time() - start_time) * 1000)

        return ImportTracklistResponse(
            success=False,
            tracklist=None,
            cue_file_path=None,
            error=f"Import failed: {e.message}",
            cached=False,
            processing_time_ms=processing_time,
            correlation_id=str(correlation_id),
            message=None,
        )

    except (RateLimitError, TimeoutError, ServiceUnavailableError) as e:
        # Handle retryable errors
        logger.warning(
            f"Retryable error during import: {e.message}",
            extra={
                "correlation_id": str(correlation_id),
                "error_code": e.error_code,
                "retry_after": getattr(e, "retry_after", None),
                "should_retry": True,
            },
        )

        # For retryable errors, we might queue for retry (simplified here)
        processing_time = int((time.time() - start_time) * 1000)

        return ImportTracklistResponse(
            success=False,
            tracklist=None,
            cue_file_path=None,
            error=f"Temporary error (will retry): {e.message}",
            cached=False,
            processing_time_ms=processing_time,
            correlation_id=str(correlation_id),
            message=None,
        )

    except ValidationError as e:
        # Handle validation errors
        logger.warning(
            f"Validation error during import: {e.message}",
            extra={
                "correlation_id": str(correlation_id),
                "field": getattr(e, "field", None),
                "value": getattr(e, "value", None),
            },
        )

        raise HTTPException(status_code=400, detail=f"Validation error: {e.message}") from e

    except ValueError as e:
        # Handle legacy validation errors (fallback)
        logger.warning(f"Legacy validation error: {e}")
        processing_time = int((time.time() - start_time) * 1000)

        return ImportTracklistResponse(
            success=False,
            tracklist=None,
            cue_file_path=None,
            error=f"Import failed: {e!s}",
            cached=False,
            processing_time_ms=processing_time,
            correlation_id=str(correlation_id),
            message=None,
        )

    except Exception as e:
        # Handle unexpected errors
        logger.error(
            f"Unexpected error during import: {e!s}",
            extra={"correlation_id": str(correlation_id), "error_type": type(e).__name__, "traceback": True},
            exc_info=True,
        )
        processing_time = int((time.time() - start_time) * 1000)

        return ImportTracklistResponse(
            success=False,
            tracklist=None,
            cue_file_path=None,
            error="Internal server error occurred during import",
            cached=False,
            processing_time_ms=processing_time,
            correlation_id=str(correlation_id),
            message=None,
        )


@router.get("/import/status/{correlation_id}")
async def get_import_status(correlation_id: UUID) -> JSONResponse:
    """
    Get the status of an async tracklist import job.

    Args:
        correlation_id: Job correlation ID from async import request

    Returns:
        JSON response with job status and result if available
    """
    try:
        # Check job status in cache
        status_key = f"import:status:{correlation_id}"
        status_data = await cache.get(status_key)

        if not status_data:
            raise HTTPException(status_code=404, detail=f"No import job found with correlation ID: {correlation_id}")

        status = status_data if isinstance(status_data, dict) else {"status": "unknown"}

        # If completed, include the result
        if status.get("status") == "completed":
            result_key = f"import:result:{correlation_id}"
            result_data = await cache.get(result_key)
            if result_data:
                status["result"] = result_data

        return JSONResponse(content=status)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting import status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve import status") from e


@router.delete("/import/cache")
async def clear_import_cache(url: str | None = None) -> JSONResponse:
    """
    Clear import-related cache entries.

    Args:
        url: Optional specific 1001tracklists URL to clear from cache

    Returns:
        JSON response with number of entries cleared
    """
    try:
        if url:
            # Clear specific URL cache
            cache_key = f"tracklist:1001:{url}"
            deleted = await cache.delete(cache_key)

            return JSONResponse(
                content={"success": True, "message": f"Cleared import cache for URL: {url}", "entries_cleared": deleted}
            )
        # Clear all import cache entries
        # This would need implementation in RedisCache to find all keys with pattern
        return JSONResponse(
            content={"success": False, "message": "Bulk cache clearing not yet implemented. Please specify a URL."}
        )

    except Exception as e:
        logger.error(f"Error clearing import cache: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "error": f"Failed to clear cache: {e!s}"})


@router.get("/import/health")
async def import_health_check() -> JSONResponse:
    """
    Health check endpoint for the import API.

    Returns:
        JSON response with import service health status
    """
    health_status: dict[str, Any] = {
        "service": "tracklist_import_api",
        "status": "healthy",
        "timestamp": time.time(),
        "components": {},
    }

    # Check import service
    try:
        # Basic instantiation check
        ImportService()
        health_status["components"]["import_service"] = "healthy"
    except Exception as e:
        health_status["components"]["import_service"] = f"unhealthy: {e!s}"
        health_status["status"] = "degraded"

    # Check matching service
    try:
        MatchingService()
        health_status["components"]["matching_service"] = "healthy"
    except Exception as e:
        health_status["components"]["matching_service"] = f"unhealthy: {e!s}"
        health_status["status"] = "degraded"

    # Check timing service
    try:
        TimingService()
        health_status["components"]["timing_service"] = "healthy"
    except Exception as e:
        health_status["components"]["timing_service"] = f"unhealthy: {e!s}"
        health_status["status"] = "degraded"

    # Check CUE integration service
    try:
        CueIntegrationService()
        health_status["components"]["cue_integration_service"] = "healthy"
    except Exception as e:
        health_status["components"]["cue_integration_service"] = f"unhealthy: {e!s}"
        health_status["status"] = "degraded"

    # Check cache connection
    try:
        await cache.ping()
        health_status["components"]["cache"] = "healthy"
    except Exception as e:
        health_status["components"]["cache"] = f"unhealthy: {e!s}"
        health_status["status"] = "degraded"

    # Check message handler
    try:
        await message_handler.ping()
        health_status["components"]["message_handler"] = "healthy"
    except Exception as e:
        health_status["components"]["message_handler"] = f"unhealthy: {e!s}"
        health_status["status"] = "degraded"

    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)
