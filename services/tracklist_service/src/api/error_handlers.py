"""Error handlers for API endpoints.

This module provides error handling middleware for converting
custom exceptions to appropriate HTTP responses.
"""

from typing import Any

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..exceptions import (
    TracklistServiceError,
    DraftNotFoundError,
    DuplicatePositionError,
    InvalidTrackPositionError,
    PublishValidationError,
    TimingError,
    ValidationError,
    AudioFileError,
    CueGenerationError,
    ConcurrentEditError,
    DatabaseError,
    ServiceUnavailableError,
)


async def tracklist_exception_handler(request: Request, exc: TracklistServiceError) -> JSONResponse:
    """Handle TracklistServiceError exceptions.

    Args:
        request: Request object.
        exc: Exception instance.

    Returns:
        JSON response with error details.
    """
    # Map exception types to HTTP status codes
    status_map = {
        DraftNotFoundError: status.HTTP_404_NOT_FOUND,
        AudioFileError: status.HTTP_404_NOT_FOUND,
        ValidationError: status.HTTP_400_BAD_REQUEST,
        DuplicatePositionError: status.HTTP_400_BAD_REQUEST,
        InvalidTrackPositionError: status.HTTP_400_BAD_REQUEST,
        PublishValidationError: status.HTTP_400_BAD_REQUEST,
        TimingError: status.HTTP_400_BAD_REQUEST,
        CueGenerationError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        ConcurrentEditError: status.HTTP_409_CONFLICT,
        DatabaseError: status.HTTP_500_INTERNAL_SERVER_ERROR,
        ServiceUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
    }

    # Get status code for this exception type
    status_code = status_map.get(type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Prepare error response
    error_response = {
        "error": {
            "code": exc.error_code,
            "message": exc.message,
            "details": exc.details,
        }
    }

    # Add retry-after header for service unavailable errors
    headers = {}
    if hasattr(exc, "retry_after") and exc.retry_after:
        headers["Retry-After"] = str(exc.retry_after)

    return JSONResponse(
        status_code=status_code,
        content=error_response,
        headers=headers,
    )


def register_exception_handlers(app: Any) -> None:
    """Register exception handlers with the FastAPI app.

    Args:
        app: FastAPI application instance.
    """
    app.add_exception_handler(TracklistServiceError, tracklist_exception_handler)

    # Also handle standard HTTP exceptions
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": f"HTTP_{exc.status_code}",
                    "message": exc.detail,
                }
            },
        )

    # Handle general exceptions
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Unhandled exception: {exc}", exc_info=True)

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred",
                }
            },
        )
