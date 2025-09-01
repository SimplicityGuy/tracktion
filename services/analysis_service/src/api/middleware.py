"""Async middleware for Analysis Service API."""

import time
import uuid
from collections.abc import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from services.analysis_service.src.structured_logging import get_logger

logger = get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add request ID to all requests and responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add request ID to request state and response headers."""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Log request start
        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
            },
        )

        # Process request
        response = await call_next(request)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response  # FastAPI middleware return type


class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware to track request processing time."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Track and log request processing time."""
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate processing time
        process_time = (time.time() - start_time) * 1000  # Convert to ms

        # Add timing to response headers
        response.headers["X-Process-Time"] = f"{process_time:.2f}ms"

        # Log request completion
        logger.info(
            "Request completed",
            extra={
                "request_id": getattr(request.state, "request_id", None),
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "process_time_ms": process_time,
            },
        )

        return response  # FastAPI middleware return type


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for global error handling."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle errors globally with proper logging."""
        try:
            return await call_next(request)  # FastAPI middleware return type  # FastAPI/Starlette response
        except Exception as e:
            # Get request ID if available
            request_id = getattr(request.state, "request_id", None)

            # Log error
            logger.error(
                "Unhandled exception in request",
                extra={"request_id": request_id, "error": str(e)},
                exc_info=True,
            )

            # Return error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal Server Error",
                    "message": "An unexpected error occurred",
                    "request_id": request_id,
                },
                headers={"X-Request-ID": request_id} if request_id else {},
            )
