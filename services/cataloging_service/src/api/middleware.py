"""API middleware for error handling and logging."""

import logging
import time
import uuid
from collections.abc import Awaitable, Callable

import structlog
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for handling errors and exceptions."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Process the request and handle any errors.

        Args:
            request: The incoming request
            call_next: The next middleware or endpoint

        Returns:
            The response
        """
        try:
            return await call_next(request)
        except Exception as e:
            logger.error(
                "Unhandled exception in request",
                exc_info=True,
                path=request.url.path,
                method=request.method,
            )
            # In production, hide error details. In debug mode, show them.
            debug_mode = logging.getLogger().isEnabledFor(logging.DEBUG)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": str(e) if debug_mode else "An error occurred",
                },
            )


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Log request details and response status.

        Args:
            request: The incoming request
            call_next: The next middleware or endpoint

        Returns:
            The response
        """
        # Generate request ID
        request_id = str(uuid.uuid4())

        # Start timer
        start_time = time.time()

        # Log request
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            query_params=dict(request.query_params),
        )

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Log response
        logger.info(
            "Request completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_seconds=duration,
        )

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id

        return response


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """Middleware to handle health check endpoints efficiently."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Skip logging for health check endpoints.

        Args:
            request: The incoming request
            call_next: The next middleware or endpoint

        Returns:
            The response
        """
        # Skip logging for health check endpoints
        if request.url.path in ["/health", "/metrics"]:
            return await call_next(request)

        # Process normally for other endpoints
        return await call_next(request)
