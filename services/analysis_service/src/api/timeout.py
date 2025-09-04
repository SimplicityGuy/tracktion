"""Request timeout and cancellation handling."""

import asyncio
import contextlib
from collections.abc import Callable
from typing import Any, cast

from fastapi import HTTPException, Request, Response, status
from starlette.middleware.base import BaseHTTPMiddleware

from services.analysis_service.src.structured_logging import get_logger

logger = get_logger(__name__)


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware to handle request timeouts."""

    def __init__(self, app: Any, default_timeout: float = 30.0) -> None:
        """Initialize timeout middleware.

        Args:
            app: FastAPI application
            default_timeout: Default timeout in seconds
        """
        super().__init__(app)
        self.default_timeout = default_timeout

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle request with timeout.

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            Response or timeout error
        """
        # Get timeout from headers or use default
        timeout = self.default_timeout
        if "X-Request-Timeout" in request.headers:
            with contextlib.suppress(ValueError):
                timeout = float(request.headers["X-Request-Timeout"])

        # Store timeout in request state for endpoint access
        request.state.timeout = timeout

        try:
            # Process request with timeout
            response = await asyncio.wait_for(call_next(request), timeout=timeout)

            # Add timeout info to response headers
            response.headers["X-Timeout"] = str(timeout)

            return cast("Response", response)

        except TimeoutError as e:
            # Log timeout
            logger.warning(
                "Request timed out",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "timeout": timeout,
                    "request_id": getattr(request.state, "request_id", None),
                },
            )

            # Return timeout error
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"Request timed out after {timeout} seconds",
            ) from e


class CancellationToken:
    """Token for request cancellation."""

    def __init__(self) -> None:
        """Initialize cancellation token."""
        self._cancelled = False
        self._event = asyncio.Event()

    def cancel(self) -> None:
        """Cancel the operation."""
        self._cancelled = True
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        """Check if operation is cancelled."""
        return self._cancelled

    async def wait_for_cancellation(self) -> None:
        """Wait for cancellation signal."""
        await self._event.wait()

    def check_cancelled(self) -> None:
        """Check and raise if cancelled."""
        if self._cancelled:
            raise asyncio.CancelledError("Operation was cancelled")


class RequestCancellationMiddleware(BaseHTTPMiddleware):
    """Middleware to handle request cancellation."""

    def __init__(self, app: Any) -> None:
        """Initialize cancellation middleware.

        Args:
            app: FastAPI application
        """
        super().__init__(app)
        # Store active requests for cancellation
        self.active_requests: dict[str, CancellationToken] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle request with cancellation support.

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            Response or cancellation error
        """
        # Get request ID
        request_id = getattr(request.state, "request_id", None)

        # Create cancellation token
        cancellation_token = CancellationToken()

        # Store in request state and active requests
        request.state.cancellation_token = cancellation_token
        if request_id:
            self.active_requests[request_id] = cancellation_token

        try:
            # Process request
            response = await call_next(request)
            return cast("Response", response)

        except asyncio.CancelledError as e:
            # Log cancellation
            logger.info(
                "Request cancelled",
                extra={
                    "path": request.url.path,
                    "method": request.method,
                    "request_id": request_id,
                },
            )

            # Return cancellation response
            raise HTTPException(status_code=499, detail="Request was cancelled") from e  # Client Closed Request

        finally:
            # Clean up
            if request_id and request_id in self.active_requests:
                del self.active_requests[request_id]

    def cancel_request(self, request_id: str) -> bool:
        """Cancel a specific request.

        Args:
            request_id: ID of request to cancel

        Returns:
            True if cancelled, False if not found
        """
        if request_id in self.active_requests:
            self.active_requests[request_id].cancel()
            return True
        return False


async def with_timeout(
    coroutine: Any,
    timeout: float,
    cancellation_token: CancellationToken | None = None,
) -> Any:
    """Execute coroutine with timeout and cancellation support.

    Args:
        coroutine: Async function to execute
        timeout: Timeout in seconds
        cancellation_token: Optional cancellation token

    Returns:
        Result of coroutine

    Raises:
        asyncio.TimeoutError: If timeout exceeded
        asyncio.CancelledError: If cancelled
    """
    # Create tasks
    tasks = [asyncio.create_task(coroutine)]

    # Add cancellation task if token provided
    if cancellation_token:
        tasks.append(asyncio.create_task(cancellation_token.wait_for_cancellation()))

    # Wait for first to complete or timeout
    done, pending = await asyncio.wait(tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)

    # Cancel pending tasks
    for task in pending:
        task.cancel()

    # Check what completed
    if not done:
        raise TimeoutError(f"Operation timed out after {timeout} seconds")

    completed_task = done.pop()

    # Check if it was cancellation
    if cancellation_token and completed_task == tasks[1]:
        tasks[0].cancel()  # Cancel the main task
        raise asyncio.CancelledError("Operation was cancelled")

    # Return result
    return completed_task.result()


# Endpoint-specific timeout configurations
ENDPOINT_TIMEOUTS = {
    "/v1/analysis": 60.0,  # Analysis can take longer
    "/v1/streaming/batch-process": 120.0,  # Batch processing needs more time
    "/v1/streaming/audio": 300.0,  # Audio streaming can be long
    "/v1/health": 5.0,  # Health checks should be fast
    "/v1/health/ready": 5.0,
    "/v1/health/live": 5.0,
}


def get_endpoint_timeout(path: str, default: float = 30.0) -> float:
    """Get timeout for specific endpoint.

    Args:
        path: Request path
        default: Default timeout if not configured

    Returns:
        Timeout in seconds
    """
    # Check exact match first
    if path in ENDPOINT_TIMEOUTS:
        return ENDPOINT_TIMEOUTS[path]

    # Check prefix match
    for endpoint, timeout in ENDPOINT_TIMEOUTS.items():
        if path.startswith(endpoint):
            return timeout

    return default
