"""
Error handling and resilience patterns for tracklist service.

Provides custom exceptions, retry logic, and circuit breaker patterns.
"""

import asyncio
import functools
import logging
import time
from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TracklistError(Exception):
    """Base exception for tracklist service errors."""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize tracklist error."""
        super().__init__(message)
        self.error_code = error_code or "TRACKLIST_ERROR"
        self.details = details or {}


class TracklistNotFoundError(TracklistError):
    """Raised when a tracklist cannot be found."""

    def __init__(self, url: str, details: dict[str, Any] | None = None):
        """Initialize not found error."""
        super().__init__(
            f"Tracklist not found at URL: {url}",
            error_code="TRACKLIST_NOT_FOUND",
            details=details or {"url": url},
        )


class ParseError(TracklistError):
    """Raised when tracklist parsing fails."""

    def __init__(
        self,
        message: str,
        element: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize parse error."""
        error_details = details or {}
        if element:
            error_details["element"] = element

        super().__init__(
            message,
            error_code="PARSE_ERROR",
            details=error_details,
        )


class ScrapingError(TracklistError):
    """Raised when scraping fails."""

    def __init__(
        self,
        message: str,
        url: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize scraping error."""
        error_details = details or {}
        if url:
            error_details["url"] = url

        super().__init__(
            message,
            error_code="SCRAPING_ERROR",
            details=error_details,
        )


class RateLimitError(TracklistError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        retry_after: int | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize rate limit error."""
        error_details = details or {}
        if retry_after:
            error_details["retry_after"] = retry_after

        super().__init__(
            (f"Rate limit exceeded. Retry after {retry_after} seconds" if retry_after else "Rate limit exceeded"),
            error_code="RATE_LIMIT_EXCEEDED",
            details=error_details,
        )


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker pattern implementation."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type[Exception] = Exception,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time: float | None = None
        self.state = CircuitBreakerState.CLOSED

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Call function with circuit breaker protection.

        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            ScrapingError: If circuit is open
            Exception: If function fails
        """
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise ScrapingError(
                    "Circuit breaker is open",
                    details={
                        "state": self.state.value,
                        "failure_count": self.failure_count,
                        "recovery_timeout": self.recovery_timeout,
                    },
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    async def call_async(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Call async function with circuit breaker protection.

        Args:
            func: Async function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            ScrapingError: If circuit is open
            Exception: If function fails
        """
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise ScrapingError(
                    "Circuit breaker is open",
                    details={
                        "state": self.state.value,
                        "failure_count": self.failure_count,
                        "recovery_timeout": self.recovery_timeout,
                    },
                )

        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        return self.last_failure_time is not None and time.time() - self.last_failure_time >= self.recovery_timeout

    def _on_success(self) -> None:
        """Handle successful call."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED

    def _on_failure(self) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[..., Any]:
    """
    Decorator for retry logic with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for each retry
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                            f"Retrying in {current_delay:.1f} seconds..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed. Last error: {e}")

            if last_exception:
                raise last_exception
            return None

        return wrapper

    return decorator


def async_retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[..., Any]:
    """
    Decorator for async retry logic with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for each retry
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                            f"Retrying in {current_delay:.1f} seconds..."
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_attempts} attempts failed. Last error: {e}")

            if last_exception:
                raise last_exception
            return None

        return wrapper

    return decorator


class PartialExtractor:
    """Helper for partial data extraction on parse errors."""

    @staticmethod
    def extract_tracks_partial(soup: Any, max_errors: int = 5) -> list[dict[str, Any]]:
        """
        Extract tracks with partial failure tolerance.

        Args:
            soup: BeautifulSoup object
            max_errors: Maximum parse errors to tolerate

        Returns:
            List of successfully parsed tracks
        """
        tracks = []
        error_count = 0

        # This would be implemented with actual parsing logic
        # For now, it's a placeholder showing the pattern
        track_elements = soup.select("div.track")

        for element in track_elements:
            try:
                # Parse individual track
                track = {
                    "artist": element.select_one(".artist").text.strip(),
                    "title": element.select_one(".title").text.strip(),
                }
                tracks.append(track)
            except Exception as e:
                error_count += 1
                logger.warning(f"Failed to parse track: {e}")

                if error_count >= max_errors:
                    logger.error(f"Too many parse errors ({error_count}), stopping")
                    break

        return tracks


def with_correlation_id(correlation_id: str | None = None) -> Callable[..., Any]:
    """
    Decorator to add correlation ID to log messages.

    Args:
        correlation_id: Correlation ID for tracking

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if correlation_id:
                logger = logging.LoggerAdapter(
                    logging.getLogger(func.__module__),
                    {"correlation_id": correlation_id},
                )
                # Store logger in function for use
                func.__dict__["logger"] = logger

            return func(*args, **kwargs)

        return wrapper

    return decorator


class HealthCheck:
    """Health check for service components."""

    def __init__(self) -> None:
        """Initialize health check."""
        self.checks: dict[str, Callable[[], bool]] = {}

    def register_check(self, name: str, check_func: Callable[[], bool]) -> None:
        """
        Register a health check.

        Args:
            name: Check name
            check_func: Function that returns True if healthy
        """
        self.checks[name] = check_func

    async def run_checks(self) -> dict[str, Any]:
        """
        Run all health checks.

        Returns:
            Dictionary with check results
        """
        results = {}
        overall_status = "healthy"

        for name, check_func in self.checks.items():
            try:
                # Handle both sync and async functions
                if asyncio.iscoroutinefunction(check_func):
                    is_healthy = await check_func()
                else:
                    is_healthy = check_func()

                results[name] = "healthy" if is_healthy else "unhealthy"

                if not is_healthy:
                    overall_status = "degraded"

            except Exception as e:
                results[name] = f"error: {e!s}"
                overall_status = "unhealthy"

        return {
            "status": overall_status,
            "checks": results,
            "timestamp": time.time(),
        }
