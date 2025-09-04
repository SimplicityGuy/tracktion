"""
Rate limiting and backpressure handling for API endpoints.

This module provides async rate limiting, backpressure management,
and request queuing for the API.
"""

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitStrategy(Enum):
    """Rate limiting strategies."""

    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    # Rate limit settings
    requests_per_second: int = 10
    requests_per_minute: int = 100
    requests_per_hour: int = 1000
    burst_size: int = 20

    # Strategy
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET

    # Backpressure settings
    max_queue_size: int = 100
    queue_timeout_seconds: float = 30.0
    enable_backpressure: bool = True

    # Connection limits
    max_concurrent_connections: int = 1000
    max_connections_per_ip: int = 10

    # Response headers
    include_headers: bool = True
    header_prefix: str = "X-RateLimit"


class TokenBucket:
    """Token bucket algorithm for rate limiting."""

    def __init__(self, rate: float, capacity: int):
        """
        Initialize token bucket.

        Args:
            rate: Tokens per second
            capacity: Maximum bucket capacity
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens: float = float(capacity)
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False otherwise
        """
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_update

            # Add new tokens based on elapsed time
            self.tokens = min(self.capacity, self.tokens + (elapsed * self.rate))
            self.last_update = now

            # Try to consume tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    async def wait_for_tokens(self, tokens: int = 1) -> float:
        """
        Calculate wait time for tokens to become available.

        Args:
            tokens: Number of tokens needed

        Returns:
            Wait time in seconds
        """
        async with self.lock:
            if self.tokens >= tokens:
                return 0.0

            needed = tokens - self.tokens
            return needed / self.rate


class SlidingWindowCounter:
    """Sliding window counter for rate limiting."""

    def __init__(self, window_size: float, max_requests: int):
        """
        Initialize sliding window counter.

        Args:
            window_size: Window size in seconds
            max_requests: Maximum requests in window
        """
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests: deque = deque()
        self.lock = asyncio.Lock()

    async def is_allowed(self) -> bool:
        """
        Check if request is allowed.

        Returns:
            True if request is allowed
        """
        async with self.lock:
            now = time.time()
            cutoff = now - self.window_size

            # Remove old requests outside window
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()

            # Check if under limit
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            return False

    async def get_reset_time(self) -> float:
        """
        Get time until oldest request expires.

        Returns:
            Reset time in seconds
        """
        async with self.lock:
            if not self.requests:
                return 0.0

            oldest = self.requests[0]
            reset_time = (oldest + self.window_size) - time.time()
            return max(0.0, reset_time)  # type: ignore[no-any-return]  # max() with float args returns float


class RequestQueue:
    """Queue for handling excess requests with backpressure."""

    def __init__(self, max_size: int, timeout: float):
        """
        Initialize request queue.

        Args:
            max_size: Maximum queue size
            timeout: Queue timeout in seconds
        """
        self.max_size = max_size
        self.timeout = timeout
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.processing = False

    async def enqueue(self, request_id: str) -> bool:
        """
        Add request to queue.

        Args:
            request_id: Unique request identifier

        Returns:
            True if queued successfully
        """
        try:
            await asyncio.wait_for(self.queue.put(request_id), timeout=1.0)
            return True
        except (TimeoutError, asyncio.QueueFull):
            return False

    async def dequeue(self) -> str | None:
        """
        Get next request from queue.

        Returns:
            Request ID or None if empty
        """
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=0.1)
        except (TimeoutError, asyncio.QueueEmpty):
            return None

    def is_full(self) -> bool:
        """Check if queue is full."""
        return self.queue.full()

    def size(self) -> int:
        """Get current queue size."""
        return self.queue.qsize()


class AsyncRateLimiter:
    """
    Async rate limiter with multiple strategies and backpressure.
    """

    def __init__(self, config: RateLimitConfig | None = None):
        """
        Initialize rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()

        # Rate limiters by client ID
        self.limiters: dict[str, Any] = {}

        # Connection tracking
        self.connections: dict[str, int] = defaultdict(int)
        self.total_connections = 0

        # Request queue for backpressure
        self.request_queue = RequestQueue(
            max_size=self.config.max_queue_size,
            timeout=self.config.queue_timeout_seconds,
        )

        # Statistics
        self.stats = {
            "total_requests": 0,
            "allowed_requests": 0,
            "rate_limited_requests": 0,
            "queued_requests": 0,
            "rejected_requests": 0,
        }

        self.lock = asyncio.Lock()

    def _get_client_id(self, request: Request) -> str:
        """
        Get client identifier from request.

        Args:
            request: FastAPI request

        Returns:
            Client identifier
        """
        # Try to get from headers first (for API keys)
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api:{api_key}"

        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"

    def _create_limiter(self) -> Any:
        """
        Create rate limiter based on strategy.

        Returns:
            Rate limiter instance
        """
        if self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return TokenBucket(rate=self.config.requests_per_second, capacity=self.config.burst_size)
        if self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return SlidingWindowCounter(
                window_size=60.0,  # 1 minute window
                max_requests=self.config.requests_per_minute,
            )
        # Default to token bucket
        return TokenBucket(rate=self.config.requests_per_second, capacity=self.config.burst_size)

    async def check_rate_limit(self, request: Request) -> tuple[bool, float | None]:
        """
        Check if request is within rate limits.

        Args:
            request: FastAPI request

        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        client_id = self._get_client_id(request)

        async with self.lock:
            # Get or create limiter for client
            if client_id not in self.limiters:
                self.limiters[client_id] = self._create_limiter()

            limiter = self.limiters[client_id]
            self.stats["total_requests"] += 1

        # Check rate limit based on strategy
        if isinstance(limiter, TokenBucket):
            allowed = await limiter.consume()
            if not allowed:
                retry_after = await limiter.wait_for_tokens()
                self.stats["rate_limited_requests"] += 1
                return False, retry_after
        elif isinstance(limiter, SlidingWindowCounter):
            allowed = await limiter.is_allowed()
            if not allowed:
                retry_after = await limiter.get_reset_time()
                self.stats["rate_limited_requests"] += 1
                return False, retry_after

        self.stats["allowed_requests"] += 1
        return True, None

    async def check_connection_limit(self, request: Request) -> bool:
        """
        Check if connection limits are exceeded.

        Args:
            request: FastAPI request

        Returns:
            True if connection is allowed
        """
        client_ip = request.client.host if request.client else "unknown"

        async with self.lock:
            # Check total connections
            if self.total_connections >= self.config.max_concurrent_connections:
                return False

            # Check per-IP connections
            if self.connections[client_ip] >= self.config.max_connections_per_ip:
                return False

            # Track connection
            self.connections[client_ip] += 1
            self.total_connections += 1
            return True

    async def release_connection(self, request: Request) -> None:
        """
        Release connection tracking.

        Args:
            request: FastAPI request
        """
        client_ip = request.client.host if request.client else "unknown"

        async with self.lock:
            if client_ip in self.connections:
                self.connections[client_ip] = max(0, self.connections[client_ip] - 1)
                if self.connections[client_ip] == 0:
                    del self.connections[client_ip]
            self.total_connections = max(0, self.total_connections - 1)

    async def handle_backpressure(self, request_id: str) -> bool:
        """
        Handle request with backpressure.

        Args:
            request_id: Unique request identifier

        Returns:
            True if request should proceed
        """
        if not self.config.enable_backpressure:
            return True

        # Try to queue request if rate limited
        if self.request_queue.is_full():
            self.stats["rejected_requests"] += 1
            return False

        queued = await self.request_queue.enqueue(request_id)
        if queued:
            self.stats["queued_requests"] += 1
        else:
            self.stats["rejected_requests"] += 1

        return queued

    def get_rate_limit_headers(self, allowed: bool, retry_after: float | None = None) -> dict[str, str]:
        """
        Get rate limit headers for response.

        Args:
            allowed: Whether request was allowed
            retry_after: Retry after seconds if rate limited

        Returns:
            Dictionary of headers
        """
        if not self.config.include_headers:
            return {}

        headers = {
            f"{self.config.header_prefix}-Limit": str(self.config.requests_per_minute),
            f"{self.config.header_prefix}-Remaining": str(
                max(
                    0,
                    self.config.requests_per_minute - self.stats["total_requests"] % self.config.requests_per_minute,
                )
            ),
            f"{self.config.header_prefix}-Reset": str(int(time.time() + 60)),
        }

        if not allowed and retry_after:
            headers["Retry-After"] = str(int(retry_after))

        return headers

    def get_stats(self) -> dict[str, Any]:
        """
        Get rate limiter statistics.

        Returns:
            Statistics dictionary
        """
        return {
            **self.stats,
            "active_connections": self.total_connections,
            "queue_size": self.request_queue.size(),
            "unique_clients": len(self.limiters),
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for applying rate limiting to all requests.
    """

    def __init__(self, app: Any, config: RateLimitConfig | None = None):
        """
        Initialize middleware.

        Args:
            app: FastAPI application
            config: Rate limit configuration
        """
        super().__init__(app)
        self.limiter = AsyncRateLimiter(config)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with rate limiting.

        Args:
            request: Incoming request
            call_next: Next middleware/handler

        Returns:
            Response with rate limit headers
        """
        # Check connection limit
        if not await self.limiter.check_connection_limit(request):
            return JSONResponse(
                status_code=503,
                content={"error": "Connection limit exceeded"},
                headers={"Retry-After": "60"},
            )

        try:
            # Check rate limit
            allowed, retry_after = await self.limiter.check_rate_limit(request)

            if not allowed:
                # Try backpressure queue
                request_id = request.headers.get("X-Request-ID", str(time.time()))
                if self.limiter.config.enable_backpressure:
                    queued = await self.limiter.handle_backpressure(request_id)
                    if not queued:
                        headers = self.limiter.get_rate_limit_headers(False, retry_after)
                        return JSONResponse(
                            status_code=429,
                            content={
                                "error": "Rate limit exceeded",
                                "retry_after": retry_after,
                            },
                            headers=headers,
                        )

                    # Wait in queue
                    await asyncio.sleep(retry_after or 1.0)
                else:
                    # No backpressure, return rate limit error
                    headers = self.limiter.get_rate_limit_headers(False, retry_after)
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": "Rate limit exceeded",
                            "retry_after": retry_after,
                        },
                        headers=headers,
                    )

            # Process request
            response = await call_next(request)

            # Add rate limit headers
            headers = self.limiter.get_rate_limit_headers(True)
            for key, value in headers.items():
                response.headers[key] = value

            return cast("Response", response)

        finally:
            # Release connection
            await self.limiter.release_connection(request)


# Decorator for per-endpoint rate limiting
def rate_limit(
    requests_per_minute: int = 60,
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET,
) -> Callable[[Callable], Callable]:
    """
    Decorator for applying rate limiting to specific endpoints.

    Args:
        requests_per_minute: Maximum requests per minute
        strategy: Rate limiting strategy
    """

    def decorator(func: Callable) -> Callable:
        # Create endpoint-specific limiter
        config = RateLimitConfig(requests_per_minute=requests_per_minute, strategy=strategy)
        limiter = AsyncRateLimiter(config)

        async def wrapper(request: Request, *args: Any, **kwargs: Any) -> Any:
            # Check rate limit
            allowed, retry_after = await limiter.check_rate_limit(request)

            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Retry after {retry_after:.1f} seconds",
                    headers={"Retry-After": str(int(retry_after or 60))},
                )

            # Call original function
            return await func(request, *args, **kwargs)

        # Preserve function metadata
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator
