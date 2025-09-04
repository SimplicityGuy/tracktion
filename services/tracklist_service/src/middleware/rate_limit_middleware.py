"""Rate limiting middleware for FastAPI applications."""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from services.tracklist_service.src.auth.dependencies import authenticate_from_request
from services.tracklist_service.src.rate_limiting.limiter import RateLimiter

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce rate limits and add rate limit headers to responses."""

    def __init__(self, app: Any, rate_limiter: RateLimiter) -> None:
        """Initialize rate limit middleware.

        Args:
            app: FastAPI application instance
            rate_limiter: Rate limiter instance
        """
        super().__init__(app)
        self.limiter = rate_limiter

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response | JSONResponse:
        """Process request through rate limiting middleware.

        Args:
            request: FastAPI request object
            call_next: Next middleware/endpoint to call

        Returns:
            Response with rate limit headers
        """
        # Skip rate limiting for health check and documentation endpoints
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        try:
            # Authenticate the request to get user information
            user = await authenticate_from_request(request)
        except Exception as e:
            # If authentication fails, still need to apply rate limiting for anonymous users
            # For now, we'll let the authentication error propagate
            # In production, you might want to apply anonymous rate limits
            logger.warning(f"Authentication failed for rate limiting: {e}")
            return await call_next(request)

        # Check rate limits
        rate_limit_result = await self.limiter.check_rate_limit(user)

        # If rate limited, return 429 response
        if not rate_limit_result.allowed:
            logger.info(
                f"Rate limit exceeded for user {user.id}",
                extra={
                    "user_id": user.id,
                    "tier": user.tier.value,
                    "remaining": rate_limit_result.remaining,
                    "retry_after": rate_limit_result.retry_after,
                },
            )

            # Build rate limit headers for 429 response
            headers = {
                "Retry-After": str(rate_limit_result.retry_after),
                "X-RateLimit-Limit": str(rate_limit_result.limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(rate_limit_result.reset_time),
                "X-RateLimit-Tier": user.tier.value,
            }

            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": f"Request rate limit exceeded. Try again in {rate_limit_result.retry_after} seconds.",
                    "retry_after": rate_limit_result.retry_after,
                },
                headers=headers,
            )

        # Process the request
        response = await call_next(request)

        # Add rate limit headers to successful response
        if rate_limit_result.headers:
            for header_name, header_value in rate_limit_result.headers.items():
                response.headers[header_name] = header_value

        # Log successful request for monitoring
        logger.debug(
            f"Request processed for user {user.id}",
            extra={
                "user_id": user.id,
                "tier": user.tier.value,
                "remaining": rate_limit_result.remaining,
                "path": request.url.path,
                "method": request.method,
            },
        )

        return response
