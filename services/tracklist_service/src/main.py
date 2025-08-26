"""
Main entry point for the tracklist service.

Provides both FastAPI web service and message queue consumer functionality.
"""

import asyncio
import logging
import signal
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable

import structlog
import uvicorn
import redis.asyncio as redis
from fastapi import FastAPI, Request, Response

from .api.search import router as search_router
from .api.developer_endpoints import router as developer_router
from .config import get_config
from .messaging.message_handler import TracklistMessageHandler
from .auth.authentication import AuthenticationManager
from .auth.dependencies import set_auth_manager
from .rate_limiting.limiter import RateLimiter
from .middleware.rate_limit_middleware import RateLimitMiddleware


# Configure structured logging
def setup_logging() -> None:
    """Configure structured logging for the service."""
    config = get_config()

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, config.log_level.upper()),
    )


# Global handlers for cleanup
message_handler: TracklistMessageHandler | None = None
redis_client: redis.Redis | None = None
rate_limiter: RateLimiter | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan events."""
    global message_handler, redis_client, rate_limiter

    logger = structlog.get_logger(__name__)
    config = get_config()

    # Startup
    logger.info("Starting tracklist service", version="0.1.0", debug_mode=config.debug_mode)

    # Initialize Redis client for rate limiting
    redis_client = redis.Redis(
        host=config.cache.redis_host,
        port=config.cache.redis_port,
        db=config.cache.redis_db,
        password=config.cache.redis_password,
        decode_responses=True,
    )

    # Initialize rate limiter
    rate_limiter = RateLimiter(redis_client)
    logger.info("Rate limiter initialized")

    # Initialize authentication manager
    auth_manager = AuthenticationManager(jwt_secret="temp-secret-key")
    set_auth_manager(auth_manager)
    logger.info("Authentication manager initialized")

    # Initialize message handler
    message_handler = TracklistMessageHandler()

    # Start message consumption in background
    consume_task = asyncio.create_task(message_handler.start_consuming())
    logger.info("Message handler started")

    try:
        yield
    finally:
        # Shutdown
        logger.info("Shutting down tracklist service")

        # Stop message handler
        if message_handler:
            await message_handler.stop()

        # Cancel consume task
        if not consume_task.done():
            consume_task.cancel()
            try:
                await consume_task
            except asyncio.CancelledError:
                pass

        # Close Redis connection
        if redis_client:
            await redis_client.close()

        logger.info("Tracklist service shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    config = get_config()

    app = FastAPI(
        title="Tracklist Service",
        description="1001tracklists.com API integration and search service",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if config.api.docs_enabled else None,
        redoc_url="/redoc" if config.api.docs_enabled else None,
    )

    # Add rate limiting middleware
    # Note: This will be initialized during lifespan startup
    @app.middleware("http")
    async def add_rate_limiting(request: Request, call_next: Callable) -> Response:
        """Add rate limiting to all requests."""
        if rate_limiter is not None:
            middleware = RateLimitMiddleware(app, rate_limiter)
            return await middleware.dispatch(request, call_next)
        # If rate limiter not initialized yet, pass through
        return await call_next(request)

    # Include API routes
    app.include_router(search_router, prefix=config.api.api_prefix)
    app.include_router(developer_router, prefix=config.api.api_prefix)

    # Health check endpoint
    @app.get("/health")
    async def health_check() -> dict:
        """Health check endpoint."""
        return {"status": "healthy", "service": "tracklist_service", "version": "0.1.0"}

    return app


def handle_shutdown(signum: int, frame: Any) -> None:
    """Handle shutdown signals gracefully."""
    logger = structlog.get_logger(__name__)
    logger.info("Received shutdown signal", signal=signum)

    # The lifespan manager will handle cleanup
    exit(0)


async def main() -> None:
    """Main async entry point."""
    setup_logging()
    config = get_config()

    # Validate configuration
    errors = config.validate()
    if errors:
        logger = structlog.get_logger(__name__)
        logger.error("Configuration validation failed", errors=errors)
        exit(1)

    # Set up signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Create and run the app
    app = create_app()

    uvicorn_config = uvicorn.Config(
        app,
        host=config.api.host,
        port=config.api.port,
        workers=config.api.workers,
        reload=config.api.reload,
        log_level=config.api.log_level,
    )

    server = uvicorn.Server(uvicorn_config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())
