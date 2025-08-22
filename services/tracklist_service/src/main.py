"""
Main entry point for the tracklist service.

Provides both FastAPI web service and message queue consumer functionality.
"""

import asyncio
import logging
import signal
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import structlog
import uvicorn
from fastapi import FastAPI

from .api.search_api import router as search_router
from .config import get_config
from .messaging.message_handler import TracklistMessageHandler


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


# Global message handler for cleanup
message_handler: TracklistMessageHandler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifespan events."""
    global message_handler

    logger = structlog.get_logger(__name__)
    config = get_config()

    # Startup
    logger.info("Starting tracklist service", version="0.1.0", debug_mode=config.debug_mode)

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

    # Include API routes
    app.include_router(search_router, prefix=config.api.api_prefix)

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
