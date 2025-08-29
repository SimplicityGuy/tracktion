"""FastAPI application for Analysis Service."""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..structured_logging import get_logger
from .endpoints import (
    analysis_router,
    health_router,
    metadata_router,
    recordings_router,
    streaming_router,
    tracklist_router,
    websocket_router,
)
from .errors import register_error_handlers
from .middleware import ErrorHandlingMiddleware, RequestIDMiddleware, TimingMiddleware
from .timeout import RequestCancellationMiddleware, TimeoutMiddleware

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Manage application lifespan with async startup and shutdown."""
    # Startup
    logger.info("Starting Analysis Service API", extra={"service": "analysis_service"})

    # Initialize async resources here
    # e.g., database connections, message queues, etc.

    yield

    # Shutdown
    logger.info("Shutting down Analysis Service API", extra={"service": "analysis_service"})

    # Clean up async resources here
    # e.g., close database connections, disconnect from message queues, etc.


# Create FastAPI app with async lifespan management
app = FastAPI(
    title="Analysis Service API",
    description="Async API for music analysis and processing",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/v1/docs",
    redoc_url="/v1/redoc",
    openapi_url="/v1/openapi.json",
)

# Configure middleware pipeline (order matters - reverse execution)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(TimeoutMiddleware, default_timeout=30.0)
app.add_middleware(RequestCancellationMiddleware)
app.add_middleware(TimingMiddleware)
app.add_middleware(RequestIDMiddleware)

# Configure CORS for async operations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Rate-Limit-Remaining", "X-Process-Time"],
)

# Register API routers
app.include_router(health_router)
app.include_router(recordings_router)
app.include_router(metadata_router)
app.include_router(tracklist_router)
app.include_router(analysis_router)
app.include_router(websocket_router)
app.include_router(streaming_router)

# Register error handlers
register_error_handlers(app)
