"""FastAPI application setup."""

from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_config

from .middleware import ErrorHandlingMiddleware, HealthCheckMiddleware, LoggingMiddleware
from .recordings import router as recordings_router
from .schemas import HealthResponse


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    get_config()

    app = FastAPI(
        title="Cataloging Service API",
        description="API for managing music file catalog",
        version="0.1.0",
    )

    # Add middleware (order matters - first added is outermost)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(HealthCheckMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # TODO: Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(recordings_router)

    # Health check endpoint
    @app.get("/health", response_model=HealthResponse)
    async def health_check() -> HealthResponse:
        """Health check endpoint."""
        return HealthResponse(
            status="healthy",
            service="cataloging-service",
            version="0.1.0",
            timestamp=datetime.now(UTC),
        )

    # Metrics endpoint (basic for now)
    @app.get("/metrics")
    async def metrics() -> dict:
        """Basic metrics endpoint."""
        return {
            "service": "cataloging-service",
            "timestamp": datetime.now(UTC).isoformat(),
            "status": "operational",
        }

    return app
