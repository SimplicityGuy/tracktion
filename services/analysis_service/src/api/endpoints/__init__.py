"""API endpoints for Analysis Service."""

from .analysis import router as analysis_router
from .health import router as health_router
from .metadata import router as metadata_router
from .recordings import router as recordings_router
from .streaming import router as streaming_router
from .tracklist import router as tracklist_router
from .websocket import router as websocket_router

__all__ = [
    "analysis_router",
    "health_router",
    "metadata_router",
    "recordings_router",
    "streaming_router",
    "tracklist_router",
    "websocket_router",
]
