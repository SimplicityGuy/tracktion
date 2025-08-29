"""Health check endpoints for Analysis Service."""

from typing import Any, Dict

from fastapi import APIRouter, status

from ...structured_logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/health", tags=["health"])


@router.get("", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, str]:
    """Basic health check endpoint.

    Returns:
        Health status response
    """
    return {"status": "healthy", "service": "analysis_service"}


@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness_check() -> Dict[str, Any]:
    """Readiness check endpoint for kubernetes.

    Checks if the service is ready to accept requests.

    Returns:
        Readiness status with component checks
    """
    # In a real implementation, check database, message queue, etc.
    checks = {"database": "ready", "message_queue": "ready", "cache": "ready"}

    all_ready = all(status == "ready" for status in checks.values())

    return {"ready": all_ready, "checks": checks, "service": "analysis_service"}


@router.get("/live", status_code=status.HTTP_200_OK)
async def liveness_check() -> Dict[str, str]:
    """Liveness check endpoint for kubernetes.

    Simple check to verify the service is alive.

    Returns:
        Liveness status
    """
    return {"status": "alive", "service": "analysis_service"}
