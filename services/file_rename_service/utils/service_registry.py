"""Service registry integration for File Rename Service."""

import logging
from typing import Any

import httpx

from services.file_rename_service.app.config import settings

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """Service registry for registering and discovering services."""

    def __init__(self) -> None:
        """Initialize service registry."""
        self.consul_url = settings.consul_url
        self.service_name = settings.service_name
        self.service_id = settings.service_id
        self.service_tags = settings.service_tags
        self.is_registered = False

    async def register(self) -> bool:
        """Register service with Consul."""
        if not self.consul_url:
            logger.info("Consul URL not configured, skipping service registration")
            return True

        try:
            async with httpx.AsyncClient() as client:
                # Prepare service registration data
                service_data = {
                    "ID": self.service_id,
                    "Name": self.service_name,
                    "Tags": self.service_tags,
                    "Address": settings.host,
                    "Port": settings.port,
                    "Check": {
                        "HTTP": f"http://{settings.host}:{settings.port}/health",
                        "Interval": "30s",
                        "Timeout": "10s",
                    },
                }

                # Register with Consul
                response = await client.put(
                    f"{self.consul_url}/v1/agent/service/register",
                    json=service_data,
                )

                if response.status_code == 200:
                    self.is_registered = True
                    logger.info(f"Successfully registered service '{self.service_name}' with Consul")
                    return True
                else:
                    logger.error(f"Failed to register service: {response.status_code} - {response.text}")
                    return False

        except Exception as e:
            logger.error(f"Error registering service with Consul: {e}")
            return False

    async def deregister(self) -> bool:
        """Deregister service from Consul."""
        if not self.consul_url or not self.is_registered:
            return True

        try:
            async with httpx.AsyncClient() as client:
                response = await client.put(f"{self.consul_url}/v1/agent/service/deregister/{self.service_id}")

                if response.status_code == 200:
                    self.is_registered = False
                    logger.info(f"Successfully deregistered service '{self.service_name}' from Consul")
                    return True
                else:
                    logger.error(f"Failed to deregister service: {response.status_code} - {response.text}")
                    return False

        except Exception as e:
            logger.error(f"Error deregistering service from Consul: {e}")
            return False

    async def update_health_check(self, status: str) -> bool:
        """Update service health check status."""
        if not self.consul_url or not self.is_registered:
            return True

        try:
            async with httpx.AsyncClient() as client:
                # Status can be: "passing", "warning", "critical"
                response = await client.put(
                    f"{self.consul_url}/v1/agent/check/update/service:{self.service_id}",
                    json={"Status": status},
                )

                if response.status_code == 200:
                    logger.debug(f"Updated health check status to '{status}'")
                    return True
                else:
                    logger.error(f"Failed to update health check: {response.status_code}")
                    return False

        except Exception as e:
            logger.error(f"Error updating health check: {e}")
            return False

    async def discover_service(self, service_name: str) -> dict[str, Any] | None:
        """Discover a service from Consul."""
        if not self.consul_url:
            return None

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.consul_url}/v1/health/service/{service_name}?passing=true")

                if response.status_code == 200:
                    services = response.json()
                    if services:
                        # Return the first healthy service instance
                        service = services[0]
                        return {
                            "address": service["Service"]["Address"],
                            "port": service["Service"]["Port"],
                            "tags": service["Service"]["Tags"],
                        }
                    else:
                        logger.warning(f"No healthy instances found for service '{service_name}'")
                        return None
                else:
                    logger.error(f"Failed to discover service: {response.status_code}")
                    return None

        except Exception as e:
            logger.error(f"Error discovering service: {e}")
            return None


# Global service registry instance
service_registry = ServiceRegistry()


# API Gateway configuration (for documentation)
API_GATEWAY_CONFIG = {
    "service": "file-rename-service",
    "version": "v1",
    "endpoints": [
        {
            "path": "/rename/analyze",
            "method": "POST",
            "description": "Analyze filename patterns",
            "rate_limit": "100/minute",
        },
        {
            "path": "/rename/propose",
            "method": "POST",
            "description": "Generate rename proposals",
            "rate_limit": "100/minute",
        },
        {
            "path": "/rename/feedback",
            "method": "POST",
            "description": "Submit user feedback",
            "rate_limit": "100/minute",
        },
        {
            "path": "/rename/patterns",
            "method": "GET",
            "description": "Retrieve learned patterns",
            "rate_limit": "1000/minute",
        },
        {
            "path": "/rename/history",
            "method": "GET",
            "description": "Get rename history",
            "rate_limit": "1000/minute",
        },
    ],
}
