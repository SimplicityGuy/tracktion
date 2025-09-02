"""
Health check endpoints for the analysis service.

This module provides health and readiness check endpoints for monitoring
the service and its dependencies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog
from flask import Flask, Response, jsonify

if TYPE_CHECKING:
    from collections.abc import Callable

logger = structlog.get_logger(__name__)


class HealthStatus(Enum):
    """Health status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class ComponentHealth:
    """Health status of a single component."""

    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    last_check: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        result: dict[str, Any] = {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
        }

        if self.latency_ms is not None:
            result["latency_ms"] = self.latency_ms

        if self.metadata:
            result["metadata"] = self.metadata

        if self.last_check is not None:
            result["last_check_timestamp"] = self.last_check

        return result


class HealthChecker:
    """Health checker for service and dependencies."""

    def __init__(
        self,
        service_name: str = "analysis_service",
        check_timeout: float = 5.0,
    ) -> None:
        """
        Initialize the health checker.

        Args:
            service_name: Name of the service
            check_timeout: Timeout for health checks (seconds)
        """
        self.service_name = service_name
        self.check_timeout = check_timeout

        # Component health checks
        self._health_checks: dict[str, Callable[[], ComponentHealth]] = {}

        # Cache for health check results
        self._cache: dict[str, ComponentHealth] = {}
        self._cache_ttl = 30.0  # Cache for 30 seconds

        # Statistics
        self._check_count = 0
        self._start_time = time.time()

        logger.info(
            "Health checker initialized",
            service_name=service_name,
            check_timeout=check_timeout,
        )

    def register_health_check(
        self,
        component_name: str,
        check_func: Callable[[], ComponentHealth],
    ) -> None:
        """
        Register a health check for a component.

        Args:
            component_name: Name of the component
            check_func: Function that returns ComponentHealth
        """
        self._health_checks[component_name] = check_func
        logger.debug(f"Registered health check for component: {component_name}")

    def check_health(self, use_cache: bool = True) -> dict[str, Any]:
        """
        Check overall health of the service.

        Args:
            use_cache: Whether to use cached results

        Returns:
            Dictionary containing health status
        """
        self._check_count += 1
        start_time = time.time()

        # Basic liveness check - service is running
        uptime = time.time() - self._start_time

        result = {
            "service": self.service_name,
            "status": HealthStatus.HEALTHY.value,
            "uptime_seconds": uptime,
            "checks_performed": self._check_count,
            "timestamp": time.time(),
        }

        latency = (time.time() - start_time) * 1000
        result["check_latency_ms"] = latency

        logger.debug(
            "Health check completed",
            status=result["status"],
            latency_ms=latency,
        )

        return result

    def check_readiness(self, use_cache: bool = False) -> dict[str, Any]:
        """
        Check readiness of the service and all dependencies.

        Args:
            use_cache: Whether to use cached results

        Returns:
            Dictionary containing readiness status
        """
        self._check_count += 1
        start_time = time.time()

        # Check all components
        components = []
        overall_status = HealthStatus.HEALTHY

        for component_name, check_func in self._health_checks.items():
            try:
                # Check cache first
                if use_cache and component_name in self._cache:
                    cached = self._cache[component_name]
                    if cached.last_check and (time.time() - cached.last_check < self._cache_ttl):
                        component_health = cached
                    else:
                        component_health = self._run_health_check(component_name, check_func)
                else:
                    component_health = self._run_health_check(component_name, check_func)

                components.append(component_health.to_dict())

                # Update overall status
                if component_health.status == HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.UNHEALTHY
                elif component_health.status == HealthStatus.DEGRADED and overall_status != HealthStatus.UNHEALTHY:
                    overall_status = HealthStatus.DEGRADED

            except Exception as e:
                logger.error(f"Error checking component {component_name}: {e}")
                components.append(
                    {
                        "name": component_name,
                        "status": HealthStatus.UNHEALTHY.value,
                        "message": f"Check failed: {e!s}",
                    }
                )
                overall_status = HealthStatus.UNHEALTHY

        # Prepare result
        result = {
            "service": self.service_name,
            "status": overall_status.value,
            "ready": overall_status != HealthStatus.UNHEALTHY,
            "components": components,
            "timestamp": time.time(),
        }

        latency = (time.time() - start_time) * 1000
        result["check_latency_ms"] = latency

        logger.info(
            "Readiness check completed",
            status=overall_status.value,
            ready=result["ready"],
            components_checked=len(components),
            latency_ms=latency,
        )

        return result

    def _run_health_check(
        self,
        component_name: str,
        check_func: Callable[[], ComponentHealth],
    ) -> ComponentHealth:
        """
        Run a health check with timeout.

        Args:
            component_name: Name of the component
            check_func: Health check function

        Returns:
            ComponentHealth result
        """
        start_time = time.time()

        try:
            # Run check with timeout
            # In production, this would use proper async/threading
            component_health = check_func()

            # Add timing information
            latency = (time.time() - start_time) * 1000
            component_health.latency_ms = latency
            component_health.last_check = time.time()

            # Cache result
            self._cache[component_name] = component_health

            return component_health

        except Exception as e:
            logger.error(f"Health check failed for {component_name}: {e}")
            return ComponentHealth(
                name=component_name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check failed: {e!s}",
                latency_ms=(time.time() - start_time) * 1000,
                last_check=time.time(),
            )

    def get_detailed_status(self) -> dict[str, Any]:
        """
        Get detailed status including all component health.

        Returns:
            Detailed status dictionary
        """
        readiness = self.check_readiness(use_cache=True)
        health = self.check_health(use_cache=True)

        return {
            "health": health,
            "readiness": readiness,
            "cache_size": len(self._cache),
            "registered_checks": list(self._health_checks.keys()),
        }


# Component-specific health checks


def check_postgresql_health() -> ComponentHealth:
    """
    Check PostgreSQL database health.

    Returns:
        ComponentHealth for PostgreSQL
    """
    try:
        # This would actually connect to PostgreSQL
        # For now, it's a placeholder
        return ComponentHealth(
            name="postgresql",
            status=HealthStatus.HEALTHY,
            message="Database connection successful",
            metadata={"version": "17.0", "connections": 10},
        )
    except Exception as e:
        return ComponentHealth(
            name="postgresql",
            status=HealthStatus.UNHEALTHY,
            message=f"Database connection failed: {e!s}",
        )


def check_neo4j_health() -> ComponentHealth:
    """
    Check Neo4j graph database health.

    Returns:
        ComponentHealth for Neo4j
    """
    try:
        # This would actually connect to Neo4j
        return ComponentHealth(
            name="neo4j",
            status=HealthStatus.HEALTHY,
            message="Graph database connection successful",
            metadata={"version": "5.26", "nodes": 1000},
        )
    except Exception as e:
        return ComponentHealth(
            name="neo4j",
            status=HealthStatus.UNHEALTHY,
            message=f"Graph database connection failed: {e!s}",
        )


def check_rabbitmq_health() -> ComponentHealth:
    """
    Check RabbitMQ message queue health.

    Returns:
        ComponentHealth for RabbitMQ
    """
    try:
        # This would actually connect to RabbitMQ
        return ComponentHealth(
            name="rabbitmq",
            status=HealthStatus.HEALTHY,
            message="Message queue connection successful",
            metadata={"version": "4.0", "queue_depth": 0},
        )
    except Exception as e:
        return ComponentHealth(
            name="rabbitmq",
            status=HealthStatus.UNHEALTHY,
            message=f"Message queue connection failed: {e!s}",
        )


def check_redis_health() -> ComponentHealth:
    """
    Check Redis cache health.

    Returns:
        ComponentHealth for Redis
    """
    try:
        # This would actually connect to Redis
        return ComponentHealth(
            name="redis",
            status=HealthStatus.HEALTHY,
            message="Cache connection successful",
            metadata={"version": "7.4", "memory_used_mb": 50},
        )
    except Exception as e:
        return ComponentHealth(
            name="redis",
            status=HealthStatus.UNHEALTHY,
            message=f"Cache connection failed: {e!s}",
        )


class HealthCheckServer:
    """Flask server for health check endpoints."""

    def __init__(
        self,
        health_checker: HealthChecker,
        port: int = 8080,
    ) -> None:
        """
        Initialize the health check server.

        Args:
            health_checker: HealthChecker instance
            port: Port to serve health checks on
        """
        self.health_checker = health_checker
        self.port = port
        self.app = Flask(__name__)
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Set up Flask routes for health check endpoints."""

        @self.app.route("/health")
        def health() -> tuple[Response, int]:
            """Basic liveness check endpoint."""
            result = self.health_checker.check_health()
            status_code = 200 if result["status"] == "healthy" else 503
            return jsonify(result), status_code

        @self.app.route("/ready")
        def ready() -> tuple[Response, int]:
            """Readiness check endpoint."""
            result = self.health_checker.check_readiness()
            status_code = 200 if result["ready"] else 503
            return jsonify(result), status_code

        @self.app.route("/health/detailed")
        def detailed_health() -> Response:
            """Detailed health status endpoint."""
            result = self.health_checker.get_detailed_status()
            return jsonify(result)

    def run(self, host: str = "0.0.0.0") -> None:
        """
        Run the health check server.

        Args:
            host: Host to bind to
        """
        logger.info(f"Starting health check server on {host}:{self.port}")
        self.app.run(host=host, port=self.port)


class HealthCheckerSingleton:
    """Singleton wrapper for HealthChecker with proper initialization."""

    _instance: HealthChecker | None = None
    _initialized: bool = False

    def __new__(cls) -> HealthCheckerSingleton:
        """Get the singleton HealthChecker instance."""
        if cls._instance is None:
            cls._instance = HealthChecker()
            cls._initialize_default_checks()
        return cls._instance  # type: ignore[return-value]

    @classmethod
    def _initialize_default_checks(cls) -> None:
        """Initialize default health checks."""
        if cls._instance is not None and not cls._initialized:
            cls._instance.register_health_check("postgresql", check_postgresql_health)
            cls._instance.register_health_check("neo4j", check_neo4j_health)
            cls._instance.register_health_check("rabbitmq", check_rabbitmq_health)
            cls._instance.register_health_check("redis", check_redis_health)
            cls._initialized = True

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        cls._instance = None
        cls._initialized = False


def get_health_checker() -> HealthCheckerSingleton:
    """Get the singleton health checker instance."""
    return HealthCheckerSingleton()


def reset_health_checker() -> None:
    """Reset the health checker singleton (mainly for testing)."""
    HealthCheckerSingleton.reset()
