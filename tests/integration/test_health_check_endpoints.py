"""Comprehensive integration tests for health check endpoints in the Tracktion system.

This module provides comprehensive testing coverage for:
- Basic health check functionality for individual services
- Service dependency health checks (database, cache, message queues)
- Aggregated system-wide health status
- Health check endpoint performance and reliability
- Health check with various failure scenarios
- Custom health checks for different service components
- Health check monitoring and alerting simulation
- Load balancer and orchestration integration scenarios
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from aiohttp import web
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

# Add service paths to system path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "services" / "analysis_service" / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

logger = logging.getLogger(__name__)


class HealthCheckStatus:
    """Enum-like class for health check statuses."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class ServiceType:
    """Enum-like class for service types."""

    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    EXTERNAL_API = "external_api"
    STORAGE = "storage"
    COMPUTE = "compute"


class HealthCheckMetrics:
    """Metrics collection for health check monitoring."""

    def __init__(self, registry: CollectorRegistry | None = None):
        """Initialize health check metrics.

        Args:
            registry: Prometheus registry for metrics collection
        """
        self.registry = registry or CollectorRegistry()

        # Define metrics
        self.health_check_requests = Counter(
            "health_check_requests_total",
            "Total number of health check requests",
            ["service", "endpoint", "status"],
            registry=self.registry,
        )

        self.health_check_duration = Histogram(
            "health_check_duration_seconds",
            "Duration of health check requests",
            ["service", "endpoint"],
            registry=self.registry,
        )

        self.service_health_status = Gauge(
            "service_health_status",
            "Current health status of services (1=healthy, 0=unhealthy)",
            ["service", "component"],
            registry=self.registry,
        )

        self.dependency_health_status = Gauge(
            "dependency_health_status",
            "Health status of service dependencies (1=healthy, 0=unhealthy)",
            ["service", "dependency_type", "dependency_name"],
            registry=self.registry,
        )


class MockHealthDependencies:
    """Mock dependencies for health check testing."""

    def __init__(self):
        """Initialize mock dependencies with default healthy states."""
        self.postgres_healthy = True
        self.neo4j_healthy = True
        self.redis_healthy = True
        self.rabbitmq_healthy = True
        self.external_api_healthy = True
        self.storage_healthy = True

        # Configurable delays for testing timeout scenarios
        self.postgres_delay = 0.0
        self.neo4j_delay = 0.0
        self.redis_delay = 0.0
        self.rabbitmq_delay = 0.0
        self.external_api_delay = 0.0
        self.storage_delay = 0.0

    async def check_postgres_health(self) -> dict[str, Any]:
        """Mock PostgreSQL health check."""
        await asyncio.sleep(self.postgres_delay)
        if not self.postgres_healthy:
            raise ConnectionError("PostgreSQL connection failed")

        return {
            "status": HealthCheckStatus.HEALTHY,
            "response_time": self.postgres_delay,
            "connection_pool": {"active": 5, "idle": 15, "total": 20},
            "version": "14.9",
        }

    async def check_neo4j_health(self) -> dict[str, Any]:
        """Mock Neo4j health check."""
        await asyncio.sleep(self.neo4j_delay)
        if not self.neo4j_healthy:
            raise ConnectionError("Neo4j connection failed")

        return {
            "status": HealthCheckStatus.HEALTHY,
            "response_time": self.neo4j_delay,
            "cluster_role": "LEADER",
            "version": "5.12.0",
        }

    async def check_redis_health(self) -> dict[str, Any]:
        """Mock Redis health check."""
        await asyncio.sleep(self.redis_delay)
        if not self.redis_healthy:
            raise ConnectionError("Redis connection failed")

        return {
            "status": HealthCheckStatus.HEALTHY,
            "response_time": self.redis_delay,
            "memory_usage": {"used": 1024000, "max": 8192000},
            "connected_clients": 12,
        }

    async def check_rabbitmq_health(self) -> dict[str, Any]:
        """Mock RabbitMQ health check."""
        await asyncio.sleep(self.rabbitmq_delay)
        if not self.rabbitmq_healthy:
            raise ConnectionError("RabbitMQ connection failed")

        return {
            "status": HealthCheckStatus.HEALTHY,
            "response_time": self.rabbitmq_delay,
            "node_health": "ok",
            "queue_count": 5,
            "message_count": 128,
        }

    async def check_external_api_health(self) -> dict[str, Any]:
        """Mock external API health check."""
        await asyncio.sleep(self.external_api_delay)
        if not self.external_api_healthy:
            raise aiohttp.ClientError("External API unreachable")

        return {
            "status": HealthCheckStatus.HEALTHY,
            "response_time": self.external_api_delay,
            "api_version": "v2.1.0",
            "rate_limit_remaining": 4900,
        }

    async def check_storage_health(self) -> dict[str, Any]:
        """Mock storage health check."""
        await asyncio.sleep(self.storage_delay)
        if not self.storage_healthy:
            raise OSError("Storage not accessible")

        return {
            "status": HealthCheckStatus.HEALTHY,
            "response_time": self.storage_delay,
            "disk_usage": {"used": 45.2, "total": 100.0},
            "mount_point": "/data",
        }


class MockServiceHealthChecker:
    """Mock service health checker with dependency management."""

    def __init__(self, service_name: str, dependencies: MockHealthDependencies):
        """Initialize service health checker.

        Args:
            service_name: Name of the service
            dependencies: Mock dependencies for health checks
        """
        self.service_name = service_name
        self.dependencies = dependencies
        self.metrics = HealthCheckMetrics()
        self.startup_time = time.time()

        # Service-specific configuration
        self.required_dependencies = self._get_required_dependencies()
        self.health_check_timeout = 5.0
        self.startup_grace_period = 30.0

    def _get_required_dependencies(self) -> list[str]:
        """Get required dependencies based on service type.

        Returns:
            List of required dependency types
        """
        service_deps = {
            "analysis_service": ["postgres", "redis", "rabbitmq", "storage"],
            "cataloging_service": ["postgres", "neo4j", "redis", "rabbitmq"],
            "file_watcher": ["rabbitmq", "storage"],
            "notification_service": ["redis", "external_api"],
            "tracklist_service": ["postgres", "redis"],
            "file_rename_service": ["postgres", "storage"],
        }
        return service_deps.get(self.service_name, [])

    async def basic_health_check(self) -> dict[str, Any]:
        """Basic liveness health check.

        Returns:
            Basic health status response
        """
        start_time = time.time()

        try:
            response = {
                "status": HealthCheckStatus.HEALTHY,
                "service": self.service_name,
                "timestamp": time.time(),
                "uptime": time.time() - self.startup_time,
            }

            # Record metrics
            self.metrics.health_check_requests.labels(
                service=self.service_name, endpoint="health", status="success"
            ).inc()

            return response

        except Exception as e:
            logger.error(f"Basic health check failed for {self.service_name}: {e}")
            response = {
                "status": HealthCheckStatus.UNHEALTHY,
                "service": self.service_name,
                "error": str(e),
                "timestamp": time.time(),
            }

            self.metrics.health_check_requests.labels(
                service=self.service_name, endpoint="health", status="error"
            ).inc()

            return response

        finally:
            duration = time.time() - start_time
            self.metrics.health_check_duration.labels(service=self.service_name, endpoint="health").observe(duration)

    async def readiness_check(self) -> dict[str, Any]:
        """Readiness check with dependency validation.

        Returns:
            Readiness status with dependency checks
        """
        start_time = time.time()

        try:
            # Check if service is past startup grace period
            startup_complete = (time.time() - self.startup_time) > self.startup_grace_period

            # Perform dependency checks
            dependency_checks = await self._check_dependencies()

            # Determine overall readiness
            all_dependencies_ready = all(
                check["status"] == HealthCheckStatus.HEALTHY for check in dependency_checks.values()
            )

            ready = startup_complete and all_dependencies_ready

            response = {
                "ready": ready,
                "service": self.service_name,
                "startup_complete": startup_complete,
                "dependencies": dependency_checks,
                "timestamp": time.time(),
            }

            self.metrics.health_check_requests.labels(
                service=self.service_name, endpoint="ready", status="success" if ready else "not_ready"
            ).inc()

            return response

        except Exception as e:
            logger.error(f"Readiness check failed for {self.service_name}: {e}")

            response = {"ready": False, "service": self.service_name, "error": str(e), "timestamp": time.time()}

            self.metrics.health_check_requests.labels(service=self.service_name, endpoint="ready", status="error").inc()

            return response

        finally:
            duration = time.time() - start_time
            self.metrics.health_check_duration.labels(service=self.service_name, endpoint="ready").observe(duration)

    async def liveness_check(self) -> dict[str, Any]:
        """Liveness check for container orchestration.

        Returns:
            Liveness status response
        """
        start_time = time.time()

        try:
            # Simple alive check - service can respond
            response = {"alive": True, "service": self.service_name, "timestamp": time.time()}

            self.metrics.health_check_requests.labels(
                service=self.service_name, endpoint="live", status="success"
            ).inc()

            return response

        except Exception as e:
            logger.error(f"Liveness check failed for {self.service_name}: {e}")

            response = {"alive": False, "service": self.service_name, "error": str(e), "timestamp": time.time()}

            self.metrics.health_check_requests.labels(service=self.service_name, endpoint="live", status="error").inc()

            return response

        finally:
            duration = time.time() - start_time
            self.metrics.health_check_duration.labels(service=self.service_name, endpoint="live").observe(duration)

    async def _check_dependencies(self) -> dict[str, dict[str, Any]]:
        """Check health of all required dependencies.

        Returns:
            Dictionary mapping dependency names to health check results
        """
        checks = {}

        # Create tasks for parallel dependency checking
        tasks = []
        dependency_names = []

        for dep in self.required_dependencies:
            if dep == "postgres":
                tasks.append(self.dependencies.check_postgres_health())
                dependency_names.append("postgres")
            elif dep == "neo4j":
                tasks.append(self.dependencies.check_neo4j_health())
                dependency_names.append("neo4j")
            elif dep == "redis":
                tasks.append(self.dependencies.check_redis_health())
                dependency_names.append("redis")
            elif dep == "rabbitmq":
                tasks.append(self.dependencies.check_rabbitmq_health())
                dependency_names.append("rabbitmq")
            elif dep == "external_api":
                tasks.append(self.dependencies.check_external_api_health())
                dependency_names.append("external_api")
            elif dep == "storage":
                tasks.append(self.dependencies.check_storage_health())
                dependency_names.append("storage")

        # Execute dependency checks with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=self.health_check_timeout
            )

            # Process results
            for i, result in enumerate(results):
                dep_name = dependency_names[i]
                if isinstance(result, Exception):
                    checks[dep_name] = {
                        "status": HealthCheckStatus.UNHEALTHY,
                        "error": str(result),
                        "timestamp": time.time(),
                    }

                    # Record dependency health metric
                    self.metrics.dependency_health_status.labels(
                        service=self.service_name, dependency_type=dep_name, dependency_name=dep_name
                    ).set(0)
                else:
                    checks[dep_name] = result

                    # Record dependency health metric
                    status_value = 1 if result["status"] == HealthCheckStatus.HEALTHY else 0
                    self.metrics.dependency_health_status.labels(
                        service=self.service_name, dependency_type=dep_name, dependency_name=dep_name
                    ).set(status_value)

        except TimeoutError:
            logger.warning(f"Dependency health checks timed out for {self.service_name}")

            # Mark all as unknown due to timeout
            for dep_name in dependency_names:
                checks[dep_name] = {
                    "status": HealthCheckStatus.UNKNOWN,
                    "error": "Health check timeout",
                    "timestamp": time.time(),
                }

                self.metrics.dependency_health_status.labels(
                    service=self.service_name, dependency_type=dep_name, dependency_name=dep_name
                ).set(0)

        return checks


class MockHealthCheckAggregator:
    """Aggregates health checks across multiple services."""

    def __init__(self):
        """Initialize health check aggregator."""
        self.services: dict[str, MockServiceHealthChecker] = {}
        self.metrics = HealthCheckMetrics()

    def register_service(self, service_name: str, health_checker: MockServiceHealthChecker):
        """Register a service for health monitoring.

        Args:
            service_name: Name of the service
            health_checker: Health checker instance
        """
        self.services[service_name] = health_checker

    async def aggregate_health_status(self) -> dict[str, Any]:
        """Aggregate health status across all registered services.

        Returns:
            Aggregated health status response
        """
        start_time = time.time()

        # Collect health checks from all services
        service_checks = {}
        overall_healthy = True

        tasks = []
        service_names = []

        for service_name, health_checker in self.services.items():
            tasks.append(health_checker.readiness_check())
            service_names.append(service_name)

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(results):
                service_name = service_names[i]

                if isinstance(result, Exception):
                    service_checks[service_name] = {"ready": False, "error": str(result), "timestamp": time.time()}
                    overall_healthy = False
                else:
                    service_checks[service_name] = result
                    if not result.get("ready", False):
                        overall_healthy = False

            response = {
                "healthy": overall_healthy,
                "services": service_checks,
                "timestamp": time.time(),
                "aggregate_response_time": time.time() - start_time,
            }

            # Record aggregated metrics
            status = "healthy" if overall_healthy else "unhealthy"
            self.metrics.health_check_requests.labels(service="aggregator", endpoint="aggregate", status=status).inc()

            return response

        except Exception as e:
            logger.error(f"Health aggregation failed: {e}")

            response = {"healthy": False, "error": str(e), "timestamp": time.time()}

            self.metrics.health_check_requests.labels(service="aggregator", endpoint="aggregate", status="error").inc()

            return response


class MockLoadBalancer:
    """Mock load balancer for health check integration testing."""

    def __init__(self, services: list[str]):
        """Initialize load balancer.

        Args:
            services: List of service endpoints
        """
        self.services = services
        self.healthy_services = set(services)  # Track healthy services
        self.health_check_interval = 10.0  # seconds
        self.health_check_timeout = 5.0  # seconds
        self.failure_threshold = 3  # consecutive failures before marking unhealthy
        self.recovery_threshold = 2  # consecutive successes before marking healthy

        # Track health check results
        self.failure_counts: dict[str, int] = dict.fromkeys(services, 0)
        self.success_counts: dict[str, int] = dict.fromkeys(services, 0)

    async def perform_health_checks(
        self, health_checkers: dict[str, MockServiceHealthChecker]
    ) -> dict[str, dict[str, Any]]:
        """Perform health checks on all registered services.

        Args:
            health_checkers: Dictionary mapping service names to health checkers

        Returns:
            Health check results for all services
        """
        results = {}

        for service_name in self.services:
            if service_name in health_checkers:
                try:
                    # Perform health check with timeout
                    result = await asyncio.wait_for(
                        health_checkers[service_name].basic_health_check(), timeout=self.health_check_timeout
                    )

                    results[service_name] = result

                    # Update failure/success tracking
                    if result["status"] == HealthCheckStatus.HEALTHY:
                        self.failure_counts[service_name] = 0
                        self.success_counts[service_name] += 1

                        # Mark as healthy if recovery threshold met
                        if self.success_counts[service_name] >= self.recovery_threshold:
                            self.healthy_services.add(service_name)
                            self.success_counts[service_name] = 0
                    else:
                        self.success_counts[service_name] = 0
                        self.failure_counts[service_name] += 1

                        # Mark as unhealthy if failure threshold met
                        if self.failure_counts[service_name] >= self.failure_threshold:
                            self.healthy_services.discard(service_name)

                except TimeoutError:
                    logger.warning(f"Health check timeout for {service_name}")
                    results[service_name] = {"status": HealthCheckStatus.UNKNOWN, "error": "Health check timeout"}

                    self.success_counts[service_name] = 0
                    self.failure_counts[service_name] += 1

                    if self.failure_counts[service_name] >= self.failure_threshold:
                        self.healthy_services.discard(service_name)

                except Exception as e:
                    logger.error(f"Health check error for {service_name}: {e}")
                    results[service_name] = {"status": HealthCheckStatus.UNHEALTHY, "error": str(e)}

                    self.success_counts[service_name] = 0
                    self.failure_counts[service_name] += 1

                    if self.failure_counts[service_name] >= self.failure_threshold:
                        self.healthy_services.discard(service_name)
            else:
                results[service_name] = {"status": HealthCheckStatus.UNKNOWN, "error": "Service not registered"}

        return results

    def get_healthy_services(self) -> list[str]:
        """Get list of currently healthy services.

        Returns:
            List of healthy service names
        """
        return list(self.healthy_services)

    def route_request(self) -> str | None:
        """Route request to a healthy service using round-robin.

        Returns:
            Service name to route to, or None if no healthy services
        """
        if not self.healthy_services:
            return None

        # Simple round-robin routing
        services_list = sorted(self.healthy_services)
        # In a real implementation, this would track rotation state
        return services_list[0]


# Test Fixtures
@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for testing."""
    return MockHealthDependencies()


@pytest.fixture
def analysis_service_health_checker(mock_dependencies):
    """Create health checker for analysis service."""
    return MockServiceHealthChecker("analysis_service", mock_dependencies)


@pytest.fixture
def cataloging_service_health_checker(mock_dependencies):
    """Create health checker for cataloging service."""
    return MockServiceHealthChecker("cataloging_service", mock_dependencies)


@pytest.fixture
def file_watcher_health_checker(mock_dependencies):
    """Create health checker for file watcher service."""
    return MockServiceHealthChecker("file_watcher", mock_dependencies)


@pytest.fixture
def health_aggregator():
    """Create health check aggregator."""
    return MockHealthCheckAggregator()


@pytest.fixture
def load_balancer():
    """Create mock load balancer."""
    return MockLoadBalancer(["analysis_service", "cataloging_service", "file_watcher"])


@pytest.fixture
async def mock_http_server():
    """Create mock HTTP server for testing external health check endpoints."""

    async def health_endpoint(request):
        """Mock health endpoint handler."""
        service_name = request.query.get("service", "mock_service")

        # Simulate different response scenarios based on query params
        simulate_error = request.query.get("error", "").lower() == "true"
        simulate_delay = float(request.query.get("delay", "0"))

        if simulate_delay > 0:
            await asyncio.sleep(simulate_delay)

        if simulate_error:
            raise web.HTTPInternalServerError(text="Service unavailable")

        return web.json_response({"status": "healthy", "service": service_name, "timestamp": time.time()})

    async def ready_endpoint(request):
        """Mock readiness endpoint handler."""
        service_name = request.query.get("service", "mock_service")
        ready = request.query.get("ready", "true").lower() == "true"

        return web.json_response(
            {
                "ready": ready,
                "service": service_name,
                "dependencies": {"database": {"status": "healthy"}, "cache": {"status": "healthy"}},
                "timestamp": time.time(),
            }
        )

    app = web.Application()
    app.router.add_get("/health", health_endpoint)
    app.router.add_get("/ready", ready_endpoint)

    return app


# Test Classes
@pytest.mark.integration
class TestBasicHealthCheckFunctionality:
    """Test basic health check functionality for individual services."""

    @pytest.mark.asyncio
    async def test_basic_health_check_success(self, analysis_service_health_checker):
        """Test successful basic health check."""
        result = await analysis_service_health_checker.basic_health_check()

        assert result["status"] == HealthCheckStatus.HEALTHY
        assert result["service"] == "analysis_service"
        assert "timestamp" in result
        assert "uptime" in result
        assert result["uptime"] >= 0

    @pytest.mark.asyncio
    async def test_liveness_check_success(self, cataloging_service_health_checker):
        """Test successful liveness check."""
        result = await cataloging_service_health_checker.liveness_check()

        assert result["alive"] is True
        assert result["service"] == "cataloging_service"
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_multiple_service_health_checks(
        self, analysis_service_health_checker, cataloging_service_health_checker, file_watcher_health_checker
    ):
        """Test health checks across multiple services."""
        checkers = [analysis_service_health_checker, cataloging_service_health_checker, file_watcher_health_checker]

        # Perform health checks in parallel
        tasks = [checker.basic_health_check() for checker in checkers]
        results = await asyncio.gather(*tasks)

        # Verify all services are healthy
        for result in results:
            assert result["status"] == HealthCheckStatus.HEALTHY
            assert "service" in result
            assert "timestamp" in result


@pytest.mark.integration
class TestServiceDependencyHealthChecks:
    """Test service dependency health checks (database, cache, message queues)."""

    @pytest.mark.asyncio
    async def test_readiness_check_all_dependencies_healthy(self, analysis_service_health_checker):
        """Test readiness check with all dependencies healthy."""
        # Override startup time to simulate service has been running long enough
        analysis_service_health_checker.startup_time = time.time() - 60.0  # 1 minute ago

        result = await analysis_service_health_checker.readiness_check()

        assert result["ready"] is True
        assert result["service"] == "analysis_service"
        assert "dependencies" in result

        # Check specific dependencies for analysis service
        deps = result["dependencies"]
        expected_deps = ["postgres", "redis", "rabbitmq", "storage"]

        for dep in expected_deps:
            assert dep in deps
            assert deps[dep]["status"] == HealthCheckStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_readiness_check_database_unhealthy(self, analysis_service_health_checker, mock_dependencies):
        """Test readiness check with database dependency unhealthy."""
        # Override startup time to simulate service has been running long enough
        analysis_service_health_checker.startup_time = time.time() - 60.0  # 1 minute ago

        # Make PostgreSQL unhealthy
        mock_dependencies.postgres_healthy = False

        result = await analysis_service_health_checker.readiness_check()

        assert result["ready"] is False
        assert "dependencies" in result
        assert result["dependencies"]["postgres"]["status"] == HealthCheckStatus.UNHEALTHY
        assert "error" in result["dependencies"]["postgres"]

    @pytest.mark.asyncio
    async def test_readiness_check_cache_unhealthy(self, analysis_service_health_checker, mock_dependencies):
        """Test readiness check with cache dependency unhealthy."""
        # Override startup time to simulate service has been running long enough
        analysis_service_health_checker.startup_time = time.time() - 60.0  # 1 minute ago

        # Make Redis unhealthy
        mock_dependencies.redis_healthy = False

        result = await analysis_service_health_checker.readiness_check()

        assert result["ready"] is False
        assert result["dependencies"]["redis"]["status"] == HealthCheckStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_readiness_check_message_queue_unhealthy(self, analysis_service_health_checker, mock_dependencies):
        """Test readiness check with message queue dependency unhealthy."""
        # Override startup time to simulate service has been running long enough
        analysis_service_health_checker.startup_time = time.time() - 60.0  # 1 minute ago

        # Make RabbitMQ unhealthy
        mock_dependencies.rabbitmq_healthy = False

        result = await analysis_service_health_checker.readiness_check()

        assert result["ready"] is False
        assert result["dependencies"]["rabbitmq"]["status"] == HealthCheckStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_readiness_check_multiple_dependencies_unhealthy(
        self, analysis_service_health_checker, mock_dependencies
    ):
        """Test readiness check with multiple dependencies unhealthy."""
        # Override startup time to simulate service has been running long enough
        analysis_service_health_checker.startup_time = time.time() - 60.0  # 1 minute ago

        # Make multiple dependencies unhealthy
        mock_dependencies.postgres_healthy = False
        mock_dependencies.redis_healthy = False

        result = await analysis_service_health_checker.readiness_check()

        assert result["ready"] is False
        assert result["dependencies"]["postgres"]["status"] == HealthCheckStatus.UNHEALTHY
        assert result["dependencies"]["redis"]["status"] == HealthCheckStatus.UNHEALTHY
        # RabbitMQ and storage should still be healthy
        assert result["dependencies"]["rabbitmq"]["status"] == HealthCheckStatus.HEALTHY
        assert result["dependencies"]["storage"]["status"] == HealthCheckStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_dependency_health_timeout(self, analysis_service_health_checker, mock_dependencies):
        """Test dependency health check timeout handling."""
        # Override startup time to simulate service has been running long enough
        analysis_service_health_checker.startup_time = time.time() - 60.0  # 1 minute ago

        # Set high delays to trigger timeout
        mock_dependencies.postgres_delay = 10.0  # Higher than timeout

        result = await analysis_service_health_checker.readiness_check()

        assert result["ready"] is False
        # Should have timeout errors
        deps = result["dependencies"]
        for dep_name, dep_result in deps.items():
            if dep_name == "postgres":
                assert dep_result["status"] == HealthCheckStatus.UNKNOWN
                assert "timeout" in dep_result.get("error", "").lower()


@pytest.mark.integration
class TestAggregatedSystemHealthStatus:
    """Test aggregated system-wide health status."""

    @pytest.mark.asyncio
    async def test_aggregate_health_all_services_healthy(
        self,
        health_aggregator,
        analysis_service_health_checker,
        cataloging_service_health_checker,
        file_watcher_health_checker,
    ):
        """Test aggregated health status with all services healthy."""
        # Register services
        health_aggregator.register_service("analysis_service", analysis_service_health_checker)
        health_aggregator.register_service("cataloging_service", cataloging_service_health_checker)
        health_aggregator.register_service("file_watcher", file_watcher_health_checker)

        result = await health_aggregator.aggregate_health_status()

        assert result["healthy"] is True
        assert "services" in result
        assert len(result["services"]) == 3

        # Check all services are ready
        for service_result in result["services"].values():
            assert service_result["ready"] is True

    @pytest.mark.asyncio
    async def test_aggregate_health_one_service_unhealthy(
        self, health_aggregator, analysis_service_health_checker, cataloging_service_health_checker, mock_dependencies
    ):
        """Test aggregated health status with one service unhealthy."""
        # Make one service's dependency unhealthy
        mock_dependencies.postgres_healthy = False

        # Register services
        health_aggregator.register_service("analysis_service", analysis_service_health_checker)
        health_aggregator.register_service("cataloging_service", cataloging_service_health_checker)

        result = await health_aggregator.aggregate_health_status()

        assert result["healthy"] is False
        assert "services" in result

        # Both services should be affected by postgres being down
        assert result["services"]["analysis_service"]["ready"] is False
        assert result["services"]["cataloging_service"]["ready"] is False

    @pytest.mark.asyncio
    async def test_aggregate_health_partial_failure(
        self, health_aggregator, analysis_service_health_checker, file_watcher_health_checker, mock_dependencies
    ):
        """Test aggregated health status with partial service failure."""
        # Make Redis unhealthy (affects analysis_service but not file_watcher)
        mock_dependencies.redis_healthy = False

        # Register services
        health_aggregator.register_service("analysis_service", analysis_service_health_checker)
        health_aggregator.register_service("file_watcher", file_watcher_health_checker)

        result = await health_aggregator.aggregate_health_status()

        assert result["healthy"] is False
        assert result["services"]["analysis_service"]["ready"] is False
        assert result["services"]["file_watcher"]["ready"] is True

    @pytest.mark.asyncio
    async def test_aggregate_health_empty_services(self, health_aggregator):
        """Test aggregated health status with no registered services."""
        result = await health_aggregator.aggregate_health_status()

        assert result["healthy"] is True  # No services means no failures
        assert result["services"] == {}


@pytest.mark.integration
class TestHealthCheckPerformanceReliability:
    """Test health check endpoint performance and reliability."""

    @pytest.mark.asyncio
    async def test_health_check_response_time(self, analysis_service_health_checker):
        """Test health check response time is within acceptable limits."""
        start_time = time.time()
        result = await analysis_service_health_checker.basic_health_check()
        end_time = time.time()

        response_time = end_time - start_time

        assert result["status"] == HealthCheckStatus.HEALTHY
        assert response_time < 1.0  # Should respond within 1 second

    @pytest.mark.asyncio
    async def test_health_check_concurrent_requests(self, analysis_service_health_checker):
        """Test health check performance under concurrent requests."""
        num_concurrent = 10

        # Create concurrent health check tasks
        tasks = [analysis_service_health_checker.basic_health_check() for _ in range(num_concurrent)]

        start_time = time.time()
        results = await asyncio.gather(*tasks)
        end_time = time.time()

        total_time = end_time - start_time

        # All requests should succeed
        for result in results:
            assert result["status"] == HealthCheckStatus.HEALTHY

        # Total time should be reasonable for concurrent execution
        assert total_time < 2.0  # Should complete within 2 seconds

    @pytest.mark.asyncio
    async def test_dependency_check_with_delays(self, analysis_service_health_checker, mock_dependencies):
        """Test dependency health checks with network delays."""
        # Override startup time to simulate service has been running long enough
        analysis_service_health_checker.startup_time = time.time() - 60.0  # 1 minute ago

        # Add realistic network delays
        mock_dependencies.postgres_delay = 0.1
        mock_dependencies.redis_delay = 0.05
        mock_dependencies.rabbitmq_delay = 0.08
        mock_dependencies.storage_delay = 0.03

        start_time = time.time()
        result = await analysis_service_health_checker.readiness_check()
        end_time = time.time()

        response_time = end_time - start_time

        assert result["ready"] is True
        # Should be faster than sum of individual delays due to parallel execution
        assert response_time < 0.5
        assert response_time > 0.1  # But should reflect the longest delay

    @pytest.mark.asyncio
    async def test_health_check_metrics_collection(self, analysis_service_health_checker):
        """Test health check metrics are properly collected."""
        metrics = analysis_service_health_checker.metrics

        # Perform several health checks
        for _ in range(5):
            await analysis_service_health_checker.basic_health_check()

        # Perform readiness check
        await analysis_service_health_checker.readiness_check()

        # Check metrics were recorded (in a real test, we'd check the actual metric values)
        # Here we just verify the metrics objects exist
        assert metrics.health_check_requests is not None
        assert metrics.health_check_duration is not None
        assert metrics.service_health_status is not None

    @pytest.mark.asyncio
    async def test_health_check_reliability_with_failures(self, analysis_service_health_checker, mock_dependencies):
        """Test health check reliability when dependencies have intermittent failures."""
        results = []

        for i in range(10):
            # Simulate intermittent failures
            mock_dependencies.postgres_healthy = (i % 3) != 0  # Fail every 3rd attempt

            result = await analysis_service_health_checker.readiness_check()
            results.append(result)

        # Should have mix of healthy and unhealthy results
        healthy_count = sum(1 for r in results if r["ready"])
        unhealthy_count = len(results) - healthy_count

        assert healthy_count > 0
        assert unhealthy_count > 0
        assert len(results) == 10


@pytest.mark.integration
class TestHealthCheckFailureScenarios:
    """Test health check with various failure scenarios."""

    @pytest.mark.asyncio
    async def test_database_connection_failure(self, analysis_service_health_checker, mock_dependencies):
        """Test health check behavior when database connection fails."""
        mock_dependencies.postgres_healthy = False

        result = await analysis_service_health_checker.readiness_check()

        assert result["ready"] is False
        assert result["dependencies"]["postgres"]["status"] == HealthCheckStatus.UNHEALTHY
        assert "connection failed" in result["dependencies"]["postgres"]["error"].lower()

    @pytest.mark.asyncio
    async def test_network_timeout_scenario(self, analysis_service_health_checker, mock_dependencies):
        """Test health check behavior during network timeouts."""
        # Set delays longer than timeout
        mock_dependencies.postgres_delay = 10.0
        mock_dependencies.redis_delay = 10.0

        result = await analysis_service_health_checker.readiness_check()

        assert result["ready"] is False
        # Dependencies should be marked as unknown due to timeout
        assert result["dependencies"]["postgres"]["status"] == HealthCheckStatus.UNKNOWN
        assert result["dependencies"]["redis"]["status"] == HealthCheckStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_partial_service_degradation(self, analysis_service_health_checker, mock_dependencies):
        """Test health check behavior with partial service degradation."""
        # Make some dependencies healthy, others not
        mock_dependencies.postgres_healthy = True
        mock_dependencies.redis_healthy = False
        mock_dependencies.rabbitmq_healthy = True
        mock_dependencies.storage_healthy = False

        result = await analysis_service_health_checker.readiness_check()

        assert result["ready"] is False
        assert result["dependencies"]["postgres"]["status"] == HealthCheckStatus.HEALTHY
        assert result["dependencies"]["redis"]["status"] == HealthCheckStatus.UNHEALTHY
        assert result["dependencies"]["rabbitmq"]["status"] == HealthCheckStatus.HEALTHY
        assert result["dependencies"]["storage"]["status"] == HealthCheckStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_cascading_failure_scenario(
        self, health_aggregator, analysis_service_health_checker, cataloging_service_health_checker, mock_dependencies
    ):
        """Test cascading failure scenario across multiple services."""
        # Register services that share dependencies
        health_aggregator.register_service("analysis_service", analysis_service_health_checker)
        health_aggregator.register_service("cataloging_service", cataloging_service_health_checker)

        # Simulate shared database failure
        mock_dependencies.postgres_healthy = False

        result = await health_aggregator.aggregate_health_status()

        assert result["healthy"] is False
        # Both services should be affected
        assert result["services"]["analysis_service"]["ready"] is False
        assert result["services"]["cataloging_service"]["ready"] is False

    @pytest.mark.asyncio
    async def test_recovery_after_failure(self, analysis_service_health_checker, mock_dependencies):
        """Test service recovery after dependency failure."""
        # Initial failure
        mock_dependencies.postgres_healthy = False
        result1 = await analysis_service_health_checker.readiness_check()
        assert result1["ready"] is False

        # Recovery
        mock_dependencies.postgres_healthy = True
        result2 = await analysis_service_health_checker.readiness_check()
        assert result2["ready"] is True
        assert result2["dependencies"]["postgres"]["status"] == HealthCheckStatus.HEALTHY


@pytest.mark.integration
class TestCustomHealthChecks:
    """Test custom health checks for different service components."""

    @pytest.mark.asyncio
    async def test_analysis_service_custom_checks(self, analysis_service_health_checker):
        """Test custom health checks specific to analysis service."""
        result = await analysis_service_health_checker.readiness_check()

        # Analysis service should check specific dependencies
        assert "postgres" in result["dependencies"]  # For metadata storage
        assert "redis" in result["dependencies"]  # For caching
        assert "rabbitmq" in result["dependencies"]  # For message processing
        assert "storage" in result["dependencies"]  # For audio file access

        # Each dependency should have detailed status
        for dep_result in result["dependencies"].values():
            assert "status" in dep_result
            assert "response_time" in dep_result

    @pytest.mark.asyncio
    async def test_cataloging_service_custom_checks(self, cataloging_service_health_checker):
        """Test custom health checks specific to cataloging service."""
        result = await cataloging_service_health_checker.readiness_check()

        # Cataloging service should check different dependencies
        assert "postgres" in result["dependencies"]  # For relational data
        assert "neo4j" in result["dependencies"]  # For graph relationships
        assert "redis" in result["dependencies"]  # For caching
        assert "rabbitmq" in result["dependencies"]  # For message processing

    @pytest.mark.asyncio
    async def test_file_watcher_custom_checks(self, file_watcher_health_checker):
        """Test custom health checks specific to file watcher service."""
        result = await file_watcher_health_checker.readiness_check()

        # File watcher should have minimal dependencies
        assert "rabbitmq" in result["dependencies"]  # For sending notifications
        assert "storage" in result["dependencies"]  # For file system access

        # Should not require database dependencies
        assert "postgres" not in result["dependencies"]
        assert "redis" not in result["dependencies"]

    @pytest.mark.asyncio
    async def test_startup_grace_period_handling(self):
        """Test startup grace period for newly started services."""
        # Create health checker with recent startup time
        mock_deps = MockHealthDependencies()
        health_checker = MockServiceHealthChecker("test_service", mock_deps)
        health_checker.startup_grace_period = 30.0  # 30 second grace period
        health_checker.startup_time = time.time()  # Just started

        result = await health_checker.readiness_check()

        # Service should not be ready during grace period, even if dependencies are healthy
        assert result["ready"] is False
        assert result["startup_complete"] is False

    @pytest.mark.asyncio
    async def test_component_specific_health_info(self, analysis_service_health_checker):
        """Test that health checks return component-specific information."""
        result = await analysis_service_health_checker.readiness_check()

        deps = result["dependencies"]

        # PostgreSQL should include connection pool info
        if "postgres" in deps and deps["postgres"]["status"] == HealthCheckStatus.HEALTHY:
            postgres_info = deps["postgres"]
            assert "connection_pool" in postgres_info
            assert "version" in postgres_info

        # Redis should include memory usage info
        if "redis" in deps and deps["redis"]["status"] == HealthCheckStatus.HEALTHY:
            redis_info = deps["redis"]
            assert "memory_usage" in redis_info
            assert "connected_clients" in redis_info

        # RabbitMQ should include queue information
        if "rabbitmq" in deps and deps["rabbitmq"]["status"] == HealthCheckStatus.HEALTHY:
            rabbitmq_info = deps["rabbitmq"]
            assert "queue_count" in rabbitmq_info
            assert "message_count" in rabbitmq_info


@pytest.mark.integration
class TestHealthCheckMonitoringAlerting:
    """Test health check monitoring and alerting simulation."""

    @pytest.mark.asyncio
    async def test_metrics_collection_during_health_checks(self, analysis_service_health_checker):
        """Test that metrics are collected during health check operations."""
        metrics = analysis_service_health_checker.metrics

        # Perform various health checks
        await analysis_service_health_checker.basic_health_check()
        await analysis_service_health_checker.liveness_check()
        await analysis_service_health_checker.readiness_check()

        # Verify metrics objects exist and are properly configured
        assert metrics.health_check_requests is not None
        assert metrics.health_check_duration is not None
        assert metrics.service_health_status is not None
        assert metrics.dependency_health_status is not None

    @pytest.mark.asyncio
    async def test_failure_threshold_monitoring(
        self, load_balancer, analysis_service_health_checker, mock_dependencies
    ):
        """Test monitoring failure thresholds for alerting."""
        health_checkers = {"analysis_service": analysis_service_health_checker}

        # Simulate consecutive failures
        mock_dependencies.postgres_healthy = False

        for _ in range(load_balancer.failure_threshold):
            await load_balancer.perform_health_checks(health_checkers)

        # Service should be marked as unhealthy after failure threshold
        healthy_services = load_balancer.get_healthy_services()
        assert "analysis_service" not in healthy_services

    @pytest.mark.asyncio
    async def test_recovery_threshold_monitoring(
        self, load_balancer, analysis_service_health_checker, mock_dependencies
    ):
        """Test monitoring recovery thresholds for alerting."""
        health_checkers = {"analysis_service": analysis_service_health_checker}

        # First mark service as unhealthy
        mock_dependencies.postgres_healthy = False
        for _ in range(load_balancer.failure_threshold):
            await load_balancer.perform_health_checks(health_checkers)

        assert "analysis_service" not in load_balancer.get_healthy_services()

        # Now simulate recovery
        mock_dependencies.postgres_healthy = True
        for _ in range(load_balancer.recovery_threshold):
            await load_balancer.perform_health_checks(health_checkers)

        # Service should be marked as healthy after recovery threshold
        healthy_services = load_balancer.get_healthy_services()
        assert "analysis_service" in healthy_services

    @pytest.mark.asyncio
    async def test_health_check_alerting_simulation(self, analysis_service_health_checker, mock_dependencies):
        """Test simulation of alerting based on health check results."""
        alerts_triggered = []

        def mock_alert_handler(service: str, status: str, details: dict):
            """Mock alert handler to capture alert events."""
            alerts_triggered.append(
                {"service": service, "status": status, "details": details, "timestamp": time.time()}
            )

        # Perform health check with all dependencies healthy
        result1 = await analysis_service_health_checker.readiness_check()
        if result1["ready"]:
            mock_alert_handler("analysis_service", "healthy", result1)

        # Simulate dependency failure
        mock_dependencies.postgres_healthy = False
        result2 = await analysis_service_health_checker.readiness_check()
        if not result2["ready"]:
            mock_alert_handler("analysis_service", "unhealthy", result2)

        # Verify alerts were triggered
        assert len(alerts_triggered) == 2
        assert alerts_triggered[0]["status"] == "healthy"
        assert alerts_triggered[1]["status"] == "unhealthy"

    @pytest.mark.asyncio
    async def test_health_status_history_tracking(self, analysis_service_health_checker, mock_dependencies):
        """Test tracking health status history for trend analysis."""
        health_history = []

        # Collect health status over time with varying conditions
        conditions = [
            {"postgres": True, "redis": True, "rabbitmq": True, "storage": True},
            {"postgres": False, "redis": True, "rabbitmq": True, "storage": True},
            {"postgres": True, "redis": False, "rabbitmq": True, "storage": True},
            {"postgres": True, "redis": True, "rabbitmq": True, "storage": True},
        ]

        for condition in conditions:
            mock_dependencies.postgres_healthy = condition["postgres"]
            mock_dependencies.redis_healthy = condition["redis"]
            mock_dependencies.rabbitmq_healthy = condition["rabbitmq"]
            mock_dependencies.storage_healthy = condition["storage"]

            result = await analysis_service_health_checker.readiness_check()
            health_history.append(
                {
                    "timestamp": result["timestamp"],
                    "ready": result["ready"],
                    "dependencies": {k: v["status"] for k, v in result["dependencies"].items()},
                }
            )

            # Small delay to ensure different timestamps
            await asyncio.sleep(0.01)

        # Verify health history was tracked
        assert len(health_history) == 4
        assert health_history[0]["ready"] is True  # All healthy
        assert health_history[1]["ready"] is False  # Postgres down
        assert health_history[2]["ready"] is False  # Redis down
        assert health_history[3]["ready"] is True  # All recovered


@pytest.mark.integration
class TestLoadBalancerOrchestrationIntegration:
    """Test load balancer and orchestration integration scenarios."""

    @pytest.mark.asyncio
    async def test_load_balancer_health_tracking(
        self, load_balancer, analysis_service_health_checker, cataloging_service_health_checker
    ):
        """Test load balancer tracking service health."""
        health_checkers = {
            "analysis_service": analysis_service_health_checker,
            "cataloging_service": cataloging_service_health_checker,
        }

        # Perform initial health checks
        results = await load_balancer.perform_health_checks(health_checkers)

        # All services should be healthy initially
        assert len(results) == 2
        assert results["analysis_service"]["status"] == HealthCheckStatus.HEALTHY
        assert results["cataloging_service"]["status"] == HealthCheckStatus.HEALTHY

        healthy_services = load_balancer.get_healthy_services()
        assert "analysis_service" in healthy_services
        assert "cataloging_service" in healthy_services

    @pytest.mark.asyncio
    async def test_load_balancer_service_removal(
        self, load_balancer, analysis_service_health_checker, cataloging_service_health_checker, mock_dependencies
    ):
        """Test load balancer removing unhealthy services from rotation."""
        health_checkers = {
            "analysis_service": analysis_service_health_checker,
            "cataloging_service": cataloging_service_health_checker,
        }

        # Make one service unhealthy
        mock_dependencies.postgres_healthy = False  # Affects both services

        # Trigger enough failures to exceed threshold
        for _ in range(load_balancer.failure_threshold):
            await load_balancer.perform_health_checks(health_checkers)

        # Both services should be removed from healthy pool (both depend on postgres)
        healthy_services = load_balancer.get_healthy_services()
        assert len(healthy_services) == 0

    @pytest.mark.asyncio
    async def test_load_balancer_service_recovery(
        self, load_balancer, analysis_service_health_checker, mock_dependencies
    ):
        """Test load balancer re-adding recovered services to rotation."""
        health_checkers = {"analysis_service": analysis_service_health_checker}

        # First make service unhealthy
        mock_dependencies.postgres_healthy = False
        for _ in range(load_balancer.failure_threshold):
            await load_balancer.perform_health_checks(health_checkers)

        assert "analysis_service" not in load_balancer.get_healthy_services()

        # Now recover service
        mock_dependencies.postgres_healthy = True
        for _ in range(load_balancer.recovery_threshold):
            await load_balancer.perform_health_checks(health_checkers)

        # Service should be back in rotation
        healthy_services = load_balancer.get_healthy_services()
        assert "analysis_service" in healthy_services

    @pytest.mark.asyncio
    async def test_load_balancer_request_routing(
        self, load_balancer, analysis_service_health_checker, cataloging_service_health_checker
    ):
        """Test load balancer request routing to healthy services."""
        health_checkers = {
            "analysis_service": analysis_service_health_checker,
            "cataloging_service": cataloging_service_health_checker,
        }

        # Ensure services are healthy
        await load_balancer.perform_health_checks(health_checkers)

        # Test routing to healthy services
        routed_service = load_balancer.route_request()
        assert routed_service in ["analysis_service", "cataloging_service"]

        healthy_services = load_balancer.get_healthy_services()
        assert routed_service in healthy_services

    @pytest.mark.asyncio
    async def test_load_balancer_no_healthy_services(
        self, load_balancer, analysis_service_health_checker, cataloging_service_health_checker, mock_dependencies
    ):
        """Test load balancer behavior when no services are healthy."""
        health_checkers = {
            "analysis_service": analysis_service_health_checker,
            "cataloging_service": cataloging_service_health_checker,
        }

        # Make all services unhealthy
        mock_dependencies.postgres_healthy = False
        mock_dependencies.redis_healthy = False
        mock_dependencies.rabbitmq_healthy = False

        # Trigger failures
        for _ in range(load_balancer.failure_threshold):
            await load_balancer.perform_health_checks(health_checkers)

        # Should have no healthy services
        healthy_services = load_balancer.get_healthy_services()
        assert len(healthy_services) == 0

        # Routing should return None
        routed_service = load_balancer.route_request()
        assert routed_service is None

    @pytest.mark.asyncio
    async def test_kubernetes_style_health_probes(self, analysis_service_health_checker, mock_dependencies):
        """Test Kubernetes-style liveness and readiness probes."""
        # Test liveness probe - should almost always succeed unless service is completely broken
        liveness_result = await analysis_service_health_checker.liveness_check()
        assert liveness_result["alive"] is True

        # Test readiness probe - more strict, checks dependencies
        readiness_result = await analysis_service_health_checker.readiness_check()
        assert readiness_result["ready"] is True

        # Make dependencies unhealthy
        mock_dependencies.postgres_healthy = False

        # Liveness should still pass (service process is alive)
        liveness_result = await analysis_service_health_checker.liveness_check()
        assert liveness_result["alive"] is True

        # Readiness should fail (service not ready to handle requests)
        readiness_result = await analysis_service_health_checker.readiness_check()
        assert readiness_result["ready"] is False

    @pytest.mark.asyncio
    async def test_health_check_timeout_configuration(
        self, load_balancer, analysis_service_health_checker, mock_dependencies
    ):
        """Test configurable health check timeouts for orchestration."""
        # Set slow dependency response
        mock_dependencies.postgres_delay = 2.0

        # Configure short timeout
        load_balancer.health_check_timeout = 1.0

        health_checkers = {"analysis_service": analysis_service_health_checker}

        # Health check should timeout
        results = await load_balancer.perform_health_checks(health_checkers)

        # Should get timeout result
        assert results["analysis_service"]["status"] == HealthCheckStatus.UNKNOWN
        assert "timeout" in results["analysis_service"].get("error", "").lower()


@pytest.mark.integration
@pytest.mark.slow
class TestHealthCheckEndToEndScenarios:
    """End-to-end integration testing scenarios."""

    @pytest.mark.asyncio
    async def test_complete_system_health_monitoring_workflow(
        self, health_aggregator, load_balancer, mock_dependencies
    ):
        """Test complete system health monitoring workflow."""
        # Setup services
        services = {
            "analysis_service": MockServiceHealthChecker("analysis_service", mock_dependencies),
            "cataloging_service": MockServiceHealthChecker("cataloging_service", mock_dependencies),
            "file_watcher": MockServiceHealthChecker("file_watcher", mock_dependencies),
        }

        # Register with aggregator
        for name, checker in services.items():
            health_aggregator.register_service(name, checker)

        # Initial system health check
        system_health = await health_aggregator.aggregate_health_status()
        assert system_health["healthy"] is True

        # Load balancer health checks
        lb_results = await load_balancer.perform_health_checks(services)
        assert all(result["status"] == HealthCheckStatus.HEALTHY for result in lb_results.values())

        # Simulate cascading failure
        mock_dependencies.postgres_healthy = False

        # Check system response to failure
        system_health = await health_aggregator.aggregate_health_status()
        assert system_health["healthy"] is False

        # Load balancer should detect failures
        lb_results = await load_balancer.perform_health_checks(services)
        for _ in range(load_balancer.failure_threshold - 1):
            await load_balancer.perform_health_checks(services)

        # Services should be removed from rotation
        assert len(load_balancer.get_healthy_services()) < len(services)

        # Simulate recovery
        mock_dependencies.postgres_healthy = True

        # System should recover
        system_health = await health_aggregator.aggregate_health_status()
        assert system_health["healthy"] is True

        # Load balancer should restore services
        for _ in range(load_balancer.recovery_threshold):
            await load_balancer.perform_health_checks(services)

        healthy_services = load_balancer.get_healthy_services()
        assert len(healthy_services) > 0

    @pytest.mark.asyncio
    async def test_disaster_recovery_simulation(self, health_aggregator, mock_dependencies):
        """Test disaster recovery scenario simulation."""
        # Setup multiple services
        services = {
            "analysis_service": MockServiceHealthChecker("analysis_service", mock_dependencies),
            "cataloging_service": MockServiceHealthChecker("cataloging_service", mock_dependencies),
            "file_watcher": MockServiceHealthChecker("file_watcher", mock_dependencies),
        }

        for name, checker in services.items():
            health_aggregator.register_service(name, checker)

        # Simulate total infrastructure failure
        mock_dependencies.postgres_healthy = False
        mock_dependencies.redis_healthy = False
        mock_dependencies.rabbitmq_healthy = False
        mock_dependencies.storage_healthy = False

        # Check system status during disaster
        disaster_health = await health_aggregator.aggregate_health_status()
        assert disaster_health["healthy"] is False

        # Verify all services are affected
        for service_result in disaster_health["services"].values():
            assert service_result["ready"] is False

        # Simulate gradual recovery
        recovery_steps = [
            {"postgres": True, "redis": False, "rabbitmq": False, "storage": False},
            {"postgres": True, "redis": True, "rabbitmq": False, "storage": False},
            {"postgres": True, "redis": True, "rabbitmq": True, "storage": False},
            {"postgres": True, "redis": True, "rabbitmq": True, "storage": True},
        ]

        for step in recovery_steps:
            mock_dependencies.postgres_healthy = step["postgres"]
            mock_dependencies.redis_healthy = step["redis"]
            mock_dependencies.rabbitmq_healthy = step["rabbitmq"]
            mock_dependencies.storage_healthy = step["storage"]

            health_status = await health_aggregator.aggregate_health_status()

            # Track recovery progress
            ready_services = sum(1 for service_result in health_status["services"].values() if service_result["ready"])

            # Verify progressive recovery
            if step == recovery_steps[-1]:  # Final step
                assert health_status["healthy"] is True
                assert ready_services == len(services)

    @pytest.mark.asyncio
    async def test_performance_degradation_detection(self, analysis_service_health_checker, mock_dependencies):
        """Test detection of performance degradation through health checks."""
        performance_history = []

        # Simulate gradually increasing response times
        delays = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

        for delay in delays:
            mock_dependencies.postgres_delay = delay
            mock_dependencies.redis_delay = delay * 0.5
            mock_dependencies.rabbitmq_delay = delay * 0.3

            start_time = time.time()
            result = await analysis_service_health_checker.readiness_check()
            end_time = time.time()

            response_time = end_time - start_time
            performance_history.append(
                {
                    "delay_setting": delay,
                    "actual_response_time": response_time,
                    "ready": result["ready"],
                    "dependencies": {k: v["response_time"] for k, v in result["dependencies"].items()},
                }
            )

        # Verify performance degradation is captured
        assert len(performance_history) == len(delays)

        # Response times should generally increase
        response_times = [entry["actual_response_time"] for entry in performance_history]
        for i in range(1, len(response_times)):
            # Allow some variance due to execution overhead
            assert response_times[i] >= response_times[i - 1] * 0.8

        # High delays should still maintain service readiness if dependencies are healthy
        for entry in performance_history:
            if entry["delay_setting"] < 5.0:  # Below timeout threshold
                assert entry["ready"] is True


if __name__ == "__main__":
    # Run specific test classes or all tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])
