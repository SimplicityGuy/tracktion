"""Service lifecycle integration tests.

This module contains integration tests for service startup, shutdown, and
lifecycle management, including dependency resolution, initialization order,
and graceful cleanup.
"""

import asyncio
import contextlib
import logging
import time
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import pytest

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAILED = "failed"


class MockService:
    """Mock service for lifecycle testing."""

    def __init__(
        self,
        name: str,
        dependencies: list[str] | None = None,
        startup_time: float = 0.1,
        shutdown_time: float = 0.05,
        failure_mode: str | None = None,
    ):
        self.name = name
        self.dependencies = dependencies or []
        self.startup_time = startup_time
        self.shutdown_time = shutdown_time
        self.failure_mode = failure_mode

        self.status = ServiceStatus.STOPPED
        self.start_time: datetime | None = None
        self.stop_time: datetime | None = None
        self.health_check_count = 0
        self.restart_count = 0
        self.initialization_data: dict[str, Any] = {}
        self.cleanup_data: dict[str, Any] = {}

        # Mock resources
        self.database_connection = None
        self.message_queue_connection = None
        self.cache_connection = None
        self.file_watchers = []

    async def start(self) -> bool:
        """Start the service."""
        if self.status != ServiceStatus.STOPPED:
            raise RuntimeError(f"Cannot start service {self.name} in status {self.status}")

        self.status = ServiceStatus.STARTING
        logger.info(f"Starting service: {self.name}")

        try:
            # Simulate startup failures
            if self.failure_mode == "startup_failure":
                await asyncio.sleep(0.01)
                raise Exception(f"Simulated startup failure for {self.name}")

            if self.failure_mode == "startup_timeout":
                await asyncio.sleep(5.0)  # Simulate timeout

            # Simulate startup time
            await asyncio.sleep(self.startup_time)

            # Initialize mock resources
            await self._initialize_resources()

            self.status = ServiceStatus.RUNNING
            self.start_time = datetime.now(UTC)

            logger.info(f"Service {self.name} started successfully")
            return True

        except Exception as e:
            self.status = ServiceStatus.FAILED
            logger.error(f"Failed to start service {self.name}: {e}")
            raise

    async def stop(self) -> bool:
        """Stop the service."""
        if self.status not in [ServiceStatus.RUNNING, ServiceStatus.FAILED]:
            logger.warning(f"Service {self.name} is not running, current status: {self.status}")
            return True

        self.status = ServiceStatus.STOPPING
        logger.info(f"Stopping service: {self.name}")

        try:
            # Simulate shutdown failures
            if self.failure_mode == "shutdown_failure":
                await asyncio.sleep(0.01)
                raise Exception(f"Simulated shutdown failure for {self.name}")

            # Simulate shutdown time
            await asyncio.sleep(self.shutdown_time)

            # Cleanup mock resources
            await self._cleanup_resources()

            self.status = ServiceStatus.STOPPED
            self.stop_time = datetime.now(UTC)

            logger.info(f"Service {self.name} stopped successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to stop service {self.name}: {e}")
            raise

    async def restart(self) -> bool:
        """Restart the service."""
        logger.info(f"Restarting service: {self.name}")

        if self.status == ServiceStatus.RUNNING:
            await self.stop()

        self.restart_count += 1
        return await self.start()

    async def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        self.health_check_count += 1

        if self.failure_mode == "health_check_failure":
            return {
                "status": "unhealthy",
                "service": self.name,
                "timestamp": datetime.now(UTC).isoformat(),
                "error": "Simulated health check failure",
            }

        return {
            "status": "healthy" if self.status == ServiceStatus.RUNNING else "unhealthy",
            "service": self.name,
            "timestamp": datetime.now(UTC).isoformat(),
            "uptime_seconds": ((datetime.now(UTC) - self.start_time).total_seconds() if self.start_time else 0),
            "health_checks": self.health_check_count,
            "restarts": self.restart_count,
        }

    async def _initialize_resources(self):
        """Initialize service resources."""
        # Simulate resource initialization
        self.database_connection = MockDatabaseConnection(f"{self.name}_db")
        self.message_queue_connection = MockMessageQueueConnection(f"{self.name}_mq")
        self.cache_connection = MockCacheConnection(f"{self.name}_cache")

        # Record initialization
        self.initialization_data = {
            "database_initialized": True,
            "message_queue_initialized": True,
            "cache_initialized": True,
            "initialization_time": datetime.now(UTC).isoformat(),
        }

    async def _cleanup_resources(self):
        """Cleanup service resources."""
        # Simulate resource cleanup
        if self.database_connection:
            await self.database_connection.close()
            self.database_connection = None

        if self.message_queue_connection:
            await self.message_queue_connection.close()
            self.message_queue_connection = None

        if self.cache_connection:
            await self.cache_connection.close()
            self.cache_connection = None

        # Clear file watchers
        self.file_watchers.clear()

        # Record cleanup
        self.cleanup_data = {
            "database_cleaned": True,
            "message_queue_cleaned": True,
            "cache_cleaned": True,
            "cleanup_time": datetime.now(UTC).isoformat(),
        }


class MockDatabaseConnection:
    """Mock database connection."""

    def __init__(self, connection_id: str):
        self.connection_id = connection_id
        self.is_connected = True
        self.query_count = 0

    async def close(self):
        """Close the connection."""
        self.is_connected = False


class MockMessageQueueConnection:
    """Mock message queue connection."""

    def __init__(self, connection_id: str):
        self.connection_id = connection_id
        self.is_connected = True
        self.message_count = 0

    async def close(self):
        """Close the connection."""
        self.is_connected = False


class MockCacheConnection:
    """Mock cache connection."""

    def __init__(self, connection_id: str):
        self.connection_id = connection_id
        self.is_connected = True
        self.cache_hits = 0

    async def close(self):
        """Close the connection."""
        self.is_connected = False


class ServiceManager:
    """Manages service lifecycle and dependencies."""

    def __init__(self):
        self.services: dict[str, MockService] = {}
        self.dependency_graph: dict[str, set[str]] = {}
        self.startup_order: list[str] = []
        self.shutdown_order: list[str] = []
        self.startup_timeout = 10.0  # seconds
        self.shutdown_timeout = 5.0  # seconds

    def register_service(self, service: MockService):
        """Register a service with the manager."""
        self.services[service.name] = service
        self.dependency_graph[service.name] = set(service.dependencies)
        logger.info(f"Registered service: {service.name} with dependencies: {service.dependencies}")

    def calculate_startup_order(self) -> list[str]:
        """Calculate service startup order based on dependencies."""
        visited = set()
        visiting = set()
        order = []

        def visit(service_name: str):
            if service_name in visiting:
                raise ValueError(f"Circular dependency detected involving {service_name}")
            if service_name in visited:
                return

            visiting.add(service_name)

            # Visit dependencies first
            for dependency in self.dependency_graph.get(service_name, set()):
                if dependency not in self.services:
                    raise ValueError(f"Service {service_name} depends on {dependency}, but it's not registered")
                visit(dependency)

            visiting.remove(service_name)
            visited.add(service_name)
            order.append(service_name)

        # Visit all services
        for service_name in self.services:
            visit(service_name)

        self.startup_order = order
        return order

    def calculate_shutdown_order(self) -> list[str]:
        """Calculate service shutdown order (reverse of startup order)."""
        if not self.startup_order:
            self.calculate_startup_order()

        self.shutdown_order = list(reversed(self.startup_order))
        return self.shutdown_order

    async def start_all_services(self) -> dict[str, bool]:
        """Start all services in dependency order."""
        logger.info("Starting all services...")

        startup_order = self.calculate_startup_order()
        results = {}

        for service_name in startup_order:
            service = self.services[service_name]

            try:
                # Wait for dependencies to be running
                await self._wait_for_dependencies(service_name)

                # Start the service with timeout
                start_task = asyncio.create_task(service.start())
                success = await asyncio.wait_for(start_task, timeout=self.startup_timeout)
                results[service_name] = success

                logger.info(f"Service {service_name} started successfully")

            except TimeoutError:
                logger.error(f"Service {service_name} startup timed out after {self.startup_timeout}s")
                results[service_name] = False
                break  # Stop starting remaining services

            except Exception as e:
                logger.error(f"Failed to start service {service_name}: {e}")
                results[service_name] = False
                break  # Stop starting remaining services

        return results

    async def stop_all_services(self) -> dict[str, bool]:
        """Stop all services in reverse dependency order."""
        logger.info("Stopping all services...")

        shutdown_order = self.calculate_shutdown_order()
        results = {}

        for service_name in shutdown_order:
            service = self.services[service_name]

            if service.status not in [ServiceStatus.RUNNING, ServiceStatus.FAILED]:
                results[service_name] = True
                continue

            try:
                # Stop the service with timeout
                stop_task = asyncio.create_task(service.stop())
                success = await asyncio.wait_for(stop_task, timeout=self.shutdown_timeout)
                results[service_name] = success

                logger.info(f"Service {service_name} stopped successfully")

            except TimeoutError:
                logger.error(f"Service {service_name} shutdown timed out after {self.shutdown_timeout}s")
                results[service_name] = False
                # Continue stopping other services even if one fails

            except Exception as e:
                logger.error(f"Failed to stop service {service_name}: {e}")
                results[service_name] = False
                # Continue stopping other services even if one fails

        return results

    async def restart_service(self, service_name: str) -> bool:
        """Restart a specific service and its dependents."""
        if service_name not in self.services:
            raise ValueError(f"Service {service_name} is not registered")

        # Find services that depend on this one
        dependents = [name for name, deps in self.dependency_graph.items() if service_name in deps]

        # Stop dependents first
        for dependent in dependents:
            if self.services[dependent].status == ServiceStatus.RUNNING:
                await self.services[dependent].stop()

        # Restart the target service
        success = await self.services[service_name].restart()

        # Start dependents again
        if success:
            for dependent in dependents:
                await self.services[dependent].start()

        return success

    async def health_check_all(self) -> dict[str, dict[str, Any]]:
        """Perform health check on all services."""
        health_results = {}

        for service_name, service in self.services.items():
            health_results[service_name] = await service.health_check()

        return health_results

    async def _wait_for_dependencies(self, service_name: str, timeout: float = 5.0):
        """Wait for service dependencies to be running."""
        dependencies = self.dependency_graph.get(service_name, set())

        if not dependencies:
            return

        start_time = time.time()

        while time.time() - start_time < timeout:
            all_ready = True

            for dependency in dependencies:
                if self.services[dependency].status != ServiceStatus.RUNNING:
                    all_ready = False
                    break

            if all_ready:
                return

            await asyncio.sleep(0.1)  # Check every 100ms

        # Raise timeout error
        unready_deps = [dep for dep in dependencies if self.services[dep].status != ServiceStatus.RUNNING]
        raise TimeoutError(f"Dependencies not ready for {service_name}: {unready_deps}")


@pytest.fixture
def service_manager():
    """Provide service manager for testing."""
    return ServiceManager()


@pytest.fixture
def sample_services():
    """Provide sample services with dependencies."""
    # Create services with dependency chain: database -> cache -> message_queue -> analysis -> api
    return {
        "database": MockService("database", startup_time=0.2, shutdown_time=0.1),
        "cache": MockService("cache", dependencies=["database"], startup_time=0.15, shutdown_time=0.08),
        "message_queue": MockService("message_queue", dependencies=["database"], startup_time=0.18, shutdown_time=0.12),
        "analysis": MockService(
            "analysis", dependencies=["database", "cache", "message_queue"], startup_time=0.25, shutdown_time=0.1
        ),
        "api": MockService("api", dependencies=["analysis", "cache"], startup_time=0.1, shutdown_time=0.05),
    }


class TestServiceLifecycle:
    """Test individual service lifecycle operations."""

    @pytest.mark.asyncio
    async def test_service_startup_success(self):
        """Test successful service startup."""
        service = MockService("test_service", startup_time=0.05)

        assert service.status == ServiceStatus.STOPPED
        assert service.start_time is None

        # Start service
        success = await service.start()

        assert success is True
        assert service.status == ServiceStatus.RUNNING
        assert service.start_time is not None
        assert service.initialization_data["database_initialized"] is True
        assert service.database_connection is not None
        assert service.message_queue_connection is not None
        assert service.cache_connection is not None

    @pytest.mark.asyncio
    async def test_service_startup_failure(self):
        """Test service startup failure."""
        service = MockService("test_service", failure_mode="startup_failure")

        with pytest.raises(Exception, match="Simulated startup failure"):
            await service.start()

        assert service.status == ServiceStatus.FAILED
        assert service.start_time is None

    @pytest.mark.asyncio
    async def test_service_shutdown_success(self):
        """Test successful service shutdown."""
        service = MockService("test_service", shutdown_time=0.03)

        # Start service first
        await service.start()
        assert service.status == ServiceStatus.RUNNING

        # Stop service
        success = await service.stop()

        assert success is True
        assert service.status == ServiceStatus.STOPPED
        assert service.stop_time is not None
        assert service.cleanup_data["database_cleaned"] is True
        assert service.database_connection is None
        assert service.message_queue_connection is None
        assert service.cache_connection is None

    @pytest.mark.asyncio
    async def test_service_shutdown_failure(self):
        """Test service shutdown failure."""
        service = MockService("test_service", failure_mode="shutdown_failure")

        # Start service first
        await service.start()

        with pytest.raises(Exception, match="Simulated shutdown failure"):
            await service.stop()

    @pytest.mark.asyncio
    async def test_service_restart(self):
        """Test service restart functionality."""
        service = MockService("test_service")

        # Start service
        await service.start()
        initial_start_time = service.start_time
        assert service.restart_count == 0

        # Restart service
        success = await service.restart()

        assert success is True
        assert service.status == ServiceStatus.RUNNING
        assert service.restart_count == 1
        assert service.start_time > initial_start_time

    @pytest.mark.asyncio
    async def test_service_health_check_healthy(self):
        """Test health check for healthy service."""
        service = MockService("test_service")
        await service.start()

        health = await service.health_check()

        assert health["status"] == "healthy"
        assert health["service"] == "test_service"
        assert health["uptime_seconds"] > 0
        assert health["health_checks"] == 1
        assert health["restarts"] == 0

        # Second health check
        health2 = await service.health_check()
        assert health2["health_checks"] == 2

    @pytest.mark.asyncio
    async def test_service_health_check_unhealthy(self):
        """Test health check for unhealthy service."""
        service = MockService("test_service", failure_mode="health_check_failure")
        await service.start()

        health = await service.health_check()

        assert health["status"] == "unhealthy"
        assert health["service"] == "test_service"
        assert "error" in health


class TestServiceManager:
    """Test service manager functionality."""

    @pytest.mark.asyncio
    async def test_dependency_order_calculation(self, service_manager, sample_services):
        """Test calculation of service startup/shutdown order."""
        # Register services
        for service in sample_services.values():
            service_manager.register_service(service)

        # Calculate startup order
        startup_order = service_manager.calculate_startup_order()

        # Verify dependency order
        assert "database" in startup_order
        assert "cache" in startup_order
        assert "analysis" in startup_order

        # Database should come before cache
        db_index = startup_order.index("database")
        cache_index = startup_order.index("cache")
        assert db_index < cache_index

        # Cache should come before analysis
        analysis_index = startup_order.index("analysis")
        assert cache_index < analysis_index

        # Calculate shutdown order (should be reverse)
        shutdown_order = service_manager.calculate_shutdown_order()
        assert shutdown_order == list(reversed(startup_order))

    @pytest.mark.asyncio
    async def test_circular_dependency_detection(self, service_manager):
        """Test detection of circular dependencies."""
        # Create circular dependency: service_a -> service_b -> service_a
        service_a = MockService("service_a", dependencies=["service_b"])
        service_b = MockService("service_b", dependencies=["service_a"])

        service_manager.register_service(service_a)
        service_manager.register_service(service_b)

        with pytest.raises(ValueError, match="Circular dependency detected"):
            service_manager.calculate_startup_order()

    @pytest.mark.asyncio
    async def test_missing_dependency_detection(self, service_manager):
        """Test detection of missing dependencies."""
        service = MockService("test_service", dependencies=["nonexistent_service"])
        service_manager.register_service(service)

        with pytest.raises(ValueError, match="depends on nonexistent_service, but it's not registered"):
            service_manager.calculate_startup_order()

    @pytest.mark.asyncio
    async def test_successful_service_startup_sequence(self, service_manager, sample_services):
        """Test successful startup of all services in correct order."""
        # Register services
        for service in sample_services.values():
            service_manager.register_service(service)

        # Start all services
        results = await service_manager.start_all_services()

        # Verify all services started successfully
        assert all(results.values())
        assert len(results) == len(sample_services)

        # Verify all services are running
        for service in sample_services.values():
            assert service.status == ServiceStatus.RUNNING
            assert service.start_time is not None

    @pytest.mark.asyncio
    async def test_successful_service_shutdown_sequence(self, service_manager, sample_services):
        """Test successful shutdown of all services in correct order."""
        # Register and start services
        for service in sample_services.values():
            service_manager.register_service(service)

        await service_manager.start_all_services()

        # Stop all services
        results = await service_manager.stop_all_services()

        # Verify all services stopped successfully
        assert all(results.values())
        assert len(results) == len(sample_services)

        # Verify all services are stopped
        for service in sample_services.values():
            assert service.status == ServiceStatus.STOPPED
            assert service.stop_time is not None

    @pytest.mark.asyncio
    async def test_startup_failure_stops_sequence(self, service_manager):
        """Test that startup failure stops the startup sequence."""
        # Create services where one will fail
        good_service = MockService("good_service")
        failing_service = MockService("failing_service", failure_mode="startup_failure")
        dependent_service = MockService("dependent_service", dependencies=["failing_service"])

        service_manager.register_service(good_service)
        service_manager.register_service(failing_service)
        service_manager.register_service(dependent_service)

        # Attempt to start all services
        results = await service_manager.start_all_services()

        # Good service should start, failing service should fail, dependent should not start
        assert results["good_service"] is True
        assert results["failing_service"] is False
        assert "dependent_service" not in results  # Should not be attempted

        assert good_service.status == ServiceStatus.RUNNING
        assert failing_service.status == ServiceStatus.FAILED
        assert dependent_service.status == ServiceStatus.STOPPED

    @pytest.mark.asyncio
    async def test_service_restart_with_dependents(self, service_manager, sample_services):
        """Test restarting a service and its dependents."""
        # Register and start all services
        for service in sample_services.values():
            service_manager.register_service(service)

        await service_manager.start_all_services()

        # Record initial start times
        initial_times = {name: service.start_time for name, service in sample_services.items()}

        # Restart cache service (database depends on it, analysis and api depend on cache)
        success = await service_manager.restart_service("cache")

        assert success is True
        assert sample_services["cache"].restart_count == 1

        # Cache should have a new start time
        assert sample_services["cache"].start_time > initial_times["cache"]

        # Database should not be affected (cache depends on database, not vice versa)
        assert sample_services["database"].start_time == initial_times["database"]

    @pytest.mark.asyncio
    async def test_health_check_all_services(self, service_manager, sample_services):
        """Test health check across all services."""
        # Register and start services
        for service in sample_services.values():
            service_manager.register_service(service)

        await service_manager.start_all_services()

        # Perform health check on all services
        health_results = await service_manager.health_check_all()

        # Verify health check results
        assert len(health_results) == len(sample_services)

        for service_name, health in health_results.items():
            assert health["service"] == service_name
            assert health["status"] == "healthy"
            assert health["uptime_seconds"] > 0
            assert health["health_checks"] == 1


class TestServiceLifecycleEdgeCases:
    """Test edge cases and error conditions in service lifecycle."""

    @pytest.mark.asyncio
    async def test_startup_timeout(self, service_manager):
        """Test service startup timeout handling."""
        slow_service = MockService("slow_service", failure_mode="startup_timeout")
        service_manager.register_service(slow_service)
        service_manager.startup_timeout = 1.0  # Short timeout

        results = await service_manager.start_all_services()

        assert results["slow_service"] is False
        # Service may be in STARTING state if timeout occurred
        assert slow_service.status in [ServiceStatus.STARTING, ServiceStatus.FAILED]

    @pytest.mark.asyncio
    async def test_graceful_shutdown_on_startup_failure(self, service_manager):
        """Test graceful shutdown when startup fails partway through."""
        good_service = MockService("good_service")
        failing_service = MockService("failing_service", failure_mode="startup_failure")

        service_manager.register_service(good_service)
        service_manager.register_service(failing_service)

        # Startup will fail
        startup_results = await service_manager.start_all_services()

        # Good service started, failing service failed
        assert startup_results["good_service"] is True
        assert startup_results["failing_service"] is False

        # Now shutdown all services
        shutdown_results = await service_manager.stop_all_services()

        # Good service should shut down cleanly
        assert shutdown_results["good_service"] is True
        assert good_service.status == ServiceStatus.STOPPED

    @pytest.mark.asyncio
    async def test_dependency_wait_timeout(self, service_manager):
        """Test timeout when waiting for dependencies."""
        slow_dependency = MockService("slow_dependency", startup_time=2.0)
        dependent_service = MockService("dependent_service", dependencies=["slow_dependency"])

        service_manager.register_service(slow_dependency)
        service_manager.register_service(dependent_service)

        # Start dependency in background (will be slow)
        dependency_task = asyncio.create_task(slow_dependency.start())

        # Try to start dependent service (should timeout waiting for dependency)
        with pytest.raises(TimeoutError, match="Dependencies not ready"):
            await service_manager._wait_for_dependencies("dependent_service", timeout=0.5)

        # Clean up
        dependency_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await dependency_task

    @pytest.mark.asyncio
    async def test_double_start_prevention(self):
        """Test prevention of starting an already running service."""
        service = MockService("test_service")
        await service.start()

        with pytest.raises(RuntimeError, match="Cannot start service test_service in status"):
            await service.start()

    @pytest.mark.asyncio
    async def test_stop_already_stopped_service(self):
        """Test stopping an already stopped service."""
        service = MockService("test_service")

        # Should not raise an error
        success = await service.stop()
        assert success is True
        assert service.status == ServiceStatus.STOPPED


class TestServiceLifecyclePerformance:
    """Test performance characteristics of service lifecycle."""

    @pytest.mark.asyncio
    async def test_parallel_independent_service_startup(self, service_manager):
        """Test that independent services start in parallel."""
        # Create multiple independent services
        services = []
        for i in range(5):
            service = MockService(f"service_{i}", startup_time=0.1)
            services.append(service)
            service_manager.register_service(service)

        # Time the startup process
        start_time = time.time()
        results = await service_manager.start_all_services()
        total_time = time.time() - start_time

        # Verify all started successfully
        assert all(results.values())

        # Should be much faster than sequential (5 * 0.1 = 0.5s)
        # Allow some overhead, but should be significantly faster
        assert total_time < 0.3  # Should be close to 0.1s with parallel execution

    @pytest.mark.asyncio
    async def test_sequential_dependent_service_startup(self, service_manager):
        """Test that dependent services start sequentially."""
        # Create chain of dependent services: service_0 -> service_1 -> service_2
        services = []
        for i in range(3):
            dependencies = [f"service_{i - 1}"] if i > 0 else []
            service = MockService(f"service_{i}", dependencies=dependencies, startup_time=0.05)
            services.append(service)
            service_manager.register_service(service)

        # Time the startup process
        start_time = time.time()
        results = await service_manager.start_all_services()
        total_time = time.time() - start_time

        # Verify all started successfully
        assert all(results.values())

        # Should take approximately 3 * 0.05 = 0.15s due to sequential dependencies
        assert 0.1 < total_time < 0.25  # Allow some overhead

    @pytest.mark.asyncio
    async def test_large_service_mesh_startup(self, service_manager):
        """Test startup performance with a large number of services."""
        # Create a larger service mesh (20 services)
        base_services = []

        # Create 5 base services (no dependencies)
        for i in range(5):
            service = MockService(f"base_{i}", startup_time=0.02)
            base_services.append(service)
            service_manager.register_service(service)

        # Create 15 dependent services (each depends on 1-2 base services)
        for i in range(15):
            deps = [f"base_{i % 5}"]
            if i % 3 == 0:  # Some services depend on multiple base services
                deps.append(f"base_{(i + 1) % 5}")

            service = MockService(f"dependent_{i}", dependencies=deps, startup_time=0.01)
            service_manager.register_service(service)

        # Time the startup process
        start_time = time.time()
        results = await service_manager.start_all_services()
        total_time = time.time() - start_time

        # Verify all 20 services started successfully
        assert len(results) == 20
        assert all(results.values())

        # Should complete reasonably quickly despite the large number of services
        assert total_time < 2.0  # Allow generous overhead for 20 services

        logger.info(f"Started 20 services in {total_time:.3f}s")
