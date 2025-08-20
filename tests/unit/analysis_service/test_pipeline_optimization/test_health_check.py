"""
Unit tests for health check endpoints.
"""

import time
from unittest.mock import Mock, patch

from services.analysis_service.src.health_check import (
    ComponentHealth,
    HealthChecker,
    HealthCheckServer,
    HealthStatus,
    check_neo4j_health,
    check_postgresql_health,
    check_rabbitmq_health,
    check_redis_health,
    get_health_checker,
    reset_health_checker,
)


class TestComponentHealth:
    """Tests for ComponentHealth class."""

    def test_component_health_creation(self) -> None:
        """Test creating a ComponentHealth instance."""
        health = ComponentHealth(
            name="test_component",
            status=HealthStatus.HEALTHY,
            message="All good",
            latency_ms=10.5,
            metadata={"version": "1.0"},
        )

        assert health.name == "test_component"
        assert health.status == HealthStatus.HEALTHY
        assert health.message == "All good"
        assert health.latency_ms == 10.5
        assert health.metadata == {"version": "1.0"}

    def test_component_health_to_dict(self) -> None:
        """Test converting ComponentHealth to dictionary."""
        health = ComponentHealth(
            name="test_component",
            status=HealthStatus.DEGRADED,
            message="Slow response",
            latency_ms=500.0,
            metadata={"connections": 5},
            last_check=1234567890.0,
        )

        result = health.to_dict()

        assert result["name"] == "test_component"
        assert result["status"] == "degraded"
        assert result["message"] == "Slow response"
        assert result["latency_ms"] == 500.0
        assert result["metadata"] == {"connections": 5}
        assert result["last_check_timestamp"] == 1234567890.0

    def test_component_health_to_dict_minimal(self) -> None:
        """Test converting minimal ComponentHealth to dictionary."""
        health = ComponentHealth(
            name="minimal",
            status=HealthStatus.UNHEALTHY,
        )

        result = health.to_dict()

        assert result["name"] == "minimal"
        assert result["status"] == "unhealthy"
        assert result["message"] == ""
        assert "latency_ms" not in result
        assert "metadata" not in result
        assert "last_check_timestamp" not in result


class TestHealthChecker:
    """Tests for HealthChecker class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.checker = HealthChecker(
            service_name="test_service",
            check_timeout=2.0,
        )

    def test_initialization(self) -> None:
        """Test health checker initialization."""
        assert self.checker.service_name == "test_service"
        assert self.checker.check_timeout == 2.0
        assert len(self.checker._health_checks) == 0
        assert self.checker._check_count == 0

    def test_register_health_check(self) -> None:
        """Test registering a health check."""

        def test_check() -> ComponentHealth:
            return ComponentHealth("test", HealthStatus.HEALTHY)

        self.checker.register_health_check("test_component", test_check)

        assert "test_component" in self.checker._health_checks
        assert self.checker._health_checks["test_component"] == test_check

    def test_check_health_basic(self) -> None:
        """Test basic health check (liveness)."""
        result = self.checker.check_health()

        assert result["service"] == "test_service"
        assert result["status"] == "healthy"
        assert "uptime_seconds" in result
        assert result["checks_performed"] == 1
        assert "timestamp" in result
        assert "check_latency_ms" in result

    def test_check_health_increments_counter(self) -> None:
        """Test that health check increments counter."""
        self.checker.check_health()
        self.checker.check_health()
        result = self.checker.check_health()

        assert result["checks_performed"] == 3

    def test_check_readiness_no_components(self) -> None:
        """Test readiness check with no components."""
        result = self.checker.check_readiness()

        assert result["service"] == "test_service"
        assert result["status"] == "healthy"
        assert result["ready"] is True
        assert result["components"] == []
        assert "timestamp" in result
        assert "check_latency_ms" in result

    def test_check_readiness_healthy_components(self) -> None:
        """Test readiness check with healthy components."""

        def healthy_check() -> ComponentHealth:
            return ComponentHealth("db", HealthStatus.HEALTHY, "Connected")

        def cache_check() -> ComponentHealth:
            return ComponentHealth("cache", HealthStatus.HEALTHY, "Ready")

        self.checker.register_health_check("database", healthy_check)
        self.checker.register_health_check("cache", cache_check)

        result = self.checker.check_readiness()

        assert result["status"] == "healthy"
        assert result["ready"] is True
        assert len(result["components"]) == 2

        # Check component details
        db_component = next(c for c in result["components"] if c["name"] == "db")
        assert db_component["status"] == "healthy"
        assert db_component["message"] == "Connected"

    def test_check_readiness_degraded_component(self) -> None:
        """Test readiness with degraded component."""

        def healthy_check() -> ComponentHealth:
            return ComponentHealth("db", HealthStatus.HEALTHY)

        def degraded_check() -> ComponentHealth:
            return ComponentHealth("cache", HealthStatus.DEGRADED, "High latency")

        self.checker.register_health_check("database", healthy_check)
        self.checker.register_health_check("cache", degraded_check)

        result = self.checker.check_readiness()

        assert result["status"] == "degraded"
        assert result["ready"] is True  # Still ready, just degraded

    def test_check_readiness_unhealthy_component(self) -> None:
        """Test readiness with unhealthy component."""

        def healthy_check() -> ComponentHealth:
            return ComponentHealth("db", HealthStatus.HEALTHY)

        def unhealthy_check() -> ComponentHealth:
            return ComponentHealth("queue", HealthStatus.UNHEALTHY, "Connection failed")

        self.checker.register_health_check("database", healthy_check)
        self.checker.register_health_check("queue", unhealthy_check)

        result = self.checker.check_readiness()

        assert result["status"] == "unhealthy"
        assert result["ready"] is False

    def test_check_readiness_component_exception(self) -> None:
        """Test readiness when component check raises exception."""

        def failing_check() -> ComponentHealth:
            raise RuntimeError("Connection error")

        self.checker.register_health_check("failing", failing_check)

        result = self.checker.check_readiness()

        assert result["status"] == "unhealthy"
        assert result["ready"] is False
        assert len(result["components"]) == 1

        component = result["components"][0]
        assert component["name"] == "failing"
        assert component["status"] == "unhealthy"
        assert "Check failed" in component["message"]

    def test_cache_functionality(self) -> None:
        """Test that health check results are cached."""
        call_count = 0

        def counting_check() -> ComponentHealth:
            nonlocal call_count
            call_count += 1
            return ComponentHealth("counter", HealthStatus.HEALTHY)

        self.checker.register_health_check("counter", counting_check)

        # First call should execute check
        self.checker.check_readiness(use_cache=False)
        assert call_count == 1

        # Second call with cache should not execute check
        self.checker.check_readiness(use_cache=True)
        assert call_count == 1  # Not incremented

        # Without cache should execute again
        self.checker.check_readiness(use_cache=False)
        assert call_count == 2

    def test_cache_expiry(self) -> None:
        """Test that cached results expire."""
        call_count = 0

        def counting_check() -> ComponentHealth:
            nonlocal call_count
            call_count += 1
            return ComponentHealth("counter", HealthStatus.HEALTHY)

        self.checker.register_health_check("counter", counting_check)
        self.checker._cache_ttl = 0.1  # Short TTL for testing

        # First call
        self.checker.check_readiness(use_cache=True)
        assert call_count == 1

        # Immediate second call uses cache
        self.checker.check_readiness(use_cache=True)
        assert call_count == 1

        # Wait for cache to expire
        time.sleep(0.15)

        # Should execute check again
        self.checker.check_readiness(use_cache=True)
        assert call_count == 2

    def test_get_detailed_status(self) -> None:
        """Test getting detailed status."""

        def test_check() -> ComponentHealth:
            return ComponentHealth("test", HealthStatus.HEALTHY)

        self.checker.register_health_check("test", test_check)

        result = self.checker.get_detailed_status()

        assert "health" in result
        assert "readiness" in result
        assert "cache_size" in result
        assert "registered_checks" in result
        assert result["registered_checks"] == ["test"]

    def test_run_health_check_with_timing(self) -> None:
        """Test that health check adds timing information."""

        def slow_check() -> ComponentHealth:
            time.sleep(0.05)
            return ComponentHealth("slow", HealthStatus.HEALTHY)

        result = self.checker._run_health_check("slow", slow_check)

        assert result.latency_ms is not None
        assert result.latency_ms >= 50  # At least 50ms
        assert result.last_check is not None

    def test_run_health_check_exception_handling(self) -> None:
        """Test health check exception handling."""

        def failing_check() -> ComponentHealth:
            raise ValueError("Test error")

        result = self.checker._run_health_check("failing", failing_check)

        assert result.status == HealthStatus.UNHEALTHY
        assert "Check failed" in result.message
        assert result.latency_ms is not None


class TestHealthCheckFunctions:
    """Tests for individual health check functions."""

    def test_check_postgresql_health(self) -> None:
        """Test PostgreSQL health check."""
        # In the actual implementation, this would mock the database connection
        result = check_postgresql_health()

        assert result.name == "postgresql"
        # Since it's a placeholder, it should return healthy
        assert result.status == HealthStatus.HEALTHY
        assert "metadata" in result.to_dict()

    def test_check_neo4j_health(self) -> None:
        """Test Neo4j health check."""
        result = check_neo4j_health()

        assert result.name == "neo4j"
        assert result.status == HealthStatus.HEALTHY
        assert "metadata" in result.to_dict()

    def test_check_rabbitmq_health(self) -> None:
        """Test RabbitMQ health check."""
        result = check_rabbitmq_health()

        assert result.name == "rabbitmq"
        assert result.status == HealthStatus.HEALTHY
        assert "metadata" in result.to_dict()

    def test_check_redis_health(self) -> None:
        """Test Redis health check."""
        result = check_redis_health()

        assert result.name == "redis"
        assert result.status == HealthStatus.HEALTHY
        assert "metadata" in result.to_dict()


class TestHealthCheckServer:
    """Tests for HealthCheckServer class."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.checker = HealthChecker("test_service")
        self.server = HealthCheckServer(self.checker, port=8081)

    def test_initialization(self) -> None:
        """Test server initialization."""
        assert self.server.health_checker == self.checker
        assert self.server.port == 8081
        assert self.server.app is not None

    def test_health_endpoint(self) -> None:
        """Test /health endpoint."""
        with self.server.app.test_client() as client:
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json
            assert data["service"] == "test_service"
            assert data["status"] == "healthy"

    def test_ready_endpoint_healthy(self) -> None:
        """Test /ready endpoint when healthy."""

        def healthy_check() -> ComponentHealth:
            return ComponentHealth("test", HealthStatus.HEALTHY)

        self.checker.register_health_check("test", healthy_check)

        with self.server.app.test_client() as client:
            response = client.get("/ready")

            assert response.status_code == 200
            data = response.json
            assert data["ready"] is True
            assert data["status"] == "healthy"

    def test_ready_endpoint_unhealthy(self) -> None:
        """Test /ready endpoint when unhealthy."""

        def unhealthy_check() -> ComponentHealth:
            return ComponentHealth("test", HealthStatus.UNHEALTHY)

        self.checker.register_health_check("test", unhealthy_check)

        with self.server.app.test_client() as client:
            response = client.get("/ready")

            assert response.status_code == 503
            data = response.json
            assert data["ready"] is False
            assert data["status"] == "unhealthy"

    def test_detailed_health_endpoint(self) -> None:
        """Test /health/detailed endpoint."""

        def test_check() -> ComponentHealth:
            return ComponentHealth("test", HealthStatus.HEALTHY)

        self.checker.register_health_check("test", test_check)

        with self.server.app.test_client() as client:
            response = client.get("/health/detailed")

            assert response.status_code == 200
            data = response.json
            assert "health" in data
            assert "readiness" in data
            assert "cache_size" in data
            assert "registered_checks" in data

    @patch("services.analysis_service.src.health_check.Flask.run")
    def test_run_server(self, mock_run: Mock) -> None:
        """Test running the health check server."""
        self.server.run(host="127.0.0.1")
        mock_run.assert_called_once_with(host="127.0.0.1", port=8081)


class TestGlobalHealthChecker:
    """Tests for global health checker functions."""

    def teardown_method(self) -> None:
        """Clean up after each test."""
        reset_health_checker()

    def test_get_health_checker_creates_instance(self) -> None:
        """Test that get_health_checker creates a singleton."""
        checker1 = get_health_checker()
        checker2 = get_health_checker()

        assert checker1 is not None
        assert checker1 is checker2

    def test_get_health_checker_registers_defaults(self) -> None:
        """Test that default health checks are registered."""
        checker = get_health_checker()

        # Should have default health checks registered
        assert "postgresql" in checker._health_checks
        assert "neo4j" in checker._health_checks
        assert "rabbitmq" in checker._health_checks
        assert "redis" in checker._health_checks

    def test_reset_health_checker(self) -> None:
        """Test resetting the global health checker."""
        checker1 = get_health_checker()
        reset_health_checker()
        checker2 = get_health_checker()

        assert checker1 is not checker2


class TestHealthCheckIntegration:
    """Integration tests for health check system."""

    def test_complete_health_check_flow(self) -> None:
        """Test complete health check flow."""
        checker = HealthChecker("integration_test")

        # Register multiple health checks
        def db_check() -> ComponentHealth:
            return ComponentHealth(
                "database",
                HealthStatus.HEALTHY,
                "Connected",
                metadata={"connections": 10},
            )

        def cache_check() -> ComponentHealth:
            return ComponentHealth(
                "cache",
                HealthStatus.DEGRADED,
                "High memory usage",
                metadata={"memory_mb": 900},
            )

        def queue_check() -> ComponentHealth:
            return ComponentHealth(
                "queue",
                HealthStatus.HEALTHY,
                "Queue empty",
                metadata={"depth": 0},
            )

        checker.register_health_check("db", db_check)
        checker.register_health_check("cache", cache_check)
        checker.register_health_check("queue", queue_check)

        # Check readiness
        readiness = checker.check_readiness()

        assert readiness["status"] == "degraded"  # Due to degraded cache
        assert readiness["ready"] is True
        assert len(readiness["components"]) == 3

        # Check health
        health = checker.check_health()
        assert health["status"] == "healthy"

        # Get detailed status
        detailed = checker.get_detailed_status()
        assert detailed["health"]["status"] == "healthy"
        assert detailed["readiness"]["status"] == "degraded"
        assert len(detailed["registered_checks"]) == 3

    def test_health_check_server_integration(self) -> None:
        """Test health check server with multiple components."""
        checker = HealthChecker("server_test")

        # Register health checks with different statuses
        checks = {
            "healthy": lambda: ComponentHealth("healthy", HealthStatus.HEALTHY),
            "degraded": lambda: ComponentHealth("degraded", HealthStatus.DEGRADED),
        }

        for name, check in checks.items():
            checker.register_health_check(name, check)

        server = HealthCheckServer(checker)

        with server.app.test_client() as client:
            # Test all endpoints
            health_response = client.get("/health")
            assert health_response.status_code == 200

            ready_response = client.get("/ready")
            assert ready_response.status_code == 200
            assert ready_response.json["status"] == "degraded"

            detailed_response = client.get("/health/detailed")
            assert detailed_response.status_code == 200
            assert len(detailed_response.json["registered_checks"]) == 2
