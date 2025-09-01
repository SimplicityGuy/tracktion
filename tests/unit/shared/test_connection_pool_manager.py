"""Unit tests for connection pool manager."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import httpx
import pytest

from shared.utils.connection_pool_manager import (
    ConnectionPoolManager,
    PoolStatistics,
    cleanup_global_pool_manager,
    get_global_pool_manager,
)


@pytest.fixture
async def pool_manager():
    """Create test connection pool manager."""
    manager = ConnectionPoolManager(
        min_connections=2,
        max_connections=5,
        keepalive_timeout=10.0,
        connection_ttl=60.0,
        health_check_interval=1.0,
        health_check_timeout=0.5,
    )
    yield manager
    await manager.close_all_pools()


class TestPoolStatistics:
    """Test pool statistics data class."""

    def test_default_statistics(self):
        """Test default statistics values."""
        stats = PoolStatistics()
        assert stats.active_connections == 0
        assert stats.idle_connections == 0
        assert stats.total_connections == 0
        assert stats.requests_served == 0
        assert stats.average_wait_time == 0.0
        assert stats.health_check_failures == 0
        assert stats.last_health_check is None
        assert stats.pool_efficiency == 0.0

    def test_statistics_with_values(self):
        """Test statistics with custom values."""
        stats = PoolStatistics(
            active_connections=5,
            idle_connections=3,
            total_connections=8,
            requests_served=100,
            average_wait_time=0.5,
        )
        assert stats.active_connections == 5
        assert stats.idle_connections == 3
        assert stats.total_connections == 8
        assert stats.requests_served == 100
        assert stats.average_wait_time == 0.5


class TestConnectionPoolManager:
    """Test connection pool manager."""

    def test_create_pool(self, pool_manager):
        """Test creating a connection pool."""
        client = pool_manager.create_pool("test_service", "https://example.com")
        assert isinstance(client, httpx.AsyncClient)
        assert client.base_url == "https://example.com"
        assert "test_service" in pool_manager._pools
        assert "test_service" in pool_manager._pool_stats

        # Creating pool again returns same client
        client2 = pool_manager.create_pool("test_service")
        assert client is client2

    def test_create_multiple_pools(self, pool_manager):
        """Test creating multiple connection pools."""
        client1 = pool_manager.create_pool("service1", "https://service1.com")
        client2 = pool_manager.create_pool("service2", "https://service2.com")

        assert client1 is not client2
        assert len(pool_manager._pools) == 2
        assert "service1" in pool_manager._pools
        assert "service2" in pool_manager._pools

    @pytest.mark.asyncio
    async def test_health_check_success(self, pool_manager):
        """Test successful health check."""
        client = pool_manager.create_pool("test_service", "https://example.com")

        with patch.object(client, "head") as mock_head:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_head.return_value = asyncio.create_task(asyncio.coroutine(lambda: mock_response)())

            result = await pool_manager._perform_health_check("test_service")
            assert result is True

            stats = pool_manager._pool_stats["test_service"]
            assert stats.last_health_check is not None
            assert stats.health_check_failures == 0

    @pytest.mark.asyncio
    async def test_health_check_failure(self, pool_manager):
        """Test failed health check."""
        client = pool_manager.create_pool("test_service", "https://example.com")

        with patch.object(client, "head") as mock_head:
            mock_response = MagicMock()
            mock_response.status_code = 503
            mock_head.return_value = asyncio.create_task(asyncio.coroutine(lambda: mock_response)())

            result = await pool_manager._perform_health_check("test_service")
            assert result is False

            stats = pool_manager._pool_stats["test_service"]
            assert stats.health_check_failures == 1

    @pytest.mark.asyncio
    async def test_health_check_error(self, pool_manager):
        """Test health check with error."""
        client = pool_manager.create_pool("test_service", "https://example.com")

        with patch.object(client, "head") as mock_head:
            mock_head.side_effect = httpx.NetworkError("Network error")

            result = await pool_manager._perform_health_check("test_service")
            assert result is False

            stats = pool_manager._pool_stats["test_service"]
            assert stats.health_check_failures == 1

    @pytest.mark.asyncio
    async def test_cleanup_stale_connections(self, pool_manager):
        """Test cleanup of stale connections."""
        pool_manager.create_pool("test_service")

        # Add some connection timestamps
        current_time = time.time()
        pool_manager._connection_timestamps["test_service"] = {
            1: current_time - 300,  # 5 minutes old (stale)
            2: current_time - 400,  # Over 6 minutes old (stale)
            3: current_time - 30,  # 30 seconds old (fresh)
        }

        await pool_manager._cleanup_stale_connections("test_service")

        # Only fresh connection should remain
        timestamps = pool_manager._connection_timestamps["test_service"]
        assert len(timestamps) == 1
        assert 3 in timestamps
        assert 1 not in timestamps
        assert 2 not in timestamps

    def test_get_statistics(self, pool_manager):
        """Test getting pool statistics."""
        pool_manager.create_pool("test_service")

        # Add some test data
        pool_manager._request_queue_times["test_service"] = [0.1, 0.2, 0.3, 0.4, 0.5]
        pool_manager._connection_timestamps["test_service"] = {
            1: time.time(),
            2: time.time(),
        }

        stats = pool_manager.get_statistics("test_service")
        assert stats.average_wait_time == 0.3  # Average of [0.1, 0.2, 0.3, 0.4, 0.5]
        assert stats.total_connections == 2

    @pytest.mark.asyncio
    async def test_warmup_pool(self, pool_manager):
        """Test pool warmup."""
        client = pool_manager.create_pool("test_service", "https://example.com")

        with patch.object(client, "head") as mock_head:
            mock_response = MagicMock()
            mock_head.return_value = asyncio.create_task(asyncio.coroutine(lambda: mock_response)())

            await pool_manager.warmup_pool("test_service", num_connections=3)
            assert mock_head.call_count == 2  # min(3, min_connections=2)

    @pytest.mark.asyncio
    async def test_warmup_pool_no_base_url(self, pool_manager):
        """Test pool warmup without base URL."""
        pool_manager.create_pool("test_service")  # No base URL
        await pool_manager.warmup_pool("test_service", num_connections=3)
        # Should complete without error

    @pytest.mark.asyncio
    async def test_request_with_tracking(self, pool_manager):
        """Test making request with tracking."""
        client = pool_manager.create_pool("test_service")

        with patch.object(client, "request") as mock_request:
            mock_response = MagicMock()
            mock_request.return_value = asyncio.create_task(asyncio.coroutine(lambda: mock_response)())

            stats_before = pool_manager._pool_stats["test_service"]
            requests_before = stats_before.requests_served

            response = await pool_manager.request_with_tracking(
                "test_service",
                "GET",
                "https://example.com/api",
            )

            assert response == mock_response
            mock_request.assert_called_once_with("GET", "https://example.com/api")

            stats_after = pool_manager._pool_stats["test_service"]
            assert stats_after.requests_served == requests_before + 1

    @pytest.mark.asyncio
    async def test_request_with_tracking_no_pool(self, pool_manager):
        """Test request with tracking when pool doesn't exist."""
        with pytest.raises(ValueError, match="No pool found for service"):
            await pool_manager.request_with_tracking(
                "nonexistent_service",
                "GET",
                "https://example.com",
            )

    @pytest.mark.asyncio
    async def test_close_pool(self, pool_manager):
        """Test closing a specific pool."""
        pool_manager.create_pool("test_service")

        # Start health check task
        pool_manager._start_health_check("test_service")
        assert "test_service" in pool_manager._health_check_tasks

        await pool_manager.close_pool("test_service")

        assert "test_service" not in pool_manager._pools
        assert "test_service" not in pool_manager._pool_stats
        assert "test_service" not in pool_manager._health_check_tasks

    @pytest.mark.asyncio
    async def test_close_all_pools(self, pool_manager):
        """Test closing all pools."""
        pool_manager.create_pool("service1")
        pool_manager.create_pool("service2")
        pool_manager.create_pool("service3")

        assert len(pool_manager._pools) == 3

        await pool_manager.close_all_pools()

        assert len(pool_manager._pools) == 0
        assert len(pool_manager._pool_stats) == 0
        assert len(pool_manager._health_check_tasks) == 0


class TestGlobalPoolManager:
    """Test global pool manager."""

    @pytest.mark.asyncio
    async def test_get_global_pool_manager(self):
        """Test getting global pool manager instance."""
        manager1 = get_global_pool_manager()
        manager2 = get_global_pool_manager()
        assert manager1 is manager2

        await cleanup_global_pool_manager()

    @pytest.mark.asyncio
    async def test_cleanup_global_pool_manager(self):
        """Test cleaning up global pool manager."""
        manager = get_global_pool_manager()
        manager.create_pool("test_service")

        await cleanup_global_pool_manager()

        # After cleanup, should create new instance
        manager2 = get_global_pool_manager()
        assert manager2 is not manager

        await cleanup_global_pool_manager()


class TestPoolEfficiency:
    """Test pool efficiency calculations."""

    def test_pool_efficiency_calculation(self, pool_manager):
        """Test pool efficiency calculation."""
        pool_manager.create_pool("test_service")
        stats = pool_manager._pool_stats["test_service"]

        # No connections
        stats.total_connections = 0
        stats.active_connections = 0
        calculated_stats = pool_manager.get_statistics("test_service")
        assert calculated_stats.pool_efficiency == 0.0

        # Half active
        stats.total_connections = 10
        stats.active_connections = 5
        pool_manager._connection_timestamps["test_service"] = {i: time.time() for i in range(10)}
        calculated_stats = pool_manager.get_statistics("test_service")
        assert calculated_stats.pool_efficiency == 50.0

        # All active
        stats.active_connections = 10
        calculated_stats = pool_manager.get_statistics("test_service")
        assert calculated_stats.pool_efficiency == 100.0
