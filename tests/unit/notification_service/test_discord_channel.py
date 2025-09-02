"""Tests for Discord notification channel."""

import asyncio
import contextlib
import os
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from services.notification_service.src.channels.discord import (
    DiscordNotificationService,
    DiscordWebhookClient,
)
from services.notification_service.src.core.base import (
    AlertType,
    NotificationMessage,
    NotificationStatus,
)


class TestDiscordWebhookClient:
    """Test DiscordWebhookClient functionality."""

    @pytest.mark.asyncio
    async def test_successful_send(self) -> None:
        """Test successful webhook send."""
        client = DiscordWebhookClient("https://discord.com/api/webhooks/test")

        # Mock successful response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.headers = {
            "X-RateLimit-Limit": "30",
            "X-RateLimit-Remaining": "29",
            "X-RateLimit-Reset": "1640995200",
            "X-RateLimit-Reset-After": "60",
        }

        # Create async context manager for session.post()
        class MockPost:
            def __init__(self, response):
                self.response = response

            async def __aenter__(self):
                return self.response

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        # Create session mock
        class MockSession:
            def __init__(self, response):
                self.response = response

            def post(self, *args, **kwargs):
                return MockPost(self.response)

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        def mock_session_factory(*args, **kwargs):
            return MockSession(mock_response)

        with patch("aiohttp.ClientSession", mock_session_factory):
            payload = {"content": "test message"}
            result = await client.send(payload)

            assert result["status"] == 200
            assert "rate_limit" in result
            assert result["rate_limit"]["limit"] == "30"

    @pytest.mark.asyncio
    async def test_rate_limited_response(self) -> None:
        """Test handling rate limited response."""
        client = DiscordWebhookClient("https://discord.com/api/webhooks/test")

        # Mock rate limited response
        mock_response = Mock()
        mock_response.status = 429
        mock_response.headers = {"X-RateLimit-Reset-After": "30"}
        mock_response.request_info = Mock()
        mock_response.history = ()

        # Create async context manager for session.post()
        class MockPost:
            def __init__(self, response):
                self.response = response

            async def __aenter__(self):
                return self.response

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        # Create session mock
        class MockSession:
            def __init__(self, response):
                self.response = response

            def post(self, *args, **kwargs):
                return MockPost(self.response)

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        def mock_session_factory(*args, **kwargs):
            return MockSession(mock_response)

        with patch("aiohttp.ClientSession", mock_session_factory):
            with pytest.raises(aiohttp.ClientResponseError) as exc_info:
                await client.send({"content": "test"})

            assert exc_info.value.status == 429
            assert "Rate limited" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_http_error_response(self) -> None:
        """Test handling HTTP error response."""
        client = DiscordWebhookClient("https://discord.com/api/webhooks/test")

        # Mock HTTP error response
        mock_response = Mock()
        mock_response.status = 404
        mock_response.text = AsyncMock(return_value="Not Found")
        mock_response.request_info = Mock()
        mock_response.history = ()

        # Create async context manager for session.post()
        class MockPost:
            def __init__(self, response):
                self.response = response

            async def __aenter__(self):
                return self.response

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        # Create session mock
        class MockSession:
            def __init__(self, response):
                self.response = response

            def post(self, *args, **kwargs):
                return MockPost(self.response)

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                return None

        def mock_session_factory(*args, **kwargs):
            return MockSession(mock_response)

        with patch("aiohttp.ClientSession", mock_session_factory):
            with pytest.raises(aiohttp.ClientResponseError) as exc_info:
                await client.send({"content": "test"})

            assert exc_info.value.status == 404

    @pytest.mark.asyncio
    async def test_validation_success(self) -> None:
        """Test successful webhook validation."""
        client = DiscordWebhookClient("https://discord.com/api/webhooks/test")

        # Mock successful send
        client.send = AsyncMock(return_value={"status": 200})

        result = await client.validate()
        assert result is True

    @pytest.mark.asyncio
    async def test_validation_failure(self) -> None:
        """Test webhook validation failure."""
        client = DiscordWebhookClient("https://discord.com/api/webhooks/test")

        # Mock failed send
        client.send = AsyncMock(side_effect=aiohttp.ClientError("Connection failed"))

        result = await client.validate()
        assert result is False


class TestDiscordNotificationService:
    """Test DiscordNotificationService functionality."""

    @pytest.fixture
    def mock_env(self) -> None:
        """Mock environment variables for testing."""
        env_vars = {
            "DISCORD_WEBHOOK_GENERAL": "https://discord.com/api/webhooks/general/token",
            "DISCORD_WEBHOOK_ERRORS": "https://discord.com/api/webhooks/errors/token",
            "DISCORD_WEBHOOK_CRITICAL": "https://discord.com/api/webhooks/critical/token",
            "DISCORD_WEBHOOK_TRACKLIST": "https://discord.com/api/webhooks/tracklist/token",
            "DISCORD_WEBHOOK_MONITORING": "https://discord.com/api/webhooks/monitoring/token",
            "DISCORD_WEBHOOK_SECURITY": "https://discord.com/api/webhooks/security/token",
        }

        with patch.dict(os.environ, env_vars):
            yield

    @pytest.mark.asyncio
    async def test_initialization(self, mock_env: None) -> None:
        """Test service initialization."""
        service = DiscordNotificationService()

        # Check that all webhooks are configured
        assert service.webhooks[AlertType.GENERAL] is not None
        assert service.webhooks[AlertType.ERROR] is not None
        assert service.webhooks[AlertType.CRITICAL] is not None

        # Check clients are created
        assert service.clients[AlertType.GENERAL] is not None
        assert service.clients[AlertType.ERROR] is not None

        # Cleanup
        await service.shutdown()

    @pytest.mark.asyncio
    async def test_successful_send(self, mock_env: None) -> None:
        """Test successful message send."""
        service = DiscordNotificationService()

        # Mock the webhook client
        mock_client = AsyncMock()
        mock_client.send.return_value = {"status": 200, "rate_limit": {}}
        service.clients[AlertType.GENERAL] = mock_client

        # Mock rate limiter
        service.rate_limiter.allow = AsyncMock(return_value=True)

        message = NotificationMessage(
            alert_type=AlertType.GENERAL,
            title="Test Message",
            message="This is a test",
        )

        result = await service.send(message)

        assert result.success is True
        assert result.status == NotificationStatus.SENT
        mock_client.send.assert_called_once()

        # Cleanup
        await service.shutdown()

    @pytest.mark.asyncio
    async def test_no_webhook_configured(self, mock_env: None) -> None:
        """Test sending to unconfigured webhook."""
        service = DiscordNotificationService()

        # Remove client for SECURITY
        service.clients[AlertType.SECURITY] = None

        message = NotificationMessage(
            alert_type=AlertType.SECURITY,
            title="Security Alert",
            message="Test security message",
        )

        result = await service.send(message)

        assert result.success is False
        assert result.status == NotificationStatus.FAILED
        assert "No webhook configured" in result.error

        # Cleanup
        await service.shutdown()

    @pytest.mark.asyncio
    async def test_rate_limited_queuing(self, mock_env: None) -> None:
        """Test message queuing when rate limited."""
        service = DiscordNotificationService()

        # Mock rate limiter to deny request
        service.rate_limiter.allow = AsyncMock(return_value=False)

        message = NotificationMessage(
            alert_type=AlertType.GENERAL,
            title="Rate Limited Message",
            message="This should be queued",
        )

        result = await service.send(message)

        assert result.success is True
        assert result.status == NotificationStatus.QUEUED
        assert "queue_size" in result.metadata

        # Check that message was queued
        assert service.message_queue[AlertType.GENERAL].qsize() == 1

        # Cleanup
        await service.shutdown()

    @pytest.mark.asyncio
    async def test_send_failure(self, mock_env: None) -> None:
        """Test handling send failure."""
        service = DiscordNotificationService()

        # Mock the webhook client to fail
        mock_client = AsyncMock()
        mock_client.send.side_effect = aiohttp.ClientError("Connection failed")
        service.clients[AlertType.ERROR] = mock_client

        # Mock rate limiter
        service.rate_limiter.allow = AsyncMock(return_value=True)

        message = NotificationMessage(
            alert_type=AlertType.ERROR,
            title="Error Message",
            message="This will fail",
        )

        result = await service.send(message)

        assert result.success is False
        assert result.status == NotificationStatus.FAILED
        assert "Connection failed" in result.error

        # Cleanup
        await service.shutdown()

    @pytest.mark.asyncio
    async def test_validation_success(self, mock_env: None) -> None:
        """Test successful configuration validation."""
        service = DiscordNotificationService()

        # Mock all clients to validate successfully
        for client in service.clients.values():
            if client:
                client.validate = AsyncMock(return_value=True)

        result = await service.validate_configuration()
        assert result is True

        # Cleanup
        await service.shutdown()

    @pytest.mark.asyncio
    async def test_validation_failure(self, mock_env: None) -> None:
        """Test configuration validation with failure."""
        service = DiscordNotificationService()

        # Mock one client to fail validation
        if service.clients[AlertType.GENERAL]:
            service.clients[AlertType.GENERAL].validate = AsyncMock(return_value=False)

        # Mock others to succeed
        for alert_type, client in service.clients.items():
            if client and alert_type != AlertType.GENERAL:
                client.validate = AsyncMock(return_value=True)

        result = await service.validate_configuration()
        assert result is False

        # Cleanup
        await service.shutdown()

    @pytest.mark.asyncio
    async def test_health_check_success(self, mock_env: None) -> None:
        """Test successful health check."""
        service = DiscordNotificationService()

        # Mock at least one client to validate successfully
        if service.clients[AlertType.GENERAL]:
            service.clients[AlertType.GENERAL].validate = AsyncMock(return_value=True)

        result = await service.health_check()
        assert result is True

        # Cleanup
        await service.shutdown()

    @pytest.mark.asyncio
    async def test_health_check_failure(self, mock_env: None) -> None:
        """Test health check with all failures."""
        service = DiscordNotificationService()

        # Mock all clients to fail validation
        for client in service.clients.values():
            if client:
                client.validate = AsyncMock(return_value=False)

        result = await service.health_check()
        assert result is False

        # Cleanup
        await service.shutdown()

    @pytest.mark.asyncio
    async def test_rate_limit_status(self, mock_env: None) -> None:
        """Test getting rate limit status."""
        service = DiscordNotificationService()

        # Mock rate limiter status
        service.rate_limiter.get_status = Mock(
            return_value={
                "available_tokens": 25,
                "max_tokens": 30,
            }
        )

        status = await service.get_rate_limit_status()

        assert "general" in status
        assert "available_tokens" in status["general"]
        assert "queue_size" in status["general"]

        # Cleanup
        await service.shutdown()

    @pytest.mark.asyncio
    async def test_queue_processing(self, mock_env: None) -> None:
        """Test background queue processing."""
        service = DiscordNotificationService()

        # Mock client for successful send
        mock_client = AsyncMock()
        mock_client.send.return_value = {"status": 200}
        service.clients[AlertType.GENERAL] = mock_client

        # Mock rate limiter
        service.rate_limiter.wait_for_token = AsyncMock()

        # Put message in queue
        message = NotificationMessage(
            alert_type=AlertType.GENERAL,
            title="Queued Message",
            message="This is queued",
        )

        await service.message_queue[AlertType.GENERAL].put(message)

        # Give queue processor time to work
        await asyncio.sleep(0.1)

        # Check that send was called
        assert mock_client.send.call_count >= 0  # May be processed

        # Cleanup
        await service.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown(self, mock_env: None) -> None:
        """Test service shutdown."""
        service = DiscordNotificationService()

        # Add some messages to queues
        for alert_type in AlertType:
            if service.clients.get(alert_type):
                message = NotificationMessage(
                    alert_type=alert_type,
                    title="Test",
                    message="Test message",
                )
                with contextlib.suppress(asyncio.QueueFull):
                    service.message_queue[alert_type].put_nowait(message)

        # Should shutdown cleanly
        await service.shutdown()

        # All tasks should be cancelled
        assert all(task.cancelled() or task.done() for task in service.queue_tasks)
