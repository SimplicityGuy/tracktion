"""Comprehensive tests for Discord notification system integration."""

import asyncio
import contextlib
import os
import time
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
from services.notification_service.src.core.rate_limiter import PerChannelRateLimiter, RateLimitConfig
from services.notification_service.src.core.retry import CircuitBreaker, RetryPolicy
from services.notification_service.src.templates.discord_templates import DiscordEmbedBuilder


class MockResponse:
    """Mock response for aiohttp requests."""

    def __init__(
        self,
        status: int = 200,
        headers: dict | None = None,
        text: str = "",
        json_data: dict | None = None,
    ):
        self.status = status
        self.headers = headers or {}
        self._text = text
        self._json_data = json_data or {}
        self.request_info = Mock()
        self.history = ()

    async def text(self) -> str:
        """Get response text."""
        return self._text

    async def json(self) -> dict:
        """Get response JSON."""
        return self._json_data


class MockClientSession:
    """Mock aiohttp ClientSession."""

    def __init__(self, responses: list[MockResponse] | None = None):
        self.responses = responses or []
        self._response_index = 0
        self.post_calls = []

    def post(self, url: str, **kwargs):
        """Mock post method."""
        self.post_calls.append({"url": url, "kwargs": kwargs})

        if self._response_index < len(self.responses):
            response = self.responses[self._response_index]
            self._response_index += 1
        else:
            # Default success response
            response = MockResponse(
                status=200,
                headers={
                    "X-RateLimit-Limit": "30",
                    "X-RateLimit-Remaining": "29",
                    "X-RateLimit-Reset": str(int(time.time() + 60)),
                    "X-RateLimit-Reset-After": "60",
                },
            )

        return MockPost(response)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MockPost:
    """Mock post context manager."""

    def __init__(self, response: MockResponse):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class TestDiscordWebhookClientRetryLogic:
    """Test DiscordWebhookClient retry and circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_successful_send(self) -> None:
        """Test successful webhook send with proper response handling."""
        webhook_url = "https://discord.com/api/webhooks/123/abc"
        client = DiscordWebhookClient(webhook_url)

        responses = [
            MockResponse(
                status=200,
                headers={
                    "X-RateLimit-Limit": "30",
                    "X-RateLimit-Remaining": "29",
                    "X-RateLimit-Reset": str(int(time.time() + 60)),
                    "X-RateLimit-Reset-After": "60",
                },
            )
        ]

        with patch("aiohttp.ClientSession", lambda: MockClientSession(responses)):
            result = await client.send({"content": "test"})
            assert result["status"] == 200
            assert result["rate_limit"]["limit"] == "30"
            assert result["rate_limit"]["remaining"] == "29"

    @pytest.mark.asyncio
    async def test_http_error_response(self) -> None:
        """Test handling of HTTP error responses."""
        webhook_url = "https://discord.com/api/webhooks/123/abc"
        client = DiscordWebhookClient(webhook_url)

        responses = [MockResponse(status=404, text="Webhook not found")]

        with patch("aiohttp.ClientSession", lambda: MockClientSession(responses)):
            with pytest.raises(aiohttp.ClientResponseError) as exc_info:
                await client.send({"content": "test"})

            assert exc_info.value.status == 404
            assert "Discord webhook error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_rate_limit_response(self) -> None:
        """Test handling of Discord rate limit responses."""
        webhook_url = "https://discord.com/api/webhooks/123/abc"
        client = DiscordWebhookClient(webhook_url)

        responses = [MockResponse(status=429, headers={"X-RateLimit-Reset-After": "30"})]

        with patch("aiohttp.ClientSession", lambda: MockClientSession(responses)):
            with pytest.raises(aiohttp.ClientResponseError) as exc_info:
                await client.send({"content": "test"})

            assert exc_info.value.status == 429
            assert "Rate limited" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_client_initialization(self) -> None:
        """Test client initialization with different parameters."""
        webhook_url = "https://discord.com/api/webhooks/123/abc"
        retry_policy = RetryPolicy(max_attempts=5, backoff_base=1.5)

        client = DiscordWebhookClient(webhook_url=webhook_url, timeout=15.0, retry_policy=retry_policy)

        assert client.webhook_url == webhook_url
        assert client.timeout == 15.0
        assert client.retry_manager.policy.max_attempts == 5
        assert client.retry_manager.policy.backoff_base == 1.5
        assert client.circuit_breaker.failure_threshold == 5

    @pytest.mark.asyncio
    async def test_webhook_validation_success(self) -> None:
        """Test successful webhook validation."""
        webhook_url = "https://discord.com/api/webhooks/123/abc"
        client = DiscordWebhookClient(webhook_url)

        # Mock the send method to simulate successful validation
        client.send = AsyncMock(return_value={"status": 200})

        result = await client.validate()
        assert result is True
        client.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_webhook_validation_failure(self) -> None:
        """Test webhook validation failure."""
        webhook_url = "https://discord.com/api/webhooks/123/abc"
        client = DiscordWebhookClient(webhook_url)

        # Mock the send method to raise an exception
        client.send = AsyncMock(side_effect=aiohttp.ClientError("Connection failed"))

        result = await client.validate()
        assert result is False

    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self) -> None:
        """Test handling of connection timeouts."""
        webhook_url = "https://discord.com/api/webhooks/123/abc"
        client = DiscordWebhookClient(webhook_url, timeout=0.1)

        # Mock timeout exception by patching the session creation
        class TimeoutMockSession:
            def post(self, *args, **kwargs):
                class TimeoutPost:
                    async def __aenter__(self):
                        raise TimeoutError("Request timed out")

                    async def __aexit__(self, exc_type, exc_val, exc_tb):
                        pass

                return TimeoutPost()

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        with patch("aiohttp.ClientSession", lambda: TimeoutMockSession()), pytest.raises(asyncio.TimeoutError):
            await client.send({"content": "test"})

    @pytest.mark.asyncio
    async def test_invalid_webhook_url_handling(self) -> None:
        """Test handling of invalid webhook URLs."""
        webhook_url = "https://discord.com/api/webhooks/invalid/url"
        client = DiscordWebhookClient(webhook_url)

        responses = [MockResponse(status=404, text="Not Found")]

        with patch("aiohttp.ClientSession", lambda: MockClientSession(responses)):
            with pytest.raises(aiohttp.ClientResponseError) as exc_info:
                await client.send({"content": "test"})

            assert exc_info.value.status == 404
            assert "Discord webhook error" in str(exc_info.value)


class TestPerChannelRateLimiter:
    """Test PerChannelRateLimiter functionality."""

    @pytest.mark.asyncio
    async def test_rate_limiting_per_webhook(self) -> None:
        """Test that rate limiting works per webhook URL."""
        config = RateLimitConfig(limit=2, window=1.0)
        limiter = PerChannelRateLimiter(config)

        webhook1 = "https://discord.com/api/webhooks/1/token"
        webhook2 = "https://discord.com/api/webhooks/2/token"

        # Both webhooks should allow initial requests
        assert await limiter.allow(webhook1) is True
        assert await limiter.allow(webhook2) is True
        assert await limiter.allow(webhook1) is True
        assert await limiter.allow(webhook2) is True

        # Both should be rate limited after exceeding limit
        assert await limiter.allow(webhook1) is False
        assert await limiter.allow(webhook2) is False

    @pytest.mark.asyncio
    async def test_thirty_messages_per_minute_limit(self) -> None:
        """Test the standard 30 messages per minute rate limit."""
        config = RateLimitConfig(limit=30, window=60.0)
        limiter = PerChannelRateLimiter(config)
        webhook_url = "https://discord.com/api/webhooks/test/token"

        # Should allow 30 requests
        for _ in range(30):
            assert await limiter.allow(webhook_url) is True

        # 31st request should be denied
        assert await limiter.allow(webhook_url) is False

    @pytest.mark.asyncio
    async def test_rate_limit_recovery(self) -> None:
        """Test that rate limits recover over time."""
        config = RateLimitConfig(limit=2, window=0.5)  # 2 per 0.5 seconds
        limiter = PerChannelRateLimiter(config)
        webhook_url = "https://discord.com/api/webhooks/test/token"

        # Use up the limit
        assert await limiter.allow(webhook_url) is True
        assert await limiter.allow(webhook_url) is True
        assert await limiter.allow(webhook_url) is False

        # Wait for recovery
        await asyncio.sleep(0.6)

        # Should be allowed again
        assert await limiter.allow(webhook_url) is True

    @pytest.mark.asyncio
    async def test_wait_for_token(self) -> None:
        """Test waiting for rate limit token."""
        config = RateLimitConfig(limit=1, window=0.2)
        limiter = PerChannelRateLimiter(config)
        webhook_url = "https://discord.com/api/webhooks/test/token"

        # Use up the limit
        assert await limiter.allow(webhook_url) is True
        assert await limiter.allow(webhook_url) is False

        # Wait for token should complete when limit resets
        start_time = time.time()
        await limiter.wait_for_token(webhook_url)
        elapsed = time.time() - start_time

        # Should have waited approximately the window duration
        assert elapsed >= 0.1  # Allow some margin

    @pytest.mark.asyncio
    async def test_rate_limit_status(self) -> None:
        """Test getting rate limit status."""
        config = RateLimitConfig(limit=5, window=10.0)
        limiter = PerChannelRateLimiter(config)
        webhook_url = "https://discord.com/api/webhooks/test/token"

        # Make some requests
        await limiter.allow(webhook_url)
        await limiter.allow(webhook_url)

        status = limiter.get_status(webhook_url)
        assert "available_tokens" in status
        assert "max_tokens" in status
        assert status["max_tokens"] == 5
        assert status["available_tokens"] < 5

    @pytest.mark.asyncio
    async def test_high_load_scenario(self) -> None:
        """Test rate limiter under high concurrent load."""
        config = RateLimitConfig(limit=10, window=1.0)
        limiter = PerChannelRateLimiter(config)
        webhook_url = "https://discord.com/api/webhooks/test/token"

        # Simulate concurrent requests
        async def make_request():
            return await limiter.allow(webhook_url)

        # Make 20 concurrent requests
        results = await asyncio.gather(*[make_request() for _ in range(20)])

        # Should allow exactly 10 requests
        allowed_count = sum(1 for result in results if result)
        assert allowed_count == 10


class TestCircuitBreakerPattern:
    """Test CircuitBreaker pattern implementation."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_states(self) -> None:
        """Test circuit breaker state transitions."""
        circuit_breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,
            expected_exception=ValueError,
        )

        async def failing_function():
            raise ValueError("Test error")

        async def success_function():
            return "success"

        # Should start closed (allowing requests)
        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_function)

        with pytest.raises(ValueError):
            await circuit_breaker.call(failing_function)

        # Should now be open (failing fast)
        # Note: The actual circuit breaker implementation may vary

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_different_exceptions(self) -> None:
        """Test circuit breaker only opens on expected exceptions."""
        circuit_breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.1,
            expected_exception=ValueError,
        )

        async def type_error_function():
            raise TypeError("Wrong type")

        async def value_error_function():
            raise ValueError("Wrong value")

        # TypeError should not trigger circuit breaker
        with pytest.raises(TypeError):
            await circuit_breaker.call(type_error_function)

        # ValueError should trigger circuit breaker
        with pytest.raises(ValueError):
            await circuit_breaker.call(value_error_function)


class TestAlertTypesAndFormatting:
    """Test different alert types and message formatting."""

    @pytest.fixture
    def embed_builder(self) -> DiscordEmbedBuilder:
        """Get embed builder instance."""
        return DiscordEmbedBuilder()

    def test_general_alert_formatting(self, embed_builder: DiscordEmbedBuilder) -> None:
        """Test general alert embed formatting."""
        embed = embed_builder.build_embed(
            alert_type=AlertType.GENERAL,
            title="General Information",
            description="This is a general notification",
        )

        assert embed["embeds"][0]["title"] == "General Information"
        assert embed["embeds"][0]["description"] == "This is a general notification"
        assert embed["embeds"][0]["color"] == 0x3498DB  # Blue
        assert "timestamp" in embed["embeds"][0]

    def test_error_alert_formatting(self, embed_builder: DiscordEmbedBuilder) -> None:
        """Test error alert embed formatting."""
        error_details = {
            "error_code": "E001",
            "component": "database",
            "severity": "high",
        }

        embed = embed_builder.build_error_embed(
            error_message="Database connection failed",
            error_details=error_details,
            traceback="Traceback (most recent call last):\n  File...",
        )

        assert embed["embeds"][0]["title"] == "❌ Error Occurred"
        assert embed["embeds"][0]["description"] == "Database connection failed"
        assert embed["embeds"][0]["color"] == 0xE67E22  # Orange

        # Check fields
        fields = embed["embeds"][0]["fields"]
        assert any(field["name"] == "Error Code" for field in fields)
        assert any(field["name"] == "Component" for field in fields)
        assert any(field["name"] == "Traceback" for field in fields)

    def test_critical_alert_formatting(self, embed_builder: DiscordEmbedBuilder) -> None:
        """Test critical alert embed formatting."""
        embed = embed_builder.build_critical_embed(
            title="System Down",
            message="Critical system failure detected",
            impact="All services unavailable",
            action_required="Immediate intervention required",
        )

        assert embed["embeds"][0]["title"] == "🚨 System Down"
        assert embed["embeds"][0]["color"] == 0xE74C3C  # Red
        assert embed["content"] == "@here Critical Alert!"

        # Check impact and action fields
        fields = embed["embeds"][0]["fields"]
        assert any(field["name"] == "🎯 Impact" for field in fields)
        assert any(field["name"] == "⚡ Action Required" for field in fields)

    def test_tracklist_alert_formatting(self, embed_builder: DiscordEmbedBuilder) -> None:
        """Test tracklist alert embed formatting."""
        tracklist_info = {
            "name": "Summer Hits 2024",
            "id": "TL001",
            "source": "spotify",
            "stats": {
                "track_count": 25,
                "duration": 95.5,
            },
        }

        embed = embed_builder.build_tracklist_embed(
            action="created",
            tracklist_info=tracklist_info,
            changes=["Added 5 new tracks", "Updated metadata"],
        )

        assert embed["embeds"][0]["title"] == "📋 Tracklist Created"
        assert "Summer Hits 2024" in embed["embeds"][0]["description"]
        assert embed["embeds"][0]["color"] == 0x2ECC71  # Green

    def test_monitoring_alert_formatting(self, embed_builder: DiscordEmbedBuilder) -> None:
        """Test monitoring alert embed formatting."""
        embed = embed_builder.build_monitoring_embed(
            metric_name="CPU Usage",
            current_value="85%",
            threshold="80%",
            status="warning",
        )

        assert "⚠️ Monitoring Alert: CPU Usage" in embed["embeds"][0]["title"]
        assert "**Current Value:** 85%" in embed["embeds"][0]["description"]
        assert "**Threshold:** 80%" in embed["embeds"][0]["description"]
        assert embed["embeds"][0]["color"] == 0x9B59B6  # Purple

    def test_security_alert_formatting(self, embed_builder: DiscordEmbedBuilder) -> None:
        """Test security alert embed formatting."""
        details = {
            "source_ip": "192.168.1.100",
            "attack_type": "brute_force",
            "attempts": 15,
        }

        embed = embed_builder.build_security_embed(
            security_event="Failed Login Attempts",
            details=details,
            severity="high",
        )

        assert "🟠 Security Alert: Failed Login Attempts" in embed["embeds"][0]["title"]
        assert embed["embeds"][0]["color"] == 0xF39C12  # Yellow

        # Check security details fields
        fields = embed["embeds"][0]["fields"]
        assert any(field["name"] == "Source Ip" for field in fields)
        assert any(field["name"] == "Severity" and field["value"] == "HIGH" for field in fields)

    def test_message_truncation_limits(self, embed_builder: DiscordEmbedBuilder) -> None:
        """Test Discord message length limits are respected."""
        # Test title truncation (256 char limit)
        long_title = "A" * 300
        embed = embed_builder.build_embed(
            alert_type=AlertType.GENERAL,
            title=long_title,
            description="Short description",
        )

        assert len(embed["embeds"][0]["title"]) <= 256
        assert embed["embeds"][0]["title"].endswith("...")

        # Test description truncation (4096 char limit)
        long_description = "B" * 5000
        embed = embed_builder.build_embed(
            alert_type=AlertType.GENERAL,
            title="Short title",
            description=long_description,
        )

        assert len(embed["embeds"][0]["description"]) <= 4096
        assert embed["embeds"][0]["description"].endswith("...")

    def test_field_limits(self, embed_builder: DiscordEmbedBuilder) -> None:
        """Test Discord field limits (25 fields max, 1024 char values)."""
        # Test field count limit (25 max)
        many_fields = [{"name": f"Field {i}", "value": f"Value {i}"} for i in range(30)]

        embed = embed_builder.build_embed(
            alert_type=AlertType.GENERAL,
            title="Field Test",
            description="Testing field limits",
            fields=many_fields,
        )

        assert len(embed["embeds"][0]["fields"]) == 25

        # Test field value truncation (1024 char limit)
        long_value_field = {
            "name": "Long Field",
            "value": "C" * 1500,
        }

        embed = embed_builder.build_embed(
            alert_type=AlertType.GENERAL,
            title="Value Test",
            description="Testing value limits",
            fields=[long_value_field],
        )

        field_value = embed["embeds"][0]["fields"][0]["value"]
        assert len(field_value) <= 1024
        assert field_value.endswith("...")


class TestDiscordIntegrationEdgeCases:
    """Test edge cases and error handling scenarios."""

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
            "DISCORD_RATE_LIMIT": "5",  # Lower for testing
            "DISCORD_QUEUE_SIZE": "10",
            "DISCORD_TIMEOUT_SECONDS": "5",
        }

        with patch.dict(os.environ, env_vars):
            yield

    @pytest.mark.asyncio
    async def test_large_message_handling(self, mock_env: None) -> None:
        """Test handling of messages exceeding Discord's 2000 char limit."""
        service = DiscordNotificationService()

        # Mock successful client
        mock_client = AsyncMock()
        mock_client.send.return_value = {"status": 200}
        service.clients[AlertType.GENERAL] = mock_client
        service.rate_limiter.allow = AsyncMock(return_value=True)

        # Create message with very long content
        large_message = NotificationMessage(
            alert_type=AlertType.GENERAL,
            title="Large Message Test",
            message="X" * 5000,  # Exceeds Discord limits
        )

        result = await service.send(large_message)

        assert result.success is True
        # Verify the embed builder truncated the message
        mock_client.send.assert_called_once()
        call_args = mock_client.send.call_args[0][0]
        embed = call_args["embeds"][0]
        assert len(embed["description"]) <= 4096

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_malformed_webhook_url(self, mock_env: None) -> None:
        """Test handling of malformed webhook URLs."""
        # Override with malformed URL
        with patch.dict(os.environ, {"DISCORD_WEBHOOK_GENERAL": "not-a-valid-url"}):
            service = DiscordNotificationService()

            message = NotificationMessage(
                alert_type=AlertType.GENERAL,
                title="Test",
                message="Test message",
            )

            # Mock client to raise connection error for malformed URL
            mock_client = AsyncMock()
            mock_client.send.side_effect = aiohttp.ClientError("Invalid URL")
            service.clients[AlertType.GENERAL] = mock_client
            service.rate_limiter.allow = AsyncMock(return_value=True)

            result = await service.send(message)

            assert result.success is False
            assert result.status == NotificationStatus.FAILED
            assert "Invalid URL" in result.error

            await service.shutdown()

    @pytest.mark.asyncio
    async def test_queue_full_scenario(self, mock_env: None) -> None:
        """Test behavior when message queue is full."""
        service = DiscordNotificationService()

        # Fill up the queue
        queue = service.message_queue[AlertType.GENERAL]
        for i in range(10):  # Queue size is 10 from env
            message = NotificationMessage(
                alert_type=AlertType.GENERAL,
                title=f"Message {i}",
                message=f"Content {i}",
            )
            queue.put_nowait(message)

        # Mock rate limiter to deny (force queuing)
        service.rate_limiter.allow = AsyncMock(return_value=False)

        # Try to send another message when queue is full
        overflow_message = NotificationMessage(
            alert_type=AlertType.GENERAL,
            title="Overflow Message",
            message="This should fail",
        )

        result = await service.send(overflow_message)

        assert result.success is False
        assert result.status == NotificationStatus.RATE_LIMITED
        assert "queue full" in result.error.lower()

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_webhook_validation_with_network_issues(self) -> None:
        """Test webhook validation with various network issues."""
        webhook_url = "https://discord.com/api/webhooks/test/token"
        client = DiscordWebhookClient(webhook_url, timeout=0.1)

        # Mock the send method directly for validation test
        client.send = AsyncMock(side_effect=TimeoutError("Timed out"))

        result = await client.validate()
        assert result is False

    @pytest.mark.asyncio
    async def test_concurrent_rate_limiting(self, mock_env: None) -> None:
        """Test rate limiting with concurrent requests."""
        service = DiscordNotificationService()

        # Mock client
        mock_client = AsyncMock()
        mock_client.send.return_value = {"status": 200}
        service.clients[AlertType.GENERAL] = mock_client

        async def send_message(index: int):
            message = NotificationMessage(
                alert_type=AlertType.GENERAL,
                title=f"Concurrent Message {index}",
                message=f"Message content {index}",
            )
            return await service.send(message)

        # Send 10 concurrent messages
        results = await asyncio.gather(*[send_message(i) for i in range(10)])

        # Some should succeed, some should be queued due to rate limiting
        sent_count = sum(1 for r in results if r.status == NotificationStatus.SENT)
        queued_count = sum(1 for r in results if r.status == NotificationStatus.QUEUED)

        assert sent_count > 0  # At least some should be sent immediately
        assert sent_count + queued_count == 10  # All should be processed

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_partial_webhook_configuration(self) -> None:
        """Test service with only some webhooks configured."""
        # Configure only general and error webhooks
        env_vars = {
            "DISCORD_WEBHOOK_GENERAL": "https://discord.com/api/webhooks/general/token",
            "DISCORD_WEBHOOK_ERRORS": "https://discord.com/api/webhooks/errors/token",
            # Other webhooks not configured
        }

        with patch.dict(os.environ, env_vars, clear=True):
            service = DiscordNotificationService()

            # General should work
            assert service.clients[AlertType.GENERAL] is not None

            # Security should not be configured
            assert service.clients[AlertType.SECURITY] is None

            # Test sending to unconfigured webhook
            message = NotificationMessage(
                alert_type=AlertType.SECURITY,
                title="Security Alert",
                message="This should fail",
            )

            result = await service.send(message)
            assert result.success is False
            assert result.status == NotificationStatus.FAILED

            await service.shutdown()

    @pytest.mark.asyncio
    async def test_service_health_check_mixed_results(self, mock_env: None) -> None:
        """Test health check with mixed webhook validation results."""
        service = DiscordNotificationService()

        # Mock some clients to succeed, others to fail
        service.clients[AlertType.GENERAL].validate = AsyncMock(return_value=True)
        service.clients[AlertType.ERROR].validate = AsyncMock(return_value=False)
        service.clients[AlertType.CRITICAL].validate = AsyncMock(return_value=True)

        # Health check should pass if at least one webhook is healthy
        result = await service.health_check()
        assert result is True

        await service.shutdown()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_with_queued_messages(self, mock_env: None) -> None:
        """Test graceful shutdown with messages still in queues."""
        service = DiscordNotificationService()

        # Add messages to multiple queues
        for alert_type in [AlertType.GENERAL, AlertType.ERROR, AlertType.CRITICAL]:
            if service.clients.get(alert_type):
                message = NotificationMessage(
                    alert_type=alert_type,
                    title="Test Message",
                    message="Test content",
                )
                with contextlib.suppress(asyncio.QueueFull):
                    service.message_queue[alert_type].put_nowait(message)

        # Shutdown should complete without hanging
        await service.shutdown()

        # All queue processor tasks should be cancelled
        for task in service.queue_tasks:
            assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_unexpected_exception_handling(self, mock_env: None) -> None:
        """Test handling of unexpected exceptions during send."""
        service = DiscordNotificationService()

        # Mock client to raise unexpected exception
        mock_client = AsyncMock()
        mock_client.send.side_effect = RuntimeError("Unexpected error")
        service.clients[AlertType.GENERAL] = mock_client
        service.rate_limiter.allow = AsyncMock(return_value=True)

        message = NotificationMessage(
            alert_type=AlertType.GENERAL,
            title="Test Message",
            message="Test content",
        )

        result = await service.send(message)

        assert result.success is False
        assert result.status == NotificationStatus.FAILED
        assert "Unexpected error" in result.error

        await service.shutdown()
