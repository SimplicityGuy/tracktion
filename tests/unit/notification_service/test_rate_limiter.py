"""Tests for rate limiting functionality."""

import asyncio
import time

import pytest

from services.notification_service.src.core.rate_limiter import (
    PerChannelRateLimiter,
    RateLimitConfig,
    RateLimiter,
    SlidingWindowRateLimiter,
)


class TestRateLimitConfig:
    """Test RateLimitConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default rate limit configuration."""
        config = RateLimitConfig()

        assert config.limit == 30
        assert config.window == 60.0
        assert config.burst_size is None

    def test_custom_config(self) -> None:
        """Test custom rate limit configuration."""
        config = RateLimitConfig(limit=100, window=120.0, burst_size=50)

        assert config.limit == 100
        assert config.window == 120.0
        assert config.burst_size == 50


class TestRateLimiter:
    """Test RateLimiter token bucket implementation."""

    @pytest.mark.asyncio
    async def test_initial_tokens(self) -> None:
        """Test initial token availability."""
        config = RateLimitConfig(limit=10, window=60.0)
        limiter = RateLimiter(config)

        # Should have full tokens initially
        assert await limiter.allow(1) is True
        assert limiter.tokens == 9.0

    @pytest.mark.asyncio
    async def test_token_consumption(self) -> None:
        """Test token consumption."""
        config = RateLimitConfig(limit=10, window=60.0)
        limiter = RateLimiter(config)

        # Consume 5 tokens
        assert await limiter.allow(5) is True
        assert abs(limiter.tokens - 5.0) < 0.001

        # Try to consume more than available
        assert await limiter.allow(6) is False
        assert abs(limiter.tokens - 5.0) < 0.001  # Unchanged

        # Consume remaining
        assert await limiter.allow(5) is True
        assert abs(limiter.tokens - 0.0) < 0.001

    @pytest.mark.asyncio
    async def test_token_replenishment(self) -> None:
        """Test token replenishment over time."""
        config = RateLimitConfig(limit=60, window=60.0)  # 1 token per second
        limiter = RateLimiter(config)

        # Consume all tokens
        limiter.tokens = 0.0
        limiter.last_update = time.monotonic()

        # Wait for replenishment
        await asyncio.sleep(0.1)

        # Should have ~0.1 tokens replenished
        allowed = await limiter.allow(0)  # Check without consuming
        assert allowed is True
        assert limiter.tokens > 0

    @pytest.mark.asyncio
    async def test_burst_size(self) -> None:
        """Test burst size limiting."""
        config = RateLimitConfig(limit=10, window=60.0, burst_size=5)
        limiter = RateLimiter(config)

        # Initial tokens should be limited to burst size
        assert limiter.tokens == 5.0

    @pytest.mark.asyncio
    async def test_wait_for_token(self) -> None:
        """Test waiting for token availability."""
        config = RateLimitConfig(limit=100, window=1.0)  # 100 tokens per second
        limiter = RateLimiter(config)

        # Consume all tokens
        limiter.tokens = 0.0

        start_time = asyncio.get_event_loop().time()
        await limiter.wait_for_token(1)
        end_time = asyncio.get_event_loop().time()

        # Should have waited for token
        assert end_time - start_time > 0
        assert limiter.tokens >= 0

    def test_get_status(self) -> None:
        """Test getting rate limiter status."""
        config = RateLimitConfig(limit=30, window=60.0, burst_size=20)
        limiter = RateLimiter(config)
        limiter.tokens = 15.0

        status = limiter.get_status()

        assert status["available_tokens"] == 15.0
        assert status["max_tokens"] == 20
        assert status["limit"] == 30
        assert status["window"] == 60.0


class TestPerChannelRateLimiter:
    """Test PerChannelRateLimiter functionality."""

    @pytest.mark.asyncio
    async def test_separate_channel_limits(self) -> None:
        """Test that channels have separate rate limits."""
        limiter = PerChannelRateLimiter()

        # Channel 1 consumption
        assert await limiter.allow("channel1", 5) is True

        # Channel 2 should have full tokens
        assert await limiter.allow("channel2", 10) is True

        # Check status
        status = limiter.get_status()
        assert "channel1" in status
        assert "channel2" in status

    @pytest.mark.asyncio
    async def test_custom_channel_config(self) -> None:
        """Test custom configuration per channel."""
        limiter = PerChannelRateLimiter()

        # Set custom config for channel1
        custom_config = RateLimitConfig(limit=100, window=60.0)
        limiter.set_channel_config("channel1", custom_config)

        # Use default for channel2
        assert await limiter.allow("channel1", 50) is True
        assert await limiter.allow("channel2", 50) is False  # Default is 30

    @pytest.mark.asyncio
    async def test_wait_for_channel_token(self) -> None:
        """Test waiting for token on specific channel."""
        config = RateLimitConfig(limit=100, window=1.0)
        limiter = PerChannelRateLimiter(default_config=config)

        # Consume all tokens for channel1
        channel1_limiter = limiter.limiters.get("channel1")
        if not channel1_limiter:
            await limiter.allow("channel1", 0)  # Initialize
            channel1_limiter = limiter.limiters["channel1"]
        channel1_limiter.tokens = 0.0

        start_time = asyncio.get_event_loop().time()
        await limiter.wait_for_token("channel1", 1)
        end_time = asyncio.get_event_loop().time()

        assert end_time - start_time > 0

    def test_get_channel_status(self) -> None:
        """Test getting status for specific channel."""
        limiter = PerChannelRateLimiter()

        # No limiter for channel yet
        status = limiter.get_status("unknown")
        assert "error" in status

        # After using channel
        asyncio.run(limiter.allow("channel1", 1))
        status = limiter.get_status("channel1")
        assert "available_tokens" in status

    def test_get_all_channels_status(self) -> None:
        """Test getting status for all channels."""
        limiter = PerChannelRateLimiter()

        # Initialize some channels
        asyncio.run(limiter.allow("channel1", 1))
        asyncio.run(limiter.allow("channel2", 1))

        status = limiter.get_status()
        assert "channel1" in status
        assert "channel2" in status
        assert isinstance(status["channel1"], dict)


class TestSlidingWindowRateLimiter:
    """Test SlidingWindowRateLimiter functionality."""

    @pytest.mark.asyncio
    async def test_allow_within_limit(self) -> None:
        """Test allowing requests within limit."""
        limiter = SlidingWindowRateLimiter(limit=5, window=1.0)

        # Should allow 5 requests
        for _ in range(5):
            assert await limiter.allow() is True

        # 6th request should be denied
        assert await limiter.allow() is False

    @pytest.mark.asyncio
    async def test_sliding_window(self) -> None:
        """Test that old requests expire from window."""
        limiter = SlidingWindowRateLimiter(limit=3, window=0.1)

        # Fill the limit
        for _ in range(3):
            assert await limiter.allow() is True

        # Should be blocked
        assert await limiter.allow() is False

        # Wait for window to slide
        await asyncio.sleep(0.15)

        # Should allow new requests
        assert await limiter.allow() is True

    @pytest.mark.asyncio
    async def test_precise_rate_limiting(self) -> None:
        """Test precise rate limiting over time."""
        limiter = SlidingWindowRateLimiter(limit=10, window=1.0)

        # Send 10 requests
        for _ in range(10):
            assert await limiter.allow() is True

        # Should be at limit
        assert await limiter.allow() is False

        # Wait for half window
        await asyncio.sleep(0.5)

        # Still blocked (requests haven't expired)
        assert await limiter.allow() is False

        # Wait for full window from first request
        await asyncio.sleep(0.6)

        # Should allow new requests
        assert await limiter.allow() is True

    def test_get_status(self) -> None:
        """Test getting sliding window status."""
        limiter = SlidingWindowRateLimiter(limit=10, window=60.0)

        # Add some requests
        current_time = time.time()
        limiter.requests.extend([current_time - 30, current_time - 20, current_time - 10])

        status = limiter.get_status()

        assert status["current_requests"] == 3
        assert status["limit"] == 10
        assert status["window"] == 60.0
        assert status["remaining"] == 7

    @pytest.mark.asyncio
    async def test_concurrent_requests(self) -> None:
        """Test handling concurrent requests."""
        limiter = SlidingWindowRateLimiter(limit=10, window=1.0)

        # Simulate concurrent requests
        async def make_request() -> bool:
            return await limiter.allow()

        # Create 15 concurrent requests
        tasks = [make_request() for _ in range(15)]
        results = await asyncio.gather(*tasks)

        # Exactly 10 should succeed
        assert sum(results) == 10
        assert results.count(True) == 10
        assert results.count(False) == 5
