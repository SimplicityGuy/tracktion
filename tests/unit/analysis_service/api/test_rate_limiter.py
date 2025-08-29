"""Unit tests for rate limiting and backpressure."""

import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi import Request

from services.analysis_service.src.api.rate_limiter import (
    AsyncRateLimiter,
    RateLimitConfig,
    RateLimitStrategy,
    RequestQueue,
    SlidingWindowCounter,
    TokenBucket,
)


class TestTokenBucket:
    """Test token bucket rate limiting algorithm."""

    @pytest.mark.asyncio
    async def test_token_bucket_initialization(self):
        """Test token bucket initializes with correct capacity."""
        bucket = TokenBucket(rate=10.0, capacity=20)
        assert bucket.rate == 10.0
        assert bucket.capacity == 20
        assert bucket.tokens == 20

    @pytest.mark.asyncio
    async def test_token_consumption(self):
        """Test consuming tokens from bucket."""
        bucket = TokenBucket(rate=10.0, capacity=20)

        # Consume single token
        assert await bucket.consume(1) is True
        assert 18.9 <= bucket.tokens <= 19.1  # Allow for small time differences

        # Consume multiple tokens
        assert await bucket.consume(5) is True
        assert 13.9 <= bucket.tokens <= 14.1  # Allow for small time differences

        # Try to consume more than available
        assert await bucket.consume(15) is False
        assert 13.9 <= bucket.tokens <= 14.1  # Unchanged (within tolerance)

    @pytest.mark.asyncio
    async def test_token_refill(self):
        """Test token refill over time."""
        bucket = TokenBucket(rate=10.0, capacity=20)

        # Consume all tokens
        assert await bucket.consume(20) is True
        assert bucket.tokens == 0

        # Wait for refill
        await asyncio.sleep(0.2)  # Should add ~2 tokens

        # Check tokens were added
        assert await bucket.consume(1) is True  # At least 1 token available

    @pytest.mark.asyncio
    async def test_wait_time_calculation(self):
        """Test calculating wait time for tokens."""
        bucket = TokenBucket(rate=10.0, capacity=20)

        # No wait when tokens available
        wait_time = await bucket.wait_for_tokens(5)
        assert wait_time == 0.0

        # Consume all tokens
        await bucket.consume(20)

        # Calculate wait time
        wait_time = await bucket.wait_for_tokens(5)
        assert wait_time == pytest.approx(0.5, rel=0.1)  # 5 tokens at 10/sec


class TestSlidingWindowCounter:
    """Test sliding window rate limiting algorithm."""

    @pytest.mark.asyncio
    async def test_sliding_window_initialization(self):
        """Test sliding window counter initialization."""
        counter = SlidingWindowCounter(window_size=60.0, max_requests=100)
        assert counter.window_size == 60.0
        assert counter.max_requests == 100
        assert len(counter.requests) == 0

    @pytest.mark.asyncio
    async def test_request_allowance(self):
        """Test allowing requests within limit."""
        counter = SlidingWindowCounter(window_size=1.0, max_requests=3)

        # Allow first 3 requests
        assert await counter.is_allowed() is True
        assert await counter.is_allowed() is True
        assert await counter.is_allowed() is True

        # Deny 4th request
        assert await counter.is_allowed() is False

    @pytest.mark.asyncio
    async def test_window_sliding(self):
        """Test window sliding over time."""
        counter = SlidingWindowCounter(window_size=0.5, max_requests=2)

        # Allow 2 requests
        assert await counter.is_allowed() is True
        assert await counter.is_allowed() is True
        assert await counter.is_allowed() is False

        # Wait for window to slide
        await asyncio.sleep(0.6)

        # Should allow new requests
        assert await counter.is_allowed() is True

    @pytest.mark.asyncio
    async def test_reset_time_calculation(self):
        """Test calculating reset time."""
        counter = SlidingWindowCounter(window_size=1.0, max_requests=1)

        # Add a request
        assert await counter.is_allowed() is True

        # Get reset time
        reset_time = await counter.get_reset_time()
        assert 0.0 <= reset_time <= 1.0


class TestRequestQueue:
    """Test request queue for backpressure handling."""

    @pytest.mark.asyncio
    async def test_queue_initialization(self):
        """Test request queue initialization."""
        queue = RequestQueue(max_size=10, timeout=5.0)
        assert queue.max_size == 10
        assert queue.timeout == 5.0
        assert queue.size() == 0
        assert queue.is_full() is False

    @pytest.mark.asyncio
    async def test_enqueue_dequeue(self):
        """Test enqueueing and dequeueing requests."""
        queue = RequestQueue(max_size=10, timeout=5.0)

        # Enqueue requests
        assert await queue.enqueue("req1") is True
        assert await queue.enqueue("req2") is True
        assert queue.size() == 2

        # Dequeue requests
        assert await queue.dequeue() == "req1"
        assert await queue.dequeue() == "req2"
        assert queue.size() == 0

        # Dequeue from empty queue
        assert await queue.dequeue() is None

    @pytest.mark.asyncio
    async def test_queue_full(self):
        """Test queue full behavior."""
        queue = RequestQueue(max_size=2, timeout=1.0)

        # Fill queue
        assert await queue.enqueue("req1") is True
        assert await queue.enqueue("req2") is True
        assert queue.is_full() is True

        # Try to enqueue when full
        assert await queue.enqueue("req3") is False


class TestAsyncRateLimiter:
    """Test async rate limiter."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return RateLimitConfig(
            requests_per_second=10,
            requests_per_minute=100,
            burst_size=20,
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            enable_backpressure=True,
            max_queue_size=10,
            max_concurrent_connections=100,
            max_connections_per_ip=5,
        )

    @pytest.fixture
    def limiter(self, config):
        """Create rate limiter instance."""
        return AsyncRateLimiter(config)

    @pytest.fixture
    def mock_request(self):
        """Create mock request."""
        request = MagicMock(spec=Request)
        request.client = MagicMock()
        request.client.host = "127.0.0.1"
        request.headers = {}
        return request

    @pytest.mark.asyncio
    async def test_rate_limiter_initialization(self, limiter, config):
        """Test rate limiter initialization."""
        assert limiter.config == config
        assert len(limiter.limiters) == 0
        assert limiter.total_connections == 0
        assert limiter.stats["total_requests"] == 0

    @pytest.mark.asyncio
    async def test_client_identification(self, limiter, mock_request):
        """Test client ID extraction from request."""
        # IP-based identification
        client_id = limiter._get_client_id(mock_request)
        assert client_id == "ip:127.0.0.1"

        # API key-based identification
        mock_request.headers["X-API-Key"] = "test-key"
        client_id = limiter._get_client_id(mock_request)
        assert client_id == "api:test-key"

    @pytest.mark.asyncio
    async def test_rate_limit_checking(self, limiter, mock_request):
        """Test rate limit checking."""
        # First request should be allowed
        allowed, retry_after = await limiter.check_rate_limit(mock_request)
        assert allowed is True
        assert retry_after is None
        assert limiter.stats["total_requests"] == 1
        assert limiter.stats["allowed_requests"] == 1

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, limiter, mock_request):
        """Test rate limit exceeded behavior."""
        # Consume all tokens
        for _ in range(20):  # Burst size
            await limiter.check_rate_limit(mock_request)

        # Next request should be denied
        allowed, retry_after = await limiter.check_rate_limit(mock_request)
        assert allowed is False
        assert retry_after is not None
        assert retry_after > 0
        assert limiter.stats["rate_limited_requests"] > 0

    @pytest.mark.asyncio
    async def test_connection_limit(self, limiter, mock_request):
        """Test connection limit enforcement."""
        # Allow first connection
        assert await limiter.check_connection_limit(mock_request) is True
        assert limiter.total_connections == 1
        assert limiter.connections["127.0.0.1"] == 1

        # Release connection
        await limiter.release_connection(mock_request)
        assert limiter.total_connections == 0

    @pytest.mark.asyncio
    async def test_per_ip_connection_limit(self, limiter, mock_request):
        """Test per-IP connection limit."""
        # Allow up to max_connections_per_ip
        for _ in range(5):
            assert await limiter.check_connection_limit(mock_request) is True

        # Next connection should be denied
        assert await limiter.check_connection_limit(mock_request) is False

    @pytest.mark.asyncio
    async def test_backpressure_handling(self, limiter):
        """Test backpressure queue handling."""
        # Queue should accept requests
        assert await limiter.handle_backpressure("req1") is True
        assert limiter.stats["queued_requests"] == 1

        # Fill queue
        for i in range(2, 11):
            await limiter.handle_backpressure(f"req{i}")

        # Queue should be full
        assert await limiter.handle_backpressure("req11") is False
        assert limiter.stats["rejected_requests"] == 1

    def test_rate_limit_headers(self, limiter):
        """Test rate limit header generation."""
        # Headers when allowed
        headers = limiter.get_rate_limit_headers(allowed=True)
        assert "X-RateLimit-Limit" in headers
        assert "X-RateLimit-Remaining" in headers
        assert "X-RateLimit-Reset" in headers

        # Headers when rate limited
        headers = limiter.get_rate_limit_headers(allowed=False, retry_after=60.0)
        assert "Retry-After" in headers
        assert headers["Retry-After"] == "60"

    def test_statistics(self, limiter):
        """Test statistics retrieval."""
        stats = limiter.get_stats()
        assert "total_requests" in stats
        assert "allowed_requests" in stats
        assert "rate_limited_requests" in stats
        assert "queued_requests" in stats
        assert "rejected_requests" in stats
        assert "active_connections" in stats
        assert "queue_size" in stats
        assert "unique_clients" in stats

    @pytest.mark.asyncio
    async def test_sliding_window_strategy(self, mock_request):
        """Test sliding window strategy."""
        config = RateLimitConfig(
            requests_per_minute=60,
            strategy=RateLimitStrategy.SLIDING_WINDOW,
        )
        limiter = AsyncRateLimiter(config)

        # Should create sliding window counter
        allowed, _ = await limiter.check_rate_limit(mock_request)
        assert allowed is True

        client_id = limiter._get_client_id(mock_request)
        assert isinstance(limiter.limiters[client_id], SlidingWindowCounter)
