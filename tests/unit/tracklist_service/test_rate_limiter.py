"""Unit tests for distributed rate limiter."""

import json
import time
from unittest.mock import AsyncMock, Mock

import pytest

from services.tracklist_service.src.auth.models import User, UserTier
from services.tracklist_service.src.rate_limiting.limiter import (
    RateLimiter,
    RateLimitResult,
    RateLimitTier,
    TokenBucket,
)


class TestTokenBucket:
    """Test TokenBucket configuration class."""

    def test_initialization(self):
        """Test token bucket initialization."""
        bucket = TokenBucket(rate=10.0, capacity=100, burst_allowance=20)

        assert bucket.rate == 10.0
        assert bucket.capacity == 100
        assert bucket.burst_allowance == 20

    def test_default_burst_allowance(self):
        """Test default burst allowance."""
        bucket = TokenBucket(rate=5.0, capacity=50)

        assert bucket.burst_allowance == 0


class TestRateLimitResult:
    """Test RateLimitResult data class."""

    def test_initialization(self):
        """Test rate limit result initialization."""
        result = RateLimitResult(
            allowed=True,
            remaining=50,
            reset_time=1234567890,
            retry_after=None,
            limit=100,
        )

        assert result.allowed is True
        assert result.remaining == 50
        assert result.reset_time == 1234567890
        assert result.retry_after is None
        assert result.limit == 100


class TestRateLimiter:
    """Test RateLimiter class."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        redis_mock = AsyncMock()
        redis_mock.ping.return_value = True
        return redis_mock

    @pytest.fixture
    def rate_limiter(self, mock_redis):
        """Create rate limiter instance."""
        return RateLimiter(mock_redis)

    @pytest.fixture
    def test_user_free(self):
        """Create free tier test user."""
        return User(id="user123", email="test@example.com", tier=UserTier.FREE)

    @pytest.fixture
    def test_user_premium(self):
        """Create premium tier test user."""
        return User(id="user456", email="premium@example.com", tier=UserTier.PREMIUM)

    def test_initialization(self, mock_redis):
        """Test rate limiter initialization."""
        limiter = RateLimiter(mock_redis)

        assert limiter.redis is mock_redis
        assert len(limiter.tiers) == 3

        # Check tier configurations
        free_bucket = limiter.tiers[RateLimitTier.FREE.value]
        assert free_bucket.rate == 10.0
        assert free_bucket.capacity == 100
        assert free_bucket.burst_allowance == 20

        premium_bucket = limiter.tiers[RateLimitTier.PREMIUM.value]
        assert premium_bucket.rate == 100.0
        assert premium_bucket.capacity == 1000
        assert premium_bucket.burst_allowance == 200

        enterprise_bucket = limiter.tiers[RateLimitTier.ENTERPRISE.value]
        assert enterprise_bucket.rate == 1000.0
        assert enterprise_bucket.capacity == 10000
        assert enterprise_bucket.burst_allowance == 2000

    @pytest.mark.asyncio
    async def test_check_rate_limit_allowed(self, rate_limiter, test_user_free, mock_redis):
        """Test rate limit check when allowed."""
        # Mock Redis eval to return allowed
        mock_redis.eval.return_value = [1, 119, int(time.time()) + 60]

        result = await rate_limiter.check_rate_limit(test_user_free, cost=1)

        assert isinstance(result, RateLimitResult)
        assert result.allowed is True
        assert result.remaining == 119
        assert result.limit == 120  # 100 + 20 burst
        assert result.retry_after is None
        assert result.headers is not None

        # Verify Redis call
        mock_redis.eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_rate_limit_denied(self, rate_limiter, test_user_free, mock_redis):
        """Test rate limit check when denied."""
        # Mock Redis eval to return denied
        mock_redis.eval.return_value = [0, 0, int(time.time()) + 60]

        result = await rate_limiter.check_rate_limit(test_user_free, cost=5)

        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after is not None
        assert result.retry_after >= 1  # At least 1 second
        assert result.headers is not None

    @pytest.mark.asyncio
    async def test_check_rate_limit_different_tiers(self, rate_limiter, test_user_free, test_user_premium, mock_redis):
        """Test rate limit checks for different user tiers."""
        mock_redis.eval.return_value = [1, 50, int(time.time()) + 60]

        # Free tier
        result_free = await rate_limiter.check_rate_limit(test_user_free)
        assert result_free.limit == 120  # 100 + 20

        # Premium tier
        result_premium = await rate_limiter.check_rate_limit(test_user_premium)
        assert result_premium.limit == 1200  # 1000 + 200

    @pytest.mark.asyncio
    async def test_consume_tokens_success(self, rate_limiter, test_user_free, mock_redis):
        """Test successful token consumption."""
        mock_redis.eval.return_value = [1, 95, int(time.time()) + 60]

        result = await rate_limiter.consume_tokens(test_user_free, 5)

        assert result is True

    @pytest.mark.asyncio
    async def test_consume_tokens_failure(self, rate_limiter, test_user_free, mock_redis):
        """Test failed token consumption."""
        mock_redis.eval.return_value = [0, 2, int(time.time()) + 60]

        result = await rate_limiter.consume_tokens(test_user_free, 5)

        assert result is False

    def test_get_limit_headers(self, rate_limiter, test_user_free):
        """Test rate limit header generation."""
        headers = rate_limiter.get_limit_headers(test_user_free, 50, 1234567890)

        assert headers["X-RateLimit-Limit"] == "120"
        assert headers["X-RateLimit-Remaining"] == "50"
        assert headers["X-RateLimit-Reset"] == "1234567890"
        assert headers["X-RateLimit-Tier"] == "free"
        assert "Retry-After" not in headers

    def test_get_limit_headers_with_retry_after(self, rate_limiter, test_user_premium):
        """Test rate limit headers with retry after."""
        headers = rate_limiter.get_limit_headers(test_user_premium, 0, 1234567890, 30)

        assert headers["X-RateLimit-Limit"] == "1200"
        assert headers["X-RateLimit-Remaining"] == "0"
        assert headers["X-RateLimit-Tier"] == "premium"
        assert headers["Retry-After"] == "30"

    @pytest.mark.asyncio
    async def test_get_user_stats(self, rate_limiter, test_user_free, mock_redis):
        """Test getting user rate limit statistics."""
        # Mock Redis response
        current_time = int(time.time())
        mock_redis.hmget.return_value = [100.0, current_time - 10]

        stats = await rate_limiter.get_user_stats(test_user_free)

        assert stats["user_id"] == "user123"
        assert stats["tier"] == "free"
        assert stats["max_tokens"] == 120
        assert stats["refill_rate"] == 10.0
        assert "current_tokens" in stats
        assert "bucket_full_at" in stats

        mock_redis.hmget.assert_called_once_with("rate_limit:user123", "tokens", "last_refill")

    @pytest.mark.asyncio
    async def test_get_user_stats_new_user(self, rate_limiter, test_user_free, mock_redis):
        """Test getting stats for user with no existing bucket."""
        mock_redis.hmget.return_value = [None, None]

        stats = await rate_limiter.get_user_stats(test_user_free)

        assert stats["current_tokens"] == 120  # Full bucket for new user

    @pytest.mark.asyncio
    async def test_reset_user_limits(self, rate_limiter, test_user_free, mock_redis):
        """Test resetting user rate limits."""
        mock_redis.delete.return_value = 1

        result = await rate_limiter.reset_user_limits(test_user_free)

        assert result is True
        mock_redis.delete.assert_called_once_with("rate_limit:user123")

    @pytest.mark.asyncio
    async def test_reset_user_limits_not_found(self, rate_limiter, test_user_free, mock_redis):
        """Test resetting limits for user with no existing limits."""
        mock_redis.delete.return_value = 0

        result = await rate_limiter.reset_user_limits(test_user_free)

        assert result is False

    @pytest.mark.asyncio
    async def test_set_custom_limit(self, rate_limiter, test_user_free, mock_redis):
        """Test setting custom rate limits."""
        mock_redis.setex.return_value = True

        result = await rate_limiter.set_custom_limit(
            test_user_free, rate=50.0, capacity=500, burst_allowance=100, ttl=7200
        )

        assert result is True

        # Verify Redis call
        mock_redis.setex.assert_called_once()
        args = mock_redis.setex.call_args

        assert args[0][0] == "rate_limit_custom:user123"
        assert args[0][1] == 7200

        # Verify JSON content
        config = json.loads(args[0][2])
        assert config["rate"] == 50.0
        assert config["capacity"] == 500
        assert config["burst_allowance"] == 100

    @pytest.mark.asyncio
    async def test_get_effective_bucket_custom(self, rate_limiter, test_user_free, mock_redis):
        """Test getting effective bucket with custom limits."""
        custom_config = json.dumps({"rate": 25.0, "capacity": 250, "burst_allowance": 50})
        mock_redis.get.return_value = custom_config

        bucket = await rate_limiter._get_effective_bucket(test_user_free)

        assert bucket.rate == 25.0
        assert bucket.capacity == 250
        assert bucket.burst_allowance == 50

        mock_redis.get.assert_called_once_with("rate_limit_custom:user123")

    @pytest.mark.asyncio
    async def test_get_effective_bucket_tier_fallback(self, rate_limiter, test_user_premium, mock_redis):
        """Test getting effective bucket falling back to tier limits."""
        mock_redis.get.return_value = None

        bucket = await rate_limiter._get_effective_bucket(test_user_premium)

        assert bucket.rate == 100.0
        assert bucket.capacity == 1000
        assert bucket.burst_allowance == 200

    @pytest.mark.asyncio
    async def test_get_effective_bucket_invalid_custom(self, rate_limiter, test_user_free, mock_redis):
        """Test getting effective bucket with invalid custom config."""
        mock_redis.get.return_value = "invalid json"

        bucket = await rate_limiter._get_effective_bucket(test_user_free)

        # Should fall back to tier config
        assert bucket.rate == 10.0
        assert bucket.capacity == 100
        assert bucket.burst_allowance == 20

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, rate_limiter, mock_redis):
        """Test health check when Redis is healthy."""
        mock_redis.ping.return_value = True

        health = await rate_limiter.health_check()

        assert health["status"] == "healthy"
        assert health["redis_connected"] is True
        assert health["tiers_configured"] == 3
        assert "free" in health["tier_names"]
        assert "premium" in health["tier_names"]
        assert "enterprise" in health["tier_names"]

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, rate_limiter, mock_redis):
        """Test health check when Redis is down."""
        mock_redis.ping.side_effect = Exception("Connection failed")

        health = await rate_limiter.health_check()

        assert health["status"] == "unhealthy"
        assert health["redis_connected"] is False
        assert "error" in health

    @pytest.mark.asyncio
    async def test_lua_script_execution(self, rate_limiter, test_user_free, mock_redis):
        """Test that Lua script is executed with correct parameters."""
        mock_redis.eval.return_value = [1, 95, int(time.time()) + 60]

        await rate_limiter.check_rate_limit(test_user_free, cost=5)

        # Verify script execution
        mock_redis.eval.assert_called_once()
        args = mock_redis.eval.call_args

        # Verify script parameters
        assert args[0][1] == 1  # Number of keys
        assert args[0][2] == "rate_limit:user123"  # Redis key
        assert args[0][3] == 10.0  # Rate
        assert args[0][4] == 100  # Capacity
        assert args[0][5] == 20  # Burst allowance
        assert args[0][6] == 5  # Cost

    @pytest.mark.asyncio
    async def test_unknown_tier_fallback(self, rate_limiter, mock_redis):
        """Test fallback to free tier for unknown user tier."""
        # Create user with custom tier (not in enum)
        user = User(id="test", email="test@example.com", tier=UserTier.FREE)
        user.tier = Mock()
        user.tier.value = "unknown_tier"

        mock_redis.eval.return_value = [1, 119, int(time.time()) + 60]

        result = await rate_limiter.check_rate_limit(user)

        # Should use free tier limits
        assert result.limit == 120  # Free tier: 100 + 20
