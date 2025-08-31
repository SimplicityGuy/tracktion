"""Distributed rate limiting with Redis using token bucket algorithm."""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, List, Any
import redis.asyncio as redis
import json

from ..auth.models import User

logger = logging.getLogger(__name__)


class RateLimitTier(Enum):
    """Rate limit tiers with different limits."""

    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


@dataclass
class TokenBucket:
    """Token bucket configuration for rate limiting."""

    rate: float  # Tokens per second
    capacity: int  # Maximum tokens in bucket
    burst_allowance: int = 0  # Additional burst capacity


@dataclass
class RateLimitResult:
    """Result of rate limit check."""

    allowed: bool
    remaining: int
    reset_time: int
    retry_after: Optional[int] = None
    limit: int = 0
    headers: Optional[Dict[str, str]] = None


class RateLimiter:
    """Distributed rate limiter using Redis and token bucket algorithm."""

    def __init__(self, redis_client: redis.Redis[str]):
        """Initialize rate limiter.

        Args:
            redis_client: Redis client for distributed counting
        """
        self.redis = redis_client

        # Define tier configurations
        self.tiers = {
            RateLimitTier.FREE.value: TokenBucket(
                rate=10.0,  # 10 requests per second
                capacity=100,  # Burst of 100 requests
                burst_allowance=20,  # Additional 20 for temporary spikes
            ),
            RateLimitTier.PREMIUM.value: TokenBucket(
                rate=100.0,  # 100 requests per second
                capacity=1000,  # Burst of 1000 requests
                burst_allowance=200,  # Additional 200 for spikes
            ),
            RateLimitTier.ENTERPRISE.value: TokenBucket(
                rate=1000.0,  # 1000 requests per second
                capacity=10000,  # Burst of 10000 requests
                burst_allowance=2000,  # Additional 2000 for spikes
            ),
        }

    async def check_rate_limit(self, user: User, cost: int = 1) -> RateLimitResult:
        """Check if user is within rate limits.

        Args:
            user: User making the request
            cost: Number of tokens to consume (default 1)

        Returns:
            Rate limit result with decision and metadata
        """
        tier = user.tier.value
        bucket = self.tiers.get(tier)

        if not bucket:
            # Default to free tier if unknown
            bucket = self.tiers[RateLimitTier.FREE.value]

        # Redis key for this user's bucket
        key = f"rate_limit:{user.id}"

        # Use Lua script for atomic operations
        result = await self._execute_token_bucket_script(key, bucket, cost, int(time.time()))

        allowed = result[0] == 1
        remaining = max(0, int(result[1]))
        reset_time = int(result[2])

        # Calculate retry after if rate limited
        retry_after = None
        if not allowed:
            # Calculate when next token will be available
            retry_after = max(1, int(cost / bucket.rate))

        headers = self.get_limit_headers(user, remaining, reset_time, retry_after)

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_time=reset_time,
            retry_after=retry_after,
            limit=bucket.capacity + bucket.burst_allowance,
            headers=headers,
        )

    async def consume_tokens(self, user: User, tokens: int) -> bool:
        """Consume tokens from user's bucket.

        Args:
            user: User consuming tokens
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if not enough available
        """
        result = await self.check_rate_limit(user, tokens)
        return result.allowed

    def get_limit_headers(
        self, user: User, remaining: int, reset_time: int, retry_after: Optional[int] = None
    ) -> Dict[str, str]:
        """Generate rate limit headers for HTTP responses.

        Args:
            user: User for rate limit info
            remaining: Remaining tokens
            reset_time: When bucket resets
            retry_after: Seconds to wait if rate limited

        Returns:
            Dictionary of rate limit headers
        """
        tier = user.tier.value
        bucket = self.tiers.get(tier, self.tiers[RateLimitTier.FREE.value])

        headers = {
            "X-RateLimit-Limit": str(bucket.capacity + bucket.burst_allowance),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset_time),
            "X-RateLimit-Tier": tier,
        }

        if retry_after:
            headers["Retry-After"] = str(retry_after)

        return headers

    async def _execute_token_bucket_script(
        self, key: str, bucket: TokenBucket, cost: int, current_time: int
    ) -> List[int]:
        """Execute Lua script for atomic token bucket operations.

        Args:
            key: Redis key for the bucket
            bucket: Token bucket configuration
            cost: Number of tokens to consume
            current_time: Current Unix timestamp

        Returns:
            List with [allowed, remaining, reset_time]
        """
        # Lua script for atomic token bucket algorithm
        script = """
        local key = KEYS[1]
        local rate = tonumber(ARGV[1])
        local capacity = tonumber(ARGV[2])
        local burst_allowance = tonumber(ARGV[3])
        local cost = tonumber(ARGV[4])
        local current_time = tonumber(ARGV[5])

        -- Get current bucket state
        local bucket_data = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket_data[1])
        local last_refill = tonumber(bucket_data[2])

        -- Initialize if bucket doesn't exist
        if tokens == nil then
            tokens = capacity + burst_allowance
            last_refill = current_time
        end

        -- Calculate tokens to add based on elapsed time
        local elapsed = current_time - last_refill
        local new_tokens = math.min(capacity + burst_allowance, tokens + (elapsed * rate))

        -- Check if we can consume the requested tokens
        local allowed = 0
        local remaining = new_tokens

        if new_tokens >= cost then
            allowed = 1
            remaining = new_tokens - cost
        end

        -- Update bucket state
        redis.call('HMSET', key, 'tokens', remaining, 'last_refill', current_time)
        redis.call('EXPIRE', key, 3600)  -- Expire after 1 hour of inactivity

        -- Calculate reset time (when bucket will be full)
        local tokens_needed = (capacity + burst_allowance) - remaining
        local reset_time = current_time + math.ceil(tokens_needed / rate)

        return {allowed, remaining, reset_time}
        """

        result = await self.redis.eval(  # type: ignore[no-untyped-call]
            script,
            1,  # Number of keys
            key,  # The key
            str(bucket.rate),  # Rate per second (as string)
            str(bucket.capacity),  # Base capacity (as string)
            str(bucket.burst_allowance),  # Burst allowance (as string)
            str(cost),  # Tokens to consume (as string)
            str(current_time),  # Current timestamp (as string)
        )
        return [int(x) for x in result]

    async def get_user_stats(self, user: User) -> Dict[str, Any]:
        """Get current rate limit stats for a user.

        Args:
            user: User to get stats for

        Returns:
            Dictionary with current rate limit status
        """
        key = f"rate_limit:{user.id}"
        tier = user.tier.value
        bucket = self.tiers.get(tier, self.tiers[RateLimitTier.FREE.value])

        # Get current bucket state
        bucket_data = await self.redis.hmget(key, ["tokens", "last_refill"])
        tokens = float(bucket_data[0]) if bucket_data[0] else bucket.capacity + bucket.burst_allowance
        last_refill = int(bucket_data[1]) if bucket_data[1] else int(time.time())

        # Calculate current tokens with refill
        current_time = int(time.time())
        elapsed = current_time - last_refill
        current_tokens = min(bucket.capacity + bucket.burst_allowance, tokens + (elapsed * bucket.rate))

        return {
            "user_id": user.id,
            "tier": tier,
            "current_tokens": int(current_tokens),
            "max_tokens": bucket.capacity + bucket.burst_allowance,
            "refill_rate": bucket.rate,
            "last_refill": last_refill,
            "bucket_full_at": current_time
            + int((bucket.capacity + bucket.burst_allowance - current_tokens) / bucket.rate),
        }

    async def reset_user_limits(self, user: User) -> bool:
        """Reset rate limits for a user (admin function).

        Args:
            user: User to reset limits for

        Returns:
            True if reset successfully
        """
        key = f"rate_limit:{user.id}"
        result = await self.redis.delete(key)

        logger.info(f"Reset rate limits for user {user.id}")
        return bool(result > 0)

    async def set_custom_limit(
        self, user: User, rate: float, capacity: int, burst_allowance: int = 0, ttl: int = 3600
    ) -> bool:
        """Set custom rate limits for a user (override tier limits).

        Args:
            user: User to set custom limits for
            rate: Tokens per second
            capacity: Base bucket capacity
            burst_allowance: Additional burst capacity
            ttl: Time to live for custom limit in seconds

        Returns:
            True if set successfully
        """
        key = f"rate_limit_custom:{user.id}"
        custom_config = {"rate": rate, "capacity": capacity, "burst_allowance": burst_allowance}

        await self.redis.setex(key, ttl, json.dumps(custom_config))
        logger.info(f"Set custom rate limits for user {user.id}: {rate}/s, {capacity}+{burst_allowance}")

        return True

    async def _get_effective_bucket(self, user: User) -> TokenBucket:
        """Get effective bucket configuration for user (including custom limits).

        Args:
            user: User to get bucket for

        Returns:
            Token bucket configuration
        """
        # Check for custom limits first
        custom_key = f"rate_limit_custom:{user.id}"
        custom_data = await self.redis.get(custom_key)

        if custom_data:
            try:
                config = json.loads(custom_data)
                return TokenBucket(
                    rate=config["rate"], capacity=config["capacity"], burst_allowance=config.get("burst_allowance", 0)
                )
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"Invalid custom rate limit config for user {user.id}")

        # Fall back to tier-based limits
        tier = user.tier.value
        return self.tiers.get(tier, self.tiers[RateLimitTier.FREE.value])

    async def health_check(self) -> Dict[str, Any]:
        """Check rate limiter health and Redis connectivity.

        Returns:
            Health status dictionary
        """
        try:
            # Test Redis connectivity
            await self.redis.ping()

            return {
                "status": "healthy",
                "redis_connected": True,
                "tiers_configured": len(self.tiers),
                "tier_names": list(self.tiers.keys()),
            }
        except Exception as e:
            logger.error(f"Rate limiter health check failed: {e}")
            return {"status": "unhealthy", "redis_connected": False, "error": str(e)}
