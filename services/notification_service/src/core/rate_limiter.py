"""Rate limiting for notification channels."""

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    limit: int = 30  # Maximum requests
    window: float = 60.0  # Time window in seconds
    burst_size: int | None = None  # Maximum burst size (defaults to limit)


class RateLimiter:
    """Token bucket rate limiter implementation."""

    def __init__(self, config: RateLimitConfig | None = None):
        """Initialize rate limiter.

        Args:
            config: Rate limit configuration
        """
        self.config = config or RateLimitConfig()
        self.burst_size = self.config.burst_size or self.config.limit
        self.tokens = float(self.burst_size)
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    async def allow(self, tokens: int = 1) -> bool:
        """Check if request is allowed under rate limit.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if request is allowed, False otherwise
        """
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now

            # Replenish tokens
            replenish_rate = self.config.limit / self.config.window
            self.tokens = min(self.burst_size, self.tokens + elapsed * replenish_rate)

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    async def wait_for_token(self, tokens: int = 1) -> None:
        """Wait until tokens are available.

        Args:
            tokens: Number of tokens needed
        """
        while not await self.allow(tokens):
            # Calculate wait time for next token
            replenish_rate = self.config.limit / self.config.window
            wait_time = (tokens - self.tokens) / replenish_rate
            await asyncio.sleep(wait_time)

    def get_status(self) -> dict[str, float]:
        """Get current rate limiter status.

        Returns:
            Dictionary with status information
        """
        return {
            "available_tokens": self.tokens,
            "max_tokens": self.burst_size,
            "limit": self.config.limit,
            "window": self.config.window,
        }


class PerChannelRateLimiter:
    """Rate limiter that tracks limits per channel."""

    def __init__(self, default_config: RateLimitConfig | None = None):
        """Initialize per-channel rate limiter.

        Args:
            default_config: Default configuration for channels
        """
        self.default_config = default_config or RateLimitConfig()
        self.limiters: dict[str, RateLimiter] = {}
        self.custom_configs: dict[str, RateLimitConfig] = {}

    def set_channel_config(self, channel: str, config: RateLimitConfig) -> None:
        """Set custom configuration for a channel.

        Args:
            channel: Channel identifier
            config: Rate limit configuration for the channel
        """
        self.custom_configs[channel] = config
        if channel in self.limiters:
            del self.limiters[channel]  # Force recreation with new config

    async def allow(self, channel: str, tokens: int = 1) -> bool:
        """Check if request is allowed for a channel.

        Args:
            channel: Channel identifier
            tokens: Number of tokens to consume

        Returns:
            True if request is allowed, False otherwise
        """
        if channel not in self.limiters:
            config = self.custom_configs.get(channel, self.default_config)
            self.limiters[channel] = RateLimiter(config)

        return await self.limiters[channel].allow(tokens)

    async def wait_for_token(self, channel: str, tokens: int = 1) -> None:
        """Wait until tokens are available for a channel.

        Args:
            channel: Channel identifier
            tokens: Number of tokens needed
        """
        if channel not in self.limiters:
            config = self.custom_configs.get(channel, self.default_config)
            self.limiters[channel] = RateLimiter(config)

        await self.limiters[channel].wait_for_token(tokens)

    def get_status(self, channel: str | None = None) -> dict[str, Any]:
        """Get rate limiter status.

        Args:
            channel: Specific channel to get status for, or None for all

        Returns:
            Status information
        """
        if channel:
            if channel in self.limiters:
                return self.limiters[channel].get_status()
            return {"error": f"No rate limiter for channel: {channel}"}
        return {ch: limiter.get_status() for ch, limiter in self.limiters.items()}


class SlidingWindowRateLimiter:
    """Sliding window rate limiter for more accurate rate limiting."""

    def __init__(self, limit: int = 30, window: float = 60.0):
        """Initialize sliding window rate limiter.

        Args:
            limit: Maximum requests in window
            window: Time window in seconds
        """
        self.limit = limit
        self.window = window
        self.requests: deque[float] = deque()
        self.lock = asyncio.Lock()

    async def allow(self) -> bool:
        """Check if request is allowed.

        Returns:
            True if request is allowed, False otherwise
        """
        async with self.lock:
            now = time.time()
            cutoff = now - self.window

            # Remove old requests outside the window
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()

            if len(self.requests) < self.limit:
                self.requests.append(now)
                return True
            return False

    def get_status(self) -> dict[str, Any]:
        """Get current status.

        Returns:
            Status information
        """
        now = time.time()
        cutoff = now - self.window
        active_count = sum(1 for t in self.requests if t >= cutoff)

        return {
            "current_requests": active_count,
            "limit": self.limit,
            "window": self.window,
            "remaining": max(0, self.limit - active_count),
        }
