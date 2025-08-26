"""Load tests for rate limiting accuracy verification."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request

from services.tracklist_service.src.auth.models import User, UserTier
from services.tracklist_service.src.rate_limiting.limiter import RateLimiter, RateLimitResult


class TestRateLimitingAccuracy:
    """Load tests to verify rate limiting accuracy under concurrent load."""

    @pytest.fixture
    async def rate_limiter(self):
        """Create rate limiter with mock Redis."""
        mock_redis = AsyncMock()
        limiter = RateLimiter(mock_redis)
        return limiter, mock_redis

    @pytest.fixture
    def test_users(self):
        """Create test users for different tiers."""
        return {
            "free": User(id="free-user", email="free@test.com", tier=UserTier.FREE),
            "premium": User(id="premium-user", email="premium@test.com", tier=UserTier.PREMIUM),
            "enterprise": User(id="enterprise-user", email="enterprise@test.com", tier=UserTier.ENTERPRISE),
        }

    @pytest.fixture
    def mock_request(self):
        """Create mock request."""
        request = MagicMock(spec=Request)
        request.client.host = "192.168.1.1"
        request.method = "GET"
        request.url.path = "/test"
        return request

    @pytest.mark.asyncio
    async def test_concurrent_requests_free_tier(self, rate_limiter, test_users, mock_request):
        """Test concurrent requests for free tier user."""
        limiter, mock_redis = rate_limiter
        user = test_users["free"]

        # Mock Redis responses for token bucket
        mock_redis.hgetall.return_value = {"tokens": "10", "last_update": str(time.time())}
        mock_redis.hmset.return_value = True

        # Simulate 100 concurrent requests within 1 minute
        tasks = []
        start_time = time.time()

        for _i in range(100):
            task = limiter.check_rate_limit(user, mock_request)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        end_time = time.time()

        # Analyze results
        allowed = sum(1 for result in results if result.allowed)
        denied = sum(1 for result in results if not result.allowed)
        duration = end_time - start_time

        # Assertions for free tier (10 requests/minute)
        assert allowed <= 12, f"Too many requests allowed: {allowed} (expected ≤12 with burst)"
        assert denied >= 88, f"Too few requests denied: {denied} (expected ≥88)"
        assert duration < 5.0, f"Load test took too long: {duration}s"

        print(f"Free tier: {allowed} allowed, {denied} denied in {duration:.3f}s")

    @pytest.mark.asyncio
    async def test_concurrent_requests_premium_tier(self, rate_limiter, test_users, mock_request):
        """Test concurrent requests for premium tier user."""
        limiter, mock_redis = rate_limiter
        user = test_users["premium"]

        # Mock Redis responses for higher limit
        mock_redis.hgetall.return_value = {"tokens": "100", "last_update": str(time.time())}
        mock_redis.hmset.return_value = True

        # Simulate 200 concurrent requests
        tasks = []
        start_time = time.time()

        for _i in range(200):
            task = limiter.check_rate_limit(user, mock_request)
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        end_time = time.time()

        # Analyze results
        allowed = sum(1 for result in results if result.allowed)
        denied = sum(1 for result in results if not result.allowed)
        duration = end_time - start_time

        # Assertions for premium tier (100 requests/minute)
        assert allowed <= 120, f"Too many requests allowed: {allowed} (expected ≤120 with burst)"
        assert denied >= 80, f"Too few requests denied: {denied} (expected ≥80)"
        assert duration < 5.0, f"Load test took too long: {duration}s"

        print(f"Premium tier: {allowed} allowed, {denied} denied in {duration:.3f}s")

    @pytest.mark.asyncio
    async def test_rate_limit_accuracy_over_time(self, rate_limiter, test_users, mock_request):
        """Test rate limit accuracy over multiple time windows."""
        limiter, mock_redis = rate_limiter
        user = test_users["free"]

        # Track requests over time
        request_times: list[float] = []
        results: list[RateLimitResult] = []

        # Mock Redis to simulate token bucket refill
        current_time = time.time()
        mock_redis.hgetall.side_effect = lambda key: {
            "tokens": str(max(0, 10 - len([t for t in request_times if current_time - t < 60]))),
            "last_update": str(current_time),
        }
        mock_redis.hmset.return_value = True

        # Send requests at regular intervals
        for _i in range(30):
            start = time.time()
            result = await limiter.check_rate_limit(user, mock_request)
            request_times.append(start)
            results.append(result)

            # Wait 2 seconds between requests
            await asyncio.sleep(2)

        # Analyze accuracy
        allowed_count = sum(1 for r in results if r.allowed)

        # Should allow approximately 10 requests (with some tolerance for timing)
        assert 8 <= allowed_count <= 12, f"Rate limit inaccuracy: {allowed_count} requests allowed"

        print(f"Over time test: {allowed_count} requests allowed out of 30")

    @pytest.mark.asyncio
    async def test_burst_allowance_accuracy(self, rate_limiter, test_users, mock_request):
        """Test burst allowance accuracy."""
        limiter, mock_redis = rate_limiter
        user = test_users["premium"]

        # Mock full token bucket
        mock_redis.hgetall.return_value = {"tokens": "120", "last_update": str(time.time())}
        mock_redis.hmset.return_value = True

        # Send burst of requests
        burst_tasks = []
        for _i in range(130):  # Above premium limit + burst
            task = limiter.check_rate_limit(user, mock_request)
            burst_tasks.append(task)

        results = await asyncio.gather(*burst_tasks)

        allowed = sum(1 for result in results if result.allowed)
        denied = sum(1 for result in results if not result.allowed)

        # Should allow burst up to 120 requests (100 + 20% burst)
        assert 115 <= allowed <= 125, f"Burst allowance inaccuracy: {allowed} allowed"
        assert denied >= 5, f"Should deny some requests in burst: {denied} denied"

        print(f"Burst test: {allowed} allowed, {denied} denied")

    @pytest.mark.asyncio
    async def test_multi_user_concurrent_load(self, rate_limiter, test_users, mock_request):
        """Test multiple users making concurrent requests."""
        limiter, mock_redis = rate_limiter

        # Mock Redis responses for all users
        def mock_hgetall(key):
            if "free-user" in key:
                return {"tokens": "10", "last_update": str(time.time())}
            elif "premium-user" in key:
                return {"tokens": "100", "last_update": str(time.time())}
            elif "enterprise-user" in key:
                return {"tokens": "1000", "last_update": str(time.time())}
            return {"tokens": "0", "last_update": str(time.time())}

        mock_redis.hgetall.side_effect = mock_hgetall
        mock_redis.hmset.return_value = True

        # Create tasks for all user tiers
        all_tasks = []

        # Free tier: 15 requests
        for _i in range(15):
            task = limiter.check_rate_limit(test_users["free"], mock_request)
            all_tasks.append(("free", task))

        # Premium tier: 150 requests
        for _i in range(150):
            task = limiter.check_rate_limit(test_users["premium"], mock_request)
            all_tasks.append(("premium", task))

        # Enterprise tier: 1200 requests
        for _i in range(1200):
            task = limiter.check_rate_limit(test_users["enterprise"], mock_request)
            all_tasks.append(("enterprise", task))

        # Execute all tasks concurrently
        start_time = time.time()
        results = await asyncio.gather(*[task for _, task in all_tasks])
        end_time = time.time()

        # Analyze results by tier
        tier_results = {"free": [], "premium": [], "enterprise": []}
        for i, (tier, _) in enumerate(all_tasks):
            tier_results[tier].append(results[i])

        # Verify each tier's limits
        free_allowed = sum(1 for r in tier_results["free"] if r.allowed)
        premium_allowed = sum(1 for r in tier_results["premium"] if r.allowed)
        enterprise_allowed = sum(1 for r in tier_results["enterprise"] if r.allowed)

        assert free_allowed <= 12, f"Free tier over limit: {free_allowed}"
        assert premium_allowed <= 120, f"Premium tier over limit: {premium_allowed}"
        assert enterprise_allowed <= 1200, f"Enterprise tier over limit: {enterprise_allowed}"

        duration = end_time - start_time
        total_requests = len(all_tasks)

        print(f"Multi-user load: {total_requests} requests in {duration:.3f}s")
        print(f"  Free: {free_allowed}/15 allowed")
        print(f"  Premium: {premium_allowed}/150 allowed")
        print(f"  Enterprise: {enterprise_allowed}/1200 allowed")

        # Performance assertion
        assert duration < 10.0, f"Multi-user test took too long: {duration}s"

    @pytest.mark.asyncio
    async def test_rate_limit_precision_timing(self, rate_limiter, test_users, mock_request):
        """Test rate limit timing precision."""
        limiter, mock_redis = rate_limiter
        user = test_users["free"]

        # Mock precise timing control
        base_time = time.time()
        call_count = 0

        def mock_time_based_tokens(*args):
            nonlocal call_count
            call_count += 1
            # Simulate token refill over time
            elapsed = (call_count - 1) * 6  # 6 seconds between calls
            tokens = min(10, max(0, 10 - call_count + (elapsed // 6)))
            return {"tokens": str(tokens), "last_update": str(base_time + elapsed)}

        mock_redis.hgetall.side_effect = mock_time_based_tokens
        mock_redis.hmset.return_value = True

        # Send requests at precise intervals
        results = []
        for i in range(20):
            result = await limiter.check_rate_limit(user, mock_request)
            results.append((time.time(), result))

            if i < 19:  # Don't sleep after last request
                await asyncio.sleep(6)  # 6 seconds = 10 requests per minute

        # Analyze timing precision
        allowed_times = [t for t, r in results if r.allowed]

        # Should allow approximately every 6 seconds (10/minute)
        allowed_count = len(allowed_times)
        assert 8 <= allowed_count <= 12, f"Timing precision issue: {allowed_count} allowed"

        print(f"Timing precision: {allowed_count} requests allowed over 2 minutes")


class TestRateLimitingStress:
    """Stress tests for rate limiting under extreme load."""

    @pytest.mark.asyncio
    async def test_extreme_concurrent_load(self):
        """Test rate limiter under extreme concurrent load."""
        mock_redis = AsyncMock()
        limiter = RateLimiter(mock_redis)
        user = User(id="stress-test", email="stress@test.com", tier=UserTier.PREMIUM)

        # Mock Redis for stress test
        mock_redis.hgetall.return_value = {"tokens": "100", "last_update": str(time.time())}
        mock_redis.hmset.return_value = True

        # Create mock request
        mock_request = MagicMock(spec=Request)
        mock_request.client.host = "192.168.1.1"

        # Generate 5000 concurrent requests
        tasks = []
        for _i in range(5000):
            task = limiter.check_rate_limit(user, mock_request)
            tasks.append(task)

        # Execute with timeout
        start_time = time.time()
        try:
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=30.0)
            end_time = time.time()

            # Analyze results
            successful_results = [r for r in results if isinstance(r, RateLimitResult)]
            exceptions = [r for r in results if isinstance(r, Exception)]

            allowed = sum(1 for r in successful_results if r.allowed)
            denied = sum(1 for r in successful_results if not r.allowed)

            print(f"Stress test: {len(successful_results)} completed, {len(exceptions)} failed")
            print(f"  {allowed} allowed, {denied} denied in {end_time - start_time:.3f}s")

            # Verify system didn't crash
            assert len(exceptions) < len(results) * 0.1, "Too many exceptions during stress test"
            assert allowed <= 120, f"Rate limit violated under stress: {allowed} allowed"

        except TimeoutError:
            pytest.fail("Stress test timed out - system may be unresponsive")

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self):
        """Test memory usage doesn't grow excessively under load."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        mock_redis = AsyncMock()
        limiter = RateLimiter(mock_redis)

        # Mock Redis responses
        mock_redis.hgetall.return_value = {"tokens": "10", "last_update": str(time.time())}
        mock_redis.hmset.return_value = True

        # Create multiple users and requests
        users = [User(id=f"user-{i}", email=f"user{i}@test.com", tier=UserTier.FREE) for i in range(100)]

        mock_request = MagicMock(spec=Request)
        mock_request.client.host = "192.168.1.1"

        # Run multiple rounds of requests
        for round_num in range(10):
            tasks = []
            for user in users:
                for _ in range(20):  # 20 requests per user
                    task = limiter.check_rate_limit(user, mock_request)
                    tasks.append(task)

            await asyncio.gather(*tasks)

            # Check memory every few rounds
            if round_num % 3 == 0:
                current_memory = process.memory_info().rss
                memory_growth = current_memory - initial_memory

                # Memory shouldn't grow by more than 50MB
                assert memory_growth < 50 * 1024 * 1024, f"Excessive memory growth: {memory_growth / 1024 / 1024:.1f}MB"

        final_memory = process.memory_info().rss
        total_growth = final_memory - initial_memory

        print(f"Memory usage: {total_growth / 1024 / 1024:.1f}MB growth")

        # Final memory check
        assert total_growth < 100 * 1024 * 1024, f"Memory leak detected: {total_growth / 1024 / 1024:.1f}MB growth"


if __name__ == "__main__":
    # Run load tests with performance reporting
    pytest.main(
        [
            __file__,
            "-v",
            "--tb=short",
            "-s",  # Show print outputs
            "--durations=10",  # Show 10 slowest tests
        ]
    )
