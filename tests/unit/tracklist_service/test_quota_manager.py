"""Unit tests for quota manager."""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from services.tracklist_service.src.auth.models import User, UserTier
from services.tracklist_service.src.quota.manager import QuotaManager
from services.tracklist_service.src.quota.models import (
    QuotaResult,
    QuotaStatus,
    QuotaType,
)


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis_mock = AsyncMock()
    redis_mock.hmget = AsyncMock()
    redis_mock.hset = AsyncMock()
    redis_mock.hincrby = AsyncMock()
    redis_mock.setex = AsyncMock()
    redis_mock.exists = AsyncMock()
    redis_mock.keys = AsyncMock()
    redis_mock.pipeline = Mock()
    redis_mock.ping = AsyncMock()
    return redis_mock


@pytest.fixture
def quota_manager(mock_redis):
    """Create QuotaManager instance with mocked Redis."""
    return QuotaManager(mock_redis)


@pytest.fixture
def free_user():
    """Create free tier user for testing."""
    return User(
        id="free-user-123",
        email="free@example.com",
        tier=UserTier.FREE,
        is_active=True,
    )


@pytest.fixture
def premium_user():
    """Create premium tier user for testing."""
    return User(
        id="premium-user-456",
        email="premium@example.com",
        tier=UserTier.PREMIUM,
        is_active=True,
    )


class TestQuotaManager:
    """Test quota manager functionality."""

    def test_quota_limits_initialization(self, quota_manager):
        """Test that quota limits are properly initialized."""
        # Verify free tier limits
        free_limits = quota_manager.quota_limits[UserTier.FREE.value]
        assert free_limits.daily_limit == 1000
        assert free_limits.monthly_limit == 25000
        assert free_limits.request_burst == 100

        # Verify premium tier limits
        premium_limits = quota_manager.quota_limits[UserTier.PREMIUM.value]
        assert premium_limits.daily_limit == 10000
        assert premium_limits.monthly_limit == 250000
        assert premium_limits.request_burst == 1000

        # Verify enterprise tier limits
        enterprise_limits = quota_manager.quota_limits[UserTier.ENTERPRISE.value]
        assert enterprise_limits.daily_limit == 100000
        assert enterprise_limits.monthly_limit == 2500000
        assert enterprise_limits.request_burst == 10000

    @pytest.mark.asyncio
    async def test_check_quota_new_user(self, quota_manager, mock_redis, free_user):
        """Test quota check for new user with no usage."""
        # Mock Redis response for new user (no existing data)
        mock_redis.hmget.return_value = [None, None, None, None]

        result = await quota_manager.check_quota(free_user, 1)

        # Assertions
        assert isinstance(result, QuotaResult)
        assert result.allowed is True
        assert result.status == QuotaStatus.OK
        assert result.daily_remaining == 999  # 1000 - 1
        assert result.monthly_remaining == 24999  # 25000 - 1
        assert result.daily_percentage == 0.1  # 1/1000 * 100
        assert result.monthly_percentage == 0.004  # 1/25000 * 100
        assert result.message is None

    @pytest.mark.asyncio
    async def test_check_quota_within_limits(self, quota_manager, mock_redis, free_user):
        """Test quota check for user within limits."""
        # Mock existing usage data
        now_ts = int(datetime.utcnow().timestamp())
        mock_redis.hmget.return_value = [
            "100",  # daily_used
            "1000",  # monthly_used
            str(now_ts),  # last_daily_reset
            str(now_ts),  # last_monthly_reset
        ]

        result = await quota_manager.check_quota(free_user, 1)

        # Assertions
        assert result.allowed is True
        assert result.status == QuotaStatus.OK
        assert result.daily_remaining == 899  # 1000 - 101
        assert result.monthly_remaining == 23999  # 25000 - 1001
        assert abs(result.daily_percentage - 10.1) < 0.001  # 101/1000 * 100
        assert abs(result.monthly_percentage - 4.004) < 0.001  # 1001/25000 * 100

    @pytest.mark.asyncio
    async def test_check_quota_warning_threshold(self, quota_manager, mock_redis, free_user):
        """Test quota check when approaching 80% threshold."""
        now_ts = int(datetime.utcnow().timestamp())
        mock_redis.hmget.return_value = [
            "800",  # daily_used - at 80%
            "1000",  # monthly_used
            str(now_ts),
            str(now_ts),
        ]

        result = await quota_manager.check_quota(free_user, 1)

        # Assertions
        assert result.allowed is True
        assert result.status == QuotaStatus.WARNING
        assert abs(result.daily_percentage - 80.1) < 0.001  # 801/1000 * 100
        assert "consider upgrading" in result.message.lower()

    @pytest.mark.asyncio
    async def test_check_quota_critical_threshold(self, quota_manager, mock_redis, free_user):
        """Test quota check when approaching 95% threshold."""
        now_ts = int(datetime.utcnow().timestamp())
        mock_redis.hmget.return_value = [
            "950",  # daily_used - at 95%
            "1000",  # monthly_used
            str(now_ts),
            str(now_ts),
        ]

        result = await quota_manager.check_quota(free_user, 1)

        # Assertions
        assert result.allowed is True
        assert result.status == QuotaStatus.CRITICAL
        assert abs(result.daily_percentage - 95.1) < 0.001  # 951/1000 * 100
        assert "approaching limits" in result.message.lower()

    @pytest.mark.asyncio
    async def test_check_quota_exceeded_daily(self, quota_manager, mock_redis, free_user):
        """Test quota check when daily limit is exceeded."""
        now_ts = int(datetime.utcnow().timestamp())
        mock_redis.hmget.return_value = [
            "1000",  # daily_used - at limit
            "5000",  # monthly_used - within limit
            str(now_ts),
            str(now_ts),
        ]

        result = await quota_manager.check_quota(free_user, 1)

        # Assertions
        assert result.allowed is False
        assert result.status == QuotaStatus.EXCEEDED
        assert result.daily_remaining == 0
        assert "daily quota limit" in result.message.lower()

    @pytest.mark.asyncio
    async def test_check_quota_exceeded_monthly(self, quota_manager, mock_redis, free_user):
        """Test quota check when monthly limit is exceeded."""
        now_ts = int(datetime.utcnow().timestamp())
        mock_redis.hmget.return_value = [
            "500",  # daily_used - within limit
            "25000",  # monthly_used - at limit
            str(now_ts),
            str(now_ts),
        ]

        result = await quota_manager.check_quota(free_user, 1)

        # Assertions
        assert result.allowed is False
        assert result.status == QuotaStatus.EXCEEDED
        assert result.monthly_remaining == 0
        assert "monthly quota limit" in result.message.lower()

    @pytest.mark.asyncio
    async def test_consume_quota_success(self, quota_manager, mock_redis, free_user):
        """Test successful quota consumption."""
        # Mock quota check to allow consumption
        now_ts = int(datetime.utcnow().timestamp())
        mock_redis.hmget.return_value = [
            "100",  # daily_used
            "1000",  # monthly_used
            str(now_ts),
            str(now_ts),
        ]

        # Mock alert checking
        mock_redis.exists.return_value = False  # No alerts sent today

        # Mock pipeline for atomic operations
        mock_pipe = AsyncMock()
        mock_redis.pipeline.return_value = mock_pipe

        result = await quota_manager.consume_quota(free_user, 5)

        # Assertions
        assert result is True
        mock_pipe.hincrby.assert_called()
        mock_pipe.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_consume_quota_denied(self, quota_manager, mock_redis, free_user):
        """Test quota consumption denial when limits exceeded."""
        now_ts = int(datetime.utcnow().timestamp())
        mock_redis.hmget.return_value = [
            "1000",  # daily_used - at limit
            "5000",  # monthly_used
            str(now_ts),
            str(now_ts),
        ]

        result = await quota_manager.consume_quota(free_user, 1)

        # Assertions
        assert result is False
        # Pipeline should not be called
        mock_redis.pipeline.assert_not_called()

    @pytest.mark.asyncio
    async def test_quota_auto_reset_daily(self, quota_manager, mock_redis, free_user):
        """Test automatic daily quota reset."""
        # Mock old timestamp (yesterday)
        yesterday_ts = int((datetime.utcnow() - timedelta(days=1)).timestamp())
        mock_redis.hmget.return_value = [
            "500",  # daily_used
            "2000",  # monthly_used
            str(yesterday_ts),  # last_daily_reset - yesterday
            str(yesterday_ts),  # last_monthly_reset
        ]

        await quota_manager.check_quota(free_user, 1)

        # Should reset daily usage but keep monthly
        calls = mock_redis.hset.call_args_list
        daily_reset_call = any("daily_used" in str(call) and "0" in str(call) for call in calls)
        assert daily_reset_call, "Daily usage should be reset to 0"

    @pytest.mark.asyncio
    async def test_quota_auto_reset_monthly(self, quota_manager, mock_redis, free_user):
        """Test automatic monthly quota reset."""
        # Mock old timestamp (last month)
        last_month = datetime.utcnow().replace(day=1) - timedelta(days=1)
        last_month_ts = int(last_month.timestamp())

        mock_redis.hmget.return_value = [
            "500",  # daily_used
            "15000",  # monthly_used
            str(last_month_ts),  # last_daily_reset
            str(last_month_ts),  # last_monthly_reset - last month
        ]

        await quota_manager.check_quota(free_user, 1)

        # Should reset both daily and monthly usage
        calls = mock_redis.hset.call_args_list
        daily_reset_call = any("daily_used" in str(call) and "0" in str(call) for call in calls)
        monthly_reset_call = any("monthly_used" in str(call) and "0" in str(call) for call in calls)

        assert daily_reset_call, "Daily usage should be reset to 0"
        assert monthly_reset_call, "Monthly usage should be reset to 0"

    @pytest.mark.asyncio
    async def test_send_quota_alert(self, quota_manager, mock_redis, free_user):
        """Test quota alert sending."""
        await quota_manager.send_quota_alert(free_user, 85.0, QuotaType.DAILY)

        # Should store alert record
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        key = call_args[0][0]
        assert "quota_alert" in key
        assert free_user.id in key

    @pytest.mark.asyncio
    async def test_upgrade_quota(self, quota_manager, mock_redis, free_user):
        """Test quota tier upgrade."""
        result = await quota_manager.upgrade_quota(free_user, "premium")

        # Assertions
        assert result is True
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        key = call_args[0][0]
        assert "quota_upgrade" in key
        assert free_user.id in key

    @pytest.mark.asyncio
    async def test_upgrade_quota_invalid_tier(self, quota_manager, mock_redis, free_user):
        """Test quota upgrade with invalid tier."""
        result = await quota_manager.upgrade_quota(free_user, "invalid_tier")

        # Assertions
        assert result is False
        mock_redis.setex.assert_not_called()

    @pytest.mark.asyncio
    async def test_reset_quotas_daily(self, quota_manager, mock_redis):
        """Test manual daily quota reset."""
        mock_redis.keys.return_value = ["quota:user1", "quota:user2", "quota:user3"]

        result = await quota_manager.reset_quotas(QuotaType.DAILY)

        # Assertions
        assert result == 3  # 3 users reset
        # Should set daily_used to 0 for all users
        assert mock_redis.hset.call_count == 6  # 2 calls per user (daily_used + last_daily_reset)

    @pytest.mark.asyncio
    async def test_reset_quotas_monthly(self, quota_manager, mock_redis):
        """Test manual monthly quota reset."""
        mock_redis.keys.return_value = ["quota:user1", "quota:user2"]

        result = await quota_manager.reset_quotas(QuotaType.MONTHLY)

        # Assertions
        assert result == 2  # 2 users reset
        # Should reset both daily and monthly usage
        assert mock_redis.hset.call_count == 8  # 4 calls per user

    @pytest.mark.asyncio
    async def test_get_quota_stats(self, quota_manager, mock_redis, premium_user):
        """Test getting quota statistics."""
        now_ts = int(datetime.utcnow().timestamp())
        mock_redis.hmget.return_value = [
            "2000",  # daily_used
            "50000",  # monthly_used
            str(now_ts),
            str(now_ts),
        ]

        stats = await quota_manager.get_quota_stats(premium_user)

        # Assertions
        assert stats["user_id"] == premium_user.id
        assert stats["tier"] == "premium"
        assert stats["daily"]["used"] == 2000
        assert stats["daily"]["limit"] == 10000
        assert stats["daily"]["remaining"] == 8000
        assert stats["daily"]["percentage"] == 20.0
        assert stats["monthly"]["used"] == 50000
        assert stats["monthly"]["limit"] == 250000
        assert stats["monthly"]["remaining"] == 200000
        assert stats["monthly"]["percentage"] == 20.0
        assert stats["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, quota_manager, mock_redis):
        """Test health check when Redis is healthy."""
        mock_redis.ping.return_value = True
        mock_redis.keys.return_value = ["quota:user1", "quota:user2", "quota:user3"]

        health = await quota_manager.health_check()

        # Assertions
        assert health["status"] == "healthy"
        assert health["redis_connected"] is True
        assert health["active_quotas"] == 3
        assert health["configured_tiers"] == 3
        assert health["alert_thresholds"] == [80, 95]

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self, quota_manager, mock_redis):
        """Test health check when Redis is unhealthy."""
        mock_redis.ping.side_effect = Exception("Connection failed")

        health = await quota_manager.health_check()

        # Assertions
        assert health["status"] == "unhealthy"
        assert health["redis_connected"] is False
        assert "error" in health

    @pytest.mark.asyncio
    async def test_alert_threshold_tracking(self, quota_manager, mock_redis, free_user):
        """Test that alerts are only sent once per day per threshold."""
        # First call - should send alert
        mock_redis.exists.return_value = False  # No alert sent today

        now_ts = int(datetime.utcnow().timestamp())
        mock_redis.hmget.return_value = [
            "850",  # 85% of daily quota
            "1000",
            str(now_ts),
            str(now_ts),
        ]

        # Mock pipeline for consumption
        mock_pipe = AsyncMock()
        mock_redis.pipeline.return_value = mock_pipe

        await quota_manager.consume_quota(free_user, 1)

        # Should record alert as sent
        setex_calls = [call for call in mock_redis.setex.call_args_list if "quota_alert_sent" in str(call)]
        assert len(setex_calls) >= 1  # At least one alert tracking record

    @pytest.mark.asyncio
    async def test_premium_user_higher_limits(self, quota_manager, mock_redis, premium_user):
        """Test that premium users get higher limits."""
        now_ts = int(datetime.utcnow().timestamp())
        mock_redis.hmget.return_value = [
            "5000",  # daily_used - would exceed free but OK for premium
            "100000",  # monthly_used - would exceed free but OK for premium
            str(now_ts),
            str(now_ts),
        ]

        result = await quota_manager.check_quota(premium_user, 1)

        # Assertions
        assert result.allowed is True
        assert result.status == QuotaStatus.OK
        assert result.daily_remaining == 4999  # 10000 - 5001
        assert result.monthly_remaining == 149999  # 250000 - 100001

    @pytest.mark.asyncio
    async def test_large_quota_consumption(self, quota_manager, mock_redis, free_user):
        """Test consuming large amount of quota at once."""
        now_ts = int(datetime.utcnow().timestamp())
        mock_redis.hmget.return_value = [
            "0",  # daily_used - starting fresh
            "0",  # monthly_used
            str(now_ts),
            str(now_ts),
        ]

        # Try to consume 500 requests at once
        result = await quota_manager.check_quota(free_user, 500)

        # Assertions
        assert result.allowed is True  # Within 1000 daily limit
        assert result.daily_remaining == 500  # 1000 - 500
        assert result.daily_percentage == 50.0  # 500/1000 * 100
