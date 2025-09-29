"""Quota management system for API usage tracking and enforcement."""

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Awaitable

import redis.asyncio as redis
from services.tracklist_service.src.auth.models import User, UserTier

from .models import QuotaAlert, QuotaLimits, QuotaResult, QuotaStatus, QuotaType, QuotaUpgrade, QuotaUsage

logger = logging.getLogger(__name__)


class QuotaManager:
    """Manages API usage quotas with daily/monthly limits and alerting."""

    def __init__(self, redis_client: redis.Redis):
        """Initialize quota manager.

        Args:
            redis_client: Redis client for quota storage
        """
        self.redis = redis_client

        # Define quota limits per tier
        self.quota_limits = {
            UserTier.FREE.value: QuotaLimits(
                daily_limit=1000,  # 1K requests per day
                monthly_limit=25000,  # 25K requests per month
                request_burst=100,
            ),
            UserTier.PREMIUM.value: QuotaLimits(
                daily_limit=10000,  # 10K requests per day
                monthly_limit=250000,  # 250K requests per month
                request_burst=1000,
            ),
            UserTier.ENTERPRISE.value: QuotaLimits(
                daily_limit=100000,  # 100K requests per day
                monthly_limit=2500000,  # 2.5M requests per month
                request_burst=10000,
            ),
        }

        # Alert thresholds
        self.alert_thresholds = [80, 95]  # Percentages

    async def check_quota(self, user: User, amount: int = 1) -> QuotaResult:
        """Check if user is within quota limits.

        Args:
            user: User making the request
            amount: Number of requests to check (default 1)

        Returns:
            Quota check result with status and remaining quotas
        """
        limits = self.quota_limits.get(user.tier.value)
        if not limits:
            # Default to free tier if unknown
            limits = self.quota_limits[UserTier.FREE.value]

        # Get current usage
        usage = await self._get_current_usage(user, limits)

        # Check if request would exceed quotas
        daily_after = usage.daily_used + amount
        monthly_after = usage.monthly_used + amount

        # Determine if request is allowed
        daily_allowed = daily_after <= limits.daily_limit
        monthly_allowed = monthly_after <= limits.monthly_limit
        allowed = daily_allowed and monthly_allowed

        # Calculate percentages and status
        daily_percentage = (daily_after / limits.daily_limit) * 100 if limits.daily_limit > 0 else 0
        monthly_percentage = (monthly_after / limits.monthly_limit) * 100 if limits.monthly_limit > 0 else 0

        # Determine overall status based on highest percentage
        max_percentage = max(daily_percentage, monthly_percentage)
        if max_percentage > 100:
            status = QuotaStatus.EXCEEDED
        elif max_percentage >= 95:
            status = QuotaStatus.CRITICAL
        elif max_percentage >= 80:
            status = QuotaStatus.WARNING
        else:
            status = QuotaStatus.OK

        # Calculate remaining quotas
        daily_remaining = max(0, limits.daily_limit - daily_after)
        monthly_remaining = max(0, limits.monthly_limit - monthly_after)

        # Calculate next reset time
        now = datetime.now(UTC)
        next_daily_reset = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        next_monthly_reset = (now.replace(day=1) + timedelta(days=32)).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )

        # Use the sooner reset time
        next_reset = min(next_daily_reset, next_monthly_reset)

        # Generate appropriate message
        message = None
        if not allowed:
            if not daily_allowed:
                message = f"Daily quota limit of {limits.daily_limit} requests exceeded"
            elif not monthly_allowed:
                message = f"Monthly quota limit of {limits.monthly_limit} requests exceeded"
        elif status == QuotaStatus.CRITICAL:
            message = f"Quota usage at {max_percentage:.1f}% - approaching limits"
        elif status == QuotaStatus.WARNING:
            message = f"Quota usage at {max_percentage:.1f}% - consider upgrading"

        return QuotaResult(
            allowed=allowed,
            status=status,
            daily_remaining=daily_remaining,
            monthly_remaining=monthly_remaining,
            daily_percentage=daily_percentage,
            monthly_percentage=monthly_percentage,
            next_reset=next_reset,
            message=message,
        )

    async def consume_quota(self, user: User, amount: int = 1) -> bool:
        """Consume quota for user if available.

        Args:
            user: User consuming quota
            amount: Amount of quota to consume

        Returns:
            True if quota was consumed, False if not enough available
        """
        # Check quota first
        quota_result = await self.check_quota(user, amount)

        if not quota_result.allowed:
            logger.warning(f"Quota consumption denied for user {user.id}: {quota_result.message}")
            return False

        # Consume the quota
        await self._increment_usage(user, amount)

        # Check if we need to send alerts
        await self._check_and_send_alerts(user, quota_result)

        logger.debug(f"Consumed {amount} quota for user {user.id}")
        return True

    async def reset_quotas(self, quota_type: QuotaType | None = None) -> int:
        """Reset quotas for all users.

        Args:
            quota_type: Type of quota to reset (daily/monthly), or None for auto-detection

        Returns:
            Number of users whose quotas were reset
        """
        now = datetime.now(UTC)

        # Auto-detect quota type if not specified
        if quota_type is None:
            if now.hour == 0 and now.minute == 0:
                quota_type = QuotaType.MONTHLY if now.day == 1 else QuotaType.DAILY
            else:
                # Manual reset - determine based on current time
                quota_type = QuotaType.DAILY

        # Get all quota keys
        pattern = "quota:*"
        keys = await self.redis.keys(pattern)

        reset_count = 0

        for key in keys:
            try:
                if quota_type == QuotaType.DAILY:
                    await cast("Awaitable[int]", self.redis.hset(key, "daily_used", "0"))
                    await cast("Awaitable[int]", self.redis.hset(key, "last_daily_reset", str(int(now.timestamp()))))
                elif quota_type == QuotaType.MONTHLY:
                    await cast("Awaitable[int]", self.redis.hset(key, "daily_used", "0"))
                    await cast("Awaitable[int]", self.redis.hset(key, "monthly_used", "0"))
                    await cast("Awaitable[int]", self.redis.hset(key, "last_daily_reset", str(int(now.timestamp()))))
                    await cast("Awaitable[int]", self.redis.hset(key, "last_monthly_reset", str(int(now.timestamp()))))
                reset_count += 1
            except Exception as e:
                key_str = key.decode() if isinstance(key, bytes) else key
                logger.error(f"Failed to reset quota for key {key_str}: {e}")

        logger.info(f"Reset {quota_type.value} quotas for {reset_count} users")
        return reset_count

    async def send_quota_alert(self, user: User, percentage: float, quota_type: QuotaType) -> None:
        """Send quota usage alert to user.

        Args:
            user: User to alert
            percentage: Current usage percentage
            quota_type: Type of quota (daily/monthly)
        """
        # Determine threshold
        threshold = 95 if percentage >= 95 else 80

        # Create alert record
        alert = QuotaAlert(
            user_id=user.id,
            quota_type=quota_type,
            percentage=percentage,
            threshold=threshold,
            current_usage=0,  # Will be filled by caller
            limit=0,  # Will be filled by caller
            timestamp=datetime.now(UTC),
        )

        # Store alert (for audit trail)
        alert_key = f"quota_alert:{user.id}:{quota_type.value}:{threshold}:{int(alert.timestamp.timestamp())}"
        await self.redis.setex(
            alert_key,
            86400 * 7,  # Keep for 7 days
            json.dumps(
                {
                    "user_id": alert.user_id,
                    "quota_type": alert.quota_type.value,
                    "percentage": alert.percentage,
                    "threshold": alert.threshold,
                    "timestamp": alert.timestamp.isoformat(),
                }
            ),
        )

        # In a real system, this would send email/webhook/notification
        logger.warning(
            f"QUOTA ALERT: User {user.id} at {percentage:.1f}% of {quota_type.value} quota (threshold: {threshold}%)"
        )

    async def upgrade_quota(self, user: User, new_tier: str) -> bool:
        """Upgrade user's quota tier.

        Args:
            user: User to upgrade
            new_tier: New tier to upgrade to

        Returns:
            True if upgrade was successful
        """
        if new_tier not in self.quota_limits:
            logger.error(f"Invalid tier for upgrade: {new_tier}")
            return False

        # Create upgrade record
        upgrade = QuotaUpgrade(
            user_id=user.id,
            current_tier=user.tier.value,
            requested_tier=new_tier,
            timestamp=datetime.now(UTC),
            approved=True,  # Auto-approve for now
            processed=False,
        )

        # Store upgrade record
        upgrade_key = f"quota_upgrade:{user.id}:{int(upgrade.timestamp.timestamp())}"
        await self.redis.setex(
            upgrade_key,
            86400 * 30,  # Keep for 30 days
            json.dumps(
                {
                    "user_id": upgrade.user_id,
                    "current_tier": upgrade.current_tier,
                    "requested_tier": upgrade.requested_tier,
                    "timestamp": upgrade.timestamp.isoformat(),
                    "approved": upgrade.approved,
                    "processed": upgrade.processed,
                }
            ),
        )

        # In a real system, this would update the user's tier in the database
        # For now, we just log the upgrade request
        logger.info(f"Quota upgrade requested for user {user.id}: {upgrade.current_tier} -> {upgrade.requested_tier}")

        return True

    async def get_quota_stats(self, user: User) -> dict[str, Any]:
        """Get current quota statistics for user.

        Args:
            user: User to get stats for

        Returns:
            Dictionary with quota statistics
        """
        limits = self.quota_limits.get(user.tier.value, self.quota_limits[UserTier.FREE.value])
        usage = await self._get_current_usage(user, limits)

        daily_percentage = (usage.daily_used / limits.daily_limit) * 100 if limits.daily_limit > 0 else 0
        monthly_percentage = (usage.monthly_used / limits.monthly_limit) * 100 if limits.monthly_limit > 0 else 0

        return {
            "user_id": user.id,
            "tier": user.tier.value,
            "daily": {
                "used": usage.daily_used,
                "limit": limits.daily_limit,
                "remaining": max(0, limits.daily_limit - usage.daily_used),
                "percentage": daily_percentage,
            },
            "monthly": {
                "used": usage.monthly_used,
                "limit": limits.monthly_limit,
                "remaining": max(0, limits.monthly_limit - usage.monthly_used),
                "percentage": monthly_percentage,
            },
            "status": self._calculate_status(daily_percentage, monthly_percentage).value,
            "last_reset": usage.last_reset_date.isoformat(),
        }

    async def _get_current_usage(self, user: User, limits: QuotaLimits) -> QuotaUsage:
        """Get current quota usage for user.

        Args:
            user: User to get usage for
            limits: Quota limits for user's tier

        Returns:
            Current quota usage
        """
        key = f"quota:{user.id}"
        now = datetime.now(UTC)

        # Get stored usage data
        usage_data = await cast(
            "Awaitable[list[Any]]",
            self.redis.hmget(
                key,
                ["daily_used", "monthly_used", "last_daily_reset", "last_monthly_reset"],
            ),
        )

        daily_used = int(usage_data[0]) if usage_data[0] else 0
        monthly_used = int(usage_data[1]) if usage_data[1] else 0
        last_daily_reset = datetime.fromtimestamp(int(usage_data[2]), tz=UTC) if usage_data[2] else now
        last_monthly_reset = datetime.fromtimestamp(int(usage_data[3]), tz=UTC) if usage_data[3] else now

        # Check if we need to reset daily usage
        if now.date() > last_daily_reset.date():
            daily_used = 0
            await cast("Awaitable[int]", self.redis.hset(key, "daily_used", "0"))
            await cast("Awaitable[int]", self.redis.hset(key, "last_daily_reset", str(int(now.timestamp()))))
            last_daily_reset = now

        # Check if we need to reset monthly usage
        if now.month != last_monthly_reset.month or now.year != last_monthly_reset.year:
            monthly_used = 0
            daily_used = 0  # Reset daily too
            await cast("Awaitable[int]", self.redis.hset(key, "monthly_used", "0"))
            await cast("Awaitable[int]", self.redis.hset(key, "daily_used", "0"))
            await cast("Awaitable[int]", self.redis.hset(key, "last_monthly_reset", str(int(now.timestamp()))))
            await cast("Awaitable[int]", self.redis.hset(key, "last_daily_reset", str(int(now.timestamp()))))
            last_monthly_reset = now
            last_daily_reset = now

        return QuotaUsage(
            user_id=user.id,
            daily_used=daily_used,
            monthly_used=monthly_used,
            daily_limit=limits.daily_limit,
            monthly_limit=limits.monthly_limit,
            last_reset_date=last_daily_reset,
            current_date=now,
        )

    async def _increment_usage(self, user: User, amount: int) -> None:
        """Increment usage counters for user.

        Args:
            user: User to increment usage for
            amount: Amount to increment by
        """
        key = f"quota:{user.id}"
        now = datetime.now(UTC)

        # Use pipeline for atomic operations
        pipe = self.redis.pipeline()
        pipe.hincrby(key, "daily_used", amount)
        pipe.hincrby(key, "monthly_used", amount)
        pipe.hset(key, "last_access", str(int(now.timestamp())))
        pipe.expire(key, 86400 * 35)  # Keep for 35 days
        await pipe.execute()

    async def _check_and_send_alerts(self, user: User, quota_result: QuotaResult) -> None:
        """Check if alerts should be sent and send them.

        Args:
            user: User to check alerts for
            quota_result: Result of quota check
        """
        for threshold in self.alert_thresholds:
            # Check daily quota
            if quota_result.daily_percentage >= threshold and await self._should_send_alert(
                user, QuotaType.DAILY, threshold
            ):
                await self.send_quota_alert(user, quota_result.daily_percentage, QuotaType.DAILY)
                await self._record_alert_sent(user, QuotaType.DAILY, threshold)

            # Check monthly quota
            if quota_result.monthly_percentage >= threshold and await self._should_send_alert(
                user, QuotaType.MONTHLY, threshold
            ):
                await self.send_quota_alert(user, quota_result.monthly_percentage, QuotaType.MONTHLY)
                await self._record_alert_sent(user, QuotaType.MONTHLY, threshold)

    async def _should_send_alert(self, user: User, quota_type: QuotaType, threshold: int) -> bool:
        """Check if an alert should be sent.

        Args:
            user: User to check
            quota_type: Type of quota
            threshold: Alert threshold

        Returns:
            True if alert should be sent
        """
        # Check if we've already sent this alert today
        alert_key = f"quota_alert_sent:{user.id}:{quota_type.value}:{threshold}:{datetime.now(UTC).date()}"
        exists = await self.redis.exists(alert_key)
        return not exists

    async def _record_alert_sent(self, user: User, quota_type: QuotaType, threshold: int) -> None:
        """Record that an alert was sent.

        Args:
            user: User the alert was sent to
            quota_type: Type of quota
            threshold: Alert threshold
        """
        alert_key = f"quota_alert_sent:{user.id}:{quota_type.value}:{threshold}:{datetime.now(UTC).date()}"
        await self.redis.setex(alert_key, 86400, "1")  # Expire at end of day

    def _calculate_status(self, daily_percentage: float, monthly_percentage: float) -> QuotaStatus:
        """Calculate quota status from percentages.

        Args:
            daily_percentage: Daily usage percentage
            monthly_percentage: Monthly usage percentage

        Returns:
            Overall quota status
        """
        max_percentage = max(daily_percentage, monthly_percentage)

        if max_percentage > 100:
            return QuotaStatus.EXCEEDED
        if max_percentage >= 95:
            return QuotaStatus.CRITICAL
        if max_percentage >= 80:
            return QuotaStatus.WARNING
        return QuotaStatus.OK

    async def health_check(self) -> dict[str, Any]:
        """Check quota manager health.

        Returns:
            Health status dictionary
        """
        try:
            # Test Redis connectivity
            await self.redis.ping()

            # Count active quota entries
            quota_keys = await self.redis.keys("quota:*")

            return {
                "status": "healthy",
                "redis_connected": True,
                "active_quotas": len(quota_keys),
                "configured_tiers": len(self.quota_limits),
                "alert_thresholds": self.alert_thresholds,
            }
        except Exception as e:
            logger.error(f"Quota manager health check failed: {e}")
            return {
                "status": "unhealthy",
                "redis_connected": False,
                "error": str(e),
            }
