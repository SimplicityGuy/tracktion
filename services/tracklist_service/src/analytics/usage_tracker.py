"""Usage tracking and analytics for API requests."""

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

import redis.asyncio as redis
from services.tracklist_service.src.auth.models import User

logger = logging.getLogger(__name__)


class AggregationPeriod(Enum):
    """Time periods for usage aggregation."""

    HOUR = "hour"
    DAY = "day"
    MONTH = "month"


@dataclass
class UsageRecord:
    """Individual usage record for tracking API requests."""

    user_id: str
    timestamp: datetime
    endpoint: str
    method: str
    response_time: float
    status_code: int
    tokens_consumed: int
    bytes_processed: int = 0
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "endpoint": self.endpoint,
            "method": self.method,
            "response_time": self.response_time,
            "status_code": self.status_code,
            "tokens_consumed": self.tokens_consumed,
            "bytes_processed": self.bytes_processed,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UsageRecord":
        """Create from dictionary."""
        return cls(
            user_id=data["user_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            endpoint=data["endpoint"],
            method=data["method"],
            response_time=data["response_time"],
            status_code=data["status_code"],
            tokens_consumed=data["tokens_consumed"],
            bytes_processed=data.get("bytes_processed", 0),
            error_message=data.get("error_message"),
        )


@dataclass
class UsageStats:
    """Aggregated usage statistics for a user and time period."""

    user_id: str
    period: str
    start_time: datetime
    end_time: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_tokens: int
    total_bytes: int
    avg_response_time: float
    endpoints: dict[str, int]
    error_rate: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "period": self.period,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_tokens": self.total_tokens,
            "total_bytes": self.total_bytes,
            "avg_response_time": self.avg_response_time,
            "endpoints": self.endpoints,
            "error_rate": self.error_rate,
        }


@dataclass
class UsageReport:
    """Comprehensive usage report with cost calculation."""

    user_id: str
    period: str
    usage_stats: UsageStats
    cost_breakdown: dict[str, Decimal]
    total_cost: Decimal
    tier: str
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "period": self.period,
            "usage_stats": self.usage_stats.to_dict(),
            "cost_breakdown": {k: str(v) for k, v in self.cost_breakdown.items()},
            "total_cost": str(self.total_cost),
            "tier": self.tier,
            "recommendations": self.recommendations,
        }


class UsageTracker:
    """Comprehensive usage tracking and analytics system."""

    def __init__(self, redis_client: redis.Redis):
        """Initialize usage tracker.

        Args:
            redis_client: Redis client for data storage
        """
        self.redis = redis_client

        # Pricing tiers (cost per 1000 tokens)
        self.pricing = {
            "free": Decimal("0.00"),  # Free tier
            "premium": Decimal("0.002"),  # $0.002 per 1000 tokens
            "enterprise": Decimal("0.001"),  # $0.001 per 1000 tokens (volume discount)
        }

        # Base costs per request type
        self.base_costs = {
            "search": Decimal("0.0001"),  # $0.0001 per search
            "tracklist": Decimal("0.0005"),  # $0.0005 per tracklist extraction
            "batch": Decimal("0.001"),  # $0.001 per batch operation
        }

    async def track_request(
        self,
        user: User,
        endpoint: str,
        method: str,
        response_time: float,
        status_code: int,
        tokens_consumed: int,
        bytes_processed: int = 0,
        error_message: str | None = None,
    ) -> None:
        """Track individual API request.

        Args:
            user: User making the request
            endpoint: API endpoint called
            method: HTTP method used
            response_time: Request processing time in seconds
            status_code: HTTP response code
            tokens_consumed: Number of rate limit tokens consumed
            bytes_processed: Amount of data processed
            error_message: Error message if request failed
        """
        try:
            record = UsageRecord(
                user_id=user.id,
                timestamp=datetime.now(UTC),
                endpoint=endpoint,
                method=method,
                response_time=response_time,
                status_code=status_code,
                tokens_consumed=tokens_consumed,
                bytes_processed=bytes_processed,
                error_message=error_message,
            )

            # Store individual record
            await self._store_usage_record(record)

            # Update real-time aggregations
            await self._update_real_time_stats(record)

            logger.debug(f"Tracked request for user {user.id}: {endpoint} ({status_code})")

        except Exception as e:
            logger.error(f"Failed to track usage for user {user.id}: {e}")

    async def _store_usage_record(self, record: UsageRecord) -> None:
        """Store individual usage record."""
        # Use sorted set for time-based queries
        timestamp = int(record.timestamp.timestamp())
        key = f"usage_records:{record.user_id}"

        await self.redis.zadd(key, {json.dumps(record.to_dict()): timestamp})

        # Set expiration for data retention (90 days)
        await self.redis.expire(key, 90 * 24 * 3600)

    async def _update_real_time_stats(self, record: UsageRecord) -> None:
        """Update real-time statistics."""
        current_hour = record.timestamp.replace(minute=0, second=0, microsecond=0)
        hour_key = f"stats:hourly:{record.user_id}:{int(current_hour.timestamp())}"

        # Use pipeline for atomic updates
        pipe = self.redis.pipeline()
        pipe.hincrby(hour_key, "total_requests", 1)
        pipe.hincrby(hour_key, "total_tokens", record.tokens_consumed)
        pipe.hincrby(hour_key, "total_bytes", record.bytes_processed)
        pipe.hincrbyfloat(hour_key, "total_response_time", record.response_time)

        if record.status_code >= 200 and record.status_code < 400:
            pipe.hincrby(hour_key, "successful_requests", 1)
        else:
            pipe.hincrby(hour_key, "failed_requests", 1)

        # Track endpoint usage
        endpoint_key = f"endpoint:{record.endpoint.replace('/', '_')}"
        pipe.hincrby(hour_key, endpoint_key, 1)

        # Set expiration
        pipe.expire(hour_key, 24 * 3600)  # Keep hourly stats for 24 hours

        await pipe.execute()

    async def get_usage_stats(
        self,
        user_id: str,
        period: AggregationPeriod,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> UsageStats:
        """Get aggregated usage statistics for a user.

        Args:
            user_id: User ID to get stats for
            period: Aggregation period (hour/day/month)
            start_time: Optional start time (defaults to current period start)
            end_time: Optional end time (defaults to current time)

        Returns:
            Aggregated usage statistics
        """
        if not start_time or not end_time:
            start_time, end_time = self._get_period_bounds(period)

        # Get usage records for the period
        records = await self._get_usage_records(user_id, start_time, end_time)

        if not records:
            return UsageStats(
                user_id=user_id,
                period=period.value,
                start_time=start_time,
                end_time=end_time,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                total_tokens=0,
                total_bytes=0,
                avg_response_time=0.0,
                endpoints={},
                error_rate=0.0,
            )

        # Aggregate statistics
        total_requests = len(records)
        successful_requests = sum(1 for r in records if 200 <= r.status_code < 400)
        failed_requests = total_requests - successful_requests
        total_tokens = sum(r.tokens_consumed for r in records)
        total_bytes = sum(r.bytes_processed for r in records)
        avg_response_time = sum(r.response_time for r in records) / total_requests

        # Endpoint breakdown
        endpoints: dict[str, int] = {}
        for record in records:
            endpoints[record.endpoint] = endpoints.get(record.endpoint, 0) + 1

        error_rate = (failed_requests / total_requests) * 100 if total_requests > 0 else 0.0

        return UsageStats(
            user_id=user_id,
            period=period.value,
            start_time=start_time,
            end_time=end_time,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            total_tokens=total_tokens,
            total_bytes=total_bytes,
            avg_response_time=avg_response_time,
            endpoints=endpoints,
            error_rate=error_rate,
        )

    async def _get_usage_records(self, user_id: str, start_time: datetime, end_time: datetime) -> list[UsageRecord]:
        """Get usage records for a time period."""
        key = f"usage_records:{user_id}"
        start_ts = int(start_time.timestamp())
        end_ts = int(end_time.timestamp())

        # Get records from sorted set
        raw_records = await self.redis.zrangebyscore(key, start_ts, end_ts)

        records = []
        for raw_record in raw_records:
            try:
                record_data = json.loads(raw_record)
                records.append(UsageRecord.from_dict(record_data))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse usage record: {e}")
                continue

        return records

    def calculate_cost(self, usage: UsageStats, tier: str) -> dict[str, Decimal]:
        """Calculate cost breakdown for usage statistics.

        Args:
            usage: Usage statistics to calculate cost for
            tier: User tier (free/premium/enterprise)

        Returns:
            Dictionary with cost breakdown
        """
        tier_lower = tier.lower()
        token_rate = self.pricing.get(tier_lower, self.pricing["premium"])

        cost_breakdown = {}

        # Token-based cost
        token_cost = (Decimal(str(usage.total_tokens)) / 1000) * token_rate
        cost_breakdown["tokens"] = token_cost

        # Request-based costs
        request_cost = Decimal("0")
        for endpoint, count in usage.endpoints.items():
            endpoint_type = self._get_endpoint_type(endpoint)
            base_cost = self.base_costs.get(endpoint_type, self.base_costs["search"])
            endpoint_cost = Decimal(str(count)) * base_cost
            cost_breakdown[f"requests_{endpoint_type}"] = endpoint_cost
            request_cost += endpoint_cost

        cost_breakdown["requests_total"] = request_cost

        # Data processing cost (per GB)
        data_cost = (Decimal(str(usage.total_bytes)) / (1024**3)) * Decimal("0.01")
        cost_breakdown["data_processing"] = data_cost

        # Free tier gets everything free
        if tier_lower == "free":
            cost_breakdown = {k: Decimal("0") for k in cost_breakdown}

        return cost_breakdown

    def _get_endpoint_type(self, endpoint: str) -> str:
        """Determine endpoint type from endpoint path."""
        if "search" in endpoint.lower():
            return "search"
        if "tracklist" in endpoint.lower():
            return "tracklist"
        if "batch" in endpoint.lower():
            return "batch"
        return "search"  # default

    async def generate_usage_report(
        self,
        user: User,
        period: AggregationPeriod,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> UsageReport:
        """Generate comprehensive usage report with cost analysis.

        Args:
            user: User to generate report for
            period: Time period for report
            start_time: Optional start time
            end_time: Optional end time

        Returns:
            Complete usage report with recommendations
        """
        usage_stats = await self.get_usage_stats(user.id, period, start_time, end_time)

        cost_breakdown = self.calculate_cost(usage_stats, user.tier.value)
        total_cost = Decimal("0")
        for cost in cost_breakdown.values():
            total_cost += cost

        recommendations = self._generate_recommendations(usage_stats, user.tier.value)

        return UsageReport(
            user_id=user.id,
            period=period.value,
            usage_stats=usage_stats,
            cost_breakdown=cost_breakdown,
            total_cost=total_cost,
            tier=user.tier.value,
            recommendations=recommendations,
        )

    def _generate_recommendations(self, usage: UsageStats, tier: str) -> list[str]:
        """Generate usage optimization recommendations."""
        recommendations = []

        # High error rate
        if usage.error_rate > 10:
            recommendations.append(
                f"High error rate ({usage.error_rate:.1f}%). Review API calls and implement proper error handling."
            )

        # Slow response times
        if usage.avg_response_time > 2.0:
            recommendations.append(
                f"Average response time is {usage.avg_response_time:.2f}s. "
                "Consider optimizing queries or using caching."
            )

        # Heavy usage patterns
        if usage.total_requests > 10000 and tier == "free":
            recommendations.append("Consider upgrading to Premium tier for better rates and higher limits.")
        elif usage.total_requests > 100000 and tier == "premium":
            recommendations.append("Consider upgrading to Enterprise tier for volume discounts.")

        # Endpoint optimization
        most_used_endpoint = max(usage.endpoints.items(), key=lambda x: x[1]) if usage.endpoints else None
        if most_used_endpoint and most_used_endpoint[1] > usage.total_requests * 0.5:
            recommendations.append(
                f"Optimize {most_used_endpoint[0]} endpoint usage - it represents "
                f"{(most_used_endpoint[1] / usage.total_requests) * 100:.1f}% of requests."
            )

        return recommendations

    def _get_period_bounds(self, period: AggregationPeriod) -> tuple[datetime, datetime]:
        """Get start and end times for aggregation period."""
        now = datetime.now(UTC)

        if period == AggregationPeriod.HOUR:
            start_time = now.replace(minute=0, second=0, microsecond=0)
            end_time = now
        elif period == AggregationPeriod.DAY:
            start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_time = now
        elif period == AggregationPeriod.MONTH:
            start_time = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            end_time = now
        else:
            # Default to current hour
            start_time = now.replace(minute=0, second=0, microsecond=0)
            end_time = now

        return start_time, end_time

    async def cleanup_old_data(self, days_to_keep: int = 90) -> int:
        """Clean up old usage data beyond retention period.

        Args:
            days_to_keep: Number of days of data to retain

        Returns:
            Number of records cleaned up
        """
        cutoff_time = datetime.now(UTC).timestamp() - (days_to_keep * 24 * 3600)
        cleanup_count = 0

        try:
            # Find all usage record keys
            pattern = "usage_records:*"
            async for key in self.redis.scan_iter(match=pattern):
                # Remove old records from sorted set
                removed = await self.redis.zremrangebyscore(key, 0, cutoff_time)
                cleanup_count += removed

            logger.info(f"Cleaned up {cleanup_count} old usage records")
            return cleanup_count

        except Exception as e:
            logger.error(f"Failed to cleanup old usage data: {e}")
            return 0

    async def health_check(self) -> dict[str, Any]:
        """Check usage tracker health and Redis connectivity.

        Returns:
            Health status dictionary
        """
        try:
            # Test Redis connectivity
            await self.redis.ping()

            # Get approximate record count
            record_count = 0
            pattern = "usage_records:*"
            async for key in self.redis.scan_iter(match=pattern):
                count = await self.redis.zcard(key)
                record_count += count

            return {
                "status": "healthy",
                "redis_connected": True,
                "total_records": record_count,
                "supported_periods": [p.value for p in AggregationPeriod],
                "pricing_tiers": list(self.pricing.keys()),
            }

        except Exception as e:
            logger.error(f"Usage tracker health check failed: {e}")
            return {
                "status": "unhealthy",
                "redis_connected": False,
                "error": str(e),
            }
