"""Abuse detection system for identifying malicious behavior patterns."""

import json
import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import redis.asyncio as redis
from fastapi import Request

from src.auth.models import User

from .models import AbuseScore, AbuseType

logger = logging.getLogger(__name__)


class AbuseDetector:
    """Advanced abuse detection using behavioral analysis."""

    def __init__(self, redis_client: redis.Redis):
        """Initialize abuse detector.

        Args:
            redis_client: Redis client for data storage
        """
        self.redis = redis_client

        # Detection thresholds
        self.thresholds = {
            "high_frequency_rpm": 1000,  # Requests per minute
            "suspicious_pattern_ratio": 0.3,  # Pattern deviation threshold
            "invalid_requests_ratio": 0.5,  # Failed requests ratio
            "resource_abuse_size": 100 * 1024 * 1024,  # 100MB per minute
            "quota_exhaustion_ratio": 0.95,  # 95% quota usage
            "consecutive_errors": 20,  # Max consecutive errors
            "velocity_window_minutes": 10,  # Velocity check window
            "geo_anomaly_threshold": 3,  # Max countries per hour
        }

        # Redis key prefixes
        self.behavior_key = "abuse:behavior"
        self.patterns_key = "abuse:patterns"
        self.scores_key = "abuse:scores"

    async def analyze_user_behavior(self, user: User, current_request: Request) -> AbuseScore:
        """Analyze user behavior for abuse patterns.

        Args:
            user: User to analyze
            current_request: Current HTTP request

        Returns:
            Abuse score with detailed analysis
        """
        try:
            # Collect behavioral data
            behavior_data = await self._collect_behavior_data(user, current_request)

            # Run all abuse detection checks
            abuse_types = []
            details = {}
            total_score = Decimal("0.0")

            # 1. High frequency detection
            freq_score, freq_abuse = await self._detect_high_frequency(user, behavior_data)
            if freq_abuse:
                abuse_types.append(AbuseType.HIGH_FREQUENCY)
            total_score += freq_score
            details["frequency"] = {
                "score": str(freq_score),
                "rpm": behavior_data.get("requests_per_minute", 0),
            }

            # 2. Suspicious pattern detection
            pattern_score, pattern_abuse = await self._detect_suspicious_patterns(user, behavior_data)
            if pattern_abuse:
                abuse_types.append(AbuseType.SUSPICIOUS_PATTERN)
            total_score += pattern_score
            details["patterns"] = {
                "score": str(pattern_score),
                "anomalies": behavior_data.get("pattern_anomalies", []),
            }

            # 3. Invalid request detection
            invalid_score, invalid_abuse = await self._detect_invalid_requests(user, behavior_data)
            if invalid_abuse:
                abuse_types.append(AbuseType.INVALID_REQUESTS)
            total_score += invalid_score
            details["invalid_requests"] = {
                "score": str(invalid_score),
                "error_rate": behavior_data.get("error_rate", 0),
            }

            # 4. Resource abuse detection
            resource_score, resource_abuse = await self._detect_resource_abuse(user, behavior_data)
            if resource_abuse:
                abuse_types.append(AbuseType.RESOURCE_ABUSE)
            total_score += resource_score
            details["resource_abuse"] = {
                "score": str(resource_score),
                "data_usage": behavior_data.get("data_usage_mb", 0),
            }

            # 5. Quota exhaustion detection
            quota_score, quota_abuse = await self._detect_quota_exhaustion(user, behavior_data)
            if quota_abuse:
                abuse_types.append(AbuseType.QUOTA_EXHAUSTION)
            total_score += quota_score
            details["quota"] = {
                "score": str(quota_score),
                "quota_usage": behavior_data.get("quota_usage_percent", 0),
            }

            # Normalize score to 0-1 range
            final_score = min(total_score, Decimal("1.0"))

            # Determine if user should be blocked
            should_block = self._should_block_user(final_score, abuse_types, behavior_data)

            # Generate recommendation
            recommendation = self._generate_recommendation(final_score, abuse_types, details)

            # Store analysis result
            await self._store_abuse_score(user, final_score, abuse_types, details)

            return AbuseScore(
                user_id=user.id,
                score=final_score,
                abuse_types=abuse_types,
                details=details,
                should_block=should_block,
                recommendation=recommendation,
            )

        except Exception as e:
            logger.error(f"Error analyzing user behavior for {user.id}: {e}")
            return AbuseScore(
                user_id=user.id,
                score=Decimal("0.0"),
                abuse_types=[],
                details={"error": str(e)},
                should_block=False,
                recommendation="Error occurred during analysis",
            )

    async def _collect_behavior_data(self, user: User, request: Request) -> dict[str, Any]:
        """Collect behavioral data for analysis."""
        now = datetime.now(UTC)
        hour_start = now.replace(minute=0, second=0, microsecond=0)
        minute_start = now.replace(second=0, microsecond=0)

        behavior_data: dict[str, Any] = {
            "timestamp": now.isoformat(),
            "ip_address": self._get_client_ip(request),
            "user_agent": request.headers.get("user-agent", ""),
            "endpoint": str(request.url.path),
            "method": request.method,
        }

        try:
            # Get recent request statistics

            # Requests in last minute
            minute_requests = await self._count_requests_in_window(user, minute_start, now)
            behavior_data["requests_per_minute"] = minute_requests

            # Requests in last hour
            hour_requests = await self._count_requests_in_window(user, hour_start, now)
            behavior_data["requests_per_hour"] = hour_requests

            # Error rate in last hour
            error_count = await self._count_errors_in_window(user, hour_start, now)
            behavior_data["error_rate"] = error_count / max(hour_requests, 1)
            behavior_data["error_count"] = error_count

            # Data usage in last hour (estimated)
            data_usage = await self._estimate_data_usage(user, hour_start, now)
            behavior_data["data_usage_mb"] = data_usage / (1024 * 1024)

            # Pattern analysis
            patterns = await self._analyze_request_patterns(user, hour_start, now)
            behavior_data["pattern_anomalies"] = patterns

            # Geographic analysis
            geo_data = await self._analyze_geographic_patterns(user, hour_start, now)
            behavior_data["geo_anomalies"] = geo_data

            # Quota usage (estimated)
            quota_usage = await self._estimate_quota_usage(user)
            behavior_data["quota_usage_percent"] = quota_usage

            # Store current behavior point
            await self._store_behavior_point(user, behavior_data)

        except Exception as e:
            logger.warning(f"Error collecting behavior data for user {user.id}: {e}")
            behavior_data["collection_error"] = str(e)

        return behavior_data

    async def _detect_high_frequency(self, user: User, data: dict[str, Any]) -> tuple[Decimal, bool]:
        """Detect high frequency abuse."""
        rpm = data.get("requests_per_minute", 0)
        threshold = self.thresholds["high_frequency_rpm"]

        if rpm > threshold:
            # Calculate score based on how much over threshold
            score = Decimal(str(min(rpm / threshold, 3.0))) * Decimal("0.4")  # Max 40% of total score
            return score, True

        return Decimal("0.0"), False

    async def _detect_suspicious_patterns(self, user: User, data: dict[str, Any]) -> tuple[Decimal, bool]:
        """Detect suspicious request patterns."""
        anomalies = data.get("pattern_anomalies", [])

        if not anomalies:
            return Decimal("0.0"), False

        # Score based on number and severity of anomalies
        anomaly_score = len(anomalies) * 0.1

        # Check for specific suspicious patterns
        suspicious_count = 0
        for anomaly in anomalies:
            if "bot_like" in anomaly or "scraping" in anomaly or "automated" in anomaly:
                suspicious_count += 1

        if suspicious_count > 0:
            score = Decimal(str(min(anomaly_score + suspicious_count * 0.1, 0.3)))  # Max 30% of total score
            return score, True

        return Decimal("0.0"), False

    async def _detect_invalid_requests(self, user: User, data: dict[str, Any]) -> tuple[Decimal, bool]:
        """Detect invalid request abuse."""
        error_rate = data.get("error_rate", 0)
        threshold = self.thresholds["invalid_requests_ratio"]

        if error_rate > threshold:
            score = Decimal(str(min(error_rate, 1.0))) * Decimal("0.25")  # Max 25% of total score
            return score, True

        return Decimal("0.0"), False

    async def _detect_resource_abuse(self, user: User, data: dict[str, Any]) -> tuple[Decimal, bool]:
        """Detect resource abuse."""
        data_usage_mb = data.get("data_usage_mb", 0)
        threshold_mb = self.thresholds["resource_abuse_size"] / (1024 * 1024)

        if data_usage_mb > threshold_mb:
            score = Decimal(str(min(data_usage_mb / threshold_mb, 2.0))) * Decimal("0.2")  # Max 20% of total score
            return score, True

        return Decimal("0.0"), False

    async def _detect_quota_exhaustion(self, user: User, data: dict[str, Any]) -> tuple[Decimal, bool]:
        """Detect quota exhaustion abuse."""
        quota_usage = data.get("quota_usage_percent", 0)
        threshold = self.thresholds["quota_exhaustion_ratio"]

        if quota_usage > threshold:
            score = Decimal(str((quota_usage - threshold) / (1.0 - threshold))) * Decimal(
                "0.15"
            )  # Max 15% of total score
            return score, True

        return Decimal("0.0"), False

    def _should_block_user(self, score: Decimal, abuse_types: list[AbuseType], data: dict[str, Any]) -> bool:
        """Determine if user should be automatically blocked."""
        # High score threshold
        if score >= Decimal("0.8"):
            return True

        # Multiple abuse types
        if len(abuse_types) >= 3:
            return True

        # Specific critical patterns
        if AbuseType.HIGH_FREQUENCY in abuse_types and AbuseType.INVALID_REQUESTS in abuse_types:
            return True

        # Very high error rate
        return bool(float(data.get("error_rate", 0)) > 0.8)

    def _generate_recommendation(self, score: Decimal, abuse_types: list[AbuseType], details: dict[str, Any]) -> str:
        """Generate recommendation based on analysis."""
        if score < Decimal("0.2"):
            return "Normal behavior - no action needed"
        if score < Decimal("0.5"):
            return "Monitor user activity - low risk"
        if score < Decimal("0.8"):
            recommendations = []

            if AbuseType.HIGH_FREQUENCY in abuse_types:
                recommendations.append("implement stricter rate limiting")
            if AbuseType.INVALID_REQUESTS in abuse_types:
                recommendations.append("review API integration")
            if AbuseType.RESOURCE_ABUSE in abuse_types:
                recommendations.append("limit resource-intensive operations")
            if AbuseType.SUSPICIOUS_PATTERN in abuse_types:
                recommendations.append("verify legitimate usage")

            return f"Medium risk - consider: {', '.join(recommendations)}"
        return "High risk - recommend immediate blocking or manual review"

    async def _count_requests_in_window(self, user: User, start: datetime, end: datetime) -> int:
        """Count requests in time window."""
        try:
            # In a real implementation, this would query actual request logs
            # For now, simulate based on stored behavior points
            user_key = f"{self.behavior_key}:{user.id}"

            start_ts = int(start.timestamp())
            end_ts = int(end.timestamp())

            # Count behavior points in window
            count = await self.redis.zcount(user_key, start_ts, end_ts)
            return int(count)

        except Exception as e:
            logger.warning(f"Error counting requests for user {user.id}: {e}")
            return 0

    async def _count_errors_in_window(self, user: User, start: datetime, end: datetime) -> int:
        """Count error requests in time window."""
        try:
            # Simulate error counting - in reality would query actual logs
            error_key = f"{self.behavior_key}:{user.id}:errors"

            start_ts = int(start.timestamp())
            end_ts = int(end.timestamp())

            count = await self.redis.zcount(error_key, start_ts, end_ts)
            return int(count)

        except Exception as e:
            logger.warning(f"Error counting errors for user {user.id}: {e}")
            return 0

    async def _estimate_data_usage(self, user: User, start: datetime, end: datetime) -> int:
        """Estimate data usage in bytes."""
        try:
            # Simulate data usage estimation
            # In reality would sum actual request/response sizes
            request_count = await self._count_requests_in_window(user, start, end)

            # Estimate average 10KB per request
            return request_count * 10 * 1024

        except Exception as e:
            logger.warning(f"Error estimating data usage for user {user.id}: {e}")
            return 0

    async def _analyze_request_patterns(self, user: User, start: datetime, end: datetime) -> list[str]:
        """Analyze request patterns for anomalies."""
        try:
            # Simulate pattern analysis
            anomalies = []

            request_count = await self._count_requests_in_window(user, start, end)

            # Check for bot-like patterns (very regular intervals)
            if request_count > 100:
                # In reality, would analyze actual request timing patterns
                anomalies.append("high_frequency_pattern")

            # Check for scraping patterns (sequential endpoint access)
            # This would require analyzing actual endpoint sequences

            return anomalies

        except Exception as e:
            logger.warning(f"Error analyzing patterns for user {user.id}: {e}")
            return []

    async def _analyze_geographic_patterns(self, user: User, start: datetime, end: datetime) -> list[str]:
        """Analyze geographic access patterns."""
        try:
            # Simulate geo analysis
            # In reality, would track IP addresses and geolocate them
            anomalies: list[str] = []

            # This is a placeholder - real implementation would:
            # 1. Track IP addresses used by user
            # 2. Geolocate each IP
            # 3. Check for impossible travel times
            # 4. Flag access from multiple countries

            return anomalies

        except Exception as e:
            logger.warning(f"Error analyzing geo patterns for user {user.id}: {e}")
            return []

    async def _estimate_quota_usage(self, user: User) -> float:
        """Estimate current quota usage percentage."""
        try:
            # Simulate quota usage calculation
            # In reality, would query actual quota system

            # For now, estimate based on tier and recent activity
            tier_limits = {"free": 1000, "premium": 10000, "enterprise": 100000}

            limit = tier_limits.get(user.tier.value, 1000)

            # Estimate usage based on recent requests
            now = datetime.now(UTC)
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            monthly_requests = await self._count_requests_in_window(user, month_start, now)

            return min(monthly_requests / limit, 1.0)

        except Exception as e:
            logger.warning(f"Error estimating quota usage for user {user.id}: {e}")
            return 0.0

    async def _store_behavior_point(self, user: User, data: dict[str, Any]) -> None:
        """Store behavior data point for future analysis."""
        try:
            user_key = f"{self.behavior_key}:{user.id}"
            timestamp = int(datetime.now(UTC).timestamp())

            # Store in sorted set for time-based queries
            await self.redis.zadd(user_key, {json.dumps(data): timestamp})

            # Keep only last 24 hours of data
            cutoff = timestamp - (24 * 3600)
            await self.redis.zremrangebyscore(user_key, 0, cutoff)

            # Store errors separately for faster counting
            if data.get("error_rate", 0) > 0:
                error_key = f"{self.behavior_key}:{user.id}:errors"
                await self.redis.zadd(error_key, {"error": timestamp})
                await self.redis.zremrangebyscore(error_key, 0, cutoff)

        except Exception as e:
            logger.warning(f"Error storing behavior point for user {user.id}: {e}")

    async def _store_abuse_score(
        self,
        user: User,
        score: Decimal,
        abuse_types: list[AbuseType],
        details: dict[str, Any],
    ) -> None:
        """Store abuse score for tracking."""
        try:
            score_data = {
                "user_id": user.id,
                "score": str(score),
                "abuse_types": [t.value for t in abuse_types],
                "details": details,
                "timestamp": datetime.now(UTC).isoformat(),
            }

            score_key = f"{self.scores_key}:{user.id}"
            timestamp = int(datetime.now(UTC).timestamp())

            await self.redis.zadd(score_key, {json.dumps(score_data): timestamp})

            # Keep only last 7 days of scores
            cutoff = timestamp - (7 * 24 * 3600)
            await self.redis.zremrangebyscore(score_key, 0, cutoff)

        except Exception as e:
            logger.warning(f"Error storing abuse score for user {user.id}: {e}")

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return str(forwarded_for.split(",")[0].strip())

        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return str(real_ip)

        # Fallback to direct connection
        if hasattr(request, "client") and request.client:
            return str(request.client.host)

        return "unknown"

    async def get_user_abuse_history(self, user: User, days: int = 7) -> list[AbuseScore]:
        """Get abuse score history for user.

        Args:
            user: User to get history for
            days: Number of days to look back

        Returns:
            List of historical abuse scores
        """
        try:
            score_key = f"{self.scores_key}:{user.id}"
            cutoff_time = datetime.now(UTC) - timedelta(days=days)
            cutoff_ts = int(cutoff_time.timestamp())

            # Get scores from Redis
            scores_data = await self.redis.zrangebyscore(score_key, cutoff_ts, "+inf")

            abuse_scores = []
            for score_json in scores_data:
                try:
                    score_data = json.loads(score_json)
                    abuse_score = AbuseScore(
                        user_id=score_data["user_id"],
                        score=Decimal(score_data["score"]),
                        abuse_types=[AbuseType(t) for t in score_data["abuse_types"]],
                        details=score_data["details"],
                        should_block=False,  # Historical data
                        recommendation=score_data.get("recommendation", "Historical record"),
                    )
                    abuse_scores.append(abuse_score)

                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Error parsing abuse score: {e}")

            return abuse_scores

        except Exception as e:
            logger.error(f"Error getting abuse history for user {user.id}: {e}")
            return []
