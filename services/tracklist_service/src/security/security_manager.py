"""Security manager for comprehensive security features."""

import hashlib
import hmac
import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from ipaddress import AddressValueError, IPv4Address, IPv6Address
from typing import Any

import redis.asyncio as redis
from fastapi import Request, Response

from src.auth.models import User

from .abuse_detector import AbuseDetector, AbuseScore
from .models import (
    AccessRuleType,
    AuditEventType,
    AuditLog,
    IPAccessRule,
    SecurityConfig,
)

logger = logging.getLogger(__name__)


class SecurityManager:
    """Comprehensive security management system."""

    def __init__(self, redis_client: redis.Redis, config: SecurityConfig | None = None):
        """Initialize security manager.

        Args:
            redis_client: Redis client for data storage
            config: Security configuration settings
        """
        self.redis = redis_client
        self.config = config or SecurityConfig(hmac_secret_key="default-secret-change-in-production")
        self.abuse_detector = AbuseDetector(redis_client)

        # Redis key prefixes
        self.ip_rules_key = "security:ip_rules"
        self.audit_logs_key = "security:audit_logs"
        self.blocked_users_key = "security:blocked_users"

    async def verify_request_signature(
        self,
        request: Request,
        secret: str | None = None,
        timestamp_header: str = "X-Timestamp",
        signature_header: str = "X-Signature",
    ) -> bool:
        """Verify HMAC signature of request.

        Args:
            request: FastAPI request object
            secret: Secret key for HMAC (uses config default if None)
            timestamp_header: Header name for timestamp
            signature_header: Header name for signature

        Returns:
            True if signature is valid and not expired
        """
        try:
            # Get headers
            timestamp = request.headers.get(timestamp_header)
            signature = request.headers.get(signature_header)

            if not timestamp or not signature:
                logger.warning(f"Missing signature headers from {self._get_client_ip(request)}")
                return False

            # Check timestamp age
            try:
                request_time = datetime.fromtimestamp(float(timestamp), UTC)
                age = (datetime.now(UTC) - request_time).total_seconds()

                if age > self.config.max_signature_age:
                    logger.warning(f"Request signature too old: {age}s from {self._get_client_ip(request)}")
                    return False

            except (ValueError, OverflowError):
                logger.warning(f"Invalid timestamp format from {self._get_client_ip(request)}")
                return False

            # Get request body
            body = await request.body()

            # Create signature payload
            method = request.method
            path = str(request.url.path)
            query = str(request.url.query) if request.url.query else ""

            payload = f"{method}\n{path}\n{query}\n{timestamp}\n{body.decode('utf-8', errors='ignore')}"

            # Calculate expected signature
            secret_key = secret or self.config.hmac_secret_key
            expected_signature = hmac.new(
                secret_key.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
            ).hexdigest()

            # Compare signatures
            is_valid = hmac.compare_digest(signature, expected_signature)

            if not is_valid:
                logger.warning(f"Invalid signature from {self._get_client_ip(request)}")
                await self.log_security_violation(
                    request,
                    None,
                    "invalid_signature",
                    {"expected_format": "HMAC-SHA256", "timestamp": timestamp},
                )

            return is_valid

        except Exception as e:
            logger.error(f"Error verifying request signature: {e}")
            return False

    async def check_ip_access(self, ip: str, user: User | None = None) -> bool:
        """Check if IP address is allowed access.

        Args:
            ip: IP address to check
            user: User making request (optional)

        Returns:
            True if access is allowed
        """
        try:
            if not self.config.ip_whitelist_enabled and not self.config.ip_blacklist_enabled:
                return True

            # Normalize IP address
            try:
                normalized_ip = str(IPv4Address(ip)) if "." in ip else str(IPv6Address(ip))
            except AddressValueError:
                logger.warning(f"Invalid IP address format: {ip}")
                return False

            # Get IP access rules from Redis
            hgetall_result = await self.redis.hgetall(f"{self.ip_rules_key}:{normalized_ip}")
            rules_data = dict(hgetall_result) if hgetall_result else {}

            if rules_data:
                rule = IPAccessRule.from_dict(rules_data)

                # Check if rule is active and not expired
                if not rule.is_active or rule.is_expired():
                    # Clean up expired/inactive rule
                    await self.redis.hdel(self.ip_rules_key, normalized_ip)
                # Apply rule
                elif rule.rule_type == AccessRuleType.BLACKLIST:
                    logger.info(f"IP {ip} blocked by blacklist rule: {rule.reason}")
                    return False
                elif rule.rule_type == AccessRuleType.WHITELIST:
                    return True

            # Check global whitelist mode
            if self.config.ip_whitelist_enabled:
                # In whitelist mode, only explicitly whitelisted IPs are allowed
                whitelist_rules = await self._get_active_rules(AccessRuleType.WHITELIST)

                for rule in whitelist_rules:
                    if rule.ip_address == normalized_ip:
                        return True

                # Not in whitelist
                logger.info(f"IP {ip} not in whitelist")
                return False

            # Default access policy
            return self.config.default_ip_access

        except Exception as e:
            logger.error(f"Error checking IP access for {ip}: {e}")
            return self.config.default_ip_access

    async def add_ip_rule(
        self,
        ip: str,
        rule_type: AccessRuleType,
        user_id: str | None = None,
        reason: str | None = None,
        expires_in_hours: int | None = None,
    ) -> IPAccessRule:
        """Add IP access rule.

        Args:
            ip: IP address
            rule_type: Whitelist or blacklist
            user_id: User who created the rule
            reason: Reason for the rule
            expires_in_hours: Hours until rule expires (None = permanent)

        Returns:
            Created IP access rule
        """
        try:
            # Normalize IP
            try:
                normalized_ip = str(IPv4Address(ip)) if "." in ip else str(IPv6Address(ip))
            except AddressValueError as e:
                raise ValueError(f"Invalid IP address format: {ip}") from e

            # Create rule
            expires_at = None
            if expires_in_hours:
                expires_at = datetime.now(UTC) + timedelta(hours=expires_in_hours)

            rule = IPAccessRule(
                ip_address=normalized_ip,
                rule_type=rule_type,
                user_id=user_id,
                reason=reason,
                expires_at=expires_at,
            )

            # Store in Redis
            await self.redis.hset(f"{self.ip_rules_key}:{normalized_ip}", mapping=rule.to_dict())

            # Set expiration if specified
            if expires_at:
                ttl = int((expires_at - datetime.now(UTC)).total_seconds())
                await self.redis.expire(f"{self.ip_rules_key}:{normalized_ip}", ttl)

            logger.info(f"Added {rule_type.value} rule for IP {ip}: {reason}")
            return rule

        except Exception as e:
            logger.error(f"Error adding IP rule for {ip}: {e}")
            raise

    async def remove_ip_rule(self, ip: str) -> bool:
        """Remove IP access rule.

        Args:
            ip: IP address to remove rule for

        Returns:
            True if rule was removed
        """
        try:
            # Normalize IP
            try:
                normalized_ip = str(IPv4Address(ip)) if "." in ip else str(IPv6Address(ip))
            except AddressValueError as e:
                raise ValueError(f"Invalid IP address format: {ip}") from e

            result = await self.redis.delete(f"{self.ip_rules_key}:{normalized_ip}")

            if result > 0:
                logger.info(f"Removed IP rule for {ip}")
                return True
            logger.info(f"No IP rule found for {ip}")
            return False

        except Exception as e:
            logger.error(f"Error removing IP rule for {ip}: {e}")
            return False

    async def detect_abuse(self, user: User, request: Request) -> AbuseScore:
        """Detect abuse patterns for user.

        Args:
            user: User to check
            request: Current request

        Returns:
            Abuse score and analysis
        """
        if not self.config.abuse_detection_enabled:
            return AbuseScore(
                user_id=user.id,
                score=Decimal("0.0"),
                abuse_types=[],
                details={},
                should_block=False,
                recommendation="Abuse detection disabled",
            )

        return await self.abuse_detector.analyze_user_behavior(user, request)

    async def auto_block_user(
        self,
        user: User,
        reason: str,
        duration_hours: int = 24,
        abuse_score: AbuseScore | None = None,
    ) -> None:
        """Automatically block user for security violations.

        Args:
            user: User to block
            reason: Reason for blocking
            duration_hours: How long to block for
            abuse_score: Abuse score that triggered the block
        """
        try:
            block_until = datetime.now(UTC) + timedelta(hours=duration_hours)

            # Store block in Redis
            block_data = {
                "user_id": user.id,
                "reason": reason,
                "blocked_at": datetime.now(UTC).isoformat(),
                "blocked_until": block_until.isoformat(),
                "abuse_score": abuse_score.to_dict() if abuse_score else None,
            }

            await self.redis.hset(
                f"{self.blocked_users_key}:{user.id}",
                mapping={k: json.dumps(v) if isinstance(v, dict) else str(v) for k, v in block_data.items()},
            )

            # Set expiration
            ttl = int((block_until - datetime.now(UTC)).total_seconds())
            await self.redis.expire(f"{self.blocked_users_key}:{user.id}", ttl)

            logger.warning(f"Auto-blocked user {user.id} for {duration_hours}h: {reason}")

        except Exception as e:
            logger.error(f"Error auto-blocking user {user.id}: {e}")

    async def is_user_blocked(self, user: User) -> tuple[bool, str | None]:
        """Check if user is currently blocked.

        Args:
            user: User to check

        Returns:
            Tuple of (is_blocked, reason)
        """
        try:
            hgetall_result = await self.redis.hgetall(f"{self.blocked_users_key}:{user.id}")
            block_data = dict(hgetall_result) if hgetall_result else {}

            if not block_data:
                return False, None

            # Check if block has expired
            blocked_until_str = block_data.get("blocked_until")
            if blocked_until_str:
                blocked_until = datetime.fromisoformat(blocked_until_str)
                if datetime.now(UTC) > blocked_until:
                    # Block expired, clean up
                    await self.redis.delete(f"{self.blocked_users_key}:{user.id}")
                    return False, None

            reason = block_data.get("reason", "Security violation")
            return True, reason

        except Exception as e:
            logger.error(f"Error checking if user {user.id} is blocked: {e}")
            return False, None

    async def log_api_access(
        self,
        request: Request,
        response: Response,
        user: User | None = None,
        api_key_id: str | None = None,
        jwt_token_id: str | None = None,
    ) -> None:
        """Log API access for audit trail.

        Args:
            request: FastAPI request
            response: FastAPI response
            user: Authenticated user (if any)
            api_key_id: API key used (if any)
            jwt_token_id: JWT token ID (if any)
        """
        if not self.config.audit_logging_enabled:
            return

        try:
            # Create audit log entry
            audit_log = AuditLog(
                event_id=str(uuid.uuid4()),
                event_type=AuditEventType.API_ACCESS,
                user_id=user.id if user else None,
                ip_address=self._get_client_ip(request),
                user_agent=request.headers.get("user-agent"),
                endpoint=str(request.url.path),
                method=request.method,
                status_code=response.status_code,
                api_key_id=api_key_id,
                jwt_token_id=jwt_token_id,
                authentication_method=("api_key" if api_key_id else "jwt" if jwt_token_id else "none"),
            )

            # Add request/response size if available
            if hasattr(request, "_body"):
                audit_log.request_size = len(request._body)

            # Store in Redis with day-based partitioning
            date_key = datetime.now(UTC).strftime("%Y-%m-%d")
            log_key = f"{self.audit_logs_key}:{date_key}:{audit_log.event_id}"

            await self.redis.hset(log_key, mapping=audit_log.to_dict())

            # Set expiration (90 days)
            await self.redis.expire(log_key, 90 * 24 * 3600)

            # Add to sorted set for time-based queries
            timestamp = datetime.now(UTC).timestamp()
            await self.redis.zadd(
                f"{self.audit_logs_key}:timeline:{date_key}",
                {audit_log.event_id: timestamp},
            )
            await self.redis.expire(f"{self.audit_logs_key}:timeline:{date_key}", 90 * 24 * 3600)

        except Exception as e:
            logger.error(f"Error logging API access: {e}")

    async def log_security_violation(
        self,
        request: Request,
        user: User | None,
        violation_type: str,
        details: dict[str, Any],
    ) -> None:
        """Log security violation.

        Args:
            request: FastAPI request
            user: User involved (if any)
            violation_type: Type of violation
            details: Additional details
        """
        try:
            audit_log = AuditLog(
                event_id=str(uuid.uuid4()),
                event_type=AuditEventType.SECURITY_VIOLATION,
                user_id=user.id if user else None,
                ip_address=self._get_client_ip(request),
                user_agent=request.headers.get("user-agent"),
                endpoint=str(request.url.path),
                method=request.method,
                status_code=None,  # No response status for security violations
                security_violation=violation_type,
                details=details,
            )

            audit_log.add_tag("security")
            audit_log.add_tag("violation")
            audit_log.add_tag(violation_type)

            # Store with higher priority (shorter key for faster access)
            date_key = datetime.now(UTC).strftime("%Y-%m-%d")
            log_key = f"{self.audit_logs_key}:violations:{date_key}:{audit_log.event_id}"

            await self.redis.hset(log_key, mapping=audit_log.to_dict())
            await self.redis.expire(log_key, 90 * 24 * 3600)

            # Add to violations timeline
            timestamp = datetime.now(UTC).timestamp()
            await self.redis.zadd(
                f"{self.audit_logs_key}:violations:timeline:{date_key}",
                {audit_log.event_id: timestamp},
            )
            await self.redis.expire(f"{self.audit_logs_key}:violations:timeline:{date_key}", 90 * 24 * 3600)

        except Exception as e:
            logger.error(f"Error logging security violation: {e}")

    async def get_audit_logs(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        event_type: AuditEventType | None = None,
        user_id: str | None = None,
        limit: int = 100,
    ) -> list[AuditLog]:
        """Retrieve audit logs with filters.

        Args:
            start_date: Start date for logs
            end_date: End date for logs
            event_type: Filter by event type
            user_id: Filter by user ID
            limit: Maximum number of logs to return

        Returns:
            List of audit logs
        """
        try:
            if not start_date:
                start_date = datetime.now(UTC) - timedelta(days=1)
            if not end_date:
                end_date = datetime.now(UTC)

            logs: list[AuditLog] = []
            current_date = start_date.date()
            end_date_date = end_date.date()

            while current_date <= end_date_date and len(logs) < limit:
                date_key = current_date.strftime("%Y-%m-%d")

                # Get log IDs from timeline
                timeline_key = f"{self.audit_logs_key}:timeline:{date_key}"
                start_ts = start_date.timestamp()
                end_ts = end_date.timestamp()

                log_ids = await self.redis.zrangebyscore(timeline_key, start_ts, end_ts, start=0, num=limit - len(logs))

                for log_id in log_ids:
                    log_key = f"{self.audit_logs_key}:{date_key}:{log_id}"
                    hgetall_result = await self.redis.hgetall(log_key)
                    log_data = dict(hgetall_result) if hgetall_result else {}

                    if log_data:
                        try:
                            audit_log = AuditLog.from_dict(log_data)

                            # Apply filters
                            if event_type and audit_log.event_type != event_type:
                                continue
                            if user_id and audit_log.user_id != user_id:
                                continue

                            logs.append(audit_log)

                        except Exception as e:
                            logger.warning(f"Error parsing audit log {log_id}: {e}")

                current_date += timedelta(days=1)

            return logs[:limit]

        except Exception as e:
            logger.error(f"Error retrieving audit logs: {e}")
            return []

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

    async def _get_active_rules(self, rule_type: AccessRuleType) -> list[IPAccessRule]:
        """Get all active rules of specified type."""
        try:
            # In a production system, this would query a database
            # For now, we'll scan Redis keys (not ideal for large datasets)
            rules = []
            pattern = f"{self.ip_rules_key}:*"

            async for key in self.redis.scan_iter(match=pattern):
                hgetall_result = await self.redis.hgetall(key)
                rule_data = dict(hgetall_result) if hgetall_result else {}
                if rule_data:
                    rule = IPAccessRule.from_dict(rule_data)
                    if rule.rule_type == rule_type and rule.is_active and not rule.is_expired():
                        rules.append(rule)

            return rules

        except Exception as e:
            logger.error(f"Error getting active {rule_type.value} rules: {e}")
            return []

    async def health_check(self) -> dict[str, Any]:
        """Check security manager health.

        Returns:
            Health status dictionary
        """
        try:
            # Test Redis connectivity
            await self.redis.ping()

            # Get stats
            ip_rules_count = len(await self.redis.keys(f"{self.ip_rules_key}:*"))
            blocked_users_count = len(await self.redis.keys(f"{self.blocked_users_key}:*"))

            return {
                "status": "healthy",
                "redis_connected": True,
                "config": {
                    "hmac_enabled": bool(self.config.hmac_secret_key != "default-secret-change-in-production"),
                    "ip_whitelist_enabled": self.config.ip_whitelist_enabled,
                    "ip_blacklist_enabled": self.config.ip_blacklist_enabled,
                    "abuse_detection_enabled": self.config.abuse_detection_enabled,
                    "audit_logging_enabled": self.config.audit_logging_enabled,
                },
                "stats": {
                    "ip_rules": ip_rules_count,
                    "blocked_users": blocked_users_count,
                },
            }

        except Exception as e:
            logger.error(f"Security manager health check failed: {e}")
            return {"status": "unhealthy", "redis_connected": False, "error": str(e)}
