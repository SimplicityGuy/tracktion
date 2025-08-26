"""Security-related data models."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
from decimal import Decimal


class AccessRuleType(Enum):
    """Type of IP access rule."""

    WHITELIST = "whitelist"
    BLACKLIST = "blacklist"


class AbuseType(Enum):
    """Types of detected abuse."""

    HIGH_FREQUENCY = "high_frequency"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    INVALID_REQUESTS = "invalid_requests"
    RESOURCE_ABUSE = "resource_abuse"
    QUOTA_EXHAUSTION = "quota_exhaustion"


class AuditEventType(Enum):
    """Types of audit events."""

    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    API_ACCESS = "api_access"
    RATE_LIMIT = "rate_limit"
    SECURITY_VIOLATION = "security_violation"
    ADMIN_ACTION = "admin_action"


@dataclass
class SecurityConfig:
    """Security configuration settings."""

    hmac_secret_key: str
    max_signature_age: int = 300  # 5 minutes
    ip_whitelist_enabled: bool = False
    ip_blacklist_enabled: bool = True
    abuse_detection_enabled: bool = True
    audit_logging_enabled: bool = True

    # Abuse detection thresholds
    max_requests_per_minute: int = 1000
    max_failed_requests_ratio: float = 0.5
    max_consecutive_errors: int = 20
    velocity_check_window_minutes: int = 10

    # IP access control
    default_ip_access: bool = True  # Allow by default
    geo_blocking_enabled: bool = False
    blocked_countries: Optional[List[str]] = None

    def __post_init__(self) -> None:
        if self.blocked_countries is None:
            self.blocked_countries = []


@dataclass
class IPAccessRule:
    """IP access control rule."""

    ip_address: str
    rule_type: AccessRuleType
    user_id: Optional[str] = None
    reason: Optional[str] = None
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

    def is_expired(self) -> bool:
        """Check if rule has expired."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "ip_address": self.ip_address,
            "rule_type": self.rule_type.value,
            "user_id": self.user_id,
            "reason": self.reason,
            "created_at": self.created_at.isoformat() if self.created_at is not None else "",
            "expires_at": self.expires_at.isoformat() if self.expires_at is not None else None,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IPAccessRule":
        """Create from dictionary."""
        return cls(
            ip_address=data["ip_address"],
            rule_type=AccessRuleType(data["rule_type"]),
            user_id=data.get("user_id"),
            reason=data.get("reason"),
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
            is_active=data.get("is_active", True),
        )


@dataclass
class AbuseScore:
    """Result of abuse detection analysis."""

    user_id: str
    score: Decimal  # 0.0 to 1.0, higher = more suspicious
    abuse_types: List[AbuseType]
    details: Dict[str, Any]
    should_block: bool
    recommendation: str

    def is_high_risk(self) -> bool:
        """Check if this is high-risk abuse."""
        return self.score >= Decimal("0.8")

    def is_medium_risk(self) -> bool:
        """Check if this is medium-risk abuse."""
        return Decimal("0.5") <= self.score < Decimal("0.8")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "score": str(self.score),
            "abuse_types": [t.value for t in self.abuse_types],
            "details": self.details,
            "should_block": self.should_block,
            "recommendation": self.recommendation,
        }


@dataclass
class AuditLog:
    """Audit log entry for security events."""

    event_id: str
    event_type: AuditEventType
    user_id: Optional[str]
    ip_address: str
    user_agent: Optional[str]
    endpoint: Optional[str]
    method: Optional[str]
    status_code: Optional[int]
    request_size: int = 0
    response_size: int = 0
    response_time: float = 0.0

    # Security-specific fields
    api_key_id: Optional[str] = None
    jwt_token_id: Optional[str] = None
    rate_limit_exceeded: bool = False
    authentication_method: Optional[str] = None
    security_violation: Optional[str] = None

    # Additional context
    details: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None

    timestamp: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if self.tags is None:
            self.tags = []
        if self.details is None:
            self.details = {}

    def add_tag(self, tag: str) -> None:
        """Add a tag to the audit log."""
        if self.tags is not None and tag not in self.tags:
            self.tags.append(tag)

    def add_detail(self, key: str, value: Any) -> None:
        """Add detail to the audit log."""
        if self.details is not None:
            self.details[key] = value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "endpoint": self.endpoint,
            "method": self.method,
            "status_code": self.status_code,
            "request_size": self.request_size,
            "response_size": self.response_size,
            "response_time": self.response_time,
            "api_key_id": self.api_key_id,
            "jwt_token_id": self.jwt_token_id,
            "rate_limit_exceeded": self.rate_limit_exceeded,
            "authentication_method": self.authentication_method,
            "security_violation": self.security_violation,
            "details": self.details,
            "tags": self.tags,
            "timestamp": self.timestamp.isoformat() if self.timestamp is not None else "",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditLog":
        """Create from dictionary."""
        return cls(
            event_id=data["event_id"],
            event_type=AuditEventType(data["event_type"]),
            user_id=data.get("user_id"),
            ip_address=data["ip_address"],
            user_agent=data.get("user_agent"),
            endpoint=data.get("endpoint"),
            method=data.get("method"),
            status_code=data.get("status_code"),
            request_size=data.get("request_size", 0),
            response_size=data.get("response_size", 0),
            response_time=data.get("response_time", 0.0),
            api_key_id=data.get("api_key_id"),
            jwt_token_id=data.get("jwt_token_id"),
            rate_limit_exceeded=data.get("rate_limit_exceeded", False),
            authentication_method=data.get("authentication_method"),
            security_violation=data.get("security_violation"),
            details=data.get("details", {}),
            tags=data.get("tags", []),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )
