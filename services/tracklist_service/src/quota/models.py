"""Quota management models."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class QuotaType(Enum):
    """Types of quota periods."""

    DAILY = "daily"
    MONTHLY = "monthly"


class QuotaStatus(Enum):
    """Quota status indicators."""

    OK = "ok"  # Under 80%
    WARNING = "warning"  # 80-95%
    CRITICAL = "critical"  # 95-100%
    EXCEEDED = "exceeded"  # Over 100%


@dataclass
class QuotaLimits:
    """Quota limits for a user tier."""

    daily_limit: int
    monthly_limit: int
    request_burst: int = 0  # Additional burst capacity


@dataclass
class QuotaUsage:
    """Current quota usage for a user."""

    user_id: str
    daily_used: int
    monthly_used: int
    daily_limit: int
    monthly_limit: int
    last_reset_date: datetime
    current_date: datetime


@dataclass
class QuotaResult:
    """Result of quota check operation."""

    allowed: bool
    status: QuotaStatus
    daily_remaining: int
    monthly_remaining: int
    daily_percentage: float
    monthly_percentage: float
    next_reset: datetime
    message: str | None = None


@dataclass
class QuotaAlert:
    """Quota alert information."""

    user_id: str
    quota_type: QuotaType
    percentage: float
    threshold: int  # 80 or 95
    current_usage: int
    limit: int
    timestamp: datetime
    sent: bool = False


@dataclass
class QuotaUpgrade:
    """Quota upgrade request."""

    user_id: str
    current_tier: str
    requested_tier: str
    timestamp: datetime
    approved: bool = False
    processed: bool = False
    metadata: dict[str, Any] | None = None
