"""Authentication models for tracklist service."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class UserTier(Enum):
    """User tier levels with different rate limits."""

    FREE = "free"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


@dataclass
class User:
    """User model for API authentication."""

    id: str
    email: str
    tier: UserTier
    is_active: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ApiKey:
    """API key model for authentication."""

    key_id: str
    user_id: str
    key_hash: str  # Store hashed version
    name: str | None = None
    is_active: bool = True
    created_at: datetime | None = None
    expires_at: datetime | None = None
    last_used_at: datetime | None = None
    permissions: dict[str, bool] | None = None


@dataclass
class TokenPair:
    """JWT token pair with access and refresh tokens."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 3600  # 1 hour default
