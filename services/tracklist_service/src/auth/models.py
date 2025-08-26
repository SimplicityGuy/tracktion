"""Authentication models for tracklist service."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any


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
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ApiKey:
    """API key model for authentication."""

    key_id: str
    user_id: str
    key_hash: str  # Store hashed version
    name: Optional[str] = None
    is_active: bool = True
    created_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    permissions: Optional[Dict[str, bool]] = None


@dataclass
class TokenPair:
    """JWT token pair with access and refresh tokens."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 3600  # 1 hour default
