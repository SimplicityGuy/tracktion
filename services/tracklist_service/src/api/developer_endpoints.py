"""
Developer API endpoints for API key management.

Provides REST endpoints for developers to manage their API keys,
view usage analytics, and control key permissions.
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import redis.asyncio as redis
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from services.tracklist_service.src.analytics.usage_tracker import AggregationPeriod, UsageTracker
from services.tracklist_service.src.auth.authentication import AuthenticationManager
from services.tracklist_service.src.auth.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/developer", tags=["developer"])


class AuthenticationManagerSingleton:
    """Singleton wrapper for AuthenticationManager."""

    _instance: AuthenticationManager | None = None

    def __new__(cls) -> AuthenticationManager:
        """Get the singleton AuthenticationManager instance."""
        if cls._instance is None:
            # In production, this would be properly initialized
            cls._instance = AuthenticationManager(jwt_secret="dev-secret-key")
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        cls._instance = None


class UsageTrackerSingleton:
    """Singleton wrapper for UsageTracker."""

    _instance: UsageTracker | None = None

    def __new__(cls) -> UsageTracker:
        """Get the singleton UsageTracker instance."""
        if cls._instance is None:
            # In production, this would be properly initialized from dependencies
            redis_client = redis.Redis.from_url("redis://localhost:6379", decode_responses=True)
            cls._instance = UsageTracker(redis_client)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        cls._instance = None


def get_auth_manager() -> AuthenticationManager:
    """Get authentication manager singleton instance."""
    return AuthenticationManagerSingleton()


def get_usage_tracker() -> UsageTracker:
    """Get usage tracker singleton instance."""
    return UsageTrackerSingleton()


# Request/Response models
class CreateKeyRequest(BaseModel):
    """Request model for creating new API keys."""

    name: str = Field(..., description="Human-readable name for the key")
    description: str | None = Field(None, description="Optional description")
    permissions: dict[str, bool] | None = Field(None, description="Custom permissions (defaults based on user tier)")
    expires_in_days: int | None = Field(None, ge=1, le=365, description="Key expiration in days (optional)")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Name cannot be empty")
        return v.strip()


class ApiKeyInfo(BaseModel):
    """API key information for listing (without sensitive data)."""

    key_id: str = Field(..., description="Unique key identifier")
    name: str | None = Field(None, description="Key name")
    key_prefix: str = Field(..., description="First 8 characters for identification")
    is_active: bool = Field(..., description="Whether key is active")
    created_at: datetime | None = Field(None, description="Creation timestamp")
    expires_at: datetime | None = Field(None, description="Expiration timestamp")
    last_used_at: datetime | None = Field(None, description="Last usage timestamp")
    permissions: dict[str, bool] | None = Field(None, description="Key permissions")
    usage_stats: dict[str, Any] | None = Field(None, description="Basic usage statistics")


class ApiKeyResponse(BaseModel):
    """Response model for API key creation."""

    key_id: str = Field(..., description="Unique key identifier")
    api_key: str = Field(..., description="The actual API key (shown only once)")
    name: str | None = Field(None, description="Key name")
    permissions: dict[str, bool] = Field(..., description="Key permissions")
    expires_at: datetime | None = Field(None, description="Expiration timestamp")
    created_at: datetime = Field(..., description="Creation timestamp")


class KeyUsageResponse(BaseModel):
    """Response model for key usage analytics."""

    key_id: str = Field(..., description="Key identifier")
    usage_stats: dict[str, Any] = Field(..., description="Detailed usage statistics")
    cost_breakdown: dict[str, Any] = Field(..., description="Cost analysis")
    recommendations: list[str] = Field(default_factory=list, description="Usage recommendations")


@router.get("/keys", response_model=list[ApiKeyInfo])
async def list_api_keys(
    user: User,  # Will be injected via Depends in route
    include_inactive: bool = False,
) -> list[ApiKeyInfo]:
    """
    List all API keys for the authenticated user.

    Args:
        include_inactive: Whether to include inactive keys in response
        user: Authenticated user from dependency

    Returns:
        List of API key information (without sensitive data)
    """
    try:
        manager = get_auth_manager()
        tracker = get_usage_tracker()

        # Get all keys for user (this would query database in production)
        user_keys = []
        for key, api_key in manager._api_keys.items():
            if api_key.user_id == user.id:
                if not include_inactive and not api_key.is_active:
                    continue

                # Get basic usage stats for the key
                usage_stats = None
                try:
                    # In production, we'd have key-specific usage tracking
                    stats = await tracker.get_usage_stats(user.id, AggregationPeriod.DAY)
                    usage_stats = {
                        "requests_today": stats.total_requests,
                        "tokens_used_today": stats.total_tokens,
                        "average_response_time": stats.avg_response_time,
                    }
                except Exception as e:
                    logger.warning(f"Failed to get usage stats for key {api_key.key_id}: {e}")

                # Create safe key info (hide sensitive data)
                key_info = ApiKeyInfo(
                    key_id=api_key.key_id,
                    name=api_key.name,
                    key_prefix=key[:8] + "..." if len(key) >= 8 else "***",
                    is_active=api_key.is_active,
                    created_at=api_key.created_at,
                    expires_at=api_key.expires_at,
                    last_used_at=api_key.last_used_at,
                    permissions=api_key.permissions,
                    usage_stats=usage_stats,
                )
                user_keys.append(key_info)

        logger.info(f"Listed {len(user_keys)} API keys for user {user.id}")
        return user_keys

    except Exception as e:
        logger.error(f"Error listing API keys for user {user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve API keys") from e


@router.post("/keys", response_model=ApiKeyResponse)
async def create_api_key(
    request: CreateKeyRequest,
    user: User,  # Will be injected via Depends in route
) -> ApiKeyResponse:
    """
    Create a new API key for the authenticated user.

    Args:
        request: Key creation request with name and permissions
        user: Authenticated user from dependency

    Returns:
        Created API key with full key value (shown only once)
    """
    try:
        manager = get_auth_manager()

        # Validate user can create more keys (implement limits)
        current_keys = sum(
            1 for _, api_key in manager._api_keys.items() if api_key.user_id == user.id and api_key.is_active
        )

        # Set limits based on user tier
        max_keys = {"free": 2, "premium": 10, "enterprise": 50}.get(user.tier.value, 2)
        if current_keys >= max_keys:
            raise HTTPException(
                status_code=403,
                detail=f"Maximum number of API keys reached for {user.tier.value} tier ({max_keys})",
            )

        # Generate new API key
        api_key = manager.generate_api_key(user_id=user.id, tier=user.tier.value, name=request.name)

        # Set custom permissions if provided
        if request.permissions:
            # Validate permissions don't exceed user tier capabilities
            default_perms = manager._get_default_permissions(user.tier.value)
            for perm, enabled in request.permissions.items():
                if enabled and perm in default_perms and not default_perms[perm]:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Permission '{perm}' not available for {user.tier.value} tier",
                    )
            api_key.permissions = request.permissions

        # Set expiration if requested
        if request.expires_in_days:
            api_key.expires_at = datetime.now(UTC) + timedelta(days=request.expires_in_days)

        # Get the raw key to return (this is only shown once)
        raw_key = None
        for key, stored_key in manager._api_keys.items():
            if stored_key.key_id == api_key.key_id:
                raw_key = key
                break

        if not raw_key:
            raise HTTPException(status_code=500, detail="Failed to retrieve generated key")

        logger.info(f"Created API key {api_key.key_id} for user {user.id}")

        return ApiKeyResponse(
            key_id=api_key.key_id,
            api_key=raw_key,
            name=api_key.name,
            permissions=api_key.permissions or {},
            expires_at=api_key.expires_at,
            created_at=api_key.created_at or datetime.now(UTC),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating API key for user {user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create API key") from e


@router.post("/keys/{key_id}/rotate", response_model=ApiKeyResponse)
async def rotate_api_key(
    key_id: str,
    user: User,  # Will be injected via Depends in route
) -> ApiKeyResponse:
    """
    Rotate an existing API key (generates new key, keeps same metadata).

    Args:
        key_id: ID of the key to rotate
        user: Authenticated user from dependency

    Returns:
        New API key with same metadata but different key value
    """
    try:
        manager = get_auth_manager()

        # Find the existing key
        old_key = None
        for api_key in manager._api_keys.values():
            if api_key.key_id == key_id and api_key.user_id == user.id:
                old_key = api_key
                break

        if not old_key:
            raise HTTPException(status_code=404, detail="API key not found")

        if not old_key.is_active:
            raise HTTPException(status_code=400, detail="Cannot rotate inactive API key")

        # Create new key with same metadata
        new_api_key = manager.generate_api_key(user_id=user.id, tier=user.tier.value, name=old_key.name)

        # Copy metadata from old key
        new_api_key.permissions = old_key.permissions
        new_api_key.expires_at = old_key.expires_at

        # Deactivate old key
        old_key.is_active = False

        # Get the raw key for response
        new_raw_key = None
        for raw_key, api_key in manager._api_keys.items():
            if api_key.key_id == new_api_key.key_id:
                new_raw_key = raw_key
                break

        logger.info(f"Rotated API key {key_id} -> {new_api_key.key_id} for user {user.id}")

        return ApiKeyResponse(
            key_id=new_api_key.key_id,
            api_key=new_raw_key or "",
            name=new_api_key.name,
            permissions=new_api_key.permissions or {},
            expires_at=new_api_key.expires_at,
            created_at=new_api_key.created_at or datetime.now(UTC),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rotating API key {key_id} for user {user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to rotate API key") from e


@router.delete("/keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    user: User,  # Will be injected via Depends in route
) -> JSONResponse:
    """
    Revoke (deactivate) an API key immediately.

    Args:
        key_id: ID of the key to revoke
        user: Authenticated user from dependency

    Returns:
        JSON response confirming revocation
    """
    try:
        manager = get_auth_manager()

        # Find and revoke the key
        key_found = False
        for api_key in manager._api_keys.values():
            if api_key.key_id == key_id and api_key.user_id == user.id:
                api_key.is_active = False
                key_found = True
                logger.info(f"Revoked API key {key_id} for user {user.id}")
                break

        if not key_found:
            raise HTTPException(status_code=404, detail="API key not found")

        return JSONResponse(
            content={
                "success": True,
                "message": f"API key {key_id} has been revoked",
                "revoked_at": datetime.now(UTC).isoformat(),
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error revoking API key {key_id} for user {user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to revoke API key") from e


@router.get("/keys/{key_id}/usage", response_model=KeyUsageResponse)
async def get_key_usage_analytics(
    key_id: str,
    user: User,  # Will be injected via Depends in route
    period: str = "month",
) -> KeyUsageResponse:
    """
    Get detailed usage analytics for a specific API key.

    Args:
        key_id: ID of the key to get analytics for
        period: Time period for analytics (day, week, month)
        user: Authenticated user from dependency

    Returns:
        Detailed usage statistics and cost breakdown
    """
    try:
        manager = get_auth_manager()
        tracker = get_usage_tracker()

        # Verify key belongs to user
        key_found = False
        api_key = None
        for stored_key in manager._api_keys.values():
            if stored_key.key_id == key_id and stored_key.user_id == user.id:
                key_found = True
                api_key = stored_key
                break

        if not key_found:
            raise HTTPException(status_code=404, detail="API key not found")

        # Convert string period to enum
        period_enum = AggregationPeriod(period)

        # Get usage stats (in production, this would be key-specific)
        usage_stats = await tracker.get_usage_stats(user.id, period_enum)

        # Calculate cost breakdown based on user tier
        tier_rates = {
            "free": 0.0,
            "premium": 0.002,  # $0.002 per 1000 tokens
            "enterprise": 0.001,  # $0.001 per 1000 tokens
        }

        token_rate = tier_rates.get(user.tier.value, 0.002)
        total_cost = (usage_stats.total_tokens / 1000) * token_rate

        cost_breakdown = {
            "base_cost": total_cost,
            "token_cost": total_cost,
            "request_cost": 0.0,  # No per-request cost in this implementation
            "tier": user.tier.value,
            "period": period,
        }

        # Generate recommendations
        recommendations = []
        if usage_stats.total_requests > 10000:
            recommendations.append("Consider upgrading to premium tier for better rates")
        if usage_stats.avg_response_time > 500:
            recommendations.append("API response times are high - consider optimizing queries")
        if api_key and not api_key.last_used_at:
            recommendations.append("This API key has never been used")
        elif api_key and api_key.last_used_at and api_key.last_used_at < datetime.now(UTC).replace(day=1):
            recommendations.append("This API key hasn't been used this month")

        # Detailed usage statistics
        detailed_stats = {
            "total_requests": usage_stats.total_requests,
            "total_tokens": usage_stats.total_tokens,
            "total_cost": total_cost,
            "average_response_time": usage_stats.avg_response_time,
            "error_rate": usage_stats.error_rate,
            "top_endpoints": list(usage_stats.endpoints.keys()),
            "period": period,
            "key_id": key_id,
            "key_name": api_key.name if api_key else None,
            "key_created": (api_key.created_at.isoformat() if api_key and api_key.created_at else None),
            "key_last_used": (api_key.last_used_at.isoformat() if api_key and api_key.last_used_at else None),
        }

        return KeyUsageResponse(
            key_id=key_id,
            usage_stats=detailed_stats,
            cost_breakdown=cost_breakdown,
            recommendations=recommendations,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting usage analytics for key {key_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve usage analytics") from e


@router.get("/usage/summary")
async def get_usage_summary(
    user: User,  # Will be injected via Depends in route
    period: str = "month",
) -> JSONResponse:
    """
    Get overall usage summary across all user's API keys.

    Args:
        period: Time period for summary (day, week, month)
        user: Authenticated user from dependency

    Returns:
        JSON response with usage summary and tier information
    """
    try:
        manager = get_auth_manager()
        tracker = get_usage_tracker()

        # Convert string period to enum
        period_enum = AggregationPeriod(period)

        # Get overall usage stats
        usage_stats = await tracker.get_usage_stats(user.id, period_enum)

        # Calculate cost based on user tier
        tier_rates = {
            "free": 0.0,
            "premium": 0.002,  # $0.002 per 1000 tokens
            "enterprise": 0.001,  # $0.001 per 1000 tokens
        }

        token_rate = tier_rates.get(user.tier.value, 0.002)
        total_cost = (usage_stats.total_tokens / 1000) * token_rate

        # Count active keys
        active_keys = sum(
            1 for _, api_key in manager._api_keys.items() if api_key.user_id == user.id and api_key.is_active
        )

        # Get tier limits (this would come from a configuration service)
        tier_limits = {
            "free": {
                "requests_per_day": 1000,
                "tokens_per_month": 25000,
                "max_keys": 2,
            },
            "premium": {
                "requests_per_day": 10000,
                "tokens_per_month": 250000,
                "max_keys": 10,
            },
            "enterprise": {
                "requests_per_day": 100000,
                "tokens_per_month": 2500000,
                "max_keys": 50,
            },
        }

        user_limits = tier_limits.get(user.tier.value, tier_limits["free"])

        summary: dict[str, Any] = {
            "user_id": user.id,
            "user_tier": user.tier.value,
            "period": period,
            "usage": {
                "total_requests": usage_stats.total_requests,
                "total_tokens": usage_stats.total_tokens,
                "total_cost": total_cost,
                "average_response_time": usage_stats.avg_response_time,
                "error_rate": usage_stats.error_rate,
            },
            "limits": user_limits,
            "utilization": {
                "requests_percent": (usage_stats.total_requests / user_limits["requests_per_day"]) * 100,
                "tokens_percent": (usage_stats.total_tokens / user_limits["tokens_per_month"]) * 100,
                "keys_percent": (active_keys / user_limits["max_keys"]) * 100,
            },
            "keys": {
                "active": active_keys,
                "total": sum(1 for _, api_key in manager._api_keys.items() if api_key.user_id == user.id),
            },
            "recommendations": [],
        }

        # Add recommendations based on usage
        utilization = summary["utilization"]
        recommendations = summary["recommendations"]

        if isinstance(utilization, dict) and utilization.get("requests_percent", 0) > 80:
            recommendations.append("Request usage is high - consider upgrading tier")
        if isinstance(utilization, dict) and utilization.get("tokens_percent", 0) > 80:
            recommendations.append("Token usage is high - consider upgrading tier")
        if active_keys == 0:
            recommendations.append("Create your first API key to start using the service")

        return JSONResponse(content=summary)

    except Exception as e:
        logger.error(f"Error getting usage summary for user {user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve usage summary") from e
