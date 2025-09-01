"""Authentication, authorization, and rate limiting for the feedback API."""

import logging
import re
from collections.abc import Callable
from functools import wraps
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic_settings import BaseSettings
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.requests import Request as StarletteRequest

logger = logging.getLogger(__name__)


# Configuration
class AuthConfig(BaseSettings):
    """Authentication configuration."""

    # API Keys - in production these would come from a secure key management system
    api_keys: set[str] = {"dev-key-123", "prod-key-456"}
    admin_keys: set[str] = {"admin-key-789"}

    # Rate limiting
    default_rate_limit: str = "100/minute"
    metrics_rate_limit: str = "50/minute"
    admin_rate_limit: str = "200/minute"

    class Config:
        env_prefix = "AUTH_"


auth_config = AuthConfig()

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

# HTTP Bearer security scheme
security = HTTPBearer(auto_error=False)


# Input sanitization helpers
def sanitize_string(value: str, max_length: int = 255) -> str:
    """Sanitize string input to prevent injection attacks."""
    if not isinstance(value, str):
        raise ValueError("Value must be a string")

    # Remove null bytes and control characters
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)

    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    # Strip whitespace
    return sanitized.strip()


def sanitize_filename(filename: str) -> str:
    """Sanitize filename input."""
    if not filename:
        raise ValueError("Filename cannot be empty")

    # Basic filename sanitization
    sanitized = sanitize_string(filename, max_length=255)

    # Remove path traversal attempts
    sanitized = re.sub(r"[/\\]+", "", sanitized)

    # Remove dangerous characters
    sanitized = re.sub(r'[<>:"|?*]', "", sanitized)

    if not sanitized:
        raise ValueError("Filename contains only invalid characters")

    return sanitized


def sanitize_error_message(error_msg: str) -> str:
    """Sanitize error messages to prevent information leakage."""
    # Generic error messages for production
    sensitive_patterns = [
        r"password",
        r"token",
        r"key",
        r"secret",
        r"database",
        r"connection",
        r"internal",
        r"file system",
        r"path",
    ]

    error_lower = error_msg.lower()
    for pattern in sensitive_patterns:
        if re.search(pattern, error_lower):
            return "An internal error occurred. Please try again."

    # Remove file paths and detailed system info
    sanitized = re.sub(r"/[/\w.-]+", "[path]", error_msg)
    return re.sub(r"[A-Za-z]:\\[\\w.-]+", "[path]", sanitized)


# Authentication dependencies
async def verify_api_key(credentials: HTTPAuthorizationCredentials | None = None) -> dict[str, Any]:
    """Verify API key and return user context."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    api_key = credentials.credentials

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if key is valid
    is_admin = api_key in auth_config.admin_keys
    is_valid = api_key in auth_config.api_keys or is_admin

    if not is_valid:
        logger.warning(f"Invalid API key attempted: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Log successful authentication
    logger.info(f"API key authenticated: {api_key[:8]}... (admin: {is_admin})")

    return {
        "api_key": api_key[:8] + "...",  # Truncated for logging
        "is_admin": is_admin,
        "permissions": ["admin"] if is_admin else ["user"],
    }


async def verify_admin_key(credentials: HTTPAuthorizationCredentials | None = None) -> dict[str, Any]:
    """Verify admin API key."""
    user_context = await verify_api_key(credentials)

    if not user_context["is_admin"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )

    return user_context


# Rate limiting decorators
def rate_limit(rate: str | None = None):
    """Rate limiting decorator."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get request from args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request | StarletteRequest):
                    request = arg
                    break

            if not request:
                # If no request found, proceed without rate limiting
                return await func(*args, **kwargs)

            # Apply rate limit
            try:
                await limiter.hit(request, rate or auth_config.default_rate_limit)
                return await func(*args, **kwargs)
            except RateLimitExceeded as e:
                logger.warning(f"Rate limit exceeded for {get_remote_address(request)}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded: {rate or auth_config.default_rate_limit}",
                    headers={"Retry-After": str(e.retry_after)},
                ) from e

        return wrapper

    return decorator


# Common rate limits
user_rate_limit = rate_limit(auth_config.default_rate_limit)
metrics_rate_limit = rate_limit(auth_config.metrics_rate_limit)
admin_rate_limit = rate_limit(auth_config.admin_rate_limit)


# Input validation helpers
def validate_proposal_id(proposal_id: str) -> str:
    """Validate and sanitize proposal ID."""
    if not proposal_id:
        raise ValueError("Proposal ID cannot be empty")

    sanitized = sanitize_string(proposal_id, max_length=100)

    # Ensure it looks like a valid UUID or similar ID format
    if not re.match(r"^[a-zA-Z0-9_-]+$", sanitized):
        raise ValueError("Invalid proposal ID format")

    return sanitized


def validate_model_version(model_version: str) -> str:
    """Validate and sanitize model version."""
    if not model_version:
        raise ValueError("Model version cannot be empty")

    sanitized = sanitize_string(model_version, max_length=50)

    # Ensure it looks like a valid version string
    if not re.match(r"^[a-zA-Z0-9._-]+$", sanitized):
        raise ValueError("Invalid model version format")

    return sanitized


def validate_confidence_score(score: float) -> float:
    """Validate confidence score."""
    if not isinstance(score, int | float):
        raise ValueError("Confidence score must be a number")

    if not 0.0 <= score <= 1.0:
        raise ValueError("Confidence score must be between 0.0 and 1.0")

    return float(score)


def validate_context_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    """Validate and sanitize context metadata."""
    if metadata is None:
        return None

    if not isinstance(metadata, dict):
        raise ValueError("Context metadata must be a dictionary")

    # Limit size and sanitize values
    if len(metadata) > 50:
        raise ValueError("Context metadata cannot have more than 50 keys")

    sanitized = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or len(key) > 100:
            continue  # Skip invalid keys

        sanitized_key = sanitize_string(key, max_length=100)

        # Sanitize values based on type
        if isinstance(value, str):
            sanitized_value = sanitize_string(str(value), max_length=1000)
        elif isinstance(value, int | float | bool):
            sanitized_value = str(value)
        elif value is None:
            sanitized_value = None
        else:
            # Convert complex types to string and sanitize
            sanitized_value = sanitize_string(str(value), max_length=1000)

        sanitized[sanitized_key] = sanitized_value

    return sanitized


# Exception handler for rate limiting
def setup_rate_limit_handler(app):
    """Setup rate limiting exception handler."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
