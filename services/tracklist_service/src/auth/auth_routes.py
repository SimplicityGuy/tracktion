"""
Authentication routes for admin access.

This module provides login, token refresh, and user management endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from .admin_auth import (
    AdminCredentials,
    AdminUser,
    TokenResponse,
    authenticate_admin,
    decode_token,
    generate_tokens,
    get_current_admin,
)

router = APIRouter(prefix="/auth", tags=["authentication"])


@router.post("/login", response_model=TokenResponse)
async def login(credentials: AdminCredentials) -> TokenResponse:
    """
    Admin login endpoint.

    Args:
        credentials: Admin username and password

    Returns:
        JWT tokens for authenticated admin

    Raises:
        HTTPException: If authentication fails
    """
    user = authenticate_admin(credentials.username, credentials.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return generate_tokens(user.username, user.role)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: str) -> TokenResponse:
    """
    Refresh access token using refresh token.

    Args:
        refresh_token: Valid refresh token

    Returns:
        New JWT tokens

    Raises:
        HTTPException: If refresh token is invalid
    """
    try:
        token_data = decode_token(refresh_token)

        # Verify it's a refresh token
        # (In a real implementation, you'd check the token type in the payload)

        return generate_tokens(token_data.username or "", token_data.role or "")

    except HTTPException as err:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from err


@router.get("/me", response_model=AdminUser)
async def get_current_user(current_user: AdminUser = Depends(get_current_admin)) -> AdminUser:  # noqa: B008 - FastAPI dependency injection requires Depends in default
    """
    Get current authenticated admin user.

    Args:
        current_user: Current authenticated admin

    Returns:
        Admin user information
    """
    return current_user


@router.post("/logout")
async def logout(current_user: AdminUser = Depends(get_current_admin)) -> dict[str, str]:  # noqa: B008 - FastAPI dependency injection requires Depends in default
    """
    Logout endpoint (client should discard tokens).

    Args:
        current_user: Current authenticated admin

    Returns:
        Logout confirmation message
    """
    # In a real implementation, you might want to blacklist the token
    # For now, just return a success message
    return {"message": f"User {current_user.username} logged out successfully"}


@router.get("/verify")
async def verify_token(current_user: AdminUser = Depends(get_current_admin)) -> dict[str, str | bool]:  # noqa: B008 - FastAPI dependency injection requires Depends in default
    """
    Verify if token is valid.

    Args:
        current_user: Current authenticated admin

    Returns:
        Token validity status
    """
    return {"valid": True, "username": current_user.username, "role": current_user.role}
