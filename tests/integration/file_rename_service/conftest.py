"""Pytest fixtures specific to file_rename_service integration tests."""

import os

import pytest


# Test database and Redis URLs for service-specific tests
@pytest.fixture(scope="session")
def postgres_test_url() -> str:
    """Get PostgreSQL test URL."""
    return os.getenv("TEST_POSTGRES_DSN", "postgresql://tracktion_user:changeme@localhost:5432/test_feedback")


@pytest.fixture(scope="session")
def redis_test_url() -> str:
    """Get Redis test URL."""
    return os.getenv("TEST_REDIS_URL", "redis://localhost:6379/1")


@pytest.fixture(scope="session")
def test_api_credentials() -> dict[str, str]:
    """Get test API credentials."""
    return {"user_key": "test-api-key-123", "admin_key": "test-admin-key-789"}
