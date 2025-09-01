"""Pytest configuration for integration tests."""

import asyncio
import os
from collections.abc import Generator

import pytest


# Configure asyncio event loop policy for tests
@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop]:
    """Create event loop for the session."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment variables."""
    # Set test environment variables
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "INFO"

    # Mock auth keys for testing
    os.environ["AUTH_API_KEYS"] = "test-api-key-123,test-api-key-456"
    os.environ["AUTH_ADMIN_KEYS"] = "test-admin-key-789"

    yield

    # Cleanup
    for key in ["TESTING", "LOG_LEVEL", "AUTH_API_KEYS", "AUTH_ADMIN_KEYS"]:
        os.environ.pop(key, None)


# Add pytest markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "requires_docker: marks tests that require Docker services")
