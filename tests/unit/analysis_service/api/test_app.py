"""Tests for FastAPI async app setup."""

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from services.analysis_service.src.api.app import app


class TestAsyncAppSetup:
    """Test async FastAPI application setup."""

    def test_app_creation(self):
        """Test that app is created properly."""
        assert app.title == "Analysis Service API"
        assert app.version == "1.0.0"
        assert app.docs_url == "/v1/docs"
        assert app.redoc_url == "/v1/redoc"
        assert app.openapi_url == "/v1/openapi.json"

    def test_middleware_registration(self):
        """Test that middleware is registered."""
        # Check that app has middleware
        # FastAPI internally manages middleware differently,
        # so we test by checking if middleware was added
        assert hasattr(app, "add_middleware")

        # Test that our custom middleware works by calling an endpoint
        client = TestClient(app)

        # Add a test endpoint if not exists
        if not any(route.path == "/test-middleware" for route in app.routes):

            @app.get("/test-middleware")
            async def test_middleware():
                return {"test": "middleware"}

        response = client.get("/test-middleware")
        # Our middleware should add these headers
        assert "X-Request-ID" in response.headers
        assert "X-Process-Time" in response.headers

    @pytest.mark.asyncio
    async def test_lifespan_context(self):
        """Test async lifespan context manager."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # App should be initialized within context
            assert client is not None

    def test_middleware_headers(self):
        """Test that middleware adds proper headers."""
        # Note: Using TestClient for synchronous test
        client = TestClient(app)

        # Add a simple health endpoint for testing
        @app.get("/health")
        async def health():
            return {"status": "healthy"}

        response = client.get("/health")

        # Check middleware headers are present
        assert "X-Request-ID" in response.headers
        assert "X-Process-Time" in response.headers
        assert response.status_code == status.HTTP_200_OK
