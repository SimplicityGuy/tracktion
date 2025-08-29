"""Unit tests for main FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from services.file_rename_service.app.main import app


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    return TestClient(app)


def test_health_check(client: TestClient) -> None:
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "File Rename Service"
    assert "version" in data
    assert "environment" in data


def test_root_endpoint(client: TestClient) -> None:
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "File Rename Service"
    assert data["status"] == "running"
    assert "version" in data


def test_cors_headers(client: TestClient) -> None:
    """Test CORS headers are properly set."""
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers


def test_404_error(client: TestClient) -> None:
    """Test 404 error handling."""
    response = client.get("/nonexistent")
    assert response.status_code == 404
