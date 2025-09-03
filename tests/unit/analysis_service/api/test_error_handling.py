"""Tests for API error handling."""

import uuid
from unittest.mock import AsyncMock, Mock, patch

from fastapi import status
from fastapi.testclient import TestClient

from services.analysis_service.src.api.app import app
from services.analysis_service.src.api.errors import (
    APIError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    ValidationError,
)


class TestErrorHandling:
    """Test error handling functionality."""

    def test_api_error_creation(self):
        """Test creating API error."""
        error = APIError(
            status_code=400,
            detail="Invalid input",
            error_code="VALIDATION_ERROR",
        )

        assert error.status_code == 400
        assert error.detail == "Invalid input"
        assert error.error_code == "VALIDATION_ERROR"

    def test_not_found_error(self):
        """Test NotFoundError."""
        error = NotFoundError(
            resource="Recording",
            resource_id="550e8400-e29b-41d4-a716-446655440000",
        )

        assert error.status_code == status.HTTP_404_NOT_FOUND
        assert "Recording with ID 550e8400-e29b-41d4-a716-446655440000 not found" in error.detail
        assert error.error_code == "RESOURCE_NOT_FOUND"

    @patch("services.analysis_service.src.api.endpoints.recordings.recording_repository")
    def test_404_error_handling(self, mock_repo):
        """Test 404 error handling."""
        mock_repo.get_by_id = AsyncMock(return_value=None)

        client = TestClient(app)
        response = client.get(f"/v1/recordings/{uuid.uuid4()}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        error_data = response.json()
        assert "detail" in error_data
        assert "not found" in error_data["detail"]

    @patch("services.analysis_service.src.api.endpoints.recordings.recording_repository")
    def test_500_error_handling(self, mock_repo):
        """Test 500 error handling."""
        mock_repo.get_by_id = AsyncMock(side_effect=Exception("Database error"))

        client = TestClient(app)
        response = client.get(f"/v1/recordings/{uuid.uuid4()}")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        error_data = response.json()
        # When an exception occurs, FastAPI's default error handler is used
        assert "detail" in error_data or "error" in error_data

    def test_validation_error_handling(self):
        """Test validation error handling."""
        client = TestClient(app)
        response = client.post("/v1/recordings", json={"invalid": "data"})

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        error_data = response.json()
        assert "detail" in error_data or "error" in error_data

    @patch("services.analysis_service.src.api.endpoints.analysis.recording_repo")
    @patch("services.analysis_service.src.api.endpoints.analysis.message_publisher")
    def test_analysis_not_found_error(self, mock_publisher, mock_repo):
        """Test analysis endpoint with recording not found."""
        mock_repo.get_by_id = AsyncMock(return_value=None)

        client = TestClient(app)
        response = client.post("/v1/analysis", json={"recording_id": str(uuid.uuid4()), "analysis_types": ["bpm"]})

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @patch("services.analysis_service.src.api.endpoints.analysis.recording_repo")
    def test_analysis_file_not_found_error(self, mock_repo):
        """Test analysis endpoint with file not found."""
        mock_recording = Mock()
        mock_recording.file_path = "/nonexistent/file.mp3"
        mock_repo.get_by_id = AsyncMock(return_value=mock_recording)

        client = TestClient(app)
        response = client.post("/v1/analysis", json={"recording_id": str(uuid.uuid4()), "analysis_types": ["bpm"]})

        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_rate_limit_error(self):
        """Test rate limit error."""
        error = RateLimitError(retry_after=60)

        assert error.status_code == status.HTTP_429_TOO_MANY_REQUESTS
        assert "Rate limit exceeded" in error.detail
        assert error.error_code == "RATE_LIMIT_EXCEEDED"
        assert error.headers["Retry-After"] == "60"

    def test_validation_error(self):
        """Test validation error."""
        error = ValidationError(
            detail="Invalid value",
            field="username",
        )

        assert error.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        assert "username" in error.detail
        assert error.error_code == "VALIDATION_ERROR"

    def test_service_unavailable_error(self):
        """Test service unavailable error."""
        error = ServiceUnavailableError(
            service="redis",
            retry_after=30,
        )

        assert error.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        assert "redis" in error.detail
        assert error.error_code == "SERVICE_UNAVAILABLE"
        assert error.headers["Retry-After"] == "30"
