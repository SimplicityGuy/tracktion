"""Tests for API error handling."""

import uuid
from unittest.mock import AsyncMock, Mock, patch

from fastapi import status
from fastapi.testclient import TestClient

from services.analysis_service.src.api.app import app
from services.analysis_service.src.api.errors import (
    ErrorCode,
    ErrorResponse,
    create_error_response,
)


class TestErrorHandling:
    """Test error handling functionality."""

    def test_create_error_response(self):
        """Test creating error response."""
        response = create_error_response(
            ErrorCode.VALIDATION_ERROR,
            "Invalid input",
            status_code=400,
            details={"field": "test"},
        )

        assert response["error"]["code"] == ErrorCode.VALIDATION_ERROR
        assert response["error"]["message"] == "Invalid input"
        assert response["error"]["status"] == 400
        assert response["error"]["details"]["field"] == "test"
        assert "request_id" in response["error"]
        assert "timestamp" in response["error"]

    def test_error_response_model(self):
        """Test ErrorResponse model."""
        error = ErrorResponse(
            code=ErrorCode.NOT_FOUND,
            message="Resource not found",
            status=404,
        )

        assert error.code == ErrorCode.NOT_FOUND
        assert error.message == "Resource not found"
        assert error.status == 404
        assert error.details is None
        assert error.request_id is not None
        assert error.timestamp is not None

    @patch("services.analysis_service.src.api.endpoints.recordings.recording_repo")
    def test_404_error_handling(self, mock_repo):
        """Test 404 error handling."""
        mock_repo.get_by_id = AsyncMock(return_value=None)

        client = TestClient(app)
        response = client.get(f"/v1/recordings/{uuid.uuid4()}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        error_data = response.json()
        assert "error" in error_data
        assert error_data["error"]["code"] == ErrorCode.NOT_FOUND

    @patch("services.analysis_service.src.api.endpoints.recordings.recording_repo")
    def test_500_error_handling(self, mock_repo):
        """Test 500 error handling."""
        mock_repo.get_by_id = AsyncMock(side_effect=Exception("Database error"))

        client = TestClient(app)
        response = client.get(f"/v1/recordings/{uuid.uuid4()}")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        error_data = response.json()
        assert "error" in error_data
        assert error_data["error"]["code"] == ErrorCode.INTERNAL_ERROR

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

    def test_rate_limit_error_response(self):
        """Test rate limit error response creation."""
        response = create_error_response(
            ErrorCode.RATE_LIMITED,
            "Too many requests",
            status_code=429,
            details={"retry_after": 60},
        )

        assert response["error"]["code"] == ErrorCode.RATE_LIMITED
        assert response["error"]["status"] == 429
        assert response["error"]["details"]["retry_after"] == 60

    def test_timeout_error_response(self):
        """Test timeout error response creation."""
        response = create_error_response(
            ErrorCode.TIMEOUT,
            "Request timed out",
            status_code=504,
        )

        assert response["error"]["code"] == ErrorCode.TIMEOUT
        assert response["error"]["status"] == 504

    def test_service_unavailable_error(self):
        """Test service unavailable error response."""
        response = create_error_response(
            ErrorCode.SERVICE_UNAVAILABLE,
            "Service temporarily unavailable",
            status_code=503,
            details={"service": "redis", "retry_after": 30},
        )

        assert response["error"]["code"] == ErrorCode.SERVICE_UNAVAILABLE
        assert response["error"]["status"] == 503
        assert response["error"]["details"]["service"] == "redis"
