"""Tests for async API endpoints."""

import uuid
from unittest.mock import AsyncMock, Mock, patch

from fastapi import status
from fastapi.testclient import TestClient

from services.analysis_service.src.api.app import app


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check(self):
        """Test basic health check."""
        client = TestClient(app)
        response = client.get("/v1/health")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "analysis_service"

    def test_readiness_check(self):
        """Test readiness check."""
        client = TestClient(app)
        response = client.get("/v1/health/ready")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "ready" in data
        assert "checks" in data
        assert data["service"] == "analysis_service"

    def test_liveness_check(self):
        """Test liveness check."""
        client = TestClient(app)
        response = client.get("/v1/health/live")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "alive"
        assert data["service"] == "analysis_service"


class TestRecordingEndpoints:
    """Test recording endpoints."""

    @patch("services.analysis_service.src.api.endpoints.recordings.recording_repository")
    @patch("services.analysis_service.src.api.endpoints.recordings.message_publisher")
    def test_submit_recording(self, mock_publisher, mock_repository):
        """Test submitting a recording."""
        # Mock the repository response
        mock_recording = Mock()
        mock_recording.id = uuid.uuid4()
        mock_repository.create = AsyncMock(return_value=mock_recording)

        # Mock the message publisher response
        mock_publisher.publish_analysis_request = AsyncMock(return_value="test-correlation-id")

        client = TestClient(app)
        response = client.post("/v1/recordings", json={"file_path": "/path/to/test.wav", "priority": 10})

        assert response.status_code == status.HTTP_202_ACCEPTED
        data = response.json()
        assert "id" in data
        assert data["status"] == "queued"
        assert "correlation_id" in data

    @patch("services.analysis_service.src.api.endpoints.recordings.recording_repository")
    def test_get_recording_status(self, mock_repository):
        """Test getting recording status."""
        # Mock the repository response
        recording_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        mock_recording = Mock()
        mock_recording.id = recording_id
        mock_recording.file_path = "/path/to/test.wav"
        mock_recording.processing_status = "processing"
        mock_repository.get_by_id = AsyncMock(return_value=mock_recording)

        # Mock metadata repository
        with patch(
            "services.analysis_service.src.api.endpoints.recordings.AsyncMetadataRepository"
        ) as mock_metadata_cls:
            mock_metadata_repo = Mock()
            mock_metadata_repo.get_by_recording_id = AsyncMock(return_value=[])
            mock_metadata_cls.return_value = mock_metadata_repo

            client = TestClient(app)
            response = client.get(f"/v1/recordings/{recording_id}")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["id"] == str(recording_id)
            assert "status" in data
            assert "file_path" in data

    @patch("services.analysis_service.src.api.endpoints.recordings.recording_repository")
    def test_list_recordings(self, mock_repository):
        """Test listing recordings."""
        # Mock the repository response
        mock_recording = Mock()
        mock_recording.id = uuid.uuid4()
        mock_recording.file_path = "/path/to/test.wav"
        mock_recording.processing_status = "completed"
        mock_recording.file_size = 1024
        mock_repository.list_paginated = AsyncMock(return_value=([mock_recording], 1))

        client = TestClient(app)
        response = client.get("/v1/recordings?limit=5&offset=0")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 5


class TestAnalysisEndpoints:
    """Test analysis endpoints."""

    def test_start_analysis(self):
        """Test starting analysis."""
        client = TestClient(app)
        response = client.post(
            "/v1/analysis",
            json={
                "recording_id": "550e8400-e29b-41d4-a716-446655440000",
                "analysis_types": ["bpm", "key"],
                "priority": 5,
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "task_id" in data
        assert data["status"] == "queued"
        assert data["analysis_types"] == ["bpm", "key"]

    def test_get_analysis_status(self):
        """Test getting analysis status."""
        client = TestClient(app)
        recording_id = "550e8400-e29b-41d4-a716-446655440000"
        response = client.get(f"/v1/analysis/{recording_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["recording_id"] == recording_id
        assert "status" in data
        assert "progress" in data
        assert "results" in data
        assert isinstance(data["results"], list)
