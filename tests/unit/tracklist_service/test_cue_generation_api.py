"""
Unit tests for CUE generation API endpoints.
"""

from datetime import timedelta
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.tracklist_service.src.api.cue_generation_api import router
from services.tracklist_service.src.models.cue_file import (
    BatchCueGenerationResponse,
    CueFormat,
    CueGenerationResponse,
    ValidationResult,
)
from services.tracklist_service.src.models.tracklist import TrackEntry, Tracklist


@pytest.fixture
def test_app():
    """Create test FastAPI app."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


@pytest.fixture
def sample_tracklist():
    """Create a sample tracklist."""
    tracks = [
        TrackEntry(
            position=1,
            start_time=timedelta(minutes=0),
            end_time=timedelta(minutes=3, seconds=30),
            artist="Test Artist 1",
            title="Test Track 1",
        ),
        TrackEntry(
            position=2,
            start_time=timedelta(minutes=3, seconds=30),
            end_time=timedelta(minutes=7),
            artist="Test Artist 2",
            title="Test Track 2",
        ),
    ]

    return Tracklist(audio_file_id=uuid4(), source="manual", tracks=tracks)


@pytest.fixture
def generation_request():
    """Create a sample generation request."""
    return {"format": "standard", "options": {}, "validate_audio": True, "audio_file_path": "test_mix.wav"}


@pytest.fixture
def batch_request():
    """Create a sample batch request."""
    return {"formats": ["standard", "cdj"], "options": {}, "validate_audio": True, "audio_file_path": "test_mix.wav"}


class TestCueGenerationAPI:
    """Test CUE Generation API endpoints."""

    @patch("services.tracklist_service.src.api.cue_generation_api.cue_generation_service")
    def test_generate_cue_file_success(self, mock_service, client, sample_tracklist, generation_request):
        """Test successful CUE file generation."""
        # Mock service responses - use AsyncMock for async methods
        mock_service.validate_tracklist_for_cue = AsyncMock(return_value=ValidationResult(valid=True))
        mock_service.generate_cue_file = AsyncMock(
            return_value=CueGenerationResponse(
                success=True,
                job_id=uuid4(),
                cue_file_id=uuid4(),
                file_path="test_file.cue",
                validation_report=ValidationResult(valid=True),
            )
        )

        # Make request
        response = client.post(
            "/api/v1/cue/generate",
            json={"request": generation_request, "tracklist_data": sample_tracklist.model_dump(mode="json")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "job_id" in data
        assert "cue_file_id" in data

    @patch("services.tracklist_service.src.api.cue_generation_api.cue_generation_service")
    def test_generate_cue_file_validation_failure(self, mock_service, client, sample_tracklist, generation_request):
        """Test CUE generation with validation failure."""
        # Mock validation failure
        mock_service.validate_tracklist_for_cue = AsyncMock(
            return_value=ValidationResult(valid=False, error="Missing required fields")
        )

        # Make request
        response = client.post(
            "/api/v1/cue/generate",
            json={"request": generation_request, "tracklist_data": sample_tracklist.model_dump(mode="json")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert "validation failed" in data["error"].lower()

    @patch("services.tracklist_service.src.api.cue_generation_api.cue_generation_service")
    def test_generate_cue_file_async(self, mock_service, client, sample_tracklist, generation_request):
        """Test asynchronous CUE generation."""
        # Mock validation success
        mock_service.validate_tracklist_for_cue = AsyncMock(return_value=ValidationResult(valid=True))

        # Make async request
        response = client.post(
            "/api/v1/cue/generate",
            json={"request": generation_request, "tracklist_data": sample_tracklist.model_dump(mode="json")},
            params={"async_processing": True},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "job_id" in data
        assert data["cue_file_id"] is None  # Not available for async

    @patch("services.tracklist_service.src.api.cue_generation_api.cue_generation_service")
    def test_batch_generate_success(self, mock_service, client, sample_tracklist, batch_request):
        """Test successful batch CUE generation."""
        # Mock service response
        mock_service.generate_multiple_formats = AsyncMock(
            return_value=BatchCueGenerationResponse(
                success=True,
                results=[
                    CueGenerationResponse(success=True, job_id=uuid4(), cue_file_id=uuid4()),
                    CueGenerationResponse(success=True, job_id=uuid4(), cue_file_id=uuid4()),
                ],
                total_files=2,
                successful_files=2,
                failed_files=0,
            )
        )

        # Make request
        response = client.post(
            "/api/v1/cue/generate/batch",
            json={"request": batch_request, "tracklist_data": sample_tracklist.model_dump(mode="json")},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total_files"] == 2
        assert data["successful_files"] == 2
        assert data["failed_files"] == 0

    @patch("services.tracklist_service.src.api.cue_generation_api.cue_generation_service")
    def test_batch_generate_async(self, mock_service, client, sample_tracklist, batch_request):
        """Test asynchronous batch CUE generation."""
        # Make async request
        response = client.post(
            "/api/v1/cue/generate/batch",
            json={"request": batch_request, "tracklist_data": sample_tracklist.model_dump(mode="json")},
            params={"async_processing": True},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total_files"] == 2
        assert data["successful_files"] == 0  # No immediate results for async

    def test_get_supported_formats(self, client):
        """Test getting supported formats."""
        with patch("services.tracklist_service.src.api.cue_generation_api.cue_generation_service") as mock_service:
            mock_service.get_supported_formats.return_value = [CueFormat.STANDARD, CueFormat.CDJ, CueFormat.TRAKTOR]

            response = client.get("/api/v1/cue/formats")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 3
            assert "standard" in data
            assert "cdj" in data
            assert "traktor" in data

    def test_get_format_capabilities_success(self, client):
        """Test getting format capabilities."""
        with patch("services.tracklist_service.src.api.cue_generation_api.cue_generation_service") as mock_service:
            mock_service.get_format_capabilities.return_value = {
                "max_tracks": 99,
                "supports_isrc": True,
                "supports_rem": True,
            }

            response = client.get("/api/v1/cue/formats/standard/capabilities")

            assert response.status_code == 200
            data = response.json()
            assert data["max_tracks"] == 99
            assert data["supports_isrc"] is True

    def test_get_format_capabilities_invalid_format(self, client):
        """Test format capabilities with invalid format."""
        response = client.get("/api/v1/cue/formats/invalid/capabilities")

        assert response.status_code == 400
        assert "Unsupported CUE format" in response.json()["detail"]

    def test_get_conversion_preview_success(self, client):
        """Test conversion preview."""
        with patch("services.tracklist_service.src.api.cue_generation_api.cue_generation_service") as mock_service:
            mock_service.get_conversion_preview.return_value = [
                "ISRC codes will be lost",
                "REM fields may be truncated",
            ]

            response = client.get(
                "/api/v1/cue/formats/conversion-preview", params={"source_format": "standard", "target_format": "cdj"}
            )

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert "ISRC codes will be lost" in data

    def test_get_conversion_preview_invalid_format(self, client):
        """Test conversion preview with invalid format."""
        response = client.get(
            "/api/v1/cue/formats/conversion-preview", params={"source_format": "invalid", "target_format": "standard"}
        )

        assert response.status_code == 400
        assert "Invalid CUE format" in response.json()["detail"]

    def test_get_job_status(self, client):
        """Test getting job status."""
        job_id = uuid4()
        response = client.get(f"/api/v1/cue/jobs/{job_id}/status")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == str(job_id)
        assert "status" in data

    def test_download_cue_file_not_implemented(self, client):
        """Test CUE file download (not yet implemented)."""
        cue_file_id = uuid4()
        response = client.get(f"/api/v1/cue/download/{cue_file_id}")

        assert response.status_code == 501
        assert "not yet implemented" in response.json()["detail"]

    def test_generate_by_tracklist_id_not_implemented(self, client, generation_request):
        """Test generation by tracklist ID (not yet implemented)."""
        tracklist_id = uuid4()
        response = client.post(f"/api/v1/cue/generate/{tracklist_id}", json=generation_request)

        assert response.status_code == 501
        assert "not yet implemented" in response.json()["detail"]

    def test_health_check_healthy(self, client):
        """Test health check with healthy services."""
        with (
            patch("services.tracklist_service.src.api.cue_generation_api.cue_generation_service") as mock_service,
            patch("services.tracklist_service.src.api.cue_generation_api.storage_service") as mock_storage,
            patch("services.tracklist_service.src.api.cue_generation_api.AudioValidationService"),
        ):
            mock_service.get_supported_formats.return_value = [CueFormat.STANDARD, CueFormat.CDJ]
            mock_storage.backend._ensure_directory_exists.return_value = True

            response = client.get("/api/v1/cue/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["service"] == "cue_generation_api"
            assert "components" in data

    def test_health_check_unhealthy(self, client):
        """Test health check with unhealthy services."""
        with patch("services.tracklist_service.src.api.cue_generation_api.cue_generation_service") as mock_service:
            mock_service.get_supported_formats.side_effect = Exception("Service unavailable")

            response = client.get("/api/v1/cue/health")

            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "unhealthy"
            assert "Service unavailable" in data["components"]["cue_service"]
