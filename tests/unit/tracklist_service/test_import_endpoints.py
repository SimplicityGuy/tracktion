"""
Tests for import API endpoints.

Comprehensive test suite for tracklist import functionality including
1001tracklists integration, validation, and error handling.
"""

from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from services.tracklist_service.src.api.import_endpoints import router
from services.tracklist_service.src.models.tracklist import ImportTracklistRequest, TrackEntry, Tracklist


@pytest.fixture
def client():
    """Create test client."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def mock_audio_file_id():
    """Mock audio file ID."""
    return uuid4()


@pytest.fixture
def mock_tracklist_request(mock_audio_file_id):
    """Create mock import request."""
    return ImportTracklistRequest(
        url="https://1001tracklists.com/tracklist/12345/test-set",
        audio_file_id=mock_audio_file_id,
        force_refresh=False,
        cue_format="standard",
    )


@pytest.fixture
def mock_imported_tracklist(mock_audio_file_id):
    """Create mock imported tracklist."""
    track_entry = TrackEntry(
        position=1,
        start_time=timedelta(minutes=1),
        end_time=timedelta(minutes=4, seconds=30),
        artist="Test Artist",
        title="Test Track",
        confidence=0.9,
    )

    return Tracklist(
        id=uuid4(),
        audio_file_id=mock_audio_file_id,
        source="1001tracklists",
        tracks=[track_entry],
        confidence_score=0.9,
    )


class TestImportEndpoints:
    """Test import API endpoints."""

    @patch("services.tracklist_service.src.api.import_endpoints.import_service")
    @patch("services.tracklist_service.src.api.import_endpoints.matching_service")
    @patch("services.tracklist_service.src.api.import_endpoints.timing_service")
    @patch("services.tracklist_service.src.api.import_endpoints.cue_integration_service")
    @patch("services.tracklist_service.src.api.import_endpoints.get_db_session")
    @patch("services.tracklist_service.src.api.import_endpoints.message_handler")
    def test_import_tracklist_success(
        self,
        mock_message_handler,
        mock_get_db_session,
        mock_cue_service,
        mock_timing_service,
        mock_matching_service,
        mock_import_service,
        client,
        mock_tracklist_request,
        mock_imported_tracklist,
        mock_audio_file_id,
    ):
        """Test successful tracklist import."""
        # Setup mocks
        mock_db = MagicMock()
        mock_get_db_session.return_value.__enter__.return_value = mock_db
        mock_get_db_session.return_value.__exit__.return_value = None

        # Mock service responses
        mock_import_service.import_tracklist.return_value = mock_imported_tracklist

        mock_matching_result = MagicMock()
        mock_matching_result.confidence_score = 0.95
        mock_matching_result.metadata = {"duration_seconds": 3600}
        mock_matching_service.match_tracklist_to_audio.return_value = mock_matching_result

        mock_timing_service.adjust_track_timings.return_value = mock_imported_tracklist

        mock_cue_result = MagicMock()
        mock_cue_result.success = True
        mock_cue_result.cue_file_id = uuid4()
        mock_cue_result.cue_file_path = "/path/to/test.cue"
        mock_cue_service.generate_cue_file.return_value = mock_cue_result

        mock_message_handler.publish = AsyncMock()

        # Make request
        response = client.post("/api/v1/tracklists/import/1001tracklists", json=mock_tracklist_request.model_dump())

        # Assertions
        assert response.status_code == 200
        response_data = response.json()

        assert response_data["success"] is True
        assert response_data["tracklist"] is not None
        assert response_data["cue_file_path"] == "/path/to/test.cue"
        assert "processing_time_ms" in response_data
        assert "correlation_id" in response_data

        # Verify service calls
        mock_import_service.import_tracklist.assert_called_once_with(
            url=mock_tracklist_request.url,
            audio_file_id=mock_tracklist_request.audio_file_id,
            force_refresh=mock_tracklist_request.force_refresh,
        )

        mock_matching_service.match_tracklist_to_audio.assert_called_once()
        mock_timing_service.adjust_track_timings.assert_called_once()
        mock_cue_service.generate_cue_file.assert_called_once()

        # Verify database operations
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once()

    @patch("services.tracklist_service.src.api.import_endpoints.import_service")
    @patch("services.tracklist_service.src.api.import_endpoints.get_db_session")
    def test_import_tracklist_validation_error(self, mock_get_db_session, mock_import_service, client):
        """Test import with invalid URL."""
        invalid_request = {
            "url": "https://invalid-site.com/tracklist",
            "audio_file_id": str(uuid4()),
            "force_refresh": False,
            "cue_format": "standard",
        }

        response = client.post("/api/v1/tracklists/import/1001tracklists", json=invalid_request)

        # Should fail validation
        assert response.status_code == 422

    @patch("services.tracklist_service.src.api.import_endpoints.import_service")
    @patch("services.tracklist_service.src.api.import_endpoints.get_db_session")
    def test_import_tracklist_service_error(
        self, mock_get_db_session, mock_import_service, client, mock_tracklist_request
    ):
        """Test import service error handling."""
        # Setup mock to raise ValueError
        mock_import_service.import_tracklist.side_effect = ValueError("Failed to fetch tracklist")

        mock_db = MagicMock()
        mock_get_db_session.return_value.__enter__.return_value = mock_db
        mock_get_db_session.return_value.__exit__.return_value = None

        response = client.post("/api/v1/tracklists/import/1001tracklists", json=mock_tracklist_request.model_dump())

        # Should return error response
        assert response.status_code == 200
        response_data = response.json()

        assert response_data["success"] is False
        assert "Import failed" in response_data["error"]
        assert response_data["tracklist"] is None
        assert "processing_time_ms" in response_data

    @patch("services.tracklist_service.src.api.import_endpoints.import_service")
    @patch("services.tracklist_service.src.api.import_endpoints.get_db_session")
    def test_import_tracklist_unexpected_error(
        self, mock_get_db_session, mock_import_service, client, mock_tracklist_request
    ):
        """Test unexpected error handling."""
        # Setup mock to raise unexpected error
        mock_import_service.import_tracklist.side_effect = RuntimeError("Unexpected error")

        mock_db = MagicMock()
        mock_get_db_session.return_value.__enter__.return_value = mock_db
        mock_get_db_session.return_value.__exit__.return_value = None

        response = client.post("/api/v1/tracklists/import/1001tracklists", json=mock_tracklist_request.model_dump())

        # Should return generic error response
        assert response.status_code == 200
        response_data = response.json()

        assert response_data["success"] is False
        assert response_data["error"] == "Internal server error occurred during import"
        assert response_data["tracklist"] is None

    @patch("services.tracklist_service.src.api.import_endpoints.process_import_async")
    def test_import_tracklist_async_processing(self, mock_process_async, client, mock_tracklist_request):
        """Test async processing option."""
        response = client.post(
            "/api/v1/tracklists/import/1001tracklists",
            json=mock_tracklist_request.model_dump(),
            params={"async_processing": True},
        )

        # Should return async response
        assert response.status_code == 200
        response_data = response.json()

        assert response_data["success"] is True
        assert response_data["tracklist"] is None
        assert "queued for async processing" in response_data.get("message", "")
        assert "correlation_id" in response_data

    @patch("services.tracklist_service.src.api.import_endpoints.cache")
    def test_get_import_status_not_found(self, mock_cache, client):
        """Test status check for non-existent job."""
        correlation_id = uuid4()
        mock_cache.get.return_value = None

        response = client.get(f"/api/v1/tracklists/import/status/{correlation_id}")

        assert response.status_code == 404
        assert "No import job found" in response.json()["detail"]

    @patch("services.tracklist_service.src.api.import_endpoints.cache")
    def test_get_import_status_completed(self, mock_cache, client):
        """Test status check for completed job."""
        correlation_id = uuid4()
        mock_status = {"status": "completed", "progress": 100, "message": "Import completed successfully"}
        mock_result = {"tracklist_id": str(uuid4())}

        mock_cache.get.side_effect = [mock_status, mock_result]

        response = client.get(f"/api/v1/tracklists/import/status/{correlation_id}")

        assert response.status_code == 200
        response_data = response.json()

        assert response_data["status"] == "completed"
        assert response_data["result"] == mock_result

    @patch("services.tracklist_service.src.api.import_endpoints.cache")
    def test_clear_import_cache_specific_url(self, mock_cache, client):
        """Test clearing cache for specific URL."""
        test_url = "https://1001tracklists.com/tracklist/12345/test-set"
        mock_cache.delete.return_value = 1

        response = client.delete("/api/v1/tracklists/import/cache", params={"url": test_url})

        assert response.status_code == 200
        response_data = response.json()

        assert response_data["success"] is True
        assert response_data["entries_cleared"] == 1
        assert test_url in response_data["message"]

        mock_cache.delete.assert_called_once()

    def test_clear_import_cache_bulk(self, client):
        """Test bulk cache clearing (not implemented)."""
        response = client.delete("/api/v1/tracklists/import/cache")

        assert response.status_code == 200
        response_data = response.json()

        assert response_data["success"] is False
        assert "not yet implemented" in response_data["message"]

    @patch("services.tracklist_service.src.api.import_endpoints.ImportService")
    @patch("services.tracklist_service.src.api.import_endpoints.MatchingService")
    @patch("services.tracklist_service.src.api.import_endpoints.TimingService")
    @patch("services.tracklist_service.src.api.import_endpoints.CueIntegrationService")
    @patch("services.tracklist_service.src.api.import_endpoints.cache")
    @patch("services.tracklist_service.src.api.import_endpoints.message_handler")
    def test_import_health_check_healthy(
        self,
        mock_message_handler,
        mock_cache,
        mock_cue_service_class,
        mock_timing_service_class,
        mock_matching_service_class,
        mock_import_service_class,
        client,
    ):
        """Test health check with all services healthy."""
        # Mock all services as healthy
        mock_import_service_class.return_value = MagicMock()
        mock_matching_service_class.return_value = MagicMock()
        mock_timing_service_class.return_value = MagicMock()
        mock_cue_service_class.return_value = MagicMock()

        mock_cache.ping = AsyncMock()
        mock_message_handler.ping = AsyncMock()

        response = client.get("/api/v1/tracklists/import/health")

        assert response.status_code == 200
        response_data = response.json()

        assert response_data["service"] == "tracklist_import_api"
        assert response_data["status"] == "healthy"
        assert response_data["components"]["import_service"] == "healthy"
        assert response_data["components"]["matching_service"] == "healthy"
        assert response_data["components"]["timing_service"] == "healthy"
        assert response_data["components"]["cue_integration_service"] == "healthy"
        assert response_data["components"]["cache"] == "healthy"
        assert response_data["components"]["message_handler"] == "healthy"

    @patch("services.tracklist_service.src.api.import_endpoints.ImportService")
    @patch("services.tracklist_service.src.api.import_endpoints.cache")
    @patch("services.tracklist_service.src.api.import_endpoints.message_handler")
    def test_import_health_check_degraded(self, mock_message_handler, mock_cache, mock_import_service_class, client):
        """Test health check with service failures."""
        # Mock import service as failing
        mock_import_service_class.side_effect = Exception("Service unavailable")

        # Mock cache and message handler as healthy
        mock_cache.ping = AsyncMock()
        mock_message_handler.ping = AsyncMock()

        response = client.get("/api/v1/tracklists/import/health")

        assert response.status_code == 503
        response_data = response.json()

        assert response_data["status"] == "degraded"
        assert "unhealthy" in response_data["components"]["import_service"]
