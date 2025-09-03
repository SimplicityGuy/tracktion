"""Tests for metadata API endpoints."""

import uuid
from unittest.mock import AsyncMock, Mock, patch

from fastapi import status
from fastapi.testclient import TestClient

from services.analysis_service.src.api.app import app


class TestMetadataEndpoints:
    """Test metadata API endpoints."""

    @patch("services.analysis_service.src.api.endpoints.metadata.metadata_repo")
    @patch("services.analysis_service.src.api.endpoints.metadata.recording_repo")
    def test_get_metadata_for_recording(self, mock_recording_repo, mock_metadata_repo):
        """Test getting metadata for a recording."""
        # Mock recording exists
        mock_recording = Mock()
        mock_recording.id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        # Mock metadata
        mock_metadata = Mock()
        mock_metadata.key = "artist"
        mock_metadata.value = "Test Artist"
        mock_metadata.source = "user"
        mock_metadata.confidence = 1.0
        mock_metadata_repo.get_by_recording_id = AsyncMock(return_value=[mock_metadata])

        client = TestClient(app)
        response = client.get("/v1/metadata/recording/550e8400-e29b-41d4-a716-446655440000")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 1
        assert data[0]["key"] == "artist"
        assert data[0]["value"] == "Test Artist"

    @patch("services.analysis_service.src.api.endpoints.metadata.metadata_repo")
    @patch("services.analysis_service.src.api.endpoints.metadata.recording_repo")
    def test_get_metadata_recording_not_found(self, mock_recording_repo, mock_metadata_repo):
        """Test getting metadata for non-existent recording."""
        mock_recording_repo.get_by_id = AsyncMock(return_value=None)

        client = TestClient(app)
        response = client.get("/v1/metadata/recording/550e8400-e29b-41d4-a716-446655440000")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @patch("services.analysis_service.src.api.endpoints.metadata.metadata_repo")
    @patch("services.analysis_service.src.api.endpoints.metadata.recording_repo")
    def test_add_metadata(self, mock_recording_repo, mock_metadata_repo):
        """Test adding metadata to a recording."""
        # Mock recording exists
        mock_recording = Mock()
        mock_recording.id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        # Mock metadata creation
        mock_metadata = Mock()
        mock_metadata.id = uuid.UUID("660e8400-e29b-41d4-a716-446655440001")
        mock_metadata.key = "genre"
        mock_metadata.value = "Electronic"
        mock_metadata.source = "api"
        mock_metadata.confidence = 0.9
        mock_metadata_repo.create = AsyncMock(return_value=mock_metadata)

        client = TestClient(app)
        response = client.post(
            "/v1/metadata",
            json={
                "recording_id": "550e8400-e29b-41d4-a716-446655440000",
                "key": "genre",
                "value": "Electronic",
                "source": "api",
                "confidence": 0.9,
            },
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["key"] == "genre"
        assert data["value"] == "Electronic"

    @patch("services.analysis_service.src.api.endpoints.metadata.metadata_repo")
    def test_update_metadata(self, mock_metadata_repo):
        """Test updating metadata."""
        # Mock existing metadata
        mock_metadata = Mock()
        mock_metadata.id = uuid.UUID("660e8400-e29b-41d4-a716-446655440001")
        mock_metadata.key = "title"
        mock_metadata.value = "Updated Title"
        mock_metadata.confidence = 0.95
        mock_metadata_repo.get_by_id = AsyncMock(return_value=mock_metadata)
        mock_metadata_repo.update = AsyncMock(return_value=mock_metadata)

        client = TestClient(app)
        response = client.put(
            "/v1/metadata/660e8400-e29b-41d4-a716-446655440001", json={"value": "Updated Title", "confidence": 0.95}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["value"] == "Updated Title"

    @patch("services.analysis_service.src.api.endpoints.metadata.metadata_repo")
    def test_update_metadata_not_found(self, mock_metadata_repo):
        """Test updating non-existent metadata."""
        mock_metadata_repo.get_by_id = AsyncMock(return_value=None)

        client = TestClient(app)
        response = client.put("/v1/metadata/660e8400-e29b-41d4-a716-446655440001", json={"value": "Updated Title"})

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @patch("services.analysis_service.src.api.endpoints.metadata.metadata_repo")
    def test_delete_metadata(self, mock_metadata_repo):
        """Test deleting metadata."""
        mock_metadata_repo.delete = AsyncMock(return_value=True)

        client = TestClient(app)
        response = client.delete("/v1/metadata/660e8400-e29b-41d4-a716-446655440001")

        assert response.status_code == status.HTTP_204_NO_CONTENT

    @patch("services.analysis_service.src.api.endpoints.metadata.metadata_repo")
    def test_delete_metadata_not_found(self, mock_metadata_repo):
        """Test deleting non-existent metadata."""
        mock_metadata_repo.delete = AsyncMock(return_value=False)

        client = TestClient(app)
        response = client.delete("/v1/metadata/660e8400-e29b-41d4-a716-446655440001")

        assert response.status_code == status.HTTP_404_NOT_FOUND

    @patch("services.analysis_service.src.api.endpoints.metadata.metadata_repo")
    def test_search_metadata(self, mock_metadata_repo):
        """Test searching metadata."""
        mock_metadata = Mock()
        mock_metadata.key = "artist"
        mock_metadata.value = "Found Artist"
        mock_metadata_repo.search = AsyncMock(return_value=[mock_metadata])

        client = TestClient(app)
        response = client.get("/v1/metadata/search", params={"key": "artist", "value": "Found"})

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data) == 1
        assert data[0]["value"] == "Found Artist"
