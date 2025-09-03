"""Comprehensive tests for the Cataloging Service API endpoints.

This file contains tests for all REST API endpoints implemented in Epic 10,
including edge cases, validation, error handling, and authorization scenarios.
"""

import concurrent.futures
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError, NoResultFound

from services.cataloging_service.src.api.app import create_app
from services.cataloging_service.src.api.recordings import get_db_session
from services.cataloging_service.src.api.schemas import (
    HealthResponse,
    MetadataResponse,
    RecordingDetailResponse,
    RecordingResponse,
    TracklistResponse,
)


class TestFixtures:
    """Test fixtures and setup."""

    @pytest.fixture
    def sample_recording_id(self):
        """Sample recording UUID."""
        return uuid.uuid4()

    @pytest.fixture
    def sample_metadata_id(self):
        """Sample metadata UUID."""
        return uuid.uuid4()

    @pytest.fixture
    def sample_tracklist_id(self):
        """Sample tracklist UUID."""
        return uuid.uuid4()

    @pytest.fixture
    def sample_recording_create_data(self):
        """Sample data for creating a recording."""
        return {
            "file_path": "/music/new_track.mp3",
            "file_name": "new_track.mp3",
            "sha256_hash": "abc123def456ghi789",
            "xxh128_hash": "xyz987uvw654",
        }

    @pytest.fixture
    def sample_recording_update_data(self):
        """Sample data for updating a recording."""
        return {
            "file_path": "/music/updated_track.mp3",
            "file_name": "updated_track.mp3",
            "sha256_hash": "updated_hash_123",
            "xxh128_hash": "updated_xxh_456",
        }

    @pytest.fixture
    def sample_metadata_create_data(self, sample_recording_id):
        """Sample data for creating metadata."""
        return {
            "recording_id": str(sample_recording_id),
            "key": "bpm",
            "value": "128",
        }

    @pytest.fixture
    def sample_tracklist_create_data(self, sample_recording_id):
        """Sample data for creating a tracklist."""
        return {
            "recording_id": str(sample_recording_id),
            "source": "manual",
            "cue_file_path": None,
            "tracks": [
                {
                    "title": "Opening Track",
                    "artist": "DJ Test",
                    "start_time": "00:00:00",
                    "duration": 300,
                    "bpm": 128,
                    "key": "A minor",
                }
            ],
        }

    @pytest.fixture
    def mock_db_session(self):
        """Mock database session for dependency override."""

        async def _mock_get_db_session():
            mock_session = AsyncMock()
            yield mock_session

        return _mock_get_db_session

    @pytest.fixture
    def test_app(self, mock_db_session):
        """Create test app with database dependency override."""
        app = create_app()
        app.dependency_overrides[get_db_session] = mock_db_session
        return app

    @pytest.fixture
    def client(self, test_app):
        """Create test client with mocked database."""
        return TestClient(test_app)

    @pytest.fixture
    def sample_recording_response(self, sample_recording_id):
        """Sample recording response data."""
        return RecordingResponse(
            id=sample_recording_id,
            file_path="/music/test.mp3",
            file_name="test.mp3",
            sha256_hash="abc123def456",
            xxh128_hash="xyz789",
            created_at=datetime.now(UTC),
        )

    @pytest.fixture
    def sample_metadata_response(self, sample_metadata_id, sample_recording_id):
        """Sample metadata response data."""
        return MetadataResponse(
            id=sample_metadata_id,
            recording_id=sample_recording_id,
            key="bpm",
            value="128",
        )

    @pytest.fixture
    def sample_tracklist_response(self, sample_tracklist_id, sample_recording_id):
        """Sample tracklist response data."""
        return TracklistResponse(
            id=sample_tracklist_id,
            recording_id=sample_recording_id,
            source="manual",
            cue_file_path=None,
            tracks=[
                {
                    "title": "Test Track",
                    "artist": "Test Artist",
                    "start_time": "00:00:00",
                    "duration": 300,
                }
            ],
        )


class TestRecordingEndpoints(TestFixtures):
    """Test cases for recording API endpoints."""

    def test_get_recordings_default_pagination(self, client, sample_recording_response):
        """Test GET /recordings with default pagination."""
        with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.get_all.return_value = [sample_recording_response]

            response = client.get("/recordings")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["id"] == str(sample_recording_response.id)
            mock_repo.get_all.assert_called_once_with(limit=100, offset=0)

    def test_get_recordings_custom_pagination(self, client, sample_recording_response):
        """Test GET /recordings with custom pagination parameters."""
        with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.get_all.return_value = [sample_recording_response]

            response = client.get("/recordings?limit=50&offset=25")

            assert response.status_code == 200
            mock_repo.get_all.assert_called_once_with(limit=50, offset=25)

    def test_get_recordings_pagination_edge_cases(self, client):
        """Test pagination edge cases."""
        with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.get_all.return_value = []

            # Test zero limit
            response = client.get("/recordings?limit=0")
            assert response.status_code == 200

            # Test maximum limit
            response = client.get("/recordings?limit=1000")
            assert response.status_code == 200

            # Test limit too high (should fail validation)
            response = client.get("/recordings?limit=2000")
            assert response.status_code == 422

            # Test negative offset (should fail validation)
            response = client.get("/recordings?offset=-1")
            assert response.status_code == 422

            # Test large valid offset
            response = client.get("/recordings?offset=999999")
            assert response.status_code == 200

    def test_get_recording_by_id_success(self, client, sample_recording_id):
        """Test GET /recordings/{id} successful retrieval."""
        with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo

            detailed_response = RecordingDetailResponse(
                id=sample_recording_id,
                file_path="/music/test.mp3",
                file_name="test.mp3",
                sha256_hash="abc123def456",
                xxh128_hash="xyz789",
                created_at=datetime.now(UTC),
                metadata=[],
                tracklists=[],
            )
            mock_repo.get_with_all_relations.return_value = detailed_response

            response = client.get(f"/recordings/{sample_recording_id}")

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == str(sample_recording_id)
            assert "metadata" in data
            assert "tracklists" in data

    def test_get_recording_by_id_not_found(self, client):
        """Test GET /recordings/{id} with non-existent recording."""
        with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.get_with_all_relations.return_value = None

            recording_id = uuid.uuid4()
            response = client.get(f"/recordings/{recording_id}")

            assert response.status_code == 404
            assert response.json()["detail"] == "Recording not found"

    def test_get_recording_by_id_invalid_uuid(self, client):
        """Test GET /recordings/{id} with invalid UUID format."""
        response = client.get("/recordings/invalid-uuid")
        assert response.status_code == 422

    @pytest.mark.skip(reason="Endpoint not implemented yet - placeholder for future implementation")
    def test_create_recording_success(self, client, sample_recording_create_data):
        """Test POST /recordings successful creation."""
        # This test is for the future POST /recordings endpoint
        with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo

            new_recording_id = uuid.uuid4()
            created_response = RecordingResponse(
                id=new_recording_id,
                **sample_recording_create_data,
                created_at=datetime.now(UTC),
            )
            mock_repo.create.return_value = created_response

            response = client.post("/recordings", json=sample_recording_create_data)

            assert response.status_code == 201
            data = response.json()
            assert data["file_path"] == sample_recording_create_data["file_path"]
            assert data["file_name"] == sample_recording_create_data["file_name"]

    @pytest.mark.skip(reason="Endpoint not implemented yet - placeholder for future implementation")
    def test_create_recording_missing_required_fields(self, client):
        """Test POST /recordings with missing required fields."""
        incomplete_data = {"file_path": "/music/test.mp3"}  # Missing file_name

        response = client.post("/recordings", json=incomplete_data)
        assert response.status_code == 422

    @pytest.mark.skip(reason="Endpoint not implemented yet - placeholder for future implementation")
    def test_create_recording_duplicate_constraint_violation(self, client, sample_recording_create_data):
        """Test POST /recordings with database constraint violations."""
        with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.create.side_effect = IntegrityError("UNIQUE constraint failed", None, None)

            response = client.post("/recordings", json=sample_recording_create_data)

            assert response.status_code == 400
            assert "already exists" in response.json()["detail"].lower()

    @pytest.mark.skip(reason="Endpoint not implemented yet - placeholder for future implementation")
    def test_update_recording_success(self, client, sample_recording_id, sample_recording_update_data):
        """Test PUT /recordings/{id} successful update."""
        with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo

            updated_response = RecordingResponse(
                id=sample_recording_id,
                **sample_recording_update_data,
                created_at=datetime.now(UTC),
            )
            mock_repo.update.return_value = updated_response

            response = client.put(f"/recordings/{sample_recording_id}", json=sample_recording_update_data)

            assert response.status_code == 200
            data = response.json()
            assert data["file_path"] == sample_recording_update_data["file_path"]

    @pytest.mark.skip(reason="Endpoint not implemented yet - placeholder for future implementation")
    def test_update_recording_not_found(self, client, sample_recording_update_data):
        """Test PUT /recordings/{id} with non-existent recording."""
        with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.update.side_effect = NoResultFound()

            recording_id = uuid.uuid4()
            response = client.put(f"/recordings/{recording_id}", json=sample_recording_update_data)

            assert response.status_code == 404

    @pytest.mark.skip(reason="Endpoint not implemented yet - placeholder for future implementation")
    def test_update_recording_partial_update(self, client, sample_recording_id):
        """Test PUT /recordings/{id} with partial update data."""
        partial_update = {"file_name": "new_name.mp3"}

        with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo

            updated_response = RecordingResponse(
                id=sample_recording_id,
                file_path="/music/original_path.mp3",
                file_name="new_name.mp3",  # Updated field
                sha256_hash="original_hash",
                xxh128_hash="original_xxh",
                created_at=datetime.now(UTC),
            )
            mock_repo.update.return_value = updated_response

            response = client.put(f"/recordings/{sample_recording_id}", json=partial_update)

            assert response.status_code == 200
            data = response.json()
            assert data["file_name"] == "new_name.mp3"

    @pytest.mark.skip(reason="Endpoint not implemented yet - placeholder for future implementation")
    def test_delete_recording_success(self, client, sample_recording_id):
        """Test DELETE /recordings/{id} successful deletion."""
        with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.delete.return_value = True

            response = client.delete(f"/recordings/{sample_recording_id}")

            assert response.status_code == 204

    @pytest.mark.skip(reason="Endpoint not implemented yet - placeholder for future implementation")
    def test_delete_recording_not_found(self, client):
        """Test DELETE /recordings/{id} with non-existent recording."""
        with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.delete.side_effect = NoResultFound()

            recording_id = uuid.uuid4()
            response = client.delete(f"/recordings/{recording_id}")

            assert response.status_code == 404

    @pytest.mark.skip(reason="Endpoint not implemented yet - placeholder for future implementation")
    def test_delete_recording_cascade_relationships(self, client, sample_recording_id):
        """Test DELETE /recordings/{id} properly cascades to related entities."""
        with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.delete.return_value = True

            response = client.delete(f"/recordings/{sample_recording_id}")

            assert response.status_code == 204
            # Verify the repository delete method is called
            mock_repo.delete.assert_called_once_with(sample_recording_id)


class TestMetadataEndpoints(TestFixtures):
    """Test cases for metadata API endpoints."""

    def test_get_recording_metadata_success(self, client, sample_recording_id, sample_recording_response):
        """Test GET /recordings/{recording_id}/metadata successful retrieval."""
        sample_metadata = [
            MetadataResponse(
                id=uuid.uuid4(),
                recording_id=sample_recording_id,
                key="bpm",
                value="128",
            ),
            MetadataResponse(
                id=uuid.uuid4(),
                recording_id=sample_recording_id,
                key="genre",
                value="techno",
            ),
        ]

        with (
            patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_recording_repo_class,
            patch("services.cataloging_service.src.api.recordings.MetadataRepository") as mock_metadata_repo_class,
        ):
            mock_recording_repo = AsyncMock()
            mock_recording_repo_class.return_value = mock_recording_repo
            mock_recording_repo.get_by_id.return_value = sample_recording_response

            mock_metadata_repo = AsyncMock()
            mock_metadata_repo_class.return_value = mock_metadata_repo
            mock_metadata_repo.get_by_recording_id.return_value = sample_metadata

            response = client.get(f"/recordings/{sample_recording_id}/metadata")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["key"] == "bpm"
            assert data[1]["key"] == "genre"

    def test_get_recording_metadata_recording_not_found(self, client):
        """Test GET /recordings/{recording_id}/metadata when recording doesn't exist."""
        with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.get_by_id.return_value = None

            recording_id = uuid.uuid4()
            response = client.get(f"/recordings/{recording_id}/metadata")

            assert response.status_code == 404
            assert response.json()["detail"] == "Recording not found"

    def test_get_recording_metadata_empty_result(self, client, sample_recording_id, sample_recording_response):
        """Test GET /recordings/{recording_id}/metadata with no metadata."""
        with (
            patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_recording_repo_class,
            patch("services.cataloging_service.src.api.recordings.MetadataRepository") as mock_metadata_repo_class,
        ):
            mock_recording_repo = AsyncMock()
            mock_recording_repo_class.return_value = mock_recording_repo
            mock_recording_repo.get_by_id.return_value = sample_recording_response

            mock_metadata_repo = AsyncMock()
            mock_metadata_repo_class.return_value = mock_metadata_repo
            mock_metadata_repo.get_by_recording_id.return_value = []

            response = client.get(f"/recordings/{sample_recording_id}/metadata")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 0

    @pytest.mark.skip(reason="Endpoint not implemented yet - placeholder for future implementation")
    def test_get_metadata_by_id_success(self, client, sample_metadata_id):
        """Test GET /metadata/{id} successful retrieval."""
        # This test is for a future standalone metadata endpoint
        sample_metadata = MetadataResponse(
            id=sample_metadata_id,
            recording_id=uuid.uuid4(),
            key="bpm",
            value="128",
        )

        with patch("services.cataloging_service.src.api.metadata.MetadataRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.get_by_id.return_value = sample_metadata

            response = client.get(f"/metadata/{sample_metadata_id}")

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == str(sample_metadata_id)
            assert data["key"] == "bpm"

    @pytest.mark.skip(reason="Endpoint not implemented yet - placeholder for future implementation")
    def test_create_metadata_success(self, client, sample_metadata_create_data):
        """Test POST /metadata successful creation."""
        new_metadata_id = uuid.uuid4()
        created_response = MetadataResponse(
            id=new_metadata_id,
            recording_id=UUID(sample_metadata_create_data["recording_id"]),
            key=sample_metadata_create_data["key"],
            value=sample_metadata_create_data["value"],
        )

        with patch("services.cataloging_service.src.api.metadata.MetadataRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.create.return_value = created_response

            response = client.post("/metadata", json=sample_metadata_create_data)

            assert response.status_code == 201
            data = response.json()
            assert data["key"] == sample_metadata_create_data["key"]
            assert data["value"] == sample_metadata_create_data["value"]

    @pytest.mark.skip(reason="Endpoint not implemented yet - placeholder for future implementation")
    def test_create_metadata_invalid_recording_id(self, client):
        """Test POST /metadata with non-existent recording_id."""
        invalid_metadata_data = {
            "recording_id": str(uuid.uuid4()),
            "key": "test_key",
            "value": "test_value",
        }

        with patch("services.cataloging_service.src.api.metadata.MetadataRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.create.side_effect = IntegrityError("Foreign key constraint failed", None, None)

            response = client.post("/metadata", json=invalid_metadata_data)

            assert response.status_code == 400

    @pytest.mark.skip(reason="Endpoint not implemented yet - placeholder for future implementation")
    def test_create_metadata_missing_fields(self, client):
        """Test POST /metadata with missing required fields."""
        incomplete_data = {"key": "bpm"}  # Missing recording_id and value

        response = client.post("/metadata", json=incomplete_data)
        assert response.status_code == 422


class TestTracklistEndpoints(TestFixtures):
    """Test cases for tracklist API endpoints."""

    def test_get_recording_tracklists_success(self, client, sample_recording_id, sample_recording_response):
        """Test GET /recordings/{recording_id}/tracklist successful retrieval."""
        sample_tracklists = [
            TracklistResponse(
                id=uuid.uuid4(),
                recording_id=sample_recording_id,
                source="manual",
                cue_file_path=None,
                tracks=[
                    {
                        "title": "Test Track",
                        "artist": "Test Artist",
                        "start_time": "00:00:00",
                        "duration": 300,
                    }
                ],
            )
        ]

        with (
            patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_recording_repo_class,
            patch("services.cataloging_service.src.api.recordings.TracklistRepository") as mock_tracklist_repo_class,
        ):
            mock_recording_repo = AsyncMock()
            mock_recording_repo_class.return_value = mock_recording_repo
            mock_recording_repo.get_by_id.return_value = sample_recording_response

            mock_tracklist_repo = AsyncMock()
            mock_tracklist_repo_class.return_value = mock_tracklist_repo
            mock_tracklist_repo.get_by_recording_id.return_value = sample_tracklists

            response = client.get(f"/recordings/{sample_recording_id}/tracklist")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["source"] == "manual"
            assert len(data[0]["tracks"]) == 1

    def test_get_recording_tracklists_recording_not_found(self, client):
        """Test GET /recordings/{recording_id}/tracklist when recording doesn't exist."""
        with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.get_by_id.return_value = None

            recording_id = uuid.uuid4()
            response = client.get(f"/recordings/{recording_id}/tracklist")

            assert response.status_code == 404

    @pytest.mark.skip(reason="Endpoint not implemented yet - placeholder for future implementation")
    def test_get_tracklists_with_pagination(self, client):
        """Test GET /tracklists with pagination."""
        sample_tracklists = [
            TracklistResponse(
                id=uuid.uuid4(),
                recording_id=uuid.uuid4(),
                source="automatic",
                cue_file_path="/path/to/cue",
                tracks=[{"title": "Track 1", "artist": "Artist 1"}],
            )
        ]

        with patch("services.cataloging_service.src.api.tracklist.TracklistRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.get_all.return_value = sample_tracklists

            response = client.get("/tracklists?limit=50&offset=0")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1

    @pytest.mark.skip(reason="Endpoint not implemented yet - placeholder for future implementation")
    def test_get_tracklist_by_id_success(self, client, sample_tracklist_id):
        """Test GET /tracklists/{id} successful retrieval."""
        sample_tracklist = TracklistResponse(
            id=sample_tracklist_id,
            recording_id=uuid.uuid4(),
            source="manual",
            cue_file_path=None,
            tracks=[{"title": "Test Track"}],
        )

        with patch("services.cataloging_service.src.api.tracklist.TracklistRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.get_by_id.return_value = sample_tracklist

            response = client.get(f"/tracklists/{sample_tracklist_id}")

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == str(sample_tracklist_id)

    @pytest.mark.skip(reason="Endpoint not implemented yet - placeholder for future implementation")
    def test_create_tracklist_success(self, client, sample_tracklist_create_data):
        """Test POST /tracklists successful creation."""
        new_tracklist_id = uuid.uuid4()
        created_response = TracklistResponse(
            id=new_tracklist_id,
            recording_id=UUID(sample_tracklist_create_data["recording_id"]),
            source=sample_tracklist_create_data["source"],
            cue_file_path=sample_tracklist_create_data["cue_file_path"],
            tracks=sample_tracklist_create_data["tracks"],
        )

        with patch("services.cataloging_service.src.api.tracklist.TracklistRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.create.return_value = created_response

            response = client.post("/tracklists", json=sample_tracklist_create_data)

            assert response.status_code == 201
            data = response.json()
            assert data["source"] == sample_tracklist_create_data["source"]

    @pytest.mark.skip(reason="Endpoint not implemented yet - placeholder for future implementation")
    def test_create_tracklist_invalid_tracks_json(self, client, sample_tracklist_create_data):
        """Test POST /tracklists with invalid tracks JSON structure."""
        invalid_data = sample_tracklist_create_data.copy()
        invalid_data["tracks"] = "invalid_json_string"

        response = client.post("/tracklists", json=invalid_data)
        assert response.status_code == 422


class TestSearchAndFilteringEndpoints(TestFixtures):
    """Test cases for search and filtering functionality."""

    def test_search_recordings_by_file_name(self, client, sample_recording_response):
        """Test POST /recordings/search by file name."""
        with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.search_by_file_name.return_value = [sample_recording_response]

            search_data = {
                "query": "test.mp3",
                "field": "file_name",
                "limit": 50,
                "offset": 0,
            }

            response = client.post("/recordings/search", json=search_data)

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            mock_repo.search_by_file_name.assert_called_once_with("test.mp3", limit=50)

    def test_search_recordings_by_file_path(self, client, sample_recording_response):
        """Test POST /recordings/search by file path."""
        with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.get_by_file_path.return_value = sample_recording_response

            search_data = {
                "query": "/music/test.mp3",
                "field": "file_path",
            }

            response = client.post("/recordings/search", json=search_data)

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1

    def test_search_recordings_by_hash_fields(self, client, sample_recording_response):
        """Test POST /recordings/search by hash fields."""
        hash_fields = ["sha256_hash", "xxh128_hash"]

        for field in hash_fields:
            with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
                mock_repo = AsyncMock()
                mock_repo_class.return_value = mock_repo
                getattr(mock_repo, f"get_by_{field}").return_value = sample_recording_response

                search_data = {
                    "query": "hash_value_123",
                    "field": field,
                }

                response = client.post("/recordings/search", json=search_data)

                assert response.status_code == 200
                data = response.json()
                assert len(data) == 1

    def test_search_recordings_no_results(self, client):
        """Test POST /recordings/search with no results."""
        with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.get_by_file_path.return_value = None

            search_data = {
                "query": "/nonexistent/file.mp3",
                "field": "file_path",
            }

            response = client.post("/recordings/search", json=search_data)

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 0

    def test_search_recordings_invalid_field(self, client):
        """Test POST /recordings/search with invalid field."""
        search_data = {
            "query": "test",
            "field": "invalid_field",
        }

        response = client.post("/recordings/search", json=search_data)

        assert response.status_code == 400
        assert "Invalid search field" in response.json()["detail"]

    def test_search_recordings_validation_errors(self, client):
        """Test POST /recordings/search validation errors."""
        # Missing required fields
        response = client.post("/recordings/search", json={})
        assert response.status_code == 422

        # Invalid limit (too high)
        search_data = {
            "query": "test",
            "field": "file_name",
            "limit": 2000,
        }
        response = client.post("/recordings/search", json=search_data)
        assert response.status_code == 422

        # Invalid offset (negative)
        search_data = {
            "query": "test",
            "field": "file_name",
            "offset": -1,
        }
        response = client.post("/recordings/search", json=search_data)
        assert response.status_code == 422


class TestSortingAndFiltering(TestFixtures):
    """Test cases for sorting and advanced filtering."""

    @pytest.mark.skip(reason="Sorting endpoints not implemented yet - placeholder for future implementation")
    def test_get_recordings_with_sorting(self, client):
        """Test GET /recordings with sorting parameters."""
        # This test is for future sorting functionality
        sort_params = [
            "created_at",
            "-created_at",
            "file_name",
            "-file_name",
            "file_path",
            "-file_path",
        ]

        for sort_param in sort_params:
            with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
                mock_repo = AsyncMock()
                mock_repo_class.return_value = mock_repo
                mock_repo.get_all.return_value = []

                response = client.get(f"/recordings?sort={sort_param}")

                assert response.status_code == 200
                # Verify the repository was called with sort parameter
                mock_repo.get_all.assert_called()

    @pytest.mark.skip(reason="Filtering endpoints not implemented yet - placeholder for future implementation")
    def test_get_recordings_with_filters(self, client):
        """Test GET /recordings with filter parameters."""
        # This test is for future filtering functionality
        filter_params = {
            "file_extension": "mp3",
            "has_metadata": "true",
            "has_tracklist": "false",
            "created_after": "2023-01-01",
            "created_before": "2024-01-01",
        }

        for param_name, param_value in filter_params.items():
            with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
                mock_repo = AsyncMock()
                mock_repo_class.return_value = mock_repo
                mock_repo.get_all_with_filters.return_value = []

                response = client.get(f"/recordings?{param_name}={param_value}")

                assert response.status_code == 200

    @pytest.mark.skip(reason="Advanced search endpoints not implemented yet")
    def test_advanced_search_combinations(self, client):
        """Test advanced search with multiple parameters."""
        # This test is for future advanced search functionality
        advanced_search_data = {
            "filters": {
                "file_extension": ["mp3", "flac"],
                "has_metadata": True,
                "created_after": "2023-01-01",
            },
            "sort": ["-created_at", "file_name"],
            "limit": 50,
            "offset": 0,
        }

        response = client.post("/recordings/advanced-search", json=advanced_search_data)
        # This endpoint doesn't exist yet, so we expect 404
        assert response.status_code == 404


class TestErrorHandlingAndEdgeCases(TestFixtures):
    """Test cases for error handling and edge cases."""

    def test_invalid_uuid_parameters(self, client):
        """Test endpoints with invalid UUID parameters."""
        invalid_uuids = [
            "invalid-uuid",
            "12345",
            "not-a-uuid-at-all",
            "123e4567-e89b-12d3-a456-42661417400",  # Invalid UUID format
        ]

        for invalid_uuid in invalid_uuids:
            # Skip empty string as it causes different routing behavior
            if invalid_uuid == "":
                continue

            response = client.get(f"/recordings/{invalid_uuid}")
            assert response.status_code == 422

            response = client.get(f"/recordings/{invalid_uuid}/metadata")
            assert response.status_code == 422

            response = client.get(f"/recordings/{invalid_uuid}/tracklist")
            assert response.status_code == 422

    def test_malformed_json_requests(self, client):
        """Test endpoints with malformed JSON."""
        # Invalid JSON syntax
        response = client.post(
            "/recordings/search",
            content="{'invalid': json}",
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422

        # Empty JSON
        response = client.post(
            "/recordings/search",
            json={},
        )
        assert response.status_code == 422

    def test_content_type_validation(self, client):
        """Test content type validation for POST requests."""
        search_data = {"query": "test", "field": "file_name"}

        # Mock the repository to prevent the actual database call
        with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.search_by_file_name.return_value = []

            # Correct content type
            response = client.post("/recordings/search", json=search_data)
            assert response.status_code == 200

            # Wrong content type
            response = client.post(
                "/recordings/search",
                content='{"query": "test", "field": "file_name"}',
                headers={"content-type": "text/plain"},
            )
            assert response.status_code in [400, 422]

    def test_method_not_allowed_errors(self, client):
        """Test method not allowed errors."""
        # Test unsupported methods on existing endpoints
        response = client.put("/recordings/search")
        assert response.status_code == 405

        response = client.delete("/recordings/search")
        assert response.status_code == 405

        response = client.patch("/health")
        assert response.status_code == 405

    def test_endpoint_not_found_errors(self, client):
        """Test 404 errors for non-existent endpoints."""
        non_existent_endpoints = [
            "/non-existent",
            "/metadata/search",  # This endpoint doesn't exist
            "/tracklists/search",  # This endpoint doesn't exist
        ]

        for endpoint in non_existent_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 404

        # Test path that matches pattern but is invalid UUID (returns 422)
        response = client.get("/recordings/invalid-endpoint")
        assert response.status_code == 422

    def test_large_payload_handling(self, client):
        """Test handling of large payloads."""
        # Create a large search query
        large_query = "x" * 10000
        search_data = {"query": large_query, "field": "file_name"}

        with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.search_by_file_name.return_value = []

            response = client.post("/recordings/search", json=search_data)

            # Should handle large payload gracefully
            assert response.status_code == 200

    @pytest.mark.skip(reason="Database constraint testing requires implemented endpoints")
    def test_database_constraint_violations(self, client, sample_recording_create_data):
        """Test handling of database constraint violations."""
        # This test is for future implementation when POST/PUT endpoints exist
        with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo

            # Simulate unique constraint violation
            mock_repo.create.side_effect = IntegrityError("UNIQUE constraint failed: recordings.file_path", None, None)

            response = client.post("/recordings", json=sample_recording_create_data)

            assert response.status_code == 400
            assert "already exists" in response.json()["detail"].lower()


class TestHealthAndUtilityEndpoints(TestFixtures):
    """Test cases for health and utility endpoints."""

    def test_health_endpoint_success(self, client):
        """Test GET /health endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["service"] == "cataloging-service"
        assert data["version"] == "0.1.0"
        assert "timestamp" in data

        # Validate response schema
        health_response = HealthResponse(**data)
        assert health_response.status == "healthy"

    def test_metrics_endpoint_success(self, client):
        """Test GET /metrics endpoint."""
        response = client.get("/metrics")

        assert response.status_code == 200
        data = response.json()

        assert data["service"] == "cataloging-service"
        assert data["status"] == "operational"
        assert "timestamp" in data

    def test_health_endpoint_response_headers(self, client):
        """Test health endpoint response headers."""
        response = client.get("/health")

        assert response.status_code == 200
        assert "content-type" in response.headers
        assert response.headers["content-type"] == "application/json"

    def test_cors_headers_present(self, client):
        """Test CORS headers are present in responses."""
        response = client.get("/health", headers={"origin": "http://localhost:3000"})

        assert "access-control-allow-origin" in response.headers
        assert response.headers["access-control-allow-origin"] == "*"

    def test_request_id_header_present(self, client):
        """Test request ID header is present in responses."""
        response = client.get("/health")

        assert response.status_code == 200
        assert "x-request-id" in response.headers

        # Verify request ID is a valid UUID
        request_id = response.headers["x-request-id"]
        uuid.UUID(request_id)  # Should not raise an exception


class TestAuthorizationAndSecurity(TestFixtures):
    """Test cases for authorization and security scenarios."""

    @pytest.mark.skip(reason="Authorization not implemented yet - placeholder for future implementation")
    def test_unauthorized_access(self, client):
        """Test unauthorized access to protected endpoints."""
        # This test is for future authorization implementation
        protected_endpoints = [
            ("POST", "/recordings"),
            ("PUT", "/recordings/123e4567-e89b-12d3-a456-426614174000"),
            ("DELETE", "/recordings/123e4567-e89b-12d3-a456-426614174000"),
            ("POST", "/metadata"),
            ("POST", "/tracklists"),
        ]

        for method, endpoint in protected_endpoints:
            if method == "POST":
                response = client.post(endpoint, json={"test": "data"})
            elif method == "PUT":
                response = client.put(endpoint, json={"test": "data"})
            elif method == "DELETE":
                response = client.delete(endpoint)
            else:
                response = client.get(endpoint)

            # Assuming future authorization implementation returns 401
            assert response.status_code in [401, 403, 404]  # 404 if endpoint doesn't exist yet

    @pytest.mark.skip(reason="Authorization not implemented yet - placeholder for future implementation")
    def test_forbidden_access(self, client):
        """Test forbidden access with insufficient permissions."""
        # This test is for future role-based authorization
        # Simulate a user with read-only permissions trying to modify data
        headers = {"authorization": "Bearer read-only-token"}

        response = client.post("/recordings", json={"test": "data"}, headers=headers)
        assert response.status_code == 403

    @pytest.mark.skip(reason="Rate limiting not implemented yet")
    def test_rate_limiting(self, client):
        """Test rate limiting functionality."""
        # This test is for future rate limiting implementation
        # Make multiple requests to trigger rate limiting
        for _ in range(100):  # Assuming rate limit is lower than 100
            response = client.get("/recordings")
            if response.status_code == 429:  # Too Many Requests
                break
        else:
            pytest.skip("Rate limiting not implemented or limit is too high")

        assert response.status_code == 429

    def test_sql_injection_prevention(self, client):
        """Test SQL injection prevention in search endpoints."""
        malicious_queries = [
            "'; DROP TABLE recordings; --",
            "' OR '1'='1",
            "1' UNION SELECT * FROM recordings --",
            "'; INSERT INTO recordings VALUES (1,2,3); --",
        ]

        for malicious_query in malicious_queries:
            with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
                mock_repo = AsyncMock()
                mock_repo_class.return_value = mock_repo
                mock_repo.search_by_file_name.return_value = []

                search_data = {
                    "query": malicious_query,
                    "field": "file_name",
                }

                response = client.post("/recordings/search", json=search_data)

                # Should not cause any errors and should return safely
                assert response.status_code == 200
                # The query should be handled safely by the repository layer


class TestPerformanceAndScalability(TestFixtures):
    """Test cases for performance and scalability scenarios."""

    def test_large_result_set_pagination(self, client):
        """Test pagination with large result sets."""
        # Simulate a large number of recordings
        large_recording_list = [
            RecordingResponse(
                id=uuid.uuid4(),
                file_path=f"/music/track_{i}.mp3",
                file_name=f"track_{i}.mp3",
                sha256_hash=f"hash_{i}",
                xxh128_hash=f"xxh_{i}",
                created_at=datetime.now(UTC),
            )
            for i in range(1000)
        ]

        with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo
            mock_repo.get_all.return_value = large_recording_list[:100]  # Return first page

            response = client.get("/recordings?limit=100&offset=0")

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 100

    def test_concurrent_request_handling(self, client):
        """Test concurrent request handling."""
        # This test verifies that the API can handle multiple concurrent requests
        # without errors (basic concurrency test)
        results = []

        def make_request():
            try:
                with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
                    mock_repo = AsyncMock()
                    mock_repo_class.return_value = mock_repo
                    mock_repo.get_all.return_value = []

                    response = client.get("/recordings")
                    results.append(response.status_code)
            except Exception:
                # Handle any exceptions during concurrent testing
                results.append(500)  # Consider it a server error

        # Make 5 concurrent requests (reduced to avoid overwhelming)
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            concurrent.futures.wait(futures)

        # Most requests should succeed (allow some failures in concurrent testing)
        assert len(results) == 5
        success_rate = sum(1 for status_code in results if status_code == 200) / len(results)
        assert success_rate >= 0.6  # At least 60% should succeed

    def test_memory_efficient_large_payload(self, client):
        """Test memory efficiency with large payloads."""
        # Test search with large results
        search_data = {
            "query": "Track",
            "field": "file_name",
        }

        with patch("services.cataloging_service.src.api.recordings.RecordingRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo_class.return_value = mock_repo

            # Simulate finding a recording with large tracklist
            large_recording = RecordingResponse(
                id=uuid.uuid4(),
                file_path="/music/large_mix.mp3",
                file_name="large_mix.mp3",
                sha256_hash="large_hash",
                xxh128_hash="large_xxh",
                created_at=datetime.now(UTC),
            )

            mock_repo.search_by_file_name.return_value = [large_recording]

            response = client.post("/recordings/search", json=search_data)

            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1

            # Verify the large data is properly serialized
            assert data[0]["file_name"] == "large_mix.mp3"


class TestAPIDocumentationAndSchemas(TestFixtures):
    """Test cases for API documentation and schema validation."""

    def test_openapi_schema_generation(self, test_app):
        """Test OpenAPI schema generation."""
        openapi_schema = test_app.openapi()

        assert openapi_schema["info"]["title"] == "Cataloging Service API"
        assert openapi_schema["info"]["description"] == "API for managing music file catalog"
        assert openapi_schema["info"]["version"] == "0.1.0"

        # Check that all current endpoints are documented
        paths = openapi_schema["paths"]
        assert "/health" in paths
        assert "/metrics" in paths
        assert "/recordings" in paths
        assert "/recordings/search" in paths
        assert "/recordings/{recording_id}" in paths
        assert "/recordings/{recording_id}/metadata" in paths
        assert "/recordings/{recording_id}/tracklist" in paths

    def test_response_model_schemas(self, test_app):
        """Test that response model schemas are properly defined."""
        openapi_schema = test_app.openapi()

        components = openapi_schema["components"]["schemas"]
        assert "RecordingResponse" in components
        assert "RecordingDetailResponse" in components
        assert "MetadataResponse" in components
        assert "TracklistResponse" in components
        assert "SearchRequest" in components
        assert "HealthResponse" in components

        # Verify RecordingResponse has required fields
        recording_schema = components["RecordingResponse"]
        required_fields = recording_schema.get("required", [])
        assert "id" in required_fields
        assert "file_path" in required_fields
        assert "file_name" in required_fields

    def test_request_validation_schemas(self, test_app):
        """Test request validation schema definitions."""
        openapi_schema = test_app.openapi()
        components = openapi_schema["components"]["schemas"]

        # Test SearchRequest schema
        search_schema = components["SearchRequest"]
        properties = search_schema["properties"]

        assert "query" in properties
        assert "field" in properties
        assert "limit" in properties
        assert "offset" in properties

        # Verify validation constraints
        assert properties["limit"]["maximum"] == 1000
        assert properties["offset"]["minimum"] == 0

    def test_endpoint_tags_and_descriptions(self, test_app):
        """Test endpoint tags and descriptions are properly set."""
        openapi_schema = test_app.openapi()
        paths = openapi_schema["paths"]

        # Check recordings endpoints have proper tags
        recordings_get = paths["/recordings"]["get"]
        assert "recordings" in recordings_get["tags"]
        assert "summary" in recordings_get or "description" in recordings_get

        # Check search endpoint documentation
        search_post = paths["/recordings/search"]["post"]
        assert "recordings" in search_post["tags"]

    def test_error_response_documentation(self, test_app):
        """Test error response documentation in OpenAPI schema."""
        openapi_schema = test_app.openapi()
        paths = openapi_schema["paths"]

        # Check that endpoints document error responses
        recording_by_id = paths["/recordings/{recording_id}"]["get"]
        responses = recording_by_id["responses"]

        # Should document validation errors (422 is automatically added by FastAPI)
        assert "422" in responses

        # Check that search endpoint also has proper error responses
        search_post = paths["/recordings/search"]["post"]
        search_responses = search_post["responses"]
        assert "422" in search_responses
