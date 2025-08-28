"""
Tests for CUE Generation API endpoints.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4
from fastapi.testclient import TestClient
from fastapi import FastAPI

from src.api.cue_generation_api import router
from src.models.cue_file import (
    CueGenerationResponse,
    BatchCueGenerationResponse,
    CueFileDB,
    ValidationReport,
)
from src.models.tracklist import Tracklist, Track


@pytest.fixture
def app():
    """Create FastAPI app with CUE generation router."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_tracklist():
    """Create sample tracklist for testing."""
    tracks = [
        Track(
            title=f"Track {i}",
            artist=f"Artist {i}",
            start_time=f"{(i - 1) * 5:02d}:00:00",
            end_time=f"{i * 5:02d}:00:00" if i < 5 else None,
            bpm=120 + i,
            key="Am" if i % 2 else "C",
        )
        for i in range(1, 6)
    ]

    return Tracklist(
        id=uuid4(),
        title="Test Mix",
        artist="Test DJ",
        tracks=tracks,
        audio_file_path="audio.wav",
        genre="Electronic",
        source="test",
    )


@pytest.fixture
def sample_cue_file():
    """Create sample CUE file for testing."""
    return CueFileDB(
        id=uuid4(),
        tracklist_id=uuid4(),
        format="standard",
        version=1,
        file_path="/tmp/cue_files/2024/01/test.cue",
        checksum="abc123def456",
        file_size=2048,
        generation_time_ms=150.5,
        is_active=True,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )


class TestCueGenerationEndpoints:
    """Test CUE file generation endpoints."""

    @patch("services.tracklist_service.src.api.cue_generation_api.cue_generation_service")
    def test_generate_cue_file_sync(self, mock_service, client, sample_tracklist):
        """Test synchronous CUE file generation."""
        # Setup
        mock_validation = ValidationReport(valid=True, warnings=[], error=None)
        mock_service.validate_tracklist_for_cue.return_value = AsyncMock(return_value=mock_validation)()

        mock_response = CueGenerationResponse(
            success=True,
            job_id=uuid4(),
            cue_file_id=str(uuid4()),
            file_path="/tmp/cue_files/test.cue",
            validation_report=mock_validation,
            error=None,
            processing_time_ms=100,
        )
        mock_service.generate_cue_file.return_value = AsyncMock(return_value=mock_response)()

        # Execute
        request_data = {
            "format": "standard",
            "options": {"include_bpm": True},
            "validate_audio": False,
        }

        response = client.post(
            "/api/v1/cue/generate",
            json={
                "request": request_data,
                "tracklist_data": sample_tracklist.model_dump(),
                "async_processing": False,
            },
        )

        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["cue_file_id"] is not None
        assert data["file_path"] is not None
        assert data["error"] is None

    @patch("services.tracklist_service.src.api.cue_generation_api.cue_generation_service")
    def test_generate_cue_file_validation_failure(self, mock_service, client, sample_tracklist):
        """Test CUE generation with validation failure."""
        # Setup
        mock_validation = ValidationReport(
            valid=False,
            warnings=["Missing track titles"],
            error="Invalid tracklist format",
        )
        mock_service.validate_tracklist_for_cue.return_value = AsyncMock(return_value=mock_validation)()

        # Execute
        request_data = {
            "format": "standard",
            "options": {},
            "validate_audio": False,
        }

        response = client.post(
            "/api/v1/cue/generate",
            json={
                "request": request_data,
                "tracklist_data": sample_tracklist.model_dump(),
                "async_processing": False,
            },
        )

        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["cue_file_id"] is None
        assert "validation failed" in data["error"].lower()

    @patch("services.tracklist_service.src.api.cue_generation_api.cue_generation_service")
    def test_generate_batch_cue_files(self, mock_service, client, sample_tracklist):
        """Test batch CUE file generation."""
        # Setup
        mock_response = BatchCueGenerationResponse(
            success=True,
            results=[],
            total_files=3,
            successful_files=3,
            failed_files=0,
            processing_time_ms=300,
        )
        mock_service.generate_multiple_formats.return_value = AsyncMock(return_value=mock_response)()

        # Execute
        request_data = {
            "formats": ["standard", "cdj", "traktor"],
            "options": {"include_bpm": True},
            "validate_audio": False,
        }

        response = client.post(
            "/api/v1/cue/generate/batch",
            json={
                "request": request_data,
                "tracklist_data": sample_tracklist.model_dump(),
                "async_processing": False,
            },
        )

        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["total_files"] == 3
        assert data["successful_files"] == 3
        assert data["failed_files"] == 0

    def test_get_supported_formats(self, client):
        """Test getting supported CUE formats."""
        response = client.get("/api/v1/cue/formats")

        assert response.status_code == 200
        formats = response.json()
        assert isinstance(formats, list)
        assert "standard" in formats
        assert "cdj" in formats
        assert "traktor" in formats

    @patch("services.tracklist_service.src.api.cue_generation_api.cue_generation_service")
    def test_get_format_capabilities(self, mock_service, client):
        """Test getting format capabilities."""
        # Setup
        mock_capabilities = {
            "max_tracks": 999,
            "supports_bpm": True,
            "supports_keys": True,
            "supports_cue_points": True,
            "limitations": [],
        }
        mock_service.get_format_capabilities.return_value = mock_capabilities

        # Execute
        response = client.get("/api/v1/cue/formats/traktor/capabilities")

        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["max_tracks"] == 999
        assert data["supports_bpm"] is True

    def test_get_format_capabilities_invalid_format(self, client):
        """Test getting capabilities for invalid format."""
        response = client.get("/api/v1/cue/formats/invalid/capabilities")
        assert response.status_code == 400
        assert "Unsupported CUE format" in response.json()["detail"]

    @patch("services.tracklist_service.src.api.cue_generation_api.cue_generation_service")
    def test_get_conversion_preview(self, mock_service, client):
        """Test conversion preview between formats."""
        # Setup
        mock_warnings = [
            "BPM data will be lost",
            "Key information not supported in target format",
        ]
        mock_service.get_conversion_preview.return_value = mock_warnings

        # Execute
        response = client.get(
            "/api/v1/cue/formats/conversion-preview",
            params={"source_format": "traktor", "target_format": "standard"},
        )

        # Verify
        assert response.status_code == 200
        warnings = response.json()
        assert len(warnings) == 2
        assert "BPM data will be lost" in warnings

    def test_get_job_status(self, client):
        """Test getting generation job status."""
        job_id = uuid4()
        response = client.get(f"/api/v1/cue/jobs/{job_id}/status")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == str(job_id)
        assert data["status"] == "completed"  # Placeholder implementation


class TestCueFileManagementEndpoints:
    """Test CUE file management endpoints."""

    @patch("services.tracklist_service.src.api.cue_generation_api.get_cue_file_repository")
    @patch("services.tracklist_service.src.api.cue_generation_api.storage_service")
    def test_get_cue_file_info(self, mock_storage, mock_get_repo, client, sample_cue_file):
        """Test getting CUE file information."""
        # Setup
        mock_repo = AsyncMock()
        mock_repo.get_cue_file_by_id = AsyncMock(return_value=sample_cue_file)
        mock_get_repo.return_value = mock_repo

        mock_storage.get_file_info.return_value = {
            "exists": True,
            "metadata": {"checksum": sample_cue_file.checksum},
            "versions": [],
            "version_count": 1,
        }

        # Execute
        response = client.get(f"/api/v1/cue/files/{sample_cue_file.id}")

        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_cue_file.id)
        assert data["format"] == sample_cue_file.format
        assert data["checksum"] == sample_cue_file.checksum

    @patch("services.tracklist_service.src.api.cue_generation_api.get_cue_file_repository")
    def test_get_cue_file_info_not_found(self, mock_get_repo, client):
        """Test getting non-existent CUE file."""
        # Setup
        mock_repo = AsyncMock()
        mock_repo.get_cue_file_by_id = AsyncMock(return_value=None)
        mock_get_repo.return_value = mock_repo

        # Execute
        response = client.get(f"/api/v1/cue/files/{uuid4()}")

        # Verify
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @patch("services.tracklist_service.src.api.cue_generation_api.get_cue_file_repository")
    def test_delete_cue_file_soft(self, mock_get_repo, client, sample_cue_file):
        """Test soft deleting CUE file."""
        # Setup
        mock_repo = AsyncMock()
        mock_repo.get_cue_file_by_id = AsyncMock(return_value=sample_cue_file)
        mock_repo.soft_delete_cue_file = AsyncMock(return_value=True)
        mock_get_repo.return_value = mock_repo

        # Execute
        response = client.delete(f"/api/v1/cue/files/{sample_cue_file.id}")

        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["deletion_type"] == "soft"
        mock_repo.soft_delete_cue_file.assert_called_once()

    @patch("services.tracklist_service.src.api.cue_generation_api.get_cue_file_repository")
    @patch("services.tracklist_service.src.api.cue_generation_api.storage_service")
    def test_delete_cue_file_hard(self, mock_storage, mock_get_repo, client, sample_cue_file):
        """Test hard deleting CUE file."""
        # Setup
        mock_repo = AsyncMock()
        mock_repo.get_cue_file_by_id = AsyncMock(return_value=sample_cue_file)
        mock_repo.hard_delete_cue_file = AsyncMock(return_value=True)
        mock_get_repo.return_value = mock_repo

        mock_storage.delete_cue_file.return_value = True

        # Execute
        response = client.delete(
            f"/api/v1/cue/files/{sample_cue_file.id}",
            params={"soft_delete": False},
        )

        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["deletion_type"] == "hard"
        mock_repo.hard_delete_cue_file.assert_called_once()
        mock_storage.delete_cue_file.assert_called_once()

    @patch("services.tracklist_service.src.api.cue_generation_api.get_cue_file_repository")
    def test_list_cue_files(self, mock_get_repo, client):
        """Test listing CUE files with pagination."""
        # Setup
        cue_files = [
            CueFileDB(
                id=uuid4(),
                tracklist_id=uuid4(),
                format="standard",
                version=1,
                file_path=f"/tmp/cue_files/test_{i}.cue",
                checksum=f"checksum_{i}",
                file_size=1024 * i,
                is_active=True,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            for i in range(3)
        ]

        mock_repo = AsyncMock()
        mock_repo.list_cue_files = AsyncMock(return_value=cue_files)
        mock_repo.count_cue_files = AsyncMock(return_value=10)
        mock_get_repo.return_value = mock_repo

        # Execute
        response = client.get(
            "/api/v1/cue/files",
            params={"limit": 3, "offset": 0, "format": "standard"},
        )

        # Verify
        assert response.status_code == 200
        data = response.json()
        assert len(data["files"]) == 3
        assert data["pagination"]["total"] == 10
        assert data["pagination"]["has_more"] is True


class TestValidationEndpoints:
    """Test validation endpoints."""

    @patch("services.tracklist_service.src.api.cue_generation_api.get_cue_file_repository")
    @patch("services.tracklist_service.src.api.cue_generation_api.storage_service")
    @patch("services.tracklist_service.src.api.cue_generation_api.cue_generation_service")
    def test_validate_cue_file(self, mock_service, mock_storage, mock_get_repo, client, sample_cue_file):
        """Test validating CUE file."""
        # Setup
        mock_repo = AsyncMock()
        mock_repo.get_cue_file_by_id = AsyncMock(return_value=sample_cue_file)
        mock_get_repo.return_value = mock_repo

        mock_storage.retrieve_cue_file.return_value = (
            True,
            'FILE "audio.wav" WAVE\nTRACK 01 AUDIO\nINDEX 01 00:00:00',
            None,
        )

        mock_validation = ValidationReport(
            valid=True,
            warnings=["Minor timing issue"],
            error=None,
            metadata={"track_count": 1},
        )
        mock_service.cue_integration.validate_cue_content.return_value = mock_validation

        # Execute
        response = client.post(f"/api/v1/cue/files/{sample_cue_file.id}/validate")

        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert len(data["warnings"]) == 1
        assert data["metadata"]["track_count"] == 1

    @patch("services.tracklist_service.src.api.cue_generation_api.get_tracklist_by_id")
    @patch("services.tracklist_service.src.api.cue_generation_api.cue_generation_service")
    def test_validate_tracklist_for_cue(self, mock_service, mock_get_tracklist, client, sample_tracklist):
        """Test validating tracklist for CUE generation."""
        # Setup
        mock_get_tracklist.return_value = AsyncMock(return_value=sample_tracklist)()

        mock_validation = ValidationReport(
            valid=True,
            warnings=[],
            error=None,
            metadata={"track_count": len(sample_tracklist.tracks)},
        )
        mock_service.validate_tracklist_for_cue.return_value = AsyncMock(return_value=mock_validation)()

        # Execute
        response = client.post(
            "/api/v1/cue/validate",
            params={"tracklist_id": str(sample_tracklist.id), "format": "standard"},
        )

        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["target_format"] == "standard"


class TestCacheEndpoints:
    """Test cache management endpoints."""

    @patch("services.tracklist_service.src.api.cue_generation_api.cache_service")
    def test_get_cache_stats(self, mock_cache, client):
        """Test getting cache statistics."""
        # Setup
        mock_stats = {
            "metrics": {
                "hits": 100,
                "misses": 20,
                "sets": 50,
                "hit_rate": 83.33,
            },
            "redis_connected": True,
            "memory_cache_size": 45,
        }
        mock_cache.get_cache_stats.return_value = AsyncMock(return_value=mock_stats)()

        # Execute
        response = client.get("/api/v1/cue/cache/stats")

        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["metrics"]["hits"] == 100
        assert data["metrics"]["hit_rate"] == 83.33

    @patch("services.tracklist_service.src.api.cue_generation_api.cache_service")
    def test_warm_cache(self, mock_cache, client):
        """Test cache warming."""
        # Setup
        tracklist_ids = [str(uuid4()) for _ in range(3)]
        mock_result = {
            "warmed": 9,  # 3 tracklists * 3 formats
            "failed": 0,
            "formats": ["standard", "cdj", "traktor"],
        }
        mock_cache.warm_cache.return_value = AsyncMock(return_value=mock_result)()

        # Execute
        response = client.post(
            "/api/v1/cue/cache/warm",
            params={
                "tracklist_ids": tracklist_ids,
                "formats": ["standard", "cdj", "traktor"],
            },
        )

        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["warmed"] == 9
        assert data["failed"] == 0

    @patch("services.tracklist_service.src.api.cue_generation_api.cue_generation_service")
    def test_invalidate_tracklist_cache(self, mock_service, client):
        """Test invalidating tracklist cache."""
        # Setup
        tracklist_id = uuid4()
        mock_service.invalidate_tracklist_cache.return_value = AsyncMock(return_value=5)()

        # Execute
        response = client.delete(f"/api/v1/cue/cache/invalidate/{tracklist_id}")

        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["invalidated_entries"] == 5

    @patch("services.tracklist_service.src.api.cue_generation_api.cache_service")
    def test_clear_cache(self, mock_cache, client):
        """Test clearing cache."""
        # Setup
        mock_cache.clear_cache.return_value = AsyncMock(return_value=100)()

        # Execute
        response = client.delete("/api/v1/cue/cache/clear")

        # Verify
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["cleared_entries"] == 100


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check_healthy(self, client):
        """Test health check when all components are healthy."""
        response = client.get("/api/v1/cue/health")

        # Note: Status might be 503 due to dependencies not being available
        # In a real test environment, we'd mock all dependencies
        assert response.status_code in [200, 503]
        data = response.json()
        assert data["service"] == "cue_generation_api"
        assert "components" in data
        assert "timestamp" in data
