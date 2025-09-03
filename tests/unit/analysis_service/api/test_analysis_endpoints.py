"""Comprehensive unit tests for analysis endpoints."""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from services.analysis_service.src.api.app import app
from services.analysis_service.src.api.endpoints.analysis import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisResult,
)


class TestAnalysisEndpoints:
    """Comprehensive tests for analysis endpoints."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        self.test_recording_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        self.test_file_path = "/path/to/test.wav"

    @pytest.fixture
    def mock_recording(self):
        """Mock recording object."""
        recording = Mock()
        recording.id = self.test_recording_id
        recording.file_path = self.test_file_path
        recording.created_at = datetime.now(UTC)
        recording.updated_at = datetime.now(UTC)
        return recording

    @pytest.fixture
    def mock_analysis_result(self):
        """Mock analysis result object."""
        result = Mock()
        result.analysis_type = "bpm"
        result.result_data = 128.5
        result.confidence_score = 0.95
        result.processing_time_ms = 1500
        result.status = "completed"
        result.created_at = datetime.now(UTC)
        return result


class TestStartAnalysis:
    """Test start_analysis endpoint."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        self.test_recording_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")

    @patch("services.analysis_service.src.api.endpoints.analysis.message_publisher")
    @patch("services.analysis_service.src.api.endpoints.analysis.recording_repo")
    def test_start_analysis_success(self, mock_recording_repo, mock_message_publisher):
        """Test successful analysis start."""
        # Mock recording exists and file exists
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording.file_path = "/path/to/test.wav"
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)
        mock_recording_repo.update_status = AsyncMock(return_value=None)

        with patch("pathlib.Path.exists", return_value=True):
            mock_message_publisher.publish_analysis_request = AsyncMock(return_value="test-correlation-id")

            response = self.client.post(
                "/v1/analysis",
                json={
                    "recording_id": str(self.test_recording_id),
                    "analysis_types": ["bpm", "key"],
                    "priority": 5,
                },
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["task_id"] == "test-correlation-id"
            assert data["recording_id"] == str(self.test_recording_id)
            assert data["status"] == "queued"
            assert data["message"] == "Analysis started"
            assert data["analysis_types"] == ["bpm", "key"]
            assert data["correlation_id"] == "test-correlation-id"

            # Verify repository calls
            mock_recording_repo.get_by_id.assert_called_once_with(self.test_recording_id)
            mock_recording_repo.update_status.assert_called_once_with(self.test_recording_id, "processing")

    @patch("services.analysis_service.src.api.endpoints.analysis.recording_repo")
    def test_start_analysis_recording_not_found(self, mock_recording_repo):
        """Test analysis start with recording not found."""
        mock_recording_repo.get_by_id = AsyncMock(return_value=None)

        response = self.client.post(
            "/v1/analysis",
            json={
                "recording_id": str(self.test_recording_id),
                "analysis_types": ["bpm"],
                "priority": 5,
            },
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert f"Recording not found: {self.test_recording_id}" in response.json()["detail"]

    @patch("services.analysis_service.src.api.endpoints.analysis.recording_repo")
    def test_start_analysis_file_not_found(self, mock_recording_repo):
        """Test analysis start with file not found."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording.file_path = "/nonexistent/file.wav"
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        with patch("pathlib.Path.exists", return_value=False):
            response = self.client.post(
                "/v1/analysis",
                json={
                    "recording_id": str(self.test_recording_id),
                    "analysis_types": ["bpm"],
                    "priority": 5,
                },
            )

            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert "Audio file not found" in response.json()["detail"]

    @patch("services.analysis_service.src.api.endpoints.analysis.message_publisher")
    @patch("services.analysis_service.src.api.endpoints.analysis.recording_repo")
    def test_start_analysis_default_values(self, mock_recording_repo, mock_message_publisher):
        """Test analysis start with default values."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording.file_path = "/path/to/test.wav"
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)
        mock_recording_repo.update_status = AsyncMock(return_value=None)

        with patch("pathlib.Path.exists", return_value=True):
            mock_message_publisher.publish_analysis_request = AsyncMock(return_value="test-correlation-id")

            # Test with minimal request (using defaults)
            response = self.client.post(
                "/v1/analysis",
                json={"recording_id": str(self.test_recording_id)},
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["analysis_types"] == ["bpm", "key", "mood", "energy"]  # default values

            # Verify message publisher was called with defaults
            mock_message_publisher.publish_analysis_request.assert_called_once_with(
                recording_id=self.test_recording_id,
                file_path="/path/to/test.wav",
                analysis_types=["bpm", "key", "mood", "energy"],
                priority=5,
            )

    @patch("services.analysis_service.src.api.endpoints.analysis.message_publisher")
    @patch("services.analysis_service.src.api.endpoints.analysis.recording_repo")
    def test_start_analysis_custom_priority(self, mock_recording_repo, mock_message_publisher):
        """Test analysis start with custom priority."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording.file_path = "/path/to/test.wav"
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)
        mock_recording_repo.update_status = AsyncMock(return_value=None)

        with patch("pathlib.Path.exists", return_value=True):
            mock_message_publisher.publish_analysis_request = AsyncMock(return_value="test-correlation-id")

            response = self.client.post(
                "/v1/analysis",
                json={
                    "recording_id": str(self.test_recording_id),
                    "analysis_types": ["mood"],
                    "priority": 10,
                },
            )

            assert response.status_code == status.HTTP_200_OK

            # Verify priority was passed correctly
            mock_message_publisher.publish_analysis_request.assert_called_once_with(
                recording_id=self.test_recording_id,
                file_path="/path/to/test.wav",
                analysis_types=["mood"],
                priority=10,
            )

    def test_start_analysis_invalid_uuid(self):
        """Test analysis start with invalid UUID."""
        response = self.client.post(
            "/v1/analysis",
            json={
                "recording_id": "invalid-uuid",
                "analysis_types": ["bpm"],
                "priority": 5,
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("services.analysis_service.src.api.endpoints.analysis.message_publisher")
    @patch("services.analysis_service.src.api.endpoints.analysis.recording_repo")
    def test_start_analysis_empty_analysis_types(self, mock_recording_repo, mock_message_publisher):
        """Test analysis start with empty analysis types."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording.file_path = "/path/to/test.wav"
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)
        mock_recording_repo.update_status = AsyncMock(return_value=None)

        with patch("pathlib.Path.exists", return_value=True):
            mock_message_publisher.publish_analysis_request = AsyncMock(return_value="test-correlation-id")

            response = self.client.post(
                "/v1/analysis",
                json={
                    "recording_id": str(self.test_recording_id),
                    "analysis_types": [],
                    "priority": 5,
                },
            )

            # Should succeed with empty list
            assert response.status_code == status.HTTP_200_OK


class TestGetAnalysisStatus:
    """Test get_analysis_status endpoint."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        self.test_recording_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")

    @patch("services.analysis_service.src.api.endpoints.analysis.analysis_repo")
    @patch("services.analysis_service.src.api.endpoints.analysis.recording_repo")
    def test_get_analysis_status_completed(self, mock_recording_repo, mock_analysis_repo):
        """Test getting analysis status for completed analysis."""
        # Mock recording exists
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording.created_at = datetime.now(UTC)
        mock_recording.updated_at = datetime.now(UTC)
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        # Mock completed analysis results
        mock_result = Mock()
        mock_result.analysis_type = "bpm"
        mock_result.result_data = 128.5
        mock_result.confidence_score = 0.95
        mock_result.processing_time_ms = 1500
        mock_result.status = "completed"
        mock_result.created_at = datetime.now(UTC)
        mock_analysis_repo.get_by_recording_id = AsyncMock(return_value=[mock_result])

        response = self.client.get(f"/v1/analysis/{self.test_recording_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["recording_id"] == str(self.test_recording_id)
        assert data["status"] == "completed"
        assert data["progress"] == 1.0
        assert len(data["results"]) == 1
        assert data["results"][0]["type"] == "bpm"
        assert data["results"][0]["value"] == 128.5
        assert data["results"][0]["confidence"] == 0.95

    @patch("services.analysis_service.src.api.endpoints.analysis.analysis_repo")
    @patch("services.analysis_service.src.api.endpoints.analysis.recording_repo")
    def test_get_analysis_status_processing(self, mock_recording_repo, mock_analysis_repo):
        """Test getting analysis status for processing analysis."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording.created_at = datetime.now(UTC)
        mock_recording.updated_at = datetime.now(UTC)
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        # Mock mixed status results (some completed, some processing)
        mock_completed = Mock()
        mock_completed.analysis_type = "bpm"
        mock_completed.result_data = 128.5
        mock_completed.confidence_score = 0.95
        mock_completed.processing_time_ms = 1500
        mock_completed.status = "completed"
        mock_completed.created_at = datetime.now(UTC)

        mock_processing = Mock()
        mock_processing.analysis_type = "key"
        mock_processing.result_data = None
        mock_processing.confidence_score = None
        mock_processing.processing_time_ms = None
        mock_processing.status = "processing"
        mock_processing.created_at = datetime.now(UTC)

        mock_analysis_repo.get_by_recording_id = AsyncMock(return_value=[mock_completed, mock_processing])

        response = self.client.get(f"/v1/analysis/{self.test_recording_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "processing"
        assert data["progress"] == 0.5  # 1 completed out of 2 total

    @patch("services.analysis_service.src.api.endpoints.analysis.analysis_repo")
    @patch("services.analysis_service.src.api.endpoints.analysis.recording_repo")
    def test_get_analysis_status_failed(self, mock_recording_repo, mock_analysis_repo):
        """Test getting analysis status for failed analysis."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording.created_at = datetime.now(UTC)
        mock_recording.updated_at = datetime.now(UTC)
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        # Mock failed result
        mock_result = Mock()
        mock_result.analysis_type = "bpm"
        mock_result.result_data = None
        mock_result.confidence_score = None
        mock_result.processing_time_ms = 2000
        mock_result.status = "failed"
        mock_result.created_at = datetime.now(UTC)
        mock_analysis_repo.get_by_recording_id = AsyncMock(return_value=[mock_result])

        response = self.client.get(f"/v1/analysis/{self.test_recording_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "failed"
        assert data["progress"] == 1.0

    @patch("services.analysis_service.src.api.endpoints.analysis.analysis_repo")
    @patch("services.analysis_service.src.api.endpoints.analysis.recording_repo")
    def test_get_analysis_status_pending(self, mock_recording_repo, mock_analysis_repo):
        """Test getting analysis status for pending analysis."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording.created_at = datetime.now(UTC)
        mock_recording.updated_at = datetime.now(UTC)
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        # No analysis results yet
        mock_analysis_repo.get_by_recording_id = AsyncMock(return_value=[])

        response = self.client.get(f"/v1/analysis/{self.test_recording_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "pending"
        assert data["progress"] == 0.0
        assert len(data["results"]) == 0

    @patch("services.analysis_service.src.api.endpoints.analysis.recording_repo")
    def test_get_analysis_status_recording_not_found(self, mock_recording_repo):
        """Test getting analysis status for non-existent recording."""
        mock_recording_repo.get_by_id = AsyncMock(return_value=None)

        response = self.client.get(f"/v1/analysis/{self.test_recording_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert f"Recording not found: {self.test_recording_id}" in response.json()["detail"]

    def test_get_analysis_status_invalid_uuid(self):
        """Test getting analysis status with invalid UUID."""
        response = self.client.get("/v1/analysis/invalid-uuid")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestSpecificAnalysisEndpoints:
    """Test specific analysis type endpoints (BPM, key, mood)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        self.test_recording_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")

    @patch("services.analysis_service.src.api.endpoints.analysis.analysis_repo")
    @patch("services.analysis_service.src.api.endpoints.analysis.recording_repo")
    def test_get_bpm_analysis_success(self, mock_recording_repo, mock_analysis_repo):
        """Test successful BPM analysis retrieval."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        mock_result = Mock()
        mock_result.result_data = 128.5
        mock_result.confidence_score = 0.95
        mock_result.status = "completed"
        mock_result.created_at = datetime.now(UTC)
        mock_result.processing_time_ms = 1500
        mock_analysis_repo.get_by_recording_and_type = AsyncMock(return_value=mock_result)

        response = self.client.get(f"/v1/analysis/{self.test_recording_id}/bpm")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["recording_id"] == str(self.test_recording_id)
        assert data["analysis_type"] == "bpm"
        assert data["result"] == 128.5
        assert data["confidence"] == 0.95
        assert data["status"] == "completed"

        mock_analysis_repo.get_by_recording_and_type.assert_called_once_with(self.test_recording_id, "bpm")

    @patch("services.analysis_service.src.api.endpoints.analysis.analysis_repo")
    @patch("services.analysis_service.src.api.endpoints.analysis.recording_repo")
    def test_get_key_analysis_success(self, mock_recording_repo, mock_analysis_repo):
        """Test successful key analysis retrieval."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        mock_result = Mock()
        mock_result.result_data = {"key": "C", "mode": "major"}
        mock_result.confidence_score = 0.88
        mock_result.status = "completed"
        mock_result.created_at = datetime.now(UTC)
        mock_result.processing_time_ms = 2000
        mock_analysis_repo.get_by_recording_and_type = AsyncMock(return_value=mock_result)

        response = self.client.get(f"/v1/analysis/{self.test_recording_id}/key")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["analysis_type"] == "key"
        assert data["result"]["key"] == "C"
        assert data["result"]["mode"] == "major"

        mock_analysis_repo.get_by_recording_and_type.assert_called_once_with(self.test_recording_id, "key")

    @patch("services.analysis_service.src.api.endpoints.analysis.analysis_repo")
    @patch("services.analysis_service.src.api.endpoints.analysis.recording_repo")
    def test_get_mood_analysis_success(self, mock_recording_repo, mock_analysis_repo):
        """Test successful mood analysis retrieval."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        mock_result = Mock()
        mock_result.result_data = {"mood": "energetic", "valence": 0.8}
        mock_result.confidence_score = 0.75
        mock_result.status = "completed"
        mock_result.created_at = datetime.now(UTC)
        mock_result.processing_time_ms = 3000
        mock_analysis_repo.get_by_recording_and_type = AsyncMock(return_value=mock_result)

        response = self.client.get(f"/v1/analysis/{self.test_recording_id}/mood")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["analysis_type"] == "mood"
        assert data["result"]["mood"] == "energetic"

        mock_analysis_repo.get_by_recording_and_type.assert_called_once_with(self.test_recording_id, "mood")

    @patch("services.analysis_service.src.api.endpoints.analysis.analysis_repo")
    @patch("services.analysis_service.src.api.endpoints.analysis.recording_repo")
    def test_get_analysis_not_found(self, mock_recording_repo, mock_analysis_repo):
        """Test analysis not found for specific type."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        mock_analysis_repo.get_by_recording_and_type = AsyncMock(return_value=None)

        response = self.client.get(f"/v1/analysis/{self.test_recording_id}/bpm")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "BPM analysis not found" in response.json()["detail"]

    @patch("services.analysis_service.src.api.endpoints.analysis.analysis_repo")
    @patch("services.analysis_service.src.api.endpoints.analysis.recording_repo")
    def test_get_analysis_not_completed(self, mock_recording_repo, mock_analysis_repo):
        """Test analysis not completed yet."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        mock_result = Mock()
        mock_result.status = "processing"  # Not completed yet
        mock_analysis_repo.get_by_recording_and_type = AsyncMock(return_value=mock_result)

        response = self.client.get(f"/v1/analysis/{self.test_recording_id}/bpm")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "BPM analysis not found" in response.json()["detail"]


class TestGenerationEndpoints:
    """Test waveform and spectrogram generation endpoints."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        self.test_recording_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")

    @patch("services.analysis_service.src.api.endpoints.analysis.message_publisher")
    @patch("services.analysis_service.src.api.endpoints.analysis.recording_repo")
    def test_generate_waveform_success(self, mock_recording_repo, mock_message_publisher):
        """Test successful waveform generation."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording.file_path = "/path/to/test.wav"
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        with patch("pathlib.Path.exists", return_value=True):
            mock_message_publisher.publish_analysis_request = AsyncMock(return_value="waveform-correlation-id")

            response = self.client.post(
                f"/v1/analysis/{self.test_recording_id}/waveform",
                params={"width": 1920, "height": 256, "color": "#ff0000"},
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["task_id"] == "waveform-correlation-id"
            assert data["recording_id"] == str(self.test_recording_id)
            assert data["status"] == "generating"
            assert data["parameters"]["width"] == 1920
            assert data["parameters"]["height"] == 256
            assert data["parameters"]["color"] == "#ff0000"

            # Verify message publisher call
            mock_message_publisher.publish_analysis_request.assert_called_once_with(
                recording_id=self.test_recording_id,
                file_path="/path/to/test.wav",
                analysis_types=["waveform"],
                priority=6,
                metadata={"width": 1920, "height": 256, "color": "#ff0000"},
            )

    @patch("services.analysis_service.src.api.endpoints.analysis.message_publisher")
    @patch("services.analysis_service.src.api.endpoints.analysis.recording_repo")
    def test_generate_waveform_default_params(self, mock_recording_repo, mock_message_publisher):
        """Test waveform generation with default parameters."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording.file_path = "/path/to/test.wav"
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        with patch("pathlib.Path.exists", return_value=True):
            mock_message_publisher.publish_analysis_request = AsyncMock(return_value="waveform-correlation-id")

            response = self.client.post(f"/v1/analysis/{self.test_recording_id}/waveform")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["parameters"]["width"] == 1920  # default
            assert data["parameters"]["height"] == 256  # default
            assert data["parameters"]["color"] == "#00ff00"  # default

    @patch("services.analysis_service.src.api.endpoints.analysis.message_publisher")
    @patch("services.analysis_service.src.api.endpoints.analysis.recording_repo")
    def test_generate_spectrogram_success(self, mock_recording_repo, mock_message_publisher):
        """Test successful spectrogram generation."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording.file_path = "/path/to/test.wav"
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        with patch("pathlib.Path.exists", return_value=True):
            mock_message_publisher.publish_analysis_request = AsyncMock(return_value="spectrogram-correlation-id")

            response = self.client.post(
                f"/v1/analysis/{self.test_recording_id}/spectrogram",
                params={"fft_size": 4096, "hop_size": 1024, "color_map": "plasma"},
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["task_id"] == "spectrogram-correlation-id"
            assert data["status"] == "generating"
            assert data["parameters"]["fft_size"] == 4096
            assert data["parameters"]["hop_size"] == 1024
            assert data["parameters"]["color_map"] == "plasma"

            # Verify message publisher call
            mock_message_publisher.publish_analysis_request.assert_called_once_with(
                recording_id=self.test_recording_id,
                file_path="/path/to/test.wav",
                analysis_types=["spectrogram"],
                priority=5,
                metadata={
                    "fft_size": 4096,
                    "hop_size": 1024,
                    "color_map": "plasma",
                },
            )

    @patch("services.analysis_service.src.api.endpoints.analysis.recording_repo")
    def test_generate_waveform_recording_not_found(self, mock_recording_repo):
        """Test waveform generation with recording not found."""
        mock_recording_repo.get_by_id = AsyncMock(return_value=None)

        response = self.client.post(f"/v1/analysis/{self.test_recording_id}/waveform")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert f"Recording not found: {self.test_recording_id}" in response.json()["detail"]

    @patch("services.analysis_service.src.api.endpoints.analysis.recording_repo")
    def test_generate_waveform_file_not_found(self, mock_recording_repo):
        """Test waveform generation with file not found."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording.file_path = "/nonexistent/file.wav"
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        with patch("pathlib.Path.exists", return_value=False):
            response = self.client.post(f"/v1/analysis/{self.test_recording_id}/waveform")

            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert "Audio file not found" in response.json()["detail"]


class TestAnalysisModels:
    """Test Pydantic models used in analysis endpoints."""

    def test_analysis_request_model(self):
        """Test AnalysisRequest model validation."""
        # Valid request
        request = AnalysisRequest(
            recording_id=uuid.uuid4(),
            analysis_types=["bpm", "key"],
            priority=7,
        )
        assert request.priority == 7
        assert len(request.analysis_types) == 2

        # Test defaults
        request_minimal = AnalysisRequest(recording_id=uuid.uuid4())
        assert request_minimal.analysis_types == ["bpm", "key", "mood", "energy"]
        assert request_minimal.priority == 5

    def test_analysis_result_model(self):
        """Test AnalysisResult model."""
        result = AnalysisResult(
            type="bpm",
            value=128.5,
            confidence=0.95,
            metadata={"processing_time_ms": 1500},
        )
        assert result.type == "bpm"
        assert result.value == 128.5
        assert result.confidence == 0.95
        assert result.metadata["processing_time_ms"] == 1500

        # Test defaults
        result_minimal = AnalysisResult(type="key", value="C major", confidence=0.8)
        assert result_minimal.metadata == {}

    def test_analysis_response_model(self):
        """Test AnalysisResponse model."""
        recording_id = uuid.uuid4()
        results = [
            AnalysisResult(type="bpm", value=128.5, confidence=0.95),
            AnalysisResult(type="key", value="C major", confidence=0.8),
        ]

        response = AnalysisResponse(
            recording_id=recording_id,
            status="completed",
            progress=1.0,
            results=results,
            started_at="2024-01-01T00:00:00Z",
            completed_at="2024-01-01T00:01:00Z",
        )

        assert response.recording_id == recording_id
        assert response.status == "completed"
        assert response.progress == 1.0
        assert len(response.results) == 2
        assert response.started_at == "2024-01-01T00:00:00Z"
        assert response.completed_at == "2024-01-01T00:01:00Z"


class TestAnalysisEndpointErrors:
    """Test error handling in analysis endpoints."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        self.test_recording_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")

    @patch("services.analysis_service.src.api.endpoints.analysis.recording_repo")
    def test_database_error_handling(self, mock_recording_repo):
        """Test database error handling."""
        mock_recording_repo.get_by_id = AsyncMock(side_effect=Exception("Database connection failed"))

        response = self.client.get(f"/v1/analysis/{self.test_recording_id}")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    @patch("services.analysis_service.src.api.endpoints.analysis.message_publisher")
    @patch("services.analysis_service.src.api.endpoints.analysis.recording_repo")
    def test_message_queue_error_handling(self, mock_recording_repo, mock_message_publisher):
        """Test message queue error handling."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording.file_path = "/path/to/test.wav"
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        with patch("pathlib.Path.exists", return_value=True):
            mock_message_publisher.publish_analysis_request = AsyncMock(
                side_effect=Exception("Message queue unavailable")
            )

            response = self.client.post(
                "/v1/analysis",
                json={"recording_id": str(self.test_recording_id), "analysis_types": ["bpm"]},
            )

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_invalid_json_handling(self):
        """Test invalid JSON handling."""
        response = self.client.post(
            "/v1/analysis",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_missing_required_fields(self):
        """Test missing required fields."""
        response = self.client.post(
            "/v1/analysis",
            json={},  # Missing recording_id
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
