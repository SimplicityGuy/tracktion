"""Comprehensive unit tests for tracklist endpoints."""

import uuid
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from services.analysis_service.src.api.app import app
from services.analysis_service.src.api.endpoints.tracklist import (
    CueSheetRequest,
    TrackInfo,
    TracklistResponse,
)


class TestTracklistEndpoints:
    """Comprehensive tests for tracklist endpoints."""

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
        return recording

    @pytest.fixture
    def mock_tracklist(self):
        """Mock tracklist object."""
        tracklist = Mock()
        tracklist.id = uuid.uuid4()
        tracklist.recording_id = self.test_recording_id
        tracklist.source = "manual"
        tracklist.cue_file_path = None
        tracklist.tracks = [
            {
                "index": 1,
                "title": "Track 1",
                "artist": "Artist 1",
                "start_time": 0.0,
                "end_time": 180.0,
                "duration": 180.0,
                "file_path": None,
            },
            {
                "index": 2,
                "title": "Track 2",
                "artist": "Artist 2",
                "start_time": 180.0,
                "end_time": 360.0,
                "duration": 180.0,
                "file_path": None,
            },
        ]
        return tracklist

    @pytest.fixture
    def sample_tracks(self):
        """Sample track data."""
        return [
            TrackInfo(
                index=1,
                title="Track 1",
                artist="Artist 1",
                start_time=0.0,
                end_time=180.0,
                duration=180.0,
                file_path=None,
            ),
            TrackInfo(
                index=2,
                title="Track 2",
                artist="Artist 2",
                start_time=180.0,
                end_time=360.0,
                duration=180.0,
                file_path=None,
            ),
        ]


class TestGetTracklist:
    """Test get_tracklist endpoint."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        self.test_recording_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")

    @patch("services.analysis_service.src.api.endpoints.tracklist.tracklist_repo")
    @patch("services.analysis_service.src.api.endpoints.tracklist.recording_repo")
    def test_get_tracklist_success(self, mock_recording_repo, mock_tracklist_repo):
        """Test successful tracklist retrieval."""
        # Mock recording exists
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        # Mock tracklist exists
        mock_tracklist = Mock()
        mock_tracklist.id = uuid.uuid4()
        mock_tracklist.recording_id = self.test_recording_id
        mock_tracklist.source = "manual"
        mock_tracklist.cue_file_path = None
        mock_tracklist.tracks = [
            {
                "index": 1,
                "title": "Track 1",
                "artist": "Artist 1",
                "start_time": 0.0,
                "end_time": 180.0,
                "duration": 180.0,
                "file_path": None,
            },
            {
                "index": 2,
                "title": "Track 2",
                "artist": "Artist 2",
                "start_time": 180.0,
                "end_time": 360.0,
                "duration": 180.0,
                "file_path": None,
            },
        ]
        mock_tracklist_repo.get_by_recording_id = AsyncMock(return_value=mock_tracklist)

        response = self.client.get(f"/v1/tracklist/{self.test_recording_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["recording_id"] == str(self.test_recording_id)
        assert data["format"] == "manual"
        assert data["total_tracks"] == 2
        assert data["total_duration"] == 360.0
        assert len(data["tracks"]) == 2

        # Verify track data
        track1 = data["tracks"][0]
        assert track1["index"] == 1
        assert track1["title"] == "Track 1"
        assert track1["artist"] == "Artist 1"
        assert track1["start_time"] == 0.0
        assert track1["end_time"] == 180.0
        assert track1["duration"] == 180.0

    @patch("services.analysis_service.src.api.endpoints.tracklist.tracklist_repo")
    @patch("services.analysis_service.src.api.endpoints.tracklist.recording_repo")
    def test_get_tracklist_cue_format(self, mock_recording_repo, mock_tracklist_repo):
        """Test tracklist retrieval with CUE format."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        # Mock CUE tracklist
        mock_tracklist = Mock()
        mock_tracklist.source = "cue_parsing"
        mock_tracklist.cue_file_path = "/path/to/test.cue"
        mock_tracklist.tracks = [
            {
                "index": 1,
                "title": "CUE Track 1",
                "artist": "CUE Artist",
                "start_time": 0.0,
                "end_time": 240.0,
                "duration": 240.0,
                "file_path": None,
            }
        ]
        mock_tracklist_repo.get_by_recording_id = AsyncMock(return_value=mock_tracklist)

        response = self.client.get(f"/v1/tracklist/{self.test_recording_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["format"] == "cue"  # Should be "cue" when cue_file_path exists

    @patch("services.analysis_service.src.api.endpoints.tracklist.tracklist_repo")
    @patch("services.analysis_service.src.api.endpoints.tracklist.recording_repo")
    def test_get_tracklist_empty_tracks(self, mock_recording_repo, mock_tracklist_repo):
        """Test tracklist retrieval with empty tracks."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        # Mock tracklist with no tracks
        mock_tracklist = Mock()
        mock_tracklist.source = "manual"
        mock_tracklist.cue_file_path = None
        mock_tracklist.tracks = []
        mock_tracklist_repo.get_by_recording_id = AsyncMock(return_value=mock_tracklist)

        response = self.client.get(f"/v1/tracklist/{self.test_recording_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_tracks"] == 0
        assert data["total_duration"] == 0.0
        assert len(data["tracks"]) == 0

    @patch("services.analysis_service.src.api.endpoints.tracklist.tracklist_repo")
    @patch("services.analysis_service.src.api.endpoints.tracklist.recording_repo")
    def test_get_tracklist_with_default_values(self, mock_recording_repo, mock_tracklist_repo):
        """Test tracklist retrieval with missing track data using defaults."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        # Mock tracklist with incomplete track data
        mock_tracklist = Mock()
        mock_tracklist.source = "detection"
        mock_tracklist.cue_file_path = None
        mock_tracklist.tracks = [
            {
                # Missing most fields - should use defaults
                "title": "Detected Track",
            }
        ]
        mock_tracklist_repo.get_by_recording_id = AsyncMock(return_value=mock_tracklist)

        response = self.client.get(f"/v1/tracklist/{self.test_recording_id}")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        track = data["tracks"][0]
        assert track["index"] == 1  # Default from enumerate
        assert track["title"] == "Detected Track"
        assert track["artist"] is None  # Default
        assert track["start_time"] == 0.0  # Default
        assert track["end_time"] == 0.0  # Default
        assert track["duration"] == 0.0  # Default

    @patch("services.analysis_service.src.api.endpoints.tracklist.recording_repo")
    def test_get_tracklist_recording_not_found(self, mock_recording_repo):
        """Test tracklist retrieval with recording not found."""
        mock_recording_repo.get_by_id = AsyncMock(return_value=None)

        response = self.client.get(f"/v1/tracklist/{self.test_recording_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert f"Recording not found: {self.test_recording_id}" in response.json()["detail"]

    @patch("services.analysis_service.src.api.endpoints.tracklist.tracklist_repo")
    @patch("services.analysis_service.src.api.endpoints.tracklist.recording_repo")
    def test_get_tracklist_not_found(self, mock_recording_repo, mock_tracklist_repo):
        """Test tracklist retrieval with tracklist not found."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        mock_tracklist_repo.get_by_recording_id = AsyncMock(return_value=None)

        response = self.client.get(f"/v1/tracklist/{self.test_recording_id}")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert f"Tracklist not found for recording: {self.test_recording_id}" in response.json()["detail"]

    def test_get_tracklist_invalid_uuid(self):
        """Test tracklist retrieval with invalid UUID."""
        response = self.client.get("/v1/tracklist/invalid-uuid")

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestDetectTracks:
    """Test detect_tracks endpoint."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        self.test_recording_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")

    @patch("services.analysis_service.src.api.endpoints.tracklist.message_publisher")
    @patch("services.analysis_service.src.api.endpoints.tracklist.recording_repo")
    def test_detect_tracks_success(self, mock_recording_repo, mock_message_publisher):
        """Test successful track detection."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording.file_path = "/path/to/test.wav"
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        with patch("pathlib.Path.exists", return_value=True):
            mock_message_publisher.publish_tracklist_generation = AsyncMock(return_value="detect-correlation-id")

            response = self.client.post(
                f"/v1/tracklist/{self.test_recording_id}/detect",
                params={"min_duration": 45.0, "sensitivity": 0.7},
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["id"] == str(self.test_recording_id)
            assert data["status"] == "detecting"
            assert data["message"] == "Track detection started"
            assert data["parameters"]["min_duration"] == 45.0
            assert data["parameters"]["sensitivity"] == 0.7
            assert data["correlation_id"] == "detect-correlation-id"

            # Verify message publisher call
            mock_message_publisher.publish_tracklist_generation.assert_called_once_with(
                recording_id=self.test_recording_id,
                source_hint="silence_detection",
                priority=6,
            )

    @patch("services.analysis_service.src.api.endpoints.tracklist.message_publisher")
    @patch("services.analysis_service.src.api.endpoints.tracklist.recording_repo")
    def test_detect_tracks_default_params(self, mock_recording_repo, mock_message_publisher):
        """Test track detection with default parameters."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording.file_path = "/path/to/test.wav"
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        with patch("pathlib.Path.exists", return_value=True):
            mock_message_publisher.publish_tracklist_generation = AsyncMock(return_value="detect-correlation-id")

            response = self.client.post(f"/v1/tracklist/{self.test_recording_id}/detect")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["parameters"]["min_duration"] == 30.0  # default
            assert data["parameters"]["sensitivity"] == 0.5  # default

    @patch("services.analysis_service.src.api.endpoints.tracklist.recording_repo")
    def test_detect_tracks_recording_not_found(self, mock_recording_repo):
        """Test track detection with recording not found."""
        mock_recording_repo.get_by_id = AsyncMock(return_value=None)

        response = self.client.post(f"/v1/tracklist/{self.test_recording_id}/detect")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert f"Recording not found: {self.test_recording_id}" in response.json()["detail"]

    @patch("services.analysis_service.src.api.endpoints.tracklist.recording_repo")
    def test_detect_tracks_file_not_found(self, mock_recording_repo):
        """Test track detection with file not found."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording.file_path = "/nonexistent/file.wav"
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        with patch("pathlib.Path.exists", return_value=False):
            response = self.client.post(f"/v1/tracklist/{self.test_recording_id}/detect")

            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert "Audio file not found" in response.json()["detail"]

    def test_detect_tracks_invalid_sensitivity(self):
        """Test track detection with invalid sensitivity parameter."""
        response = self.client.post(
            f"/v1/tracklist/{self.test_recording_id}/detect",
            params={"sensitivity": 1.5},  # Outside valid range
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestSplitTracks:
    """Test split_tracks endpoint."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        self.test_recording_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")

    @patch("services.analysis_service.src.api.endpoints.tracklist.message_publisher")
    @patch("services.analysis_service.src.api.endpoints.tracklist.tracklist_repo")
    @patch("services.analysis_service.src.api.endpoints.tracklist.recording_repo")
    def test_split_tracks_success(self, mock_recording_repo, mock_tracklist_repo, mock_message_publisher):
        """Test successful track splitting."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording.file_path = "/path/to/test.wav"
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        with patch("pathlib.Path.exists", return_value=True):
            # Mock tracklist exists
            mock_tracklist = Mock()
            mock_tracklist.id = uuid.uuid4()
            mock_tracklist.tracks = [{"title": "Track 1"}, {"title": "Track 2"}]
            mock_tracklist_repo.get_by_recording_id = AsyncMock(return_value=mock_tracklist)

            mock_message_publisher.publish_analysis_request = AsyncMock(return_value="split-correlation-id")

            response = self.client.post(
                f"/v1/tracklist/{self.test_recording_id}/split",
                params={"output_format": "mp3"},
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["id"] == str(self.test_recording_id)
            assert data["status"] == "splitting"
            assert data["message"] == "Track splitting started"
            assert data["output_format"] == "mp3"
            assert data["correlation_id"] == "split-correlation-id"

            # Verify message publisher call
            mock_message_publisher.publish_analysis_request.assert_called_once_with(
                recording_id=self.test_recording_id,
                file_path="/path/to/test.wav",
                analysis_types=["track_splitting"],
                priority=7,
                metadata={"output_format": "mp3", "tracklist_id": str(mock_tracklist.id)},
            )

    @patch("services.analysis_service.src.api.endpoints.tracklist.message_publisher")
    @patch("services.analysis_service.src.api.endpoints.tracklist.tracklist_repo")
    @patch("services.analysis_service.src.api.endpoints.tracklist.recording_repo")
    def test_split_tracks_default_format(self, mock_recording_repo, mock_tracklist_repo, mock_message_publisher):
        """Test track splitting with default format."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording.file_path = "/path/to/test.wav"
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        with patch("pathlib.Path.exists", return_value=True):
            mock_tracklist = Mock()
            mock_tracklist.id = uuid.uuid4()
            mock_tracklist.tracks = [{"title": "Track 1"}]
            mock_tracklist_repo.get_by_recording_id = AsyncMock(return_value=mock_tracklist)

            mock_message_publisher.publish_analysis_request = AsyncMock(return_value="split-correlation-id")

            response = self.client.post(f"/v1/tracklist/{self.test_recording_id}/split")

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["output_format"] == "flac"  # default

    @patch("services.analysis_service.src.api.endpoints.tracklist.tracklist_repo")
    @patch("services.analysis_service.src.api.endpoints.tracklist.recording_repo")
    def test_split_tracks_no_tracklist(self, mock_recording_repo, mock_tracklist_repo):
        """Test track splitting with no tracklist."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording.file_path = "/path/to/test.wav"
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        with patch("pathlib.Path.exists", return_value=True):
            mock_tracklist_repo.get_by_recording_id = AsyncMock(return_value=None)

            response = self.client.post(f"/v1/tracklist/{self.test_recording_id}/split")

            assert response.status_code == status.HTTP_400_BAD_REQUEST
            assert "No tracklist found - run track detection first" in response.json()["detail"]

    @patch("services.analysis_service.src.api.endpoints.tracklist.tracklist_repo")
    @patch("services.analysis_service.src.api.endpoints.tracklist.recording_repo")
    def test_split_tracks_empty_tracklist(self, mock_recording_repo, mock_tracklist_repo):
        """Test track splitting with empty tracklist."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording.file_path = "/path/to/test.wav"
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        with patch("pathlib.Path.exists", return_value=True):
            # Mock tracklist with no tracks
            mock_tracklist = Mock()
            mock_tracklist.tracks = []
            mock_tracklist_repo.get_by_recording_id = AsyncMock(return_value=mock_tracklist)

            response = self.client.post(f"/v1/tracklist/{self.test_recording_id}/split")

            assert response.status_code == status.HTTP_400_BAD_REQUEST
            assert "No tracklist found - run track detection first" in response.json()["detail"]

    @patch("services.analysis_service.src.api.endpoints.tracklist.recording_repo")
    def test_split_tracks_recording_not_found(self, mock_recording_repo):
        """Test track splitting with recording not found."""
        mock_recording_repo.get_by_id = AsyncMock(return_value=None)

        response = self.client.post(f"/v1/tracklist/{self.test_recording_id}/split")

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert f"Recording not found: {self.test_recording_id}" in response.json()["detail"]


class TestParseCueSheet:
    """Test parse_cue_sheet endpoint."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        self.test_file_path = "/path/to/test.wav"
        self.sample_cue_content = """FILE "test.wav" WAVE
  TRACK 01 AUDIO
    TITLE "Track 1"
    PERFORMER "Artist 1"
    INDEX 01 00:00:00
  TRACK 02 AUDIO
    TITLE "Track 2"
    PERFORMER "Artist 2"
    INDEX 01 03:00:00"""

    @patch("services.analysis_service.src.api.endpoints.tracklist.message_publisher")
    @patch("services.analysis_service.src.api.endpoints.tracklist.recording_repo")
    def test_parse_cue_sheet_success_existing_recording(self, mock_recording_repo, mock_message_publisher):
        """Test CUE sheet parsing with existing recording."""
        # Mock existing recording
        mock_recording = Mock()
        mock_recording.id = uuid.uuid4()
        mock_recording_repo.get_by_file_path = AsyncMock(return_value=mock_recording)

        with patch("pathlib.Path.exists", return_value=True):
            mock_message_publisher.publish_analysis_request = AsyncMock(return_value="cue-correlation-id")

            response = self.client.post(
                "/v1/tracklist/parse-cue",
                json={
                    "cue_content": self.sample_cue_content,
                    "audio_file_path": self.test_file_path,
                    "validate_cue": True,
                },
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["status"] == "parsing"
            assert data["format"] == "cue"
            assert data["recording_id"] == str(mock_recording.id)
            assert data["audio_file"] == self.test_file_path
            assert data["message"] == "CUE sheet parsing started"
            assert data["correlation_id"] == "cue-correlation-id"

            # Should not create new recording
            mock_recording_repo.create.assert_not_called()

    @patch("services.analysis_service.src.api.endpoints.tracklist.message_publisher")
    @patch("services.analysis_service.src.api.endpoints.tracklist.recording_repo")
    def test_parse_cue_sheet_create_new_recording(self, mock_recording_repo, mock_message_publisher):
        """Test CUE sheet parsing creating new recording."""
        # Mock no existing recording
        mock_recording_repo.get_by_file_path = AsyncMock(return_value=None)

        # Mock new recording creation
        new_recording = Mock()
        new_recording.id = uuid.uuid4()
        mock_recording_repo.create = AsyncMock(return_value=new_recording)

        with patch("pathlib.Path.exists", return_value=True), patch("pathlib.Path.stat") as mock_stat:
            mock_stat_result = Mock()
            mock_stat_result.st_size = 1024000
            mock_stat.return_value = mock_stat_result

            mock_message_publisher.publish_analysis_request = AsyncMock(return_value="cue-correlation-id")

            response = self.client.post(
                "/v1/tracklist/parse-cue",
                json={
                    "cue_content": self.sample_cue_content,
                    "audio_file_path": self.test_file_path,
                    "validate_cue": False,
                },
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["recording_id"] == str(new_recording.id)

            # Verify recording was created
            mock_recording_repo.create.assert_called_once_with(
                file_path=self.test_file_path,
                file_name="test.wav",
                file_size=1024000,
            )

            # Verify message publisher call
            mock_message_publisher.publish_analysis_request.assert_called_once_with(
                recording_id=new_recording.id,
                file_path=self.test_file_path,
                analysis_types=["cue_parsing"],
                priority=7,
                metadata={
                    "cue_content": self.sample_cue_content,
                    "validate_cue": False,
                },
            )

    @patch("services.analysis_service.src.api.endpoints.tracklist.message_publisher")
    @patch("services.analysis_service.src.api.endpoints.tracklist.recording_repo")
    def test_parse_cue_sheet_default_validation(self, mock_recording_repo, mock_message_publisher):
        """Test CUE sheet parsing with default validation setting."""
        mock_recording = Mock()
        mock_recording.id = uuid.uuid4()
        mock_recording_repo.get_by_file_path = AsyncMock(return_value=mock_recording)

        with patch("pathlib.Path.exists", return_value=True):
            mock_message_publisher.publish_analysis_request = AsyncMock(return_value="cue-correlation-id")

            response = self.client.post(
                "/v1/tracklist/parse-cue",
                json={
                    "cue_content": self.sample_cue_content,
                    "audio_file_path": self.test_file_path,
                    # validate_cue not specified - should use default True
                },
            )

            assert response.status_code == status.HTTP_200_OK

            # Verify message publisher called with default validation
            args = mock_message_publisher.publish_analysis_request.call_args
            assert args[1]["metadata"]["validate_cue"] is True  # default

    def test_parse_cue_sheet_file_not_found(self):
        """Test CUE sheet parsing with file not found."""
        with patch("pathlib.Path.exists", return_value=False):
            response = self.client.post(
                "/v1/tracklist/parse-cue",
                json={
                    "cue_content": self.sample_cue_content,
                    "audio_file_path": "/nonexistent/file.wav",
                    "validate_cue": True,
                },
            )

            assert response.status_code == status.HTTP_404_NOT_FOUND
            assert "Audio file not found" in response.json()["detail"]

    def test_parse_cue_sheet_invalid_json(self):
        """Test CUE sheet parsing with invalid JSON."""
        response = self.client.post(
            "/v1/tracklist/parse-cue",
            json={
                "cue_content": self.sample_cue_content,
                # Missing required audio_file_path
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @patch("services.analysis_service.src.api.endpoints.tracklist.message_publisher")
    @patch("services.analysis_service.src.api.endpoints.tracklist.recording_repo")
    def test_parse_cue_sheet_empty_cue_content(self, mock_recording_repo, mock_message_publisher):
        """Test CUE sheet parsing with empty CUE content."""
        mock_recording = Mock()
        mock_recording.id = uuid.uuid4()
        mock_recording_repo.get_by_file_path = AsyncMock(return_value=mock_recording)

        with patch("pathlib.Path.exists", return_value=True):
            mock_message_publisher.publish_analysis_request = AsyncMock(return_value="cue-correlation-id")

            response = self.client.post(
                "/v1/tracklist/parse-cue",
                json={
                    "cue_content": "",
                    "audio_file_path": self.test_file_path,
                    "validate_cue": True,
                },
            )

            # Should still accept empty content (validation happens in processor)
            assert response.status_code == status.HTTP_200_OK


class TestUpdateTracklist:
    """Test update_tracklist endpoint."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        self.test_recording_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")
        self.sample_tracks = [
            {
                "index": 1,
                "title": "Updated Track 1",
                "artist": "Updated Artist 1",
                "start_time": 0.0,
                "end_time": 200.0,
                "duration": 200.0,
                "file_path": None,
            },
            {
                "index": 2,
                "title": "Updated Track 2",
                "artist": "Updated Artist 2",
                "start_time": 200.0,
                "end_time": 400.0,
                "duration": 200.0,
                "file_path": "/path/to/track2.flac",
            },
        ]

    @patch("services.analysis_service.src.api.endpoints.tracklist.tracklist_repo")
    @patch("services.analysis_service.src.api.endpoints.tracklist.recording_repo")
    def test_update_tracklist_existing(self, mock_recording_repo, mock_tracklist_repo):
        """Test updating existing tracklist."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        # Mock existing tracklist
        mock_existing_tracklist = Mock()
        mock_tracklist_repo.get_by_recording_id = AsyncMock(return_value=mock_existing_tracklist)
        mock_tracklist_repo.update_tracks = AsyncMock(return_value=None)

        response = self.client.put(
            f"/v1/tracklist/{self.test_recording_id}/tracks",
            json=self.sample_tracks,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == str(self.test_recording_id)
        assert data["status"] == "updated"
        assert "Tracklist updated with 2 tracks" in data["message"]

        # Verify update was called
        mock_tracklist_repo.update_tracks.assert_called_once()
        args = mock_tracklist_repo.update_tracks.call_args
        assert args[0][0] == self.test_recording_id
        updated_tracks = args[0][1]
        assert len(updated_tracks) == 2
        assert updated_tracks[0]["title"] == "Updated Track 1"

    @patch("services.analysis_service.src.api.endpoints.tracklist.tracklist_repo")
    @patch("services.analysis_service.src.api.endpoints.tracklist.recording_repo")
    def test_update_tracklist_create_new(self, mock_recording_repo, mock_tracklist_repo):
        """Test creating new tracklist when none exists."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        # Mock no existing tracklist
        mock_tracklist_repo.get_by_recording_id = AsyncMock(return_value=None)
        mock_tracklist_repo.create = AsyncMock(return_value=None)

        response = self.client.put(
            f"/v1/tracklist/{self.test_recording_id}/tracks",
            json=self.sample_tracks,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "updated"
        assert "Tracklist updated with 2 tracks" in data["message"]

        # Verify create was called
        mock_tracklist_repo.create.assert_called_once()
        args = mock_tracklist_repo.create.call_args
        assert args[1]["recording_id"] == self.test_recording_id
        assert args[1]["source"] == "manual"
        created_tracks = args[1]["tracks"]
        assert len(created_tracks) == 2
        assert created_tracks[0]["title"] == "Updated Track 1"

    @patch("services.analysis_service.src.api.endpoints.tracklist.recording_repo")
    def test_update_tracklist_recording_not_found(self, mock_recording_repo):
        """Test updating tracklist with recording not found."""
        mock_recording_repo.get_by_id = AsyncMock(return_value=None)

        response = self.client.put(
            f"/v1/tracklist/{self.test_recording_id}/tracks",
            json=self.sample_tracks,
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert f"Recording not found: {self.test_recording_id}" in response.json()["detail"]

    @patch("services.analysis_service.src.api.endpoints.tracklist.recording_repo")
    def test_update_tracklist_empty_tracks(self, mock_recording_repo):
        """Test updating tracklist with empty tracks list."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        response = self.client.put(
            f"/v1/tracklist/{self.test_recording_id}/tracks",
            json=[],  # Empty tracks list
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Track list cannot be empty" in response.json()["detail"]

    def test_update_tracklist_invalid_track_data(self):
        """Test updating tracklist with invalid track data."""
        response = self.client.put(
            f"/v1/tracklist/{self.test_recording_id}/tracks",
            json=[
                {
                    "index": "invalid",  # Should be int
                    "title": "Track 1",
                    "artist": None,
                    "start_time": 0.0,
                    "end_time": 100.0,
                    "duration": 100.0,
                    "file_path": None,
                }
            ],
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_update_tracklist_invalid_uuid(self):
        """Test updating tracklist with invalid UUID."""
        response = self.client.put(
            "/v1/tracklist/invalid-uuid/tracks",
            json=self.sample_tracks,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestTracklistModels:
    """Test Pydantic models used in tracklist endpoints."""

    def test_track_info_model(self):
        """Test TrackInfo model validation."""
        track = TrackInfo(
            index=1,
            title="Test Track",
            artist="Test Artist",
            start_time=0.0,
            end_time=180.0,
            duration=180.0,
            file_path="/path/to/track.flac",
        )

        assert track.index == 1
        assert track.title == "Test Track"
        assert track.artist == "Test Artist"
        assert track.start_time == 0.0
        assert track.end_time == 180.0
        assert track.duration == 180.0
        assert track.file_path == "/path/to/track.flac"

        # Test with None values
        track_minimal = TrackInfo(
            index=1,
            title="Minimal Track",
            artist=None,
            start_time=0.0,
            end_time=100.0,
            duration=100.0,
            file_path=None,
        )
        assert track_minimal.artist is None
        assert track_minimal.file_path is None

    def test_tracklist_response_model(self):
        """Test TracklistResponse model."""
        recording_id = uuid.uuid4()
        tracks = [
            TrackInfo(
                index=1,
                title="Track 1",
                artist="Artist 1",
                start_time=0.0,
                end_time=180.0,
                duration=180.0,
                file_path=None,
            )
        ]

        response = TracklistResponse(
            recording_id=recording_id,
            format="cue",
            total_tracks=1,
            total_duration=180.0,
            tracks=tracks,
        )

        assert response.recording_id == recording_id
        assert response.format == "cue"
        assert response.total_tracks == 1
        assert response.total_duration == 180.0
        assert len(response.tracks) == 1

    def test_cue_sheet_request_model(self):
        """Test CueSheetRequest model."""
        request = CueSheetRequest(
            cue_content='FILE "test.wav" WAVE',
            audio_file_path="/path/to/test.wav",
            validate_cue=False,
        )

        assert request.cue_content == 'FILE "test.wav" WAVE'
        assert request.audio_file_path == "/path/to/test.wav"
        assert request.validate_cue is False

        # Test default validation
        request_default = CueSheetRequest(
            cue_content='FILE "test.wav" WAVE',
            audio_file_path="/path/to/test.wav",
        )
        assert request_default.validate_cue is True  # default


class TestTracklistEndpointErrors:
    """Test error handling in tracklist endpoints."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        self.test_recording_id = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")

    @patch("services.analysis_service.src.api.endpoints.tracklist.recording_repo")
    def test_database_error_handling(self, mock_recording_repo):
        """Test database error handling."""
        mock_recording_repo.get_by_id = AsyncMock(side_effect=Exception("Database connection failed"))

        response = self.client.get(f"/v1/tracklist/{self.test_recording_id}")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    @patch("services.analysis_service.src.api.endpoints.tracklist.message_publisher")
    @patch("services.analysis_service.src.api.endpoints.tracklist.recording_repo")
    def test_message_queue_error_handling(self, mock_recording_repo, mock_message_publisher):
        """Test message queue error handling."""
        mock_recording = Mock()
        mock_recording.id = self.test_recording_id
        mock_recording.file_path = "/path/to/test.wav"
        mock_recording_repo.get_by_id = AsyncMock(return_value=mock_recording)

        with patch("pathlib.Path.exists", return_value=True):
            mock_message_publisher.publish_tracklist_generation = AsyncMock(
                side_effect=Exception("Message queue unavailable")
            )

            response = self.client.post(f"/v1/tracklist/{self.test_recording_id}/detect")

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_invalid_json_handling(self):
        """Test invalid JSON handling."""
        response = self.client.post(
            "/v1/tracklist/parse-cue",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_missing_required_fields(self):
        """Test missing required fields."""
        response = self.client.post(
            "/v1/tracklist/parse-cue",
            json={"cue_content": "test"},  # Missing audio_file_path
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
