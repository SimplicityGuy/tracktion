"""Unit tests for API endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from services.file_rename_service.app.main import app


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    return TestClient(app)


def test_analyze_patterns_endpoint(client: TestClient) -> None:
    """Test the analyze patterns endpoint."""
    request_data = {
        "filenames": ["track01.mp3", "track02.mp3", "track03.mp3"],
        "context": {"album": "Test Album"},
    }

    with patch("services.file_rename_service.api.routers.rabbitmq_manager") as mock_rabbitmq:
        mock_rabbitmq.is_connected = False

        response = client.post("/rename/analyze", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "patterns" in data
        assert "categories" in data
        assert "confidence" in data
        assert "suggestions" in data
        assert data["confidence"] > 0


def test_propose_rename_endpoint(client: TestClient) -> None:
    """Test the propose rename endpoint."""
    request_data = {
        "original_name": "test_file.mp3",
        "metadata": {"title": "Test Song", "artist": "Test Artist"},
    }

    with patch("services.file_rename_service.api.routers.rabbitmq_manager") as mock_rabbitmq:
        mock_rabbitmq.is_connected = False

        with patch("services.file_rename_service.api.routers.get_db") as mock_get_db:
            mock_db = MagicMock()
            mock_get_db.return_value = mock_db

            response = client.post("/rename/propose", json=request_data)

            assert response.status_code == 200
            data = response.json()
            assert "original_name" in data
            assert data["original_name"] == "test_file.mp3"
            assert "proposals" in data
            assert len(data["proposals"]) > 0
            assert "recommended" in data


def test_submit_feedback_endpoint(client: TestClient) -> None:
    """Test the submit feedback endpoint."""
    request_data = {
        "original_name": "test_file.mp3",
        "proposed_name": "renamed_test_file.mp3",
        "was_accepted": True,
        "rating": 5,
    }

    with patch("services.file_rename_service.api.routers.rabbitmq_manager") as mock_rabbitmq:
        mock_rabbitmq.is_connected = False

        with patch("services.file_rename_service.api.routers.get_db") as mock_get_db:
            mock_db = MagicMock()
            mock_history = MagicMock()
            mock_history.id = 1
            mock_db.query().filter().first.return_value = None
            mock_db.flush.return_value = None
            mock_db.commit.return_value = None
            mock_get_db.return_value = mock_db

            response = client.post("/rename/feedback", json=request_data)

            assert response.status_code == 200
            data = response.json()
            assert "feedback_id" in data
            assert "message" in data
            assert "patterns_updated" in data


def test_get_patterns_endpoint(client: TestClient) -> None:
    """Test the get patterns endpoint."""
    with patch("services.file_rename_service.api.routers.get_db") as mock_get_db:
        mock_db = MagicMock()
        mock_pattern = MagicMock()
        mock_pattern.id = 1
        mock_pattern.pattern_type = "regex"
        mock_pattern.pattern_value = ".*\\.mp3$"
        mock_pattern.description = "MP3 files"
        mock_pattern.category = "music"
        mock_pattern.frequency = 10
        mock_pattern.confidence_score = 0.9
        mock_pattern.created_at = "2024-01-01T00:00:00"
        mock_pattern.updated_at = "2024-01-01T00:00:00"

        mock_db.query().filter().order_by().offset().limit().all.return_value = [mock_pattern]
        mock_get_db.return_value = mock_db

        response = client.get("/rename/patterns")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if data:
            assert "pattern_type" in data[0]
            assert "pattern_value" in data[0]


def test_get_patterns_with_filters(client: TestClient) -> None:
    """Test the get patterns endpoint with filters."""
    with patch("services.file_rename_service.api.routers.get_db") as mock_get_db:
        mock_db = MagicMock()
        mock_db.query().filter().filter().filter().order_by().offset().limit().all.return_value = []
        mock_get_db.return_value = mock_db

        response = client.get("/rename/patterns?category=music&pattern_type=regex")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


def test_get_history_endpoint(client: TestClient) -> None:
    """Test the get history endpoint."""
    with patch("services.file_rename_service.api.routers.get_db") as mock_get_db:
        mock_db = MagicMock()
        mock_history = MagicMock()
        mock_history.id = 1
        mock_history.original_name = "original.mp3"
        mock_history.proposed_name = "renamed.mp3"
        mock_history.final_name = None
        mock_history.confidence_score = 0.8
        mock_history.was_accepted = True
        mock_history.feedback_rating = 4
        mock_history.created_at = "2024-01-01T00:00:00"

        mock_db.query().order_by().offset().limit().all.return_value = [mock_history]
        mock_get_db.return_value = mock_db

        response = client.get("/rename/history")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if data:
            assert "original_name" in data[0]
            assert "proposed_name" in data[0]


def test_get_history_with_filter(client: TestClient) -> None:
    """Test the get history endpoint with acceptance filter."""
    with patch("services.file_rename_service.api.routers.get_db") as mock_get_db:
        mock_db = MagicMock()
        mock_db.query().filter().order_by().offset().limit().all.return_value = []
        mock_get_db.return_value = mock_db

        response = client.get("/rename/history?was_accepted=true")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


def test_pagination_parameters(client: TestClient) -> None:
    """Test pagination parameters."""
    with patch("services.file_rename_service.api.routers.get_db") as mock_get_db:
        mock_db = MagicMock()
        mock_db.query().filter().order_by().offset().limit().all.return_value = []
        mock_get_db.return_value = mock_db

        response = client.get("/rename/patterns?skip=10&limit=50")

        assert response.status_code == 200
        # Check that offset and limit were called with correct values
        mock_db.query().filter().order_by().offset.assert_called_with(10)
        mock_db.query().filter().order_by().offset().limit.assert_called_with(50)
