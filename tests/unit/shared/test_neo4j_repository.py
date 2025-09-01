"""Unit tests for Neo4j repository."""

from unittest.mock import MagicMock, Mock, patch
from uuid import uuid4

import pytest
from neo4j.exceptions import AuthError, ServiceUnavailable

from shared.core_types.src.neo4j_repository import Neo4jRepository


class TestNeo4jRepository:
    """Test cases for Neo4jRepository."""

    @pytest.fixture
    def mock_driver(self):
        """Create a mock Neo4j driver."""
        driver = Mock()
        driver.verify_connectivity = Mock()
        driver.close = Mock()
        return driver

    @pytest.fixture
    def mock_session(self):
        """Create a mock Neo4j session."""
        session = MagicMock()
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=None)
        return session

    @pytest.fixture
    def repository(self, mock_driver):
        """Create a repository instance with mock driver."""
        with patch("shared.core_types.src.neo4j_repository.GraphDatabase") as mock_graph_db:
            mock_graph_db.driver.return_value = mock_driver
            return Neo4jRepository("bolt://localhost:7687", "neo4j", "password")

    def test_initialization_success(self, mock_driver):
        """Test successful initialization."""
        with patch("shared.core_types.src.neo4j_repository.GraphDatabase") as mock_graph_db:
            mock_graph_db.driver.return_value = mock_driver

            Neo4jRepository("bolt://localhost:7687", "neo4j", "password")

            mock_graph_db.driver.assert_called_once_with("bolt://localhost:7687", auth=("neo4j", "password"))
            mock_driver.verify_connectivity.assert_called_once()

    def test_initialization_service_unavailable(self, mock_driver):
        """Test initialization when Neo4j is unavailable."""
        mock_driver.verify_connectivity.side_effect = ServiceUnavailable("Service unavailable")

        with patch("shared.core_types.src.neo4j_repository.GraphDatabase") as mock_graph_db:
            mock_graph_db.driver.return_value = mock_driver

            with pytest.raises(ServiceUnavailable):
                Neo4jRepository("bolt://localhost:7687", "neo4j", "password")

    def test_initialization_auth_error(self, mock_driver):
        """Test initialization with authentication error."""
        mock_driver.verify_connectivity.side_effect = AuthError("Authentication failed")

        with patch("shared.core_types.src.neo4j_repository.GraphDatabase") as mock_graph_db:
            mock_graph_db.driver.return_value = mock_driver

            with pytest.raises(AuthError):
                Neo4jRepository("bolt://localhost:7687", "neo4j", "password")

    def test_close(self, repository, mock_driver):
        """Test closing the driver connection."""
        repository.close()
        mock_driver.close.assert_called_once()

    def test_create_recording_node(self, repository, mock_driver, mock_session):
        """Test creating a recording node."""
        mock_driver.session.return_value = mock_session

        # Mock the result
        mock_record = {
            "uuid": "test-uuid",
            "file_name": "test.wav",
            "file_path": "/path/test.wav",
        }
        mock_result = Mock()
        mock_result.single.return_value = {"r": mock_record}
        mock_session.run.return_value = mock_result

        # Call create_recording_node
        recording_id = uuid4()
        result = repository.create_recording_node(
            recording_id=recording_id, file_name="test.wav", file_path="/path/test.wav"
        )

        # Verify the query was executed
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "CREATE (r:Recording" in call_args[0][0]
        assert call_args[1]["uuid"] == str(recording_id)
        assert call_args[1]["file_name"] == "test.wav"
        assert call_args[1]["file_path"] == "/path/test.wav"

        assert result == mock_record

    def test_create_recording_node_no_result(self, repository, mock_driver, mock_session):
        """Test creating a recording node with no result."""
        mock_driver.session.return_value = mock_session

        # Mock no result
        mock_result = Mock()
        mock_result.single.return_value = None
        mock_session.run.return_value = mock_result

        # Call create_recording_node
        recording_id = uuid4()
        result = repository.create_recording_node(
            recording_id=recording_id, file_name="test.wav", file_path="/path/test.wav"
        )

        assert result == {}

    def test_get_recording_node(self, repository, mock_driver, mock_session):
        """Test getting a recording node."""
        mock_driver.session.return_value = mock_session

        # Mock the result
        mock_record = {
            "uuid": "test-uuid",
            "file_name": "test.wav",
            "file_path": "/path/test.wav",
        }
        mock_result = Mock()
        mock_result.single.return_value = {"r": mock_record}
        mock_session.run.return_value = mock_result

        # Call get_recording_node
        recording_id = uuid4()
        result = repository.get_recording_node(recording_id)

        # Verify the query was executed
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "MATCH (r:Recording {uuid: $uuid})" in call_args[0][0]
        assert call_args[1]["uuid"] == str(recording_id)

        assert result == mock_record

    def test_get_recording_node_not_found(self, repository, mock_driver, mock_session):
        """Test getting a recording node that doesn't exist."""
        mock_driver.session.return_value = mock_session

        # Mock no result
        mock_result = Mock()
        mock_result.single.return_value = None
        mock_session.run.return_value = mock_result

        # Call get_recording_node
        recording_id = uuid4()
        result = repository.get_recording_node(recording_id)

        assert result is None

    def test_add_metadata_relationship(self, repository, mock_driver, mock_session):
        """Test adding metadata relationship."""
        mock_driver.session.return_value = mock_session

        # Call add_metadata_relationship
        recording_id = uuid4()
        repository.add_metadata_relationship(recording_id=recording_id, key="artist", value="Test Artist")

        # Verify the query was executed
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "CREATE (r)-[:HAS_METADATA]->" in call_args[0][0]
        assert call_args[1]["uuid"] == str(recording_id)
        assert call_args[1]["key"] == "artist"
        assert call_args[1]["value"] == "Test Artist"

    def test_get_recording_metadata(self, repository, mock_driver, mock_session):
        """Test getting recording metadata."""
        mock_driver.session.return_value = mock_session

        # Mock the result
        mock_result = [
            {"key": "artist", "value": "Test Artist"},
            {"key": "album", "value": "Test Album"},
        ]
        mock_session.run.return_value = mock_result

        # Call get_recording_metadata
        recording_id = uuid4()
        result = repository.get_recording_metadata(recording_id)

        # Verify the query was executed
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "MATCH (r:Recording {uuid: $uuid})-[:HAS_METADATA]->" in call_args[0][0]
        assert call_args[1]["uuid"] == str(recording_id)

        assert result == mock_result

    def test_add_tracklist_with_tracks(self, repository, mock_driver, mock_session):
        """Test adding tracklist with tracks."""
        mock_driver.session.return_value = mock_session

        # Call add_tracklist_with_tracks
        recording_id = uuid4()
        tracks = [
            {"title": "Track 1", "artist": "Artist 1", "start_time": "00:00"},
            {"title": "Track 2", "artist": "Artist 2", "start_time": "03:30"},
        ]

        repository.add_tracklist_with_tracks(recording_id=recording_id, source="manual", tracks=tracks)

        # Verify the query was executed
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "CREATE (r)-[:HAS_TRACKLIST]->" in call_args[0][0]
        assert "UNWIND $tracks AS track" in call_args[0][0]
        assert call_args[1]["recording_uuid"] == str(recording_id)
        assert call_args[1]["source"] == "manual"
        assert call_args[1]["tracks"] == tracks

    def test_get_tracklist_tracks(self, repository, mock_driver, mock_session):
        """Test getting tracklist tracks."""
        mock_driver.session.return_value = mock_session

        # Mock the result
        mock_result = [
            {"title": "Track 1", "artist": "Artist 1", "start_time": "00:00"},
            {"title": "Track 2", "artist": "Artist 2", "start_time": "03:30"},
        ]
        mock_session.run.return_value = mock_result

        # Call get_tracklist_tracks
        recording_id = uuid4()
        result = repository.get_tracklist_tracks(recording_id)

        # Verify the query was executed
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "MATCH (r:Recording {uuid: $uuid})-[:HAS_TRACKLIST]->" in call_args[0][0]
        assert call_args[1]["uuid"] == str(recording_id)

        assert result == mock_result

    def test_delete_recording_node(self, repository, mock_driver, mock_session):
        """Test deleting a recording node."""
        mock_driver.session.return_value = mock_session

        # Mock the result (1 node deleted)
        mock_result = Mock()
        mock_result.single.return_value = {"deleted": 1}
        mock_session.run.return_value = mock_result

        # Call delete_recording_node
        recording_id = uuid4()
        result = repository.delete_recording_node(recording_id)

        # Verify the query was executed
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args
        assert "DETACH DELETE r, rel, n" in call_args[0][0]
        assert call_args[1]["uuid"] == str(recording_id)

        assert result is True

    def test_delete_recording_node_not_found(self, repository, mock_driver, mock_session):
        """Test deleting a recording node that doesn't exist."""
        mock_driver.session.return_value = mock_session

        # Mock the result (0 nodes deleted)
        mock_result = Mock()
        mock_result.single.return_value = {"deleted": 0}
        mock_session.run.return_value = mock_result

        # Call delete_recording_node
        recording_id = uuid4()
        result = repository.delete_recording_node(recording_id)

        assert result is False

    def test_create_constraints(self, repository, mock_driver, mock_session):
        """Test creating database constraints."""
        mock_driver.session.return_value = mock_session

        # Call create_constraints
        repository.create_constraints()

        # Verify both constraint queries were executed
        assert mock_session.run.call_count == 2

        # Check first call (uniqueness constraint)
        first_call = mock_session.run.call_args_list[0]
        assert "CREATE CONSTRAINT recording_uuid_unique" in first_call[0][0]

        # Check second call (index)
        second_call = mock_session.run.call_args_list[1]
        assert "CREATE INDEX metadata_key_index" in second_call[0][0]

    def test_clear_database(self, repository, mock_driver, mock_session):
        """Test clearing the database."""
        mock_driver.session.return_value = mock_session

        # Call clear_database
        repository.clear_database()

        # Verify the delete query was executed
        mock_session.run.assert_called_once_with("MATCH (n) DETACH DELETE n")
