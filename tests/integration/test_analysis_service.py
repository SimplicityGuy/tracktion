"""Integration tests for the analysis service."""

import os
import sys
import time
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0,
    str(Path(__file__).parent.parent.parent / "services" / "analysis_service" / "src"),
)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from services.analysis_service.src.exceptions import InvalidAudioFileError, MetadataExtractionError, StorageError
from services.analysis_service.src.main import AnalysisService
from services.analysis_service.src.message_consumer import MessageConsumer
from services.analysis_service.src.storage_handler import StorageHandler


@pytest.fixture
def analysis_service():
    """Create an analysis service instance."""
    return AnalysisService()


@pytest.fixture
def mock_rabbitmq_message():
    """Create a mock RabbitMQ message."""
    return {
        "recording_id": str(uuid.uuid4()),
        "file_path": "/path/to/test.mp3",
        "timestamp": time.time(),
    }


class TestAnalysisServiceIntegration:
    """Integration tests for the analysis service."""

    @patch("main.StorageHandler")
    @patch("main.MetadataExtractor")
    @patch("main.MessageConsumer")
    def test_service_initialization(self, mock_consumer, mock_extractor, mock_storage, analysis_service):
        """Test service initialization."""
        analysis_service.initialize()

        assert analysis_service.extractor is not None
        assert analysis_service.storage is not None
        assert analysis_service.consumer is not None
        mock_consumer.assert_called_once()
        mock_extractor.assert_called_once()
        mock_storage.assert_called_once()

    @patch("main.StorageHandler")
    @patch("main.MetadataExtractor")
    def test_process_message_success(
        self,
        mock_extractor_class,
        mock_storage_class,
        analysis_service,
        mock_rabbitmq_message,
    ):
        """Test successful message processing."""
        # Setup mocks
        mock_extractor = MagicMock()
        mock_storage = MagicMock()
        mock_extractor_class.return_value = mock_extractor
        mock_storage_class.return_value = mock_storage

        mock_extractor.extract.return_value = {
            "title": "Test Song",
            "artist": "Test Artist",
            "duration": "180.0",
            "format": "mp3",
        }
        mock_storage.store_metadata.return_value = True
        mock_storage.update_recording_status.return_value = True

        # Initialize and process
        analysis_service.initialize()
        analysis_service.process_message(mock_rabbitmq_message, "test-correlation-id")

        # Verify calls
        mock_extractor.extract.assert_called_once_with(mock_rabbitmq_message["file_path"])
        mock_storage.store_metadata.assert_called_once()
        mock_storage.update_recording_status.assert_called_with(
            uuid.UUID(mock_rabbitmq_message["recording_id"]),
            "processed",
            None,
            "test-correlation-id",
        )

    @patch("main.StorageHandler")
    @patch("main.MetadataExtractor")
    def test_process_message_invalid_file(
        self,
        mock_extractor_class,
        mock_storage_class,
        analysis_service,
        mock_rabbitmq_message,
    ):
        """Test processing with invalid audio file."""
        # Setup mocks
        mock_extractor = MagicMock()
        mock_storage = MagicMock()
        mock_extractor_class.return_value = mock_extractor
        mock_storage_class.return_value = mock_storage

        mock_extractor.extract.side_effect = InvalidAudioFileError("Unsupported format")
        mock_storage.update_recording_status.return_value = True

        # Initialize and process
        analysis_service.initialize()
        analysis_service.process_message(mock_rabbitmq_message, "test-correlation-id")

        # Verify error handling
        mock_storage.update_recording_status.assert_called_with(
            uuid.UUID(mock_rabbitmq_message["recording_id"]),
            "invalid",
            "Unsupported format",
            "test-correlation-id",
        )

    @patch("main.StorageHandler")
    @patch("main.MetadataExtractor")
    def test_process_message_extraction_error(
        self,
        mock_extractor_class,
        mock_storage_class,
        analysis_service,
        mock_rabbitmq_message,
    ):
        """Test processing with extraction error."""
        # Setup mocks
        mock_extractor = MagicMock()
        mock_storage = MagicMock()
        mock_extractor_class.return_value = mock_extractor
        mock_storage_class.return_value = mock_storage

        mock_extractor.extract.side_effect = [
            MetadataExtractionError("Extraction failed"),
            MetadataExtractionError("Extraction failed"),
            MetadataExtractionError("Extraction failed"),
            MetadataExtractionError("Extraction failed"),  # Exceeds max retries
        ]
        mock_storage.update_recording_status.return_value = True

        # Initialize and process
        analysis_service.initialize()
        analysis_service.process_message(mock_rabbitmq_message, "test-correlation-id")

        # Verify retry logic was attempted
        assert mock_extractor.extract.call_count == 4  # 1 initial + 3 retries

        # Verify failure status update
        mock_storage.update_recording_status.assert_called_with(
            uuid.UUID(mock_rabbitmq_message["recording_id"]),
            "failed",
            "Extraction failed",
            "test-correlation-id",
        )

    @patch("main.StorageHandler")
    @patch("main.MetadataExtractor")
    def test_process_message_storage_error(
        self,
        mock_extractor_class,
        mock_storage_class,
        analysis_service,
        mock_rabbitmq_message,
    ):
        """Test processing with storage error."""
        # Setup mocks
        mock_extractor = MagicMock()
        mock_storage = MagicMock()
        mock_extractor_class.return_value = mock_extractor
        mock_storage_class.return_value = mock_storage

        mock_extractor.extract.return_value = {
            "title": "Test Song",
            "artist": "Test Artist",
        }

        mock_storage.store_metadata.side_effect = StorageError("Database error")
        mock_storage.update_recording_status.return_value = True

        # Initialize and process
        analysis_service.initialize()
        analysis_service.process_message(mock_rabbitmq_message, "test-correlation-id")

        # Verify failure status update
        calls = mock_storage.update_recording_status.call_args_list
        assert len(calls) > 0
        last_call = calls[-1]
        assert last_call[0][1] == "failed"  # Status should be "failed"

    def test_process_message_missing_fields(self, analysis_service):
        """Test processing with missing required fields."""
        # Message missing recording_id
        invalid_message1 = {"file_path": "/path/to/file.mp3"}
        analysis_service.process_message(invalid_message1, "test-correlation-id")

        # Message missing file_path
        invalid_message2 = {"recording_id": str(uuid.uuid4())}
        analysis_service.process_message(invalid_message2, "test-correlation-id")

        # Both messages should be handled gracefully without crashing

    @patch("main.StorageHandler")
    @patch("main.MetadataExtractor")
    @patch("main.MessageConsumer")
    def test_service_shutdown(
        self,
        mock_consumer_class,
        mock_extractor_class,
        mock_storage_class,
        analysis_service,
    ):
        """Test service shutdown."""
        # Setup mocks
        mock_consumer = MagicMock()
        mock_storage = MagicMock()
        mock_consumer_class.return_value = mock_consumer
        mock_storage_class.return_value = mock_storage

        # Initialize and shutdown
        analysis_service.initialize()
        analysis_service.running = True
        analysis_service.shutdown()

        # Verify cleanup
        assert analysis_service.running is False
        mock_consumer.stop.assert_called_once()
        mock_storage.close.assert_called_once()

    @patch("main.StorageHandler")
    @patch("main.MetadataExtractor")
    @patch("main.MessageConsumer")
    def test_health_check(
        self,
        mock_consumer_class,
        mock_extractor_class,
        mock_storage_class,
        analysis_service,
    ):
        """Test health check functionality."""
        # Setup mocks
        mock_consumer = MagicMock()
        mock_connection = MagicMock()
        mock_connection.is_closed = False
        mock_consumer.connection = mock_connection
        mock_consumer_class.return_value = mock_consumer
        mock_storage_class.return_value = MagicMock()

        # Initialize and check health
        analysis_service.initialize()
        analysis_service.running = True
        health = analysis_service.health_check()

        assert health["service"] == "analysis_service"
        assert health["status"] == "healthy"
        assert health["healthy"] is True
        assert "components" in health
        assert health["components"]["rabbitmq"]["status"] == "connected"
        assert health["components"]["storage"]["status"] == "initialized"

    @patch("message_consumer.pika")
    def test_message_consumer_connection_retry(self, mock_pika):
        """Test RabbitMQ connection retry logic."""
        # Simulate connection failures then success
        mock_pika.exceptions.AMQPConnectionError = Exception
        mock_pika.BlockingConnection.side_effect = [
            Exception("Connection failed"),
            Exception("Connection failed"),
            MagicMock(),  # Success on third attempt
        ]

        consumer = MessageConsumer("amqp://localhost:5672/")
        consumer.connect()

        # Verify retry attempts
        assert mock_pika.BlockingConnection.call_count == 3

    def test_end_to_end_flow(self, analysis_service):
        """Test end-to-end message processing flow."""
        # This test would require actual RabbitMQ and database connections
        # It's marked for manual/integration environment testing
        pytest.skip("Requires live RabbitMQ and database connections")


class TestStorageHandlerIntegration:
    """Integration tests for storage handler."""

    @patch("storage_handler.Neo4jRepository")
    @patch("storage_handler.MetadataRepository")
    @patch("storage_handler.RecordingRepository")
    @patch("storage_handler.get_db_session")
    def test_storage_initialization(self, mock_get_session, mock_recording_repo, mock_metadata_repo, mock_neo4j_repo):
        """Test storage handler initialization."""
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        # Set environment variables
        with patch.dict(
            os.environ,
            {
                "NEO4J_URI": "bolt://localhost:7687",
                "NEO4J_USER": "neo4j",
                "NEO4J_PASSWORD": "password",
            },
        ):
            storage = StorageHandler()

            assert storage.recording_repo is not None
            assert storage.metadata_repo is not None
            assert storage.neo4j_repo is not None
            mock_recording_repo.assert_called_once_with(mock_session)
            mock_metadata_repo.assert_called_once_with(mock_session)

    @patch("storage_handler.Neo4jRepository")
    @patch("storage_handler.MetadataRepository")
    @patch("storage_handler.RecordingRepository")
    @patch("storage_handler.get_db_session")
    def test_store_metadata(
        self,
        mock_get_session,
        mock_recording_repo_class,
        mock_metadata_repo_class,
        mock_neo4j_repo_class,
    ):
        """Test metadata storage."""
        # Setup mocks
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session

        mock_recording_repo = MagicMock()
        mock_metadata_repo = MagicMock()
        mock_neo4j_repo = MagicMock()

        mock_recording_repo_class.return_value = mock_recording_repo
        mock_metadata_repo_class.return_value = mock_metadata_repo
        mock_neo4j_repo_class.return_value = mock_neo4j_repo

        mock_recording = MagicMock()
        mock_recording.file_path = "/path/to/file.mp3"
        mock_recording.file_hash = "abc123"
        mock_recording_repo.get.return_value = mock_recording
        mock_metadata_repo.get_by_recording.return_value = []
        mock_neo4j_repo.recording_exists.return_value = True

        # Set environment variables
        with patch.dict(
            os.environ,
            {
                "NEO4J_URI": "bolt://localhost:7687",
                "NEO4J_USER": "neo4j",
                "NEO4J_PASSWORD": "password",
            },
        ):
            storage = StorageHandler()

            recording_id = uuid.uuid4()
            metadata = {
                "title": "Test Song",
                "artist": "Test Artist",
                "duration": "180.0",
            }

            result = storage.store_metadata(recording_id, metadata, "test-correlation")

            assert result is True
            mock_recording_repo.get.assert_called_once_with(recording_id)
            assert mock_metadata_repo.create.call_count == 3  # Three metadata items
            assert mock_neo4j_repo.create_metadata.call_count == 3


class TestMessageConsumerIntegration:
    """Integration tests for message consumer."""

    @patch("message_consumer.pika")
    def test_consumer_initialization(self, mock_pika):
        """Test message consumer initialization."""
        consumer = MessageConsumer("amqp://localhost:5672/", "test_queue", "test_exchange", "test.routing")

        assert consumer.rabbitmq_url == "amqp://localhost:5672/"
        assert consumer.queue_name == "test_queue"
        assert consumer.exchange_name == "test_exchange"
        assert consumer.routing_key == "test.routing"

    @patch("message_consumer.pika")
    def test_consumer_connect(self, mock_pika):
        """Test consumer connection."""
        mock_connection = MagicMock()
        mock_channel = MagicMock()
        mock_connection.channel.return_value = mock_channel
        mock_pika.BlockingConnection.return_value = mock_connection

        consumer = MessageConsumer("amqp://localhost:5672/")
        consumer.connect()

        mock_channel.exchange_declare.assert_called_once()
        mock_channel.queue_declare.assert_called_once()
        mock_channel.queue_bind.assert_called_once()
        mock_channel.basic_qos.assert_called_once_with(prefetch_count=1)

    @patch("message_consumer.pika")
    def test_consumer_message_processing(self, mock_pika):
        """Test message consumption and processing."""
        mock_connection = MagicMock()
        mock_channel = MagicMock()
        mock_connection.channel.return_value = mock_channel
        mock_pika.BlockingConnection.return_value = mock_connection

        consumer = MessageConsumer("amqp://localhost:5672/")
        consumer.connect()

        # Create mock callback
        mock_callback = MagicMock()

        # Simulate message consumption
        consumer.consume(mock_callback)

        # Verify consumer setup
        mock_channel.basic_consume.assert_called_once()
        call_kwargs = mock_channel.basic_consume.call_args[1]
        assert call_kwargs["queue"] == consumer.queue_name
        assert call_kwargs["auto_ack"] is False
