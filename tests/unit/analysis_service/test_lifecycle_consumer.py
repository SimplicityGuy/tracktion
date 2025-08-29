"""Unit tests for the lifecycle event consumer in analysis service."""

import json
import unittest
from unittest.mock import MagicMock, Mock, patch

from services.analysis_service.src.lifecycle_consumer import LifecycleEventConsumer


class TestLifecycleEventConsumer(unittest.TestCase):
    """Test cases for LifecycleEventConsumer."""

    @patch("services.analysis_service.src.lifecycle_consumer.StorageHandler")
    def setUp(self, mock_storage_handler_class: Mock) -> None:
        """Set up test fixtures."""
        # Mock StorageHandler to prevent database initialization
        mock_storage_handler = MagicMock()
        mock_storage_handler_class.return_value = mock_storage_handler

        self.consumer = LifecycleEventConsumer(
            rabbitmq_url="amqp://guest:guest@localhost:5672/",
            redis_host="localhost",
            redis_port=6379,
            enable_cache=False,  # Disable cache for testing
        )

        # Set the mocked storage handler
        self.consumer.storage_handler = mock_storage_handler

    @patch("services.analysis_service.src.lifecycle_consumer.pika.BlockingConnection")
    def test_connect(self, mock_connection: Mock) -> None:
        """Test RabbitMQ connection establishment."""
        mock_channel = MagicMock()
        mock_connection.return_value.channel.return_value = mock_channel

        self.consumer.connect()

        mock_connection.assert_called_once()
        mock_channel.exchange_declare.assert_called_once_with(
            exchange="file_events",
            exchange_type="topic",
            durable=True,
        )
        mock_channel.queue_declare.assert_called_once_with(
            queue="analysis.lifecycle.events",
            durable=True,
        )
        mock_channel.queue_bind.assert_called_once()

    def test_handle_file_deleted_no_cache(self) -> None:
        """Test file deletion handling without cache."""
        # Mock Neo4j repository
        mock_neo4j_repo = MagicMock()
        mock_neo4j_repo.delete_recording_by_filepath.return_value = True
        assert self.consumer.storage_handler is not None  # For mypy
        self.consumer.storage_handler.neo4j_repo = mock_neo4j_repo

        # Handle file deletion
        self.consumer.handle_file_deleted("/path/to/file.mp3", "test-correlation-id")

        # Verify Neo4j deletion was called
        mock_neo4j_repo.delete_recording_by_filepath.assert_called_once_with("/path/to/file.mp3")

    @patch("services.analysis_service.src.lifecycle_consumer.StorageHandler")
    @patch("services.analysis_service.src.lifecycle_consumer.AudioCache")
    def test_handle_file_deleted_with_cache(self, mock_audio_cache: Mock, mock_storage_handler_class: Mock) -> None:
        """Test file deletion handling with cache cleanup."""
        # Mock StorageHandler
        mock_storage_handler = MagicMock()
        mock_storage_handler_class.return_value = mock_storage_handler

        # Create consumer with cache enabled
        consumer = LifecycleEventConsumer(
            rabbitmq_url="amqp://guest:guest@localhost:5672/",
            redis_host="localhost",
            redis_port=6379,
            enable_cache=True,
        )

        # Mock cache
        mock_cache_instance = MagicMock()
        mock_cache_instance.redis_client = MagicMock()
        mock_cache_instance._generate_file_hash.return_value = "test_hash"
        mock_cache_instance._build_cache_key.side_effect = lambda prefix, hash: f"{prefix}:{hash}:1.0"
        mock_cache_instance.redis_client.delete.return_value = 1
        mock_cache_instance.BPM_PREFIX = "bpm"
        mock_cache_instance.TEMPORAL_PREFIX = "temporal"
        mock_cache_instance.KEY_PREFIX = "key"
        mock_cache_instance.MOOD_PREFIX = "mood"
        consumer.cache = mock_cache_instance

        # Mock Neo4j
        mock_neo4j_repo = MagicMock()
        mock_neo4j_repo.delete_recording_by_filepath.return_value = True
        consumer.storage_handler = mock_storage_handler
        consumer.storage_handler.neo4j_repo = mock_neo4j_repo

        # Handle file deletion
        consumer.handle_file_deleted("/path/to/file.mp3", "test-correlation-id")

        # Verify cache cleanup
        mock_cache_instance._generate_file_hash.assert_called_once_with("/path/to/file.mp3")
        assert mock_cache_instance.redis_client.delete.call_count == 4  # 4 cache prefixes

        # Verify Neo4j deletion
        mock_neo4j_repo.delete_recording_by_filepath.assert_called_once_with("/path/to/file.mp3")

    def test_handle_file_moved(self) -> None:
        """Test file move event handling."""
        # Mock cache
        mock_cache = MagicMock()
        mock_cache.redis_client = MagicMock()
        mock_cache._generate_file_hash.return_value = "test_hash"
        mock_cache._build_cache_key.side_effect = lambda prefix, hash: f"{prefix}:{hash}:1.0"
        mock_cache.redis_client.delete.return_value = 1
        mock_cache.BPM_PREFIX = "bpm"
        mock_cache.TEMPORAL_PREFIX = "temporal"
        mock_cache.KEY_PREFIX = "key"
        mock_cache.MOOD_PREFIX = "mood"
        self.consumer.cache = mock_cache

        # Handle file move
        self.consumer.handle_file_moved("/old/path/file.mp3", "/new/path/file.mp3", "test-correlation-id")

        # Verify old cache entries were cleared
        mock_cache._generate_file_hash.assert_called_once_with("/old/path/file.mp3")
        assert mock_cache.redis_client.delete.call_count == 4

    def test_handle_file_renamed(self) -> None:
        """Test file rename event handling."""
        # Mock cache
        mock_cache = MagicMock()
        mock_cache.redis_client = MagicMock()
        mock_cache._generate_file_hash.return_value = "test_hash"
        mock_cache._build_cache_key.side_effect = lambda prefix, hash: f"{prefix}:{hash}:1.0"
        mock_cache.redis_client.delete.return_value = 1
        mock_cache.BPM_PREFIX = "bpm"
        mock_cache.TEMPORAL_PREFIX = "temporal"
        mock_cache.KEY_PREFIX = "key"
        mock_cache.MOOD_PREFIX = "mood"
        self.consumer.cache = mock_cache

        # Handle file rename
        self.consumer.handle_file_renamed("/path/old_name.mp3", "/path/new_name.mp3", "test-correlation-id")

        # Verify old cache entries were cleared
        mock_cache._generate_file_hash.assert_called_once_with("/path/old_name.mp3")
        assert mock_cache.redis_client.delete.call_count == 4

    @patch("services.analysis_service.src.lifecycle_consumer.pika.BlockingConnection")
    def test_process_message_deleted_event(self, mock_connection: Mock) -> None:
        """Test processing of deleted event message."""
        # Set up mock channel
        mock_channel = MagicMock()
        mock_method = MagicMock()
        mock_method.delivery_tag = 123
        mock_properties = MagicMock()

        # Mock Neo4j
        mock_neo4j_repo = MagicMock()
        mock_neo4j_repo.delete_recording_by_filepath.return_value = True
        assert self.consumer.storage_handler is not None  # For mypy
        self.consumer.storage_handler.neo4j_repo = mock_neo4j_repo

        # Create message body
        message = {
            "event_type": "deleted",
            "file_path": "/path/to/file.mp3",
            "correlation_id": "test-correlation-id",
        }
        body = json.dumps(message).encode("utf-8")

        # Process message
        self.consumer.process_message(mock_channel, mock_method, mock_properties, body)

        # Verify message was acknowledged
        mock_channel.basic_ack.assert_called_once_with(delivery_tag=123)

        # Verify Neo4j deletion was called
        mock_neo4j_repo.delete_recording_by_filepath.assert_called_once_with("/path/to/file.mp3")

    @patch("services.analysis_service.src.lifecycle_consumer.pika.BlockingConnection")
    def test_process_message_moved_event(self, mock_connection: Mock) -> None:
        """Test processing of moved event message with new structure."""
        # Set up mock channel
        mock_channel = MagicMock()
        mock_method = MagicMock()
        mock_method.delivery_tag = 123
        mock_properties = MagicMock()
        mock_properties.correlation_id = None

        # Mock cache
        mock_cache = MagicMock()
        mock_cache.redis_client = MagicMock()
        mock_cache._generate_file_hash.return_value = "test_hash"
        mock_cache._build_cache_key.side_effect = lambda prefix, hash: f"{prefix}:{hash}:1.0"
        mock_cache.redis_client.delete.return_value = 1
        mock_cache.BPM_PREFIX = "bpm"
        mock_cache.TEMPORAL_PREFIX = "temporal"
        mock_cache.KEY_PREFIX = "key"
        mock_cache.MOOD_PREFIX = "mood"
        self.consumer.cache = mock_cache

        # Mock Neo4j for update
        mock_neo4j_repo = MagicMock()
        mock_neo4j_repo.delete_recording_by_filepath.return_value = True
        assert self.consumer.storage_handler is not None
        self.consumer.storage_handler.neo4j_repo = mock_neo4j_repo

        # Create message body with new structure
        message = {
            "event_type": "moved",
            "file_path": "/new/path/file.mp3",  # New path is in file_path
            "old_path": "/old/path/file.mp3",  # Old path is in old_path
            "timestamp": "2025-08-28T10:00:00Z",
            "instance_id": "watcher1",
            "sha256_hash": "hash123",
            "xxh128_hash": "hash456",
        }
        body = json.dumps(message).encode("utf-8")

        # Process message
        self.consumer.process_message(mock_channel, mock_method, mock_properties, body)

        # Verify message was acknowledged
        mock_channel.basic_ack.assert_called_once_with(delivery_tag=123)

        # Verify cache cleanup for old path
        mock_cache._generate_file_hash.assert_called_once_with("/old/path/file.mp3")

        # Verify Neo4j update (currently deletes old)
        mock_neo4j_repo.delete_recording_by_filepath.assert_called_once_with("/old/path/file.mp3")

    @patch("services.analysis_service.src.lifecycle_consumer.pika.BlockingConnection")
    def test_process_message_invalid_json(self, mock_connection: Mock) -> None:
        """Test handling of invalid JSON message."""
        # Set up mock channel
        mock_channel = MagicMock()
        mock_method = MagicMock()
        mock_method.delivery_tag = 123
        mock_properties = MagicMock()

        # Invalid JSON body
        body = b"invalid json"

        # Process message
        self.consumer.process_message(mock_channel, mock_method, mock_properties, body)

        # Verify message was rejected without requeue
        mock_channel.basic_nack.assert_called_once_with(delivery_tag=123, requeue=False)

    def test_clear_cache_entries_no_hash(self) -> None:
        """Test cache clearing when file hash cannot be generated."""
        # Mock cache with failing hash generation
        mock_cache = MagicMock()
        mock_cache.redis_client = MagicMock()
        mock_cache._generate_file_hash.return_value = None
        self.consumer.cache = mock_cache

        # Clear cache entries
        self.consumer.clear_cache_entries("/path/to/file.mp3", "test-correlation-id")

        # Verify no Redis operations were performed
        mock_cache.redis_client.delete.assert_not_called()

    def test_remove_neo4j_data_no_repo(self) -> None:
        """Test Neo4j removal when repository is not initialized."""
        # Set Neo4j repo to None
        assert self.consumer.storage_handler is not None  # For mypy
        self.consumer.storage_handler.neo4j_repo = None

        # This should not raise an exception
        self.consumer.remove_neo4j_data("/path/to/file.mp3", "test-correlation-id")

    def test_remove_neo4j_data_exception(self) -> None:
        """Test Neo4j removal error handling."""
        # Mock Neo4j repo that raises exception
        mock_neo4j_repo = MagicMock()
        mock_neo4j_repo.delete_recording_by_filepath.side_effect = Exception("DB error")
        assert self.consumer.storage_handler is not None  # For mypy
        self.consumer.storage_handler.neo4j_repo = mock_neo4j_repo

        # This should not raise an exception (error is logged)
        self.consumer.remove_neo4j_data("/path/to/file.mp3", "test-correlation-id")


if __name__ == "__main__":
    unittest.main()
