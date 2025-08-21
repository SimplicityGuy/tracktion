"""Unit tests for the message publisher module."""

import json
import uuid
from unittest.mock import MagicMock, patch

import pytest

from services.file_watcher.src.message_publisher import MessagePublisher


class TestMessagePublisher:
    """Tests for MessagePublisher class."""

    @patch("services.file_watcher.src.message_publisher.pika.BlockingConnection")
    def test_connect_success(self, mock_connection_class):
        """Test successful connection to RabbitMQ."""
        mock_connection = MagicMock()
        mock_channel = MagicMock()
        mock_connection.channel.return_value = mock_channel
        mock_connection_class.return_value = mock_connection

        publisher = MessagePublisher()
        publisher.connect()

        # Verify connection was established
        mock_connection_class.assert_called_once()
        mock_connection.channel.assert_called_once()

        # Verify exchange was declared
        mock_channel.exchange_declare.assert_called_once_with(
            exchange="file_events", exchange_type="topic", durable=True
        )

    @patch("services.file_watcher.src.message_publisher.pika.BlockingConnection")
    def test_connect_failure(self, mock_connection_class):
        """Test handling of connection failure."""
        mock_connection_class.side_effect = Exception("Connection failed")

        publisher = MessagePublisher()

        with pytest.raises(Exception) as exc_info:
            publisher.connect()
        assert "Connection failed" in str(exc_info.value)

    @patch("services.file_watcher.src.message_publisher.pika.BlockingConnection")
    def test_disconnect(self, mock_connection_class):
        """Test disconnection from RabbitMQ."""
        mock_connection = MagicMock()
        mock_connection.is_closed = False
        mock_connection_class.return_value = mock_connection

        publisher = MessagePublisher()
        publisher.connect()
        publisher.disconnect()

        mock_connection.close.assert_called_once()

    def test_determine_file_type_ogg(self):
        """Test file type determination for OGG files."""
        publisher = MessagePublisher()

        assert publisher._determine_file_type(".ogg") == "ogg"
        assert publisher._determine_file_type(".OGG") == "ogg"
        assert publisher._determine_file_type(".oga") == "ogg"
        assert publisher._determine_file_type(".OGA") == "ogg"

    def test_determine_file_type_other_formats(self):
        """Test file type determination for other formats."""
        publisher = MessagePublisher()

        assert publisher._determine_file_type(".mp3") == "mp3"
        assert publisher._determine_file_type(".flac") == "flac"
        assert publisher._determine_file_type(".wav") == "wav"
        assert publisher._determine_file_type(".m4a") == "mp4"
        assert publisher._determine_file_type(".unknown") == "unknown"

    @patch("services.file_watcher.src.message_publisher.pika.BlockingConnection")
    @patch("services.file_watcher.src.message_publisher.uuid.uuid4")
    @patch("services.file_watcher.src.message_publisher.datetime")
    def test_publish_ogg_file_discovered(self, mock_datetime, mock_uuid, mock_connection_class):
        """Test publishing OGG file discovery event."""
        # Setup mocks
        mock_uuid.return_value = uuid.UUID("12345678-1234-5678-1234-567812345678")
        mock_datetime.now.return_value.isoformat.return_value = "2024-01-01T00:00:00"

        mock_connection = MagicMock()
        mock_channel = MagicMock()
        mock_connection.channel.return_value = mock_channel
        mock_connection.is_closed = False
        mock_connection_class.return_value = mock_connection

        publisher = MessagePublisher()
        publisher.connect()

        # Test data for OGG file
        file_info = {
            "path": "/music/test.ogg",
            "name": "test.ogg",
            "extension": ".ogg",
            "size_bytes": "1024",
            "modified_time": "1234567890",
            "hash": "abc123",
        }

        result = publisher.publish_file_discovered(file_info)

        assert result is True

        # Verify message was published
        mock_channel.basic_publish.assert_called_once()
        call_args = mock_channel.basic_publish.call_args

        # Check exchange and routing key
        assert call_args.kwargs["exchange"] == "file_events"
        assert call_args.kwargs["routing_key"] == "file.discovered"

        # Check message body
        body = json.loads(call_args.kwargs["body"])
        assert body["correlation_id"] == "12345678-1234-5678-1234-567812345678"
        assert body["event_type"] == "file_discovered"
        assert body["file_info"] == file_info
        assert body["file_type"] == "ogg"
        assert body["format_family"] == "ogg_vorbis"

        # Check message properties
        props = call_args.kwargs["properties"]
        assert props.delivery_mode == 2  # Persistent
        assert props.correlation_id == "12345678-1234-5678-1234-567812345678"
        assert props.content_type == "application/json"

    @patch("services.file_watcher.src.message_publisher.pika.BlockingConnection")
    @patch("services.file_watcher.src.message_publisher.uuid.uuid4")
    def test_publish_oga_file_discovered(self, mock_uuid, mock_connection_class):
        """Test publishing OGA file discovery event."""
        mock_uuid.return_value = uuid.UUID("12345678-1234-5678-1234-567812345678")

        mock_connection = MagicMock()
        mock_channel = MagicMock()
        mock_connection.channel.return_value = mock_channel
        mock_connection.is_closed = False
        mock_connection_class.return_value = mock_connection

        publisher = MessagePublisher()
        publisher.connect()

        # Test data for OGA file
        file_info = {
            "path": "/music/test.oga",
            "name": "test.oga",
            "extension": ".oga",
            "size_bytes": "2048",
            "modified_time": "1234567890",
            "hash": "def456",
        }

        result = publisher.publish_file_discovered(file_info)

        assert result is True

        # Check message body for OGA-specific handling
        call_args = mock_channel.basic_publish.call_args
        body = json.loads(call_args.kwargs["body"])
        assert body["file_type"] == "ogg"
        assert body["format_family"] == "ogg_vorbis"

    @patch("services.file_watcher.src.message_publisher.pika.BlockingConnection")
    def test_publish_with_connection_closed(self, mock_connection_class):
        """Test publishing when connection is closed (reconnect)."""
        mock_connection = MagicMock()
        mock_channel = MagicMock()
        mock_connection.channel.return_value = mock_channel
        mock_connection.is_closed = True  # Connection is closed
        mock_connection_class.return_value = mock_connection

        publisher = MessagePublisher()
        publisher.connection = mock_connection
        publisher.channel = mock_channel

        file_info = {"path": "/music/test.ogg", "extension": ".ogg"}

        # Should attempt to reconnect
        publisher.publish_file_discovered(file_info)

        # Verify reconnection attempt
        mock_connection.channel.assert_called()

    @patch("services.file_watcher.src.message_publisher.pika.BlockingConnection")
    @patch("services.file_watcher.src.message_publisher.logger")
    def test_publish_failure(self, mock_logger, mock_connection_class):
        """Test handling of publish failure."""
        mock_connection = MagicMock()
        mock_channel = MagicMock()
        mock_channel.basic_publish.side_effect = Exception("Publish failed")
        mock_connection.channel.return_value = mock_channel
        mock_connection.is_closed = False
        mock_connection_class.return_value = mock_connection

        publisher = MessagePublisher()
        publisher.connect()

        file_info = {"path": "/music/test.ogg", "extension": ".ogg"}

        result = publisher.publish_file_discovered(file_info)

        assert result is False

        # Verify error was logged
        mock_logger.error.assert_called()

    @patch("services.file_watcher.src.message_publisher.pika.BlockingConnection")
    def test_context_manager(self, mock_connection_class):
        """Test using publisher as context manager."""
        mock_connection = MagicMock()
        mock_connection.is_closed = False
        mock_connection_class.return_value = mock_connection

        with MessagePublisher() as publisher:
            # Should be connected
            assert publisher.connection is not None

        # Should be disconnected after exit
        mock_connection.close.assert_called_once()

    @patch("services.file_watcher.src.message_publisher.logger")
    def test_publish_ogg_file_logging(self, mock_logger):
        """Test that OGG files trigger specific logging."""
        with patch("services.file_watcher.src.message_publisher.pika.BlockingConnection"):
            publisher = MessagePublisher()
            publisher.channel = MagicMock()
            publisher.connection = MagicMock()
            publisher.connection.is_closed = False

            file_info = {"path": "/music/test.ogg", "extension": ".ogg"}

            publisher.publish_file_discovered(file_info)

            # Check for OGG-specific log message
            mock_logger.info.assert_any_call(
                "Publishing OGG file discovery event",
                correlation_id=mock_logger.info.call_args_list[-1].kwargs["correlation_id"],
                file_path="/music/test.ogg",
                extension=".ogg",
            )
