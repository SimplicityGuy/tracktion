"""Unit tests for cataloging service message consumer."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aio_pika import IncomingMessage

from services.cataloging_service.src.config import Config, DatabaseConfig, RabbitMQConfig, ServiceConfig
from services.cataloging_service.src.message_consumer import CatalogingMessageConsumer


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    return Config(
        database=DatabaseConfig(
            host="localhost",
            port=5432,
            name="test_db",
            user="test_user",
            password="test_pass",
        ),
        rabbitmq=RabbitMQConfig(
            host="localhost",
            port=5672,
            username="guest",
            password="guest",
            exchange="file_events",
            queue="cataloging.test",
        ),
        service=ServiceConfig(
            soft_delete_enabled=True,
            cleanup_interval_days=30,
            log_level="INFO",
        ),
    )


@pytest.fixture
def consumer(mock_config):
    """Create cataloging message consumer."""
    with patch("services.cataloging_service.src.message_consumer.get_config", return_value=mock_config):
        with patch("services.cataloging_service.src.message_consumer.create_async_engine"):
            return CatalogingMessageConsumer()


class TestCatalogingMessageConsumer:
    """Test CatalogingMessageConsumer class."""

    @pytest.mark.asyncio
    async def test_connect(self, consumer):
        """Test connecting to RabbitMQ."""
        with patch("aio_pika.connect_robust") as mock_connect:
            mock_connection = AsyncMock()
            mock_channel = AsyncMock()
            mock_exchange = AsyncMock()
            mock_queue = AsyncMock()

            mock_connection.channel.return_value = mock_channel
            mock_channel.declare_exchange.return_value = mock_exchange
            mock_channel.declare_queue.return_value = mock_queue
            mock_connect.return_value = mock_connection

            await consumer.connect()

            # Verify connection was established
            mock_connect.assert_called_once()
            mock_channel.set_qos.assert_called_once_with(prefetch_count=10)
            mock_channel.declare_exchange.assert_called_once()
            mock_channel.declare_queue.assert_called_once()

            # Verify queue bindings
            assert mock_queue.bind.call_count == 5  # 5 routing keys

    @pytest.mark.asyncio
    async def test_disconnect(self, consumer):
        """Test disconnecting from RabbitMQ."""
        mock_connection = AsyncMock()
        mock_connection.is_closed = False
        consumer.connection = mock_connection

        await consumer.disconnect()

        mock_connection.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_message_created(self, consumer):
        """Test processing a file created message."""
        # Create mock message
        message_body = {
            "event_type": "created",
            "file_path": "/music/test.mp3",
            "sha256_hash": "abc123",
            "xxh128_hash": "def456",
            "size_bytes": "1024",
            "correlation_id": "test-123",
        }

        mock_message = MagicMock(spec=IncomingMessage)
        mock_message.body = json.dumps(message_body).encode()

        # Mock session and lifecycle service
        mock_session = AsyncMock()
        consumer.SessionLocal = MagicMock(return_value=mock_session)

        with patch("services.cataloging_service.src.message_consumer.FileLifecycleService") as mock_service_class:
            mock_service = AsyncMock()
            mock_service.handle_file_created.return_value = (True, None)
            mock_service_class.return_value = mock_service

            await consumer.process_message(mock_message)

            # Verify service was called
            mock_service.handle_file_created.assert_called_once_with("/music/test.mp3", "abc123", "def456", 1024)

    @pytest.mark.asyncio
    async def test_process_message_modified(self, consumer):
        """Test processing a file modified message."""
        message_body = {
            "event_type": "modified",
            "file_path": "/music/test.mp3",
            "sha256_hash": "newhash",
            "xxh128_hash": "newhash2",
            "size_bytes": "2048",
            "correlation_id": "test-124",
        }

        mock_message = MagicMock(spec=IncomingMessage)
        mock_message.body = json.dumps(message_body).encode()

        mock_session = AsyncMock()
        consumer.SessionLocal = MagicMock(return_value=mock_session)

        with patch("services.cataloging_service.src.message_consumer.FileLifecycleService") as mock_service_class:
            mock_service = AsyncMock()
            mock_service.handle_file_modified.return_value = (True, None)
            mock_service_class.return_value = mock_service

            await consumer.process_message(mock_message)

            mock_service.handle_file_modified.assert_called_once_with("/music/test.mp3", "newhash", "newhash2", 2048)

    @pytest.mark.asyncio
    async def test_process_message_deleted(self, consumer):
        """Test processing a file deleted message."""
        message_body = {
            "event_type": "deleted",
            "file_path": "/music/test.mp3",
            "correlation_id": "test-125",
        }

        mock_message = MagicMock(spec=IncomingMessage)
        mock_message.body = json.dumps(message_body).encode()

        mock_session = AsyncMock()
        consumer.SessionLocal = MagicMock(return_value=mock_session)

        with patch("services.cataloging_service.src.message_consumer.FileLifecycleService") as mock_service_class:
            mock_service = AsyncMock()
            mock_service.handle_file_deleted.return_value = (True, None)
            mock_service_class.return_value = mock_service

            await consumer.process_message(mock_message)

            # Should use soft delete by default
            mock_service.handle_file_deleted.assert_called_once_with("/music/test.mp3", soft_delete=True)

    @pytest.mark.asyncio
    async def test_process_message_moved(self, consumer):
        """Test processing a file moved message."""
        message_body = {
            "event_type": "moved",
            "file_path": "/music/new/test.mp3",
            "old_path": "/music/old/test.mp3",
            "sha256_hash": "abc123",
            "xxh128_hash": "def456",
            "correlation_id": "test-126",
        }

        mock_message = MagicMock(spec=IncomingMessage)
        mock_message.body = json.dumps(message_body).encode()

        mock_session = AsyncMock()
        consumer.SessionLocal = MagicMock(return_value=mock_session)

        with patch("services.cataloging_service.src.message_consumer.FileLifecycleService") as mock_service_class:
            mock_service = AsyncMock()
            mock_service.handle_file_moved.return_value = (True, None)
            mock_service_class.return_value = mock_service

            await consumer.process_message(mock_message)

            mock_service.handle_file_moved.assert_called_once_with(
                "/music/old/test.mp3", "/music/new/test.mp3", "abc123", "def456"
            )

    @pytest.mark.asyncio
    async def test_process_message_renamed(self, consumer):
        """Test processing a file renamed message."""
        message_body = {
            "event_type": "renamed",
            "file_path": "/music/renamed.mp3",
            "old_path": "/music/test.mp3",
            "sha256_hash": "abc123",
            "xxh128_hash": "def456",
            "correlation_id": "test-127",
        }

        mock_message = MagicMock(spec=IncomingMessage)
        mock_message.body = json.dumps(message_body).encode()

        mock_session = AsyncMock()
        consumer.SessionLocal = MagicMock(return_value=mock_session)

        with patch("services.cataloging_service.src.message_consumer.FileLifecycleService") as mock_service_class:
            mock_service = AsyncMock()
            mock_service.handle_file_renamed.return_value = (True, None)
            mock_service_class.return_value = mock_service

            await consumer.process_message(mock_message)

            mock_service.handle_file_renamed.assert_called_once_with(
                "/music/test.mp3", "/music/renamed.mp3", "abc123", "def456"
            )

    @pytest.mark.asyncio
    async def test_process_message_unknown_event(self, consumer):
        """Test processing an unknown event type."""
        message_body = {
            "event_type": "unknown",
            "file_path": "/music/test.mp3",
            "correlation_id": "test-128",
        }

        mock_message = MagicMock(spec=IncomingMessage)
        mock_message.body = json.dumps(message_body).encode()

        mock_session = AsyncMock()
        consumer.SessionLocal = MagicMock(return_value=mock_session)

        with patch("services.cataloging_service.src.message_consumer.FileLifecycleService") as mock_service_class:
            mock_service = AsyncMock()
            mock_service_class.return_value = mock_service

            # Should not raise an exception for unknown event
            await consumer.process_message(mock_message)

            # No service methods should be called
            mock_service.handle_file_created.assert_not_called()
            mock_service.handle_file_modified.assert_not_called()
            mock_service.handle_file_deleted.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_message_invalid_json(self, consumer):
        """Test processing a message with invalid JSON."""
        mock_message = MagicMock(spec=IncomingMessage)
        mock_message.body = b"invalid json"

        # Should handle JSON decode error gracefully
        await consumer.process_message(mock_message)

    @pytest.mark.asyncio
    async def test_cleanup_old_deletes(self, consumer):
        """Test cleanup of old soft-deleted records."""
        mock_session = AsyncMock()
        consumer.SessionLocal = MagicMock(return_value=mock_session)

        with patch("services.cataloging_service.src.message_consumer.FileLifecycleService") as mock_service_class:
            mock_service = AsyncMock()
            mock_service.cleanup_old_soft_deletes.return_value = 5
            mock_service_class.return_value = mock_service

            await consumer.cleanup_old_deletes()

            mock_service.cleanup_old_soft_deletes.assert_called_once_with(30)
