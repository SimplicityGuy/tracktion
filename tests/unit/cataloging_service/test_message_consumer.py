"""Comprehensive unit tests for cataloging service message consumers."""

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from aio_pika import IncomingMessage

from services.cataloging_service.src.config import Config, DatabaseConfig, RabbitMQConfig, ServiceConfig
from services.cataloging_service.src.consumers.tracklist_consumer import TracklistMessageConsumer
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
def mock_db_manager():
    """Create mock database manager."""
    mock_manager = MagicMock()
    mock_session = AsyncMock()

    # Create a proper async context manager
    async_context_manager = AsyncMock()
    async_context_manager.__aenter__ = AsyncMock(return_value=mock_session)
    async_context_manager.__aexit__ = AsyncMock(return_value=None)

    mock_manager.get_session.return_value = async_context_manager
    return mock_manager, mock_session


@pytest.fixture
def cataloging_consumer(mock_config, mock_db_manager):
    """Create cataloging message consumer."""
    mock_manager, mock_session = mock_db_manager

    with (
        patch("services.cataloging_service.src.message_consumer.get_config", return_value=mock_config),
        patch("services.cataloging_service.src.message_consumer.get_db_manager", return_value=mock_manager),
    ):
        consumer = CatalogingMessageConsumer()
        consumer.session = mock_session  # Add session for convenience
        return consumer


@pytest.fixture
def tracklist_consumer(mock_config, mock_db_manager):
    """Create tracklist message consumer."""
    mock_manager, mock_session = mock_db_manager

    with (
        patch("services.cataloging_service.src.consumers.tracklist_consumer.get_config", return_value=mock_config),
        patch("services.cataloging_service.src.consumers.tracklist_consumer.get_db_manager", return_value=mock_manager),
    ):
        consumer = TracklistMessageConsumer()
        consumer.session = mock_session  # Add session for convenience
        return consumer


@pytest.fixture
def mock_message():
    """Create mock RabbitMQ message."""
    message = MagicMock(spec=IncomingMessage)
    message.process.return_value.__aenter__ = AsyncMock()
    message.process.return_value.__aexit__ = AsyncMock()
    return message


@pytest.fixture
def mock_repositories():
    """Create mock repositories."""
    recording_repo = AsyncMock()
    metadata_repo = AsyncMock()
    tracklist_repo = AsyncMock()

    return {
        "recording": recording_repo,
        "metadata": metadata_repo,
        "tracklist": tracklist_repo,
    }


class TestCatalogingMessageConsumer:
    """Test CatalogingMessageConsumer class."""

    @pytest.mark.asyncio
    async def test_connect_success(self, cataloging_consumer):
        """Test successful connection to RabbitMQ."""
        with patch("aio_pika.connect_robust") as mock_connect:
            mock_connection = AsyncMock()
            mock_channel = AsyncMock()
            mock_exchange = AsyncMock()
            mock_queue = AsyncMock()

            mock_connection.channel.return_value = mock_channel
            mock_channel.declare_exchange.return_value = mock_exchange
            mock_channel.declare_queue.return_value = mock_queue
            mock_connect.return_value = mock_connection

            await cataloging_consumer.connect()

            # Verify connection establishment
            mock_connect.assert_called_once_with(
                cataloging_consumer.config.rabbitmq.url,
                client_properties={"connection_name": "cataloging-service"},
            )

            # Verify channel setup
            mock_channel.set_qos.assert_called_once_with(prefetch_count=10)
            mock_channel.declare_exchange.assert_called_once_with(
                cataloging_consumer.config.rabbitmq.exchange,
                "topic",  # ExchangeType.TOPIC
                durable=True,
            )

            # Verify queue setup
            mock_channel.declare_queue.assert_called_once_with(
                cataloging_consumer.config.rabbitmq.queue,
                durable=True,
                arguments={
                    "x-dead-letter-exchange": f"{cataloging_consumer.config.rabbitmq.exchange}.dlx",
                    "x-message-ttl": 86400000,
                },
            )

            # Verify all routing keys are bound
            expected_routing_keys = [
                "file.created",
                "file.modified",
                "file.deleted",
                "file.moved",
                "file.renamed",
            ]
            assert mock_queue.bind.call_count == len(expected_routing_keys)

    @pytest.mark.asyncio
    async def test_connect_failure(self, cataloging_consumer):
        """Test connection failure handling."""
        with (
            patch("aio_pika.connect_robust", side_effect=Exception("Connection failed")),
            pytest.raises(Exception, match="Connection failed"),
        ):
            await cataloging_consumer.connect()

    @pytest.mark.asyncio
    async def test_disconnect_success(self, cataloging_consumer):
        """Test successful disconnection from RabbitMQ."""
        mock_connection = AsyncMock()
        mock_connection.is_closed = False
        cataloging_consumer.connection = mock_connection

        await cataloging_consumer.disconnect()

        mock_connection.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_already_closed(self, cataloging_consumer):
        """Test disconnection when connection is already closed."""
        mock_connection = AsyncMock()
        mock_connection.is_closed = True
        cataloging_consumer.connection = mock_connection

        await cataloging_consumer.disconnect()

        mock_connection.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_disconnect_with_error(self, cataloging_consumer):
        """Test disconnection with error."""
        mock_connection = AsyncMock()
        mock_connection.is_closed = False
        mock_connection.close.side_effect = Exception("Close failed")
        cataloging_consumer.connection = mock_connection

        # Should not raise exception, just log the error
        await cataloging_consumer.disconnect()

        mock_connection.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_message_created_new_file(self, cataloging_consumer, mock_message, mock_repositories):
        """Test processing a file created message for new file."""
        message_body = {
            "event_type": "created",
            "file_path": "/music/test.mp3",
            "sha256_hash": "abc123",
            "xxh128_hash": "def456",
            "size_bytes": "1024",
            "correlation_id": "test-123",
            "metadata": {"artist": "Test Artist", "title": "Test Song"},
        }
        mock_message.body = json.dumps(message_body).encode()

        # Mock repository behavior
        recording_repo = mock_repositories["recording"]
        metadata_repo = mock_repositories["metadata"]
        recording_repo.get_by_file_path.return_value = None  # File doesn't exist

        mock_recording = MagicMock()
        mock_recording.id = str(uuid4())
        recording_repo.create.return_value = mock_recording

        with (
            patch("services.cataloging_service.src.message_consumer.RecordingRepository", return_value=recording_repo),
            patch("services.cataloging_service.src.message_consumer.MetadataRepository", return_value=metadata_repo),
        ):
            await cataloging_consumer.process_message(mock_message)

        # Verify recording creation
        recording_repo.create.assert_called_once_with(
            file_path="/music/test.mp3",
            file_name="test.mp3",
            sha256_hash="abc123",
            xxh128_hash="def456",
        )

        # Verify metadata creation
        metadata_repo.bulk_create.assert_called_once_with(
            mock_recording.id, {"artist": "Test Artist", "title": "Test Song"}
        )

    @pytest.mark.asyncio
    async def test_process_message_created_existing_file(self, cataloging_consumer, mock_message, mock_repositories):
        """Test processing a file created message for existing file."""
        message_body = {
            "event_type": "created",
            "file_path": "/music/test.mp3",
            "sha256_hash": "new_hash",
            "xxh128_hash": "new_hash2",
            "correlation_id": "test-124",
        }
        mock_message.body = json.dumps(message_body).encode()

        # Mock repository behavior
        recording_repo = mock_repositories["recording"]
        metadata_repo = mock_repositories["metadata"]

        existing_recording = MagicMock()
        existing_recording.id = str(uuid4())
        recording_repo.get_by_file_path.return_value = existing_recording

        with (
            patch("services.cataloging_service.src.message_consumer.RecordingRepository", return_value=recording_repo),
            patch("services.cataloging_service.src.message_consumer.MetadataRepository", return_value=metadata_repo),
        ):
            await cataloging_consumer.process_message(mock_message)

        # Verify recording update instead of creation
        recording_repo.create.assert_not_called()
        recording_repo.update.assert_called_once_with(
            existing_recording.id,
            sha256_hash="new_hash",
            xxh128_hash="new_hash2",
        )

    @pytest.mark.asyncio
    async def test_process_message_modified_existing_file(self, cataloging_consumer, mock_message, mock_repositories):
        """Test processing a file modified message for existing file."""
        message_body = {
            "event_type": "modified",
            "file_path": "/music/test.mp3",
            "sha256_hash": "updated_hash",
            "xxh128_hash": "updated_hash2",
            "correlation_id": "test-125",
            "metadata": {"last_modified": "2024-01-01"},
        }
        mock_message.body = json.dumps(message_body).encode()

        # Mock repository behavior
        recording_repo = mock_repositories["recording"]
        metadata_repo = mock_repositories["metadata"]

        existing_recording = MagicMock()
        existing_recording.id = str(uuid4())
        recording_repo.get_by_file_path.return_value = existing_recording

        with (
            patch("services.cataloging_service.src.message_consumer.RecordingRepository", return_value=recording_repo),
            patch("services.cataloging_service.src.message_consumer.MetadataRepository", return_value=metadata_repo),
        ):
            await cataloging_consumer.process_message(mock_message)

        # Verify recording update
        recording_repo.update.assert_called_once_with(
            existing_recording.id,
            sha256_hash="updated_hash",
            xxh128_hash="updated_hash2",
        )

        # Verify metadata upsert
        metadata_repo.upsert.assert_called_once_with(existing_recording.id, "last_modified", "2024-01-01")

    @pytest.mark.asyncio
    async def test_process_message_modified_nonexistent_file(
        self, cataloging_consumer, mock_message, mock_repositories
    ):
        """Test processing a file modified message for nonexistent file."""
        message_body = {
            "event_type": "modified",
            "file_path": "/music/test.mp3",
            "sha256_hash": "new_hash",
            "xxh128_hash": "new_hash2",
            "correlation_id": "test-126",
        }
        mock_message.body = json.dumps(message_body).encode()

        # Mock repository behavior
        recording_repo = mock_repositories["recording"]
        metadata_repo = mock_repositories["metadata"]
        recording_repo.get_by_file_path.return_value = None  # File doesn't exist

        mock_recording = MagicMock()
        mock_recording.id = str(uuid4())
        recording_repo.create.return_value = mock_recording

        with (
            patch("services.cataloging_service.src.message_consumer.RecordingRepository", return_value=recording_repo),
            patch("services.cataloging_service.src.message_consumer.MetadataRepository", return_value=metadata_repo),
        ):
            await cataloging_consumer.process_message(mock_message)

        # Should create new record
        recording_repo.create.assert_called_once_with(
            file_path="/music/test.mp3",
            file_name="test.mp3",
            sha256_hash="new_hash",
            xxh128_hash="new_hash2",
        )

    @pytest.mark.asyncio
    async def test_process_message_deleted(self, cataloging_consumer, mock_message, mock_repositories):
        """Test processing a file deleted message."""
        message_body = {
            "event_type": "deleted",
            "file_path": "/music/test.mp3",
            "correlation_id": "test-127",
        }
        mock_message.body = json.dumps(message_body).encode()

        # Mock repository behavior
        recording_repo = mock_repositories["recording"]
        metadata_repo = mock_repositories["metadata"]

        existing_recording = MagicMock()
        existing_recording.id = str(uuid4())
        recording_repo.get_by_file_path.return_value = existing_recording

        with (
            patch("services.cataloging_service.src.message_consumer.RecordingRepository", return_value=recording_repo),
            patch("services.cataloging_service.src.message_consumer.MetadataRepository", return_value=metadata_repo),
        ):
            await cataloging_consumer.process_message(mock_message)

        # Verify recording deletion
        recording_repo.delete.assert_called_once_with(existing_recording.id)

    @pytest.mark.asyncio
    async def test_process_message_deleted_nonexistent(self, cataloging_consumer, mock_message, mock_repositories):
        """Test processing a file deleted message for nonexistent file."""
        message_body = {
            "event_type": "deleted",
            "file_path": "/music/nonexistent.mp3",
            "correlation_id": "test-128",
        }
        mock_message.body = json.dumps(message_body).encode()

        # Mock repository behavior
        recording_repo = mock_repositories["recording"]
        metadata_repo = mock_repositories["metadata"]
        recording_repo.get_by_file_path.return_value = None  # File doesn't exist

        with (
            patch("services.cataloging_service.src.message_consumer.RecordingRepository", return_value=recording_repo),
            patch("services.cataloging_service.src.message_consumer.MetadataRepository", return_value=metadata_repo),
        ):
            await cataloging_consumer.process_message(mock_message)

        # Should not attempt to delete
        recording_repo.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_message_moved(self, cataloging_consumer, mock_message, mock_repositories):
        """Test processing a file moved message."""
        message_body = {
            "event_type": "moved",
            "file_path": "/music/new/test.mp3",
            "old_path": "/music/old/test.mp3",
            "sha256_hash": "abc123",
            "xxh128_hash": "def456",
            "correlation_id": "test-129",
        }
        mock_message.body = json.dumps(message_body).encode()

        # Mock repository behavior
        recording_repo = mock_repositories["recording"]
        metadata_repo = mock_repositories["metadata"]

        existing_recording = MagicMock()
        existing_recording.id = str(uuid4())
        recording_repo.get_by_file_path.return_value = existing_recording

        with (
            patch("services.cataloging_service.src.message_consumer.RecordingRepository", return_value=recording_repo),
            patch("services.cataloging_service.src.message_consumer.MetadataRepository", return_value=metadata_repo),
        ):
            await cataloging_consumer.process_message(mock_message)

        # Verify recording update
        recording_repo.update.assert_called_once_with(
            existing_recording.id,
            file_path="/music/new/test.mp3",
            file_name="test.mp3",
        )

    @pytest.mark.asyncio
    async def test_process_message_moved_missing_old_path(self, cataloging_consumer, mock_message, mock_repositories):
        """Test processing a file moved message with missing old_path."""
        message_body = {
            "event_type": "moved",
            "file_path": "/music/new/test.mp3",
            "correlation_id": "test-130",
        }
        mock_message.body = json.dumps(message_body).encode()

        # Mock repository behavior
        recording_repo = mock_repositories["recording"]
        metadata_repo = mock_repositories["metadata"]

        with (
            patch("services.cataloging_service.src.message_consumer.RecordingRepository", return_value=recording_repo),
            patch("services.cataloging_service.src.message_consumer.MetadataRepository", return_value=metadata_repo),
            pytest.raises(ValueError, match="Missing old_path for moved event"),
        ):
            await cataloging_consumer.process_message(mock_message)

    @pytest.mark.asyncio
    async def test_process_message_renamed(self, cataloging_consumer, mock_message, mock_repositories):
        """Test processing a file renamed message."""
        message_body = {
            "event_type": "renamed",
            "file_path": "/music/renamed.mp3",
            "old_path": "/music/test.mp3",
            "correlation_id": "test-131",
        }
        mock_message.body = json.dumps(message_body).encode()

        # Mock repository behavior
        recording_repo = mock_repositories["recording"]
        metadata_repo = mock_repositories["metadata"]

        existing_recording = MagicMock()
        existing_recording.id = str(uuid4())
        recording_repo.get_by_file_path.return_value = existing_recording

        with (
            patch("services.cataloging_service.src.message_consumer.RecordingRepository", return_value=recording_repo),
            patch("services.cataloging_service.src.message_consumer.MetadataRepository", return_value=metadata_repo),
        ):
            await cataloging_consumer.process_message(mock_message)

        # Verify recording update
        recording_repo.update.assert_called_once_with(
            existing_recording.id,
            file_path="/music/renamed.mp3",
            file_name="renamed.mp3",
        )

    @pytest.mark.asyncio
    async def test_process_message_renamed_missing_old_path(self, cataloging_consumer, mock_message, mock_repositories):
        """Test processing a file renamed message with missing old_path."""
        message_body = {
            "event_type": "renamed",
            "file_path": "/music/renamed.mp3",
            "correlation_id": "test-132",
        }
        mock_message.body = json.dumps(message_body).encode()

        # Mock repository behavior
        recording_repo = mock_repositories["recording"]
        metadata_repo = mock_repositories["metadata"]

        with (
            patch("services.cataloging_service.src.message_consumer.RecordingRepository", return_value=recording_repo),
            patch("services.cataloging_service.src.message_consumer.MetadataRepository", return_value=metadata_repo),
            pytest.raises(ValueError, match="Missing old_path for renamed event"),
        ):
            await cataloging_consumer.process_message(mock_message)

    @pytest.mark.asyncio
    async def test_process_message_unknown_event_type(self, cataloging_consumer, mock_message, mock_repositories):
        """Test processing an unknown event type."""
        message_body = {
            "event_type": "unknown_event",
            "file_path": "/music/test.mp3",
            "correlation_id": "test-133",
        }
        mock_message.body = json.dumps(message_body).encode()

        # Mock repository behavior
        recording_repo = mock_repositories["recording"]
        metadata_repo = mock_repositories["metadata"]

        with (
            patch("services.cataloging_service.src.message_consumer.RecordingRepository", return_value=recording_repo),
            patch("services.cataloging_service.src.message_consumer.MetadataRepository", return_value=metadata_repo),
        ):
            # Should not raise an exception, just log warning
            await cataloging_consumer.process_message(mock_message)

        # No repository methods should be called
        recording_repo.create.assert_not_called()
        recording_repo.update.assert_not_called()
        recording_repo.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_message_invalid_json(self, cataloging_consumer, mock_message):
        """Test processing a message with invalid JSON."""
        mock_message.body = b"invalid json content"

        # Should handle JSON decode error gracefully
        await cataloging_consumer.process_message(mock_message)

    @pytest.mark.asyncio
    async def test_process_message_repository_error(self, cataloging_consumer, mock_message, mock_repositories):
        """Test processing a message when repository raises an error."""
        message_body = {
            "event_type": "created",
            "file_path": "/music/test.mp3",
            "sha256_hash": "abc123",
            "xxh128_hash": "def456",
            "correlation_id": "test-134",
        }
        mock_message.body = json.dumps(message_body).encode()

        # Mock repository to raise an error
        recording_repo = mock_repositories["recording"]
        metadata_repo = mock_repositories["metadata"]
        recording_repo.get_by_file_path.side_effect = Exception("Database error")

        with (
            patch("services.cataloging_service.src.message_consumer.RecordingRepository", return_value=recording_repo),
            patch("services.cataloging_service.src.message_consumer.MetadataRepository", return_value=metadata_repo),
            pytest.raises(Exception, match="Database error"),
        ):
            # Should re-raise the exception for message requeuing
            await cataloging_consumer.process_message(mock_message)

    @pytest.mark.asyncio
    async def test_process_message_size_string_conversion(self, cataloging_consumer, mock_message, mock_repositories):
        """Test processing message with size_bytes as string."""
        message_body = {
            "event_type": "created",
            "file_path": "/music/test.mp3",
            "sha256_hash": "abc123",
            "xxh128_hash": "def456",
            "size_bytes": "2048",  # String instead of int
            "correlation_id": "test-135",
        }
        mock_message.body = json.dumps(message_body).encode()

        recording_repo = mock_repositories["recording"]
        metadata_repo = mock_repositories["metadata"]
        recording_repo.get_by_file_path.return_value = None

        mock_recording = MagicMock()
        mock_recording.id = str(uuid4())
        recording_repo.create.return_value = mock_recording

        with (
            patch("services.cataloging_service.src.message_consumer.RecordingRepository", return_value=recording_repo),
            patch("services.cataloging_service.src.message_consumer.MetadataRepository", return_value=metadata_repo),
        ):
            await cataloging_consumer.process_message(mock_message)

        # Should still work with string size conversion
        recording_repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_old_deletes(self, cataloging_consumer):
        """Test cleanup old deletes method."""
        # This is currently a placeholder that doesn't implement anything
        await cataloging_consumer.cleanup_old_deletes()
        # Just verify it doesn't raise an exception


class TestTracklistMessageConsumer:
    """Test TracklistMessageConsumer class."""

    @pytest.mark.asyncio
    async def test_connect_success(self, tracklist_consumer):
        """Test successful connection to RabbitMQ."""
        with patch("aio_pika.connect_robust") as mock_connect:
            mock_connection = AsyncMock()
            mock_channel = AsyncMock()
            mock_exchange = AsyncMock()
            mock_queue = AsyncMock()

            mock_connection.channel.return_value = mock_channel
            mock_channel.declare_exchange.return_value = mock_exchange
            mock_channel.declare_queue.return_value = mock_queue
            mock_connect.return_value = mock_connection

            await tracklist_consumer.connect()

            # Verify connection establishment
            mock_connect.assert_called_once_with(
                tracklist_consumer.config.rabbitmq.url,
                client_properties={"connection_name": "cataloging-tracklist-consumer"},
            )

            # Verify exchange declaration for tracklist events
            mock_channel.declare_exchange.assert_called_once_with(
                "tracklist_events",
                "topic",  # ExchangeType.TOPIC
                durable=True,
            )

            # Verify queue setup
            mock_channel.declare_queue.assert_called_once_with(
                "cataloging.tracklist.events",
                durable=True,
                arguments={
                    "x-dead-letter-exchange": "tracklist_events.dlx",
                    "x-message-ttl": 86400000,
                },
            )

            # Verify routing key bindings
            expected_routing_keys = ["tracklist.generated", "metadata.extracted"]
            assert mock_queue.bind.call_count == len(expected_routing_keys)

    @pytest.mark.asyncio
    async def test_connect_failure(self, tracklist_consumer):
        """Test connection failure handling."""
        with (
            patch("aio_pika.connect_robust", side_effect=Exception("Connection failed")),
            pytest.raises(Exception, match="Connection failed"),
        ):
            await tracklist_consumer.connect()

    @pytest.mark.asyncio
    async def test_disconnect_success(self, tracklist_consumer):
        """Test successful disconnection from RabbitMQ."""
        mock_connection = AsyncMock()
        mock_connection.is_closed = False
        tracklist_consumer.connection = mock_connection

        await tracklist_consumer.disconnect()

        mock_connection.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_message_tracklist_generated(self, tracklist_consumer, mock_message, mock_repositories):
        """Test processing a tracklist generated message."""
        message_body = {
            "event_type": "tracklist.generated",
            "file_path": "/music/test.mp3",
            "source": "cue_file",
            "tracks": [
                {"title": "Track 1", "start_time": "00:00"},
                {"title": "Track 2", "start_time": "03:30"},
            ],
            "cue_file_path": "/music/test.cue",
            "correlation_id": "test-200",
        }
        mock_message.body = json.dumps(message_body).encode()

        # Mock repository behavior
        recording_repo = mock_repositories["recording"]
        tracklist_repo = mock_repositories["tracklist"]

        mock_recording = MagicMock()
        mock_recording.id = str(uuid4())
        recording_repo.get_by_file_path.return_value = mock_recording

        with (
            patch(
                "services.cataloging_service.src.consumers.tracklist_consumer.RecordingRepository",
                return_value=recording_repo,
            ),
            patch(
                "services.cataloging_service.src.consumers.tracklist_consumer.TracklistRepository",
                return_value=tracklist_repo,
            ),
        ):
            await tracklist_consumer.process_message(mock_message)

        # Verify tracklist upsert
        tracklist_repo.upsert.assert_called_once_with(
            recording_id=mock_recording.id,
            source="cue_file",
            tracks=[
                {"title": "Track 1", "start_time": "00:00"},
                {"title": "Track 2", "start_time": "03:30"},
            ],
            cue_file_path="/music/test.cue",
        )

    @pytest.mark.asyncio
    async def test_process_message_tracklist_generated_recording_not_found(
        self, tracklist_consumer, mock_message, mock_repositories
    ):
        """Test processing tracklist generated when recording doesn't exist."""
        message_body = {
            "event_type": "tracklist.generated",
            "file_path": "/music/nonexistent.mp3",
            "source": "generated",
            "tracks": [],
            "correlation_id": "test-201",
        }
        mock_message.body = json.dumps(message_body).encode()

        # Mock repository behavior
        recording_repo = mock_repositories["recording"]
        tracklist_repo = mock_repositories["tracklist"]
        recording_repo.get_by_file_path.return_value = None  # Recording not found

        with (
            patch(
                "services.cataloging_service.src.consumers.tracklist_consumer.RecordingRepository",
                return_value=recording_repo,
            ),
            patch(
                "services.cataloging_service.src.consumers.tracklist_consumer.TracklistRepository",
                return_value=tracklist_repo,
            ),
        ):
            # Should not raise exception, just log warning
            await tracklist_consumer.process_message(mock_message)

        # Should not attempt tracklist upsert
        tracklist_repo.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_message_metadata_extracted(self, tracklist_consumer, mock_message, mock_repositories):
        """Test processing a metadata extracted message."""
        message_body = {
            "event_type": "metadata.extracted",
            "file_path": "/music/test.mp3",
            "metadata": {
                "artist": "Test Artist",
                "album": "Test Album",
                "year": 2024,
                "genre": "Electronic",
            },
            "correlation_id": "test-202",
        }
        mock_message.body = json.dumps(message_body).encode()

        # Mock repository behavior
        recording_repo = mock_repositories["recording"]

        mock_recording = MagicMock()
        mock_recording.id = str(uuid4())
        recording_repo.get_by_file_path.return_value = mock_recording

        # Mock metadata repo
        metadata_repo = AsyncMock()

        with (
            patch(
                "services.cataloging_service.src.consumers.tracklist_consumer.RecordingRepository",
                return_value=recording_repo,
            ),
            patch("services.cataloging_service.src.repositories.MetadataRepository", return_value=metadata_repo),
        ):
            await tracklist_consumer.process_message(mock_message)

        # Verify metadata upserts for each key-value pair
        expected_calls = [
            (mock_recording.id, "artist", "Test Artist"),
            (mock_recording.id, "album", "Test Album"),
            (mock_recording.id, "year", "2024"),  # Converted to string
            (mock_recording.id, "genre", "Electronic"),
        ]

        assert metadata_repo.upsert.call_count == len(expected_calls)
        for call_args in metadata_repo.upsert.call_args_list:
            assert call_args[0] in expected_calls

    @pytest.mark.asyncio
    async def test_process_message_metadata_extracted_recording_not_found(
        self, tracklist_consumer, mock_message, mock_repositories
    ):
        """Test processing metadata extracted when recording doesn't exist."""
        message_body = {
            "event_type": "metadata.extracted",
            "file_path": "/music/nonexistent.mp3",
            "metadata": {"artist": "Test Artist"},
            "correlation_id": "test-203",
        }
        mock_message.body = json.dumps(message_body).encode()

        # Mock repository behavior
        recording_repo = mock_repositories["recording"]
        recording_repo.get_by_file_path.return_value = None  # Recording not found

        metadata_repo = AsyncMock()

        with (
            patch(
                "services.cataloging_service.src.consumers.tracklist_consumer.RecordingRepository",
                return_value=recording_repo,
            ),
            patch("services.cataloging_service.src.repositories.MetadataRepository", return_value=metadata_repo),
        ):
            # Should not raise exception, just log warning
            await tracklist_consumer.process_message(mock_message)

        # Should not attempt metadata upsert
        metadata_repo.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_message_unknown_event_type(self, tracklist_consumer, mock_message, mock_repositories):
        """Test processing an unknown event type."""
        message_body = {
            "event_type": "unknown.event",
            "file_path": "/music/test.mp3",
            "correlation_id": "test-204",
        }
        mock_message.body = json.dumps(message_body).encode()

        # Mock repository behavior
        recording_repo = mock_repositories["recording"]
        tracklist_repo = mock_repositories["tracklist"]

        with (
            patch(
                "services.cataloging_service.src.consumers.tracklist_consumer.RecordingRepository",
                return_value=recording_repo,
            ),
            patch(
                "services.cataloging_service.src.consumers.tracklist_consumer.TracklistRepository",
                return_value=tracklist_repo,
            ),
        ):
            # Should not raise an exception, just log warning
            await tracklist_consumer.process_message(mock_message)

        # No repository methods should be called
        recording_repo.get_by_file_path.assert_not_called()
        tracklist_repo.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_message_invalid_json(self, tracklist_consumer, mock_message):
        """Test processing a message with invalid JSON."""
        mock_message.body = b"invalid json content"

        # Should raise JSON decode error for requeuing
        with pytest.raises(json.JSONDecodeError):
            await tracklist_consumer.process_message(mock_message)

    @pytest.mark.asyncio
    async def test_process_message_repository_error(self, tracklist_consumer, mock_message, mock_repositories):
        """Test processing a message when repository raises an error."""
        message_body = {
            "event_type": "tracklist.generated",
            "file_path": "/music/test.mp3",
            "source": "generated",
            "tracks": [],
            "correlation_id": "test-205",
        }
        mock_message.body = json.dumps(message_body).encode()

        # Mock repository to raise an error
        recording_repo = mock_repositories["recording"]
        tracklist_repo = mock_repositories["tracklist"]
        recording_repo.get_by_file_path.side_effect = Exception("Database error")

        with (
            patch(
                "services.cataloging_service.src.consumers.tracklist_consumer.RecordingRepository",
                return_value=recording_repo,
            ),
            patch(
                "services.cataloging_service.src.consumers.tracklist_consumer.TracklistRepository",
                return_value=tracklist_repo,
            ),
            pytest.raises(Exception, match="Database error"),
        ):
            # Should re-raise the exception for message requeuing
            await tracklist_consumer.process_message(mock_message)


class TestMessageConsumerIntegration:
    """Integration tests for message consumer interactions."""

    @pytest.mark.asyncio
    async def test_consumers_can_be_created_simultaneously(self, mock_config, mock_db_manager):
        """Test that both consumers can be created without conflicts."""
        mock_manager, mock_session = mock_db_manager

        with (
            patch("services.cataloging_service.src.message_consumer.get_config", return_value=mock_config),
            patch("services.cataloging_service.src.message_consumer.get_db_manager", return_value=mock_manager),
            patch("services.cataloging_service.src.consumers.tracklist_consumer.get_config", return_value=mock_config),
            patch(
                "services.cataloging_service.src.consumers.tracklist_consumer.get_db_manager", return_value=mock_manager
            ),
        ):
            cataloging_consumer = CatalogingMessageConsumer()
            tracklist_consumer = TracklistMessageConsumer()

            # Both should be created successfully with different configurations
            assert cataloging_consumer.config.rabbitmq.exchange == "file_events"
            assert (
                tracklist_consumer.config.rabbitmq.exchange == "file_events"
            )  # Uses same config but different hardcoded exchange

    @pytest.mark.asyncio
    async def test_error_recovery_patterns(self, cataloging_consumer, mock_message):
        """Test that errors are handled correctly for message requeuing."""
        mock_message.body = b"invalid json"

        # JSON errors should be handled gracefully (logged but not re-raised)
        await cataloging_consumer.process_message(mock_message)

        # Database errors should be re-raised for requeuing
        mock_message.body = json.dumps({"event_type": "created", "file_path": "/test"}).encode()

        with patch("services.cataloging_service.src.message_consumer.RecordingRepository") as mock_repo_class:
            mock_repo = AsyncMock()
            mock_repo.get_by_file_path.side_effect = Exception("DB connection failed")
            mock_repo_class.return_value = mock_repo

            with pytest.raises(Exception, match="DB connection failed"):
                await cataloging_consumer.process_message(mock_message)
