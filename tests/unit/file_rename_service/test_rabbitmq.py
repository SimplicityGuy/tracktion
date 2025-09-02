"""Unit tests for RabbitMQ integration."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aio_pika import DeliveryMode

from services.file_rename_service.utils.rabbitmq import MessageTopics, QueueNames, RabbitMQManager, get_rabbitmq_manager


@pytest.fixture
def rabbitmq_manager() -> RabbitMQManager:
    """Create RabbitMQ manager instance."""
    return RabbitMQManager()


@pytest.mark.asyncio
async def test_connect_success(rabbitmq_manager: RabbitMQManager) -> None:
    """Test successful connection to RabbitMQ."""
    with patch("services.file_rename_service.utils.rabbitmq.aio_pika.connect_robust") as mock_connect:
        # Setup mocks
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_exchange = AsyncMock()

        mock_connect.return_value = mock_connection
        mock_connection.channel.return_value = mock_channel
        mock_channel.declare_exchange.return_value = mock_exchange

        # Connect
        await rabbitmq_manager.connect()

        # Assertions
        assert rabbitmq_manager.connection == mock_connection
        assert rabbitmq_manager.channel == mock_channel
        assert rabbitmq_manager.exchange == mock_exchange
        assert rabbitmq_manager.is_connected is True

        mock_connect.assert_called_once()
        mock_channel.set_qos.assert_called_once()
        mock_channel.declare_exchange.assert_called_once()


@pytest.mark.asyncio
async def test_connect_retry_on_failure(rabbitmq_manager: RabbitMQManager) -> None:
    """Test retry logic on connection failure."""
    with patch("services.file_rename_service.utils.rabbitmq.aio_pika.connect_robust") as mock_connect:
        # First attempt fails, second succeeds
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_exchange = AsyncMock()

        mock_connect.side_effect = [Exception("Connection failed"), mock_connection]
        mock_connection.channel.return_value = mock_channel
        mock_channel.declare_exchange.return_value = mock_exchange

        # Connect with reduced retry delay for faster test
        await rabbitmq_manager.connect(max_retries=2, retry_delay=0.1)

        # Should succeed on second attempt
        assert rabbitmq_manager.is_connected is True
        assert mock_connect.call_count == 2


@pytest.mark.asyncio
async def test_disconnect(rabbitmq_manager: RabbitMQManager) -> None:
    """Test disconnection from RabbitMQ."""
    # Setup connected state
    rabbitmq_manager.connection = AsyncMock()
    rabbitmq_manager.channel = AsyncMock()
    rabbitmq_manager._is_connected = True
    rabbitmq_manager._consumers = {"test_queue": "consumer_tag"}

    # Disconnect
    await rabbitmq_manager.disconnect()

    # Assertions
    rabbitmq_manager.channel.basic_cancel.assert_called_once_with("consumer_tag")
    rabbitmq_manager.channel.close.assert_called_once()
    rabbitmq_manager.connection.close.assert_called_once()
    assert rabbitmq_manager.is_connected is False


@pytest.mark.asyncio
async def test_declare_queue(rabbitmq_manager: RabbitMQManager) -> None:
    """Test queue declaration."""
    # Setup
    mock_queue = AsyncMock()
    rabbitmq_manager.channel = AsyncMock()
    rabbitmq_manager.channel.declare_queue.return_value = mock_queue
    rabbitmq_manager._is_connected = True

    # Declare queue
    queue = await rabbitmq_manager.declare_queue("test_queue", durable=True)

    # Assertions
    assert queue == mock_queue
    assert rabbitmq_manager.queues["test_queue"] == mock_queue
    rabbitmq_manager.channel.declare_queue.assert_called_once_with(
        "test_queue",
        durable=True,
        auto_delete=False,
        exclusive=False,
    )


@pytest.mark.asyncio
async def test_bind_queue(rabbitmq_manager: RabbitMQManager) -> None:
    """Test queue binding to exchange."""
    # Setup
    mock_queue = AsyncMock()
    rabbitmq_manager.queues = {"test_queue": mock_queue}
    rabbitmq_manager.exchange = AsyncMock()
    rabbitmq_manager._is_connected = True

    # Bind queue
    await rabbitmq_manager.bind_queue("test_queue", "test.routing.key")

    # Assertions
    mock_queue.bind.assert_called_once_with(
        rabbitmq_manager.exchange,
        routing_key="test.routing.key",
    )


@pytest.mark.asyncio
async def test_publish_message(rabbitmq_manager: RabbitMQManager) -> None:
    """Test message publishing."""
    # Setup
    rabbitmq_manager.exchange = AsyncMock()
    rabbitmq_manager._is_connected = True

    message_data = {"type": "test", "data": "test_data"}

    # Publish message
    with patch("services.file_rename_service.utils.rabbitmq.Message") as mock_message_class:
        mock_message = MagicMock()
        mock_message_class.return_value = mock_message

        await rabbitmq_manager.publish("test.routing.key", message_data)

        # Assertions
        mock_message_class.assert_called_once()
        call_args = mock_message_class.call_args
        assert json.loads(call_args[1]["body"].decode()) == message_data
        assert call_args[1]["delivery_mode"] == DeliveryMode.PERSISTENT
        assert call_args[1]["content_type"] == "application/json"

        rabbitmq_manager.exchange.publish.assert_called_once_with(
            mock_message,
            routing_key="test.routing.key",
        )


@pytest.mark.asyncio
async def test_consume_messages(rabbitmq_manager: RabbitMQManager) -> None:
    """Test message consumption."""
    # Setup
    mock_queue = AsyncMock()
    mock_queue.consume.return_value = "consumer_tag"
    rabbitmq_manager.queues = {"test_queue": mock_queue}
    rabbitmq_manager._is_connected = True

    # Mock callback
    callback = AsyncMock()

    # Start consuming
    await rabbitmq_manager.consume("test_queue", callback, auto_ack=False)

    # Assertions
    assert rabbitmq_manager._consumers["test_queue"] == "consumer_tag"
    mock_queue.consume.assert_called_once()


@pytest.mark.asyncio
async def test_stop_consuming(rabbitmq_manager: RabbitMQManager) -> None:
    """Test stopping message consumption."""
    # Setup
    rabbitmq_manager.channel = AsyncMock()
    rabbitmq_manager._consumers = {"test_queue": "consumer_tag"}

    # Stop consuming
    await rabbitmq_manager.stop_consuming("test_queue")

    # Assertions
    rabbitmq_manager.channel.basic_cancel.assert_called_once_with("consumer_tag")
    assert "test_queue" not in rabbitmq_manager._consumers


def test_message_topics() -> None:
    """Test message topics constants."""
    assert MessageTopics.RENAME_REQUEST == "rename.request"
    assert MessageTopics.RENAME_RESPONSE == "rename.response"
    assert MessageTopics.RENAME_FEEDBACK == "rename.feedback"
    assert MessageTopics.RENAME_ERROR == "rename.error"
    assert MessageTopics.PATTERN_ANALYZE == "rename.pattern.analyze"
    assert MessageTopics.PATTERN_RESPONSE == "rename.pattern.response"


def test_queue_names() -> None:
    """Test queue names constants."""
    assert QueueNames.RENAME_REQUEST_QUEUE == "file_rename.request"
    assert QueueNames.RENAME_RESPONSE_QUEUE == "file_rename.response"
    assert QueueNames.RENAME_FEEDBACK_QUEUE == "file_rename.feedback"
    assert QueueNames.PATTERN_ANALYZE_QUEUE == "file_rename.pattern"


@pytest.mark.asyncio
async def test_get_rabbitmq_manager_context() -> None:
    """Test RabbitMQ manager context manager."""
    with patch("services.file_rename_service.utils.rabbitmq.rabbitmq_manager") as mock_manager:
        mock_manager.is_connected = False
        mock_manager.connect = AsyncMock()

        async with get_rabbitmq_manager() as manager:
            assert manager == mock_manager
            mock_manager.connect.assert_called_once()
