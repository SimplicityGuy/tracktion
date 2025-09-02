"""Tests for message queue handler."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from aio_pika import DeliveryMode
from aio_pika.abc import AbstractIncomingMessage

from services.tracklist_service.src.messaging.message_handler import TracklistMessageHandler
from services.tracklist_service.src.models.search_models import (
    PaginationInfo,
    SearchRequest,
    SearchRequestMessage,
    SearchResponse,
    SearchResponseMessage,
    SearchResult,
    SearchType,
)


@pytest.fixture
def message_handler():
    """Create message handler instance."""
    return TracklistMessageHandler()


@pytest.fixture
def mock_search_request():
    """Create mock search request."""
    return SearchRequest(
        query="test",
        search_type=SearchType.DJ,
        page=1,
        limit=20,
        correlation_id=uuid4(),
    )


@pytest.fixture
def mock_search_response():
    """Create mock search response."""
    return SearchResponse(
        results=[
            SearchResult(
                dj_name="Test DJ",
                url="https://1001tracklists.com/test",
                source_url="https://1001tracklists.com/test",
            )
        ],
        pagination=PaginationInfo(
            page=1,
            limit=20,
            total_pages=1,
            total_items=1,
            has_next=False,
            has_previous=False,
        ),
        query_info={"query": "test"},
        cache_hit=False,
        response_time_ms=100.0,
        correlation_id=uuid4(),
    )


class TestTracklistMessageHandler:
    """Test TracklistMessageHandler functionality."""

    @pytest.mark.asyncio
    async def test_connect_success(self, message_handler):
        """Test successful connection to RabbitMQ."""
        with patch("aio_pika.connect_robust") as mock_connect:
            mock_connection = AsyncMock()
            mock_channel = AsyncMock()
            mock_exchange = AsyncMock()
            mock_queue = AsyncMock()

            mock_connect.return_value = mock_connection
            mock_connection.channel.return_value = mock_channel
            mock_channel.declare_exchange.return_value = mock_exchange
            mock_channel.declare_queue.return_value = mock_queue

            await message_handler.connect()

            assert message_handler._connection == mock_connection
            assert message_handler._channel == mock_channel
            assert message_handler._exchange == mock_exchange
            assert message_handler._queue == mock_queue

            mock_connect.assert_called_once()
            mock_channel.set_qos.assert_called_once_with(prefetch_count=1)

    @pytest.mark.asyncio
    async def test_connect_failure(self, message_handler):
        """Test connection failure handling."""
        with patch("aio_pika.connect_robust") as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")

            with pytest.raises(Exception) as exc_info:
                await message_handler.connect()

            assert "Connection failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_disconnect(self, message_handler):
        """Test disconnection from RabbitMQ."""
        message_handler._channel = AsyncMock()
        message_handler._connection = AsyncMock()

        message_handler._channel.is_closed = False
        message_handler._connection.is_closed = False

        await message_handler.disconnect()

        message_handler._channel.close.assert_called_once()
        message_handler._connection.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_search_request_cached(self, message_handler, mock_search_request, mock_search_response):
        """Test processing search request with cached response."""
        # Create mock message
        request_msg = SearchRequestMessage(
            request=mock_search_request,
            reply_to="test_reply_queue",
            timeout_seconds=30,
        )

        mock_message = AsyncMock(spec=AbstractIncomingMessage)
        mock_message.body = request_msg.model_dump_json().encode()
        mock_message.ack = AsyncMock()

        # Mock cache to return cached response
        message_handler._cache = MagicMock()
        message_handler._cache.get_cached_response.return_value = mock_search_response

        # Mock publish method
        message_handler._publish_response = AsyncMock()

        # Process request
        await message_handler.process_search_request(mock_message)

        # Verify cache was checked
        message_handler._cache.get_cached_response.assert_called_once()

        # Verify response was published
        message_handler._publish_response.assert_called_once()
        published_response = message_handler._publish_response.call_args[0][0]
        assert published_response.success is True
        assert published_response.response == mock_search_response

        # Verify message was acknowledged
        mock_message.ack.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_search_request_not_cached(self, message_handler, mock_search_request, mock_search_response):
        """Test processing search request without cached response."""
        # Create mock message
        request_msg = SearchRequestMessage(
            request=mock_search_request,
            reply_to="test_reply_queue",
            timeout_seconds=30,
        )

        mock_message = AsyncMock(spec=AbstractIncomingMessage)
        mock_message.body = request_msg.model_dump_json().encode()
        mock_message.ack = AsyncMock()

        # Mock cache to return no cached response
        message_handler._cache = MagicMock()
        message_handler._cache.get_cached_response.return_value = None
        message_handler._cache.is_search_failed_recently.return_value = None
        message_handler._cache.cache_response = MagicMock()

        # Mock scraper to return response
        message_handler._scraper = MagicMock()
        message_handler._scraper.search.return_value = mock_search_response

        # Mock publish method
        message_handler._publish_response = AsyncMock()

        # Process request
        await message_handler.process_search_request(mock_message)

        # Verify scraper was called
        message_handler._scraper.search.assert_called_once_with(mock_search_request)

        # Verify response was cached
        message_handler._cache.cache_response.assert_called_once()

        # Verify response was published
        message_handler._publish_response.assert_called_once()

        # Verify message was acknowledged
        mock_message.ack.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_search_request_recently_failed(self, message_handler, mock_search_request):
        """Test processing search request that recently failed."""
        # Create mock message
        request_msg = SearchRequestMessage(
            request=mock_search_request,
            reply_to="test_reply_queue",
            timeout_seconds=30,
        )

        mock_message = AsyncMock(spec=AbstractIncomingMessage)
        mock_message.body = request_msg.model_dump_json().encode()
        mock_message.ack = AsyncMock()

        # Mock cache to indicate recent failure
        message_handler._cache = MagicMock()
        message_handler._cache.get_cached_response.return_value = None
        message_handler._cache.is_search_failed_recently.return_value = "Previous error"

        # Mock publish method
        message_handler._publish_response = AsyncMock()

        # Process request
        await message_handler.process_search_request(mock_message)

        # Verify error response was published
        message_handler._publish_response.assert_called_once()
        published_response = message_handler._publish_response.call_args[0][0]
        assert published_response.success is False
        assert published_response.error.error_code == "RECENTLY_FAILED"

        # Verify message was acknowledged
        mock_message.ack.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_search_request_exception(self, message_handler, mock_search_request):
        """Test processing search request with exception."""
        # Create mock message
        request_msg = SearchRequestMessage(
            request=mock_search_request,
            reply_to="test_reply_queue",
            timeout_seconds=30,
        )

        mock_message = AsyncMock(spec=AbstractIncomingMessage)
        mock_message.body = request_msg.model_dump_json().encode()
        mock_message.reject = AsyncMock()
        mock_message.redelivered_count = 0

        # Mock cache to return no cached response
        message_handler._cache = MagicMock()
        message_handler._cache.get_cached_response.return_value = None
        message_handler._cache.is_search_failed_recently.return_value = None
        message_handler._cache.cache_failed_search = MagicMock()

        # Mock scraper to raise exception
        message_handler._scraper = MagicMock()
        message_handler._scraper.search.side_effect = Exception("Scraping failed")

        # Mock publish method
        message_handler._publish_response = AsyncMock()

        # Process request
        await message_handler.process_search_request(mock_message)

        # Verify error response was published
        message_handler._publish_response.assert_called_once()
        published_response = message_handler._publish_response.call_args[0][0]
        assert published_response.success is False
        assert "Scraping failed" in published_response.error.error_message

        # Verify failed search was cached
        message_handler._cache.cache_failed_search.assert_called_once()

        # Verify message was rejected with requeue
        mock_message.reject.assert_called_once_with(requeue=True)

    @pytest.mark.asyncio
    async def test_publish_response(self, message_handler, mock_search_response):
        """Test publishing response message."""
        # Setup mock exchange and channel
        mock_exchange = AsyncMock()
        mock_channel = AsyncMock()
        message_handler._exchange = mock_exchange
        message_handler._channel = mock_channel

        # Create response message with a proper mock response
        mock_response = mock_search_response
        response_msg = SearchResponseMessage(
            success=True,
            response=mock_response,
            processing_time_ms=100.0,
        )

        # Publish response
        await message_handler._publish_response(response_msg, "reply_queue")

        # Verify message was published
        mock_exchange.publish.assert_called_once()
        published_message = mock_exchange.publish.call_args[0][0]
        assert published_message.content_type == "application/json"
        assert published_message.delivery_mode == DeliveryMode.PERSISTENT

    @pytest.mark.asyncio
    async def test_publish_response_not_connected(self, message_handler, mock_search_response):
        """Test publishing response when not connected."""
        message_handler._exchange = None
        message_handler._channel = None

        response_msg = SearchResponseMessage(
            success=True,
            response=mock_search_response,
            processing_time_ms=100.0,
        )

        with pytest.raises(RuntimeError) as exc_info:
            await message_handler._publish_response(response_msg)

        assert "Not connected to RabbitMQ" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_publish_search_request(self, message_handler, mock_search_request):
        """Test publishing search request."""
        # Setup mock exchange and channel
        mock_exchange = AsyncMock()
        mock_channel = AsyncMock()
        message_handler._exchange = mock_exchange
        message_handler._channel = mock_channel

        # Publish request
        await message_handler.publish_search_request(
            mock_search_request,
            reply_to="reply_queue",
            timeout_seconds=60,
        )

        # Verify message was published
        mock_exchange.publish.assert_called_once()
        published_message = mock_exchange.publish.call_args[0][0]
        assert published_message.content_type == "application/json"
        assert published_message.correlation_id == str(mock_search_request.correlation_id)

    @pytest.mark.asyncio
    async def test_start_consuming(self, message_handler):
        """Test starting message consumption."""
        with (
            patch.object(message_handler, "connect", new_callable=AsyncMock),
            patch.object(message_handler, "disconnect", new_callable=AsyncMock),
        ):
            # Setup mock queue
            mock_queue = AsyncMock()
            message_handler._queue = mock_queue

            # Create mock iterator that yields one message then stops
            mock_message = AsyncMock()

            class MockAsyncIterator:
                def __init__(self):
                    self.yielded = False

                def __aiter__(self):
                    return self

                async def __anext__(self):
                    if not self.yielded:
                        self.yielded = True
                        return mock_message
                    message_handler._running = False
                    raise StopAsyncIteration

            # Create a mock context manager
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__.return_value = MockAsyncIterator()
            mock_context_manager.__aexit__.return_value = None
            # Make iterator a regular method that returns the context manager
            mock_queue.iterator = MagicMock(return_value=mock_context_manager)

            with patch.object(message_handler, "process_search_request", new_callable=AsyncMock):
                await message_handler.start_consuming()

                # Verify message was processed
                message_handler.process_search_request.assert_called_once_with(mock_message)

                # Verify disconnect was called
                message_handler.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop(self, message_handler):
        """Test stopping message handler."""
        message_handler._running = True

        with patch.object(message_handler, "disconnect", new_callable=AsyncMock):
            await message_handler.stop()

            assert message_handler._running is False
            message_handler.disconnect.assert_called_once()
