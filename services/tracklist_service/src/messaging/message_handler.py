"""
Message queue handler for tracklist service.

Handles RabbitMQ message consumption and publishing.
"""

import asyncio
import json
import logging
import time
from uuid import uuid4

import aio_pika
from aio_pika import ExchangeType, Message
from aio_pika.abc import (
    AbstractChannel,
    AbstractExchange,
    AbstractIncomingMessage,
    AbstractQueue,
    AbstractRobustConnection,
)
from services.tracklist_service.src.cache.redis_cache import get_cache
from services.tracklist_service.src.config import get_config
from services.tracklist_service.src.models.search_models import (
    SearchError,
    SearchRequest,
    SearchRequestMessage,
    SearchResponseMessage,
)
from services.tracklist_service.src.scraper.search_scraper import SearchScraper

logger = logging.getLogger(__name__)


class TracklistMessageHandler:
    """Handles message queue operations for the tracklist service."""

    def __init__(self) -> None:
        """Initialize the message handler."""
        self.config = get_config().message_queue
        self._running = False
        self._connection: AbstractRobustConnection | None = None
        self._channel: AbstractChannel | None = None
        self._exchange: AbstractExchange | None = None
        self._queue: AbstractQueue | None = None
        self._scraper = SearchScraper()
        self._cache = get_cache()

    async def connect(self) -> None:
        """Establish connection to RabbitMQ."""
        try:
            # Create connection
            self._connection = await aio_pika.connect_robust(
                self.config.rabbitmq_url,
                client_properties={
                    "connection_name": "tracklist-service",
                },
            )

            # Create channel
            self._channel = await self._connection.channel()
            await self._channel.set_qos(prefetch_count=self.config.prefetch_count)

            # Declare exchange
            self._exchange = await self._channel.declare_exchange(
                self.config.exchange_name,
                ExchangeType.TOPIC,
                durable=True,
            )

            # Declare and bind queue
            self._queue = await self._channel.declare_queue(
                self.config.search_queue,
                durable=True,
                arguments={
                    "x-message-ttl": 300000,  # 5 minutes TTL
                    "x-max-length": 1000,  # Max 1000 messages
                },
            )

            # Bind queue to exchange
            if self._exchange:
                await self._queue.bind(
                    self._exchange,
                    routing_key=self.config.search_routing_key,
                )

            logger.info("Connected to RabbitMQ successfully")

        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    async def disconnect(self) -> None:
        """Close RabbitMQ connection."""
        try:
            if self._channel and not self._channel.is_closed:
                await self._channel.close()
            if self._connection and not self._connection.is_closed:
                await self._connection.close()
            logger.info("Disconnected from RabbitMQ")
        except Exception as e:
            logger.error(f"Error disconnecting from RabbitMQ: {e}")

    async def process_search_request(self, message: AbstractIncomingMessage) -> None:
        """Process a search request message.

        Args:
            message: Incoming RabbitMQ message
        """
        start_time = time.time()

        try:
            # Parse message body
            body = json.loads(message.body.decode())

            # Parse request message
            request_msg = SearchRequestMessage.model_validate(body)
            request = request_msg.request

            logger.info(
                f"Processing search request: query='{request.query}', "
                f"type={request.search_type.value}, "
                f"correlation_id={request.correlation_id}"
            )

            # Check cache first
            cached_response = self._cache.get_cached_response(request)

            if cached_response:
                # Send cached response
                response = cached_response
                response.cache_hit = True
                logger.info(f"Cache hit for correlation_id={request.correlation_id}")
            else:
                # Check if search failed recently
                recent_error = self._cache.is_search_failed_recently(request)
                if recent_error:
                    # Send error response
                    error = SearchError(
                        error_code="RECENTLY_FAILED",
                        error_message=f"Search recently failed: {recent_error}",
                        correlation_id=request.correlation_id,
                        details=None,
                        retry_after=300,  # 5 minutes
                    )

                    response_msg = SearchResponseMessage(
                        success=False,
                        response=None,
                        error=error,
                        processing_time_ms=(time.time() - start_time) * 1000,
                    )

                    await self._publish_response(response_msg, request_msg.reply_to)
                    await message.ack()
                    return

                # Execute search
                response = self._scraper.search(request)

                # Cache successful response
                self._cache.cache_response(request, response)

            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            response.response_time_ms = processing_time_ms

            # Create response message
            response_msg = SearchResponseMessage(
                success=True,
                response=response,
                error=None,
                processing_time_ms=processing_time_ms,
            )

            # Publish response
            await self._publish_response(response_msg, request_msg.reply_to)

            # Acknowledge message
            await message.ack()

            logger.info(
                f"Search request processed: correlation_id={request.correlation_id}, "
                f"results={len(response.results)}, time={processing_time_ms:.2f}ms"
            )

        except Exception as e:
            logger.error(f"Error processing search request: {e}")

            # Try to send error response
            try:
                if "request_msg" in locals() and request_msg:
                    error = SearchError(
                        error_code="PROCESSING_ERROR",
                        error_message=str(e),
                        correlation_id=(request.correlation_id if "request" in locals() and request else uuid4()),
                        details=None,
                        retry_after=None,
                    )

                    error_response = SearchResponseMessage(
                        success=False,
                        response=None,
                        error=error,
                        processing_time_ms=0.0,  # Error occurred, no meaningful processing time
                    )

                    await self._publish_response(error_response, request_msg.reply_to)

                    # Cache failed search
                    if "request" in locals():
                        self._cache.cache_failed_search(request, str(e))

            except Exception as publish_error:
                logger.error(f"Failed to publish error response: {publish_error}")

            # Reject message with requeue based on retry count
            requeue = (message.redelivered or 0) < self.config.max_retries
            await message.reject(requeue=requeue)

    async def _publish_response(self, response: SearchResponseMessage, reply_to: str | None = None) -> None:
        """Publish a response message.

        Args:
            response: Response message to publish
            reply_to: Reply queue name (optional)
        """
        if not self._exchange or not self._channel:
            raise RuntimeError("Not connected to RabbitMQ")

        # Determine routing key
        routing_key = reply_to or self.config.result_routing_key

        # Create message
        message = Message(
            body=response.model_dump_json().encode(),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            correlation_id=(str(response.response.correlation_id) if response.response else None),
        )

        # Publish message
        await self._exchange.publish(
            message,
            routing_key=routing_key,
        )

        logger.debug(f"Published response to {routing_key}")

    async def start_consuming(self) -> None:
        """Start consuming messages from the queue."""
        self._running = True

        # Connect to RabbitMQ
        await self.connect()

        if not self._queue:
            raise RuntimeError("Queue not initialized")

        try:
            # Start consuming messages
            async with self._queue.iterator() as queue_iter:
                logger.info("Started consuming messages from queue")

                async for message in queue_iter:
                    if not self._running:
                        break

                    # Process message
                    await self.process_search_request(message)

        except asyncio.CancelledError:
            logger.info("Message consumption cancelled")
        except Exception as e:
            logger.error(f"Error in message consumption: {e}")
            raise
        finally:
            await self.disconnect()

    async def stop(self) -> None:
        """Stop the message handler."""
        logger.info("Stopping message handler")
        self._running = False
        await self.disconnect()

    async def publish_search_request(
        self,
        request: SearchRequest,
        reply_to: str | None = None,
        timeout_seconds: int = 30,
    ) -> None:
        """Publish a search request to the queue.

        Args:
            request: Search request to publish
            reply_to: Reply queue name for response
            timeout_seconds: Request timeout in seconds
        """
        if not self._exchange or not self._channel:
            await self.connect()
            if not self._exchange:
                raise RuntimeError("Failed to initialize exchange")

        # Create request message
        request_msg = SearchRequestMessage(
            request=request,
            reply_to=reply_to,
            timeout_seconds=timeout_seconds,
        )

        # Create AMQP message
        message = Message(
            body=request_msg.model_dump_json().encode(),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            correlation_id=str(request.correlation_id),
            expiration=timeout_seconds * 1000,  # Convert to milliseconds
        )

        # Publish message
        await self._exchange.publish(
            message,
            routing_key=self.config.search_routing_key,
        )

        logger.info(f"Published search request: correlation_id={request.correlation_id}, query='{request.query}'")
