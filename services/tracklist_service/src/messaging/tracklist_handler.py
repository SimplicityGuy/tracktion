"""
Message queue handler for tracklist retrieval.

Handles async tracklist retrieval requests via RabbitMQ.
"""

import asyncio
import json
import logging
import time
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

import aio_pika
from aio_pika import ExchangeType, Message
from aio_pika.abc import (
    AbstractChannel,
    AbstractExchange,
    AbstractIncomingMessage,
    AbstractQueue,
    AbstractRobustConnection,
)

from src.cache.redis_cache import RedisCache
from src.config import get_config
from src.models.tracklist_models import Tracklist, TracklistRequest
from src.scraper.tracklist_scraper import TracklistScraper

logger = logging.getLogger(__name__)


class TracklistMessageHandler:
    """Handles tracklist retrieval via message queue."""

    def __init__(self) -> None:
        """Initialize the message handler."""
        self.config = get_config().message_queue
        self._running = False
        self._connection: AbstractRobustConnection | None = None
        self._channel: AbstractChannel | None = None
        self._exchange: AbstractExchange | None = None
        self._queue: AbstractQueue | None = None
        self._scraper = TracklistScraper()
        self._cache = RedisCache()

    async def connect(self) -> None:
        """Establish connection to RabbitMQ."""
        try:
            # Create connection
            self._connection = await aio_pika.connect_robust(
                self.config.rabbitmq_url,
                client_properties={
                    "connection_name": "tracklist-service-retrieval",
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

            # Declare tracklist retrieval queue
            self._queue = await self._channel.declare_queue(
                "tracklist.retrieval",
                durable=True,
                arguments={
                    "x-message-ttl": 600000,  # 10 minutes TTL
                    "x-max-length": 500,  # Max 500 messages
                    "x-dead-letter-exchange": f"{self.config.exchange_name}.dlx",
                },
            )

            # Bind queue to exchange
            if self._exchange:
                await self._queue.bind(
                    self._exchange,
                    routing_key="tracklist.retrieval",
                )

            logger.info("Connected to RabbitMQ for tracklist retrieval")

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

    async def process_tracklist_request(self, message: AbstractIncomingMessage) -> None:
        """
        Process a tracklist retrieval request.

        Args:
            message: Incoming RabbitMQ message
        """
        start_time = time.time()
        correlation_id = None

        try:
            # Parse message body
            body = json.loads(message.body.decode())

            # Extract request
            request_data = body.get("request", {})
            correlation_id = UUID(request_data.get("correlation_id", ""))

            # Update job status to processing
            status_key = f"job:status:{correlation_id}"
            await self._cache.set(
                status_key,
                json.dumps(
                    {
                        "status": "processing",
                        "started_at": datetime.now(UTC).isoformat(),
                    }
                ),
                ttl=3600,  # 1 hour
            )

            # Create TracklistRequest
            request = TracklistRequest(**request_data)

            logger.info(f"Processing tracklist request: url={request.url}, correlation_id={correlation_id}")

            # Check cache first
            if request.url and not request.force_refresh:
                cache_key = f"tracklist:{request.url}"
                cached_data = await self._cache.get(cache_key)

                if cached_data:
                    # Update status with cached result
                    await self._update_job_status(
                        correlation_id,
                        "completed",
                        cached_data,
                        processing_time_ms=int((time.time() - start_time) * 1000),
                        cached=True,
                    )
                    await message.ack()
                    logger.info(f"Returned cached tracklist for {correlation_id}")
                    return

            # Scrape the tracklist
            if not request.url:
                raise ValueError("URL is required for tracklist retrieval")

            tracklist = self._scraper.scrape_tracklist(request.url)

            # Filter transitions if not requested
            if not request.include_transitions:
                tracklist.transitions = []

            # Cache the result
            tracklist_json = tracklist.model_dump_json()
            cache_key = f"tracklist:{request.url}"
            await self._cache.set(
                cache_key,
                tracklist_json,
                ttl=7 * 24 * 60 * 60,  # 7 days
            )

            # Update job status with result
            processing_time_ms = int((time.time() - start_time) * 1000)
            await self._update_job_status(
                correlation_id,
                "completed",
                tracklist_json,
                processing_time_ms=processing_time_ms,
                cached=False,
            )

            # Publish completion event
            await self._publish_completion_event(tracklist, correlation_id)

            # Acknowledge message
            await message.ack()

            logger.info(
                f"Tracklist processed: correlation_id={correlation_id}, "
                f"tracks={len(tracklist.tracks)}, time={processing_time_ms}ms"
            )

        except Exception as e:
            logger.error(f"Error processing tracklist request: {e}")

            # Update job status with error
            if correlation_id:
                await self._update_job_status(
                    correlation_id,
                    "failed",
                    None,
                    error=str(e),
                    processing_time_ms=int((time.time() - start_time) * 1000),
                )

            # Reject message with requeue based on retry count
            requeue = (message.redelivered or 0) < self.config.max_retries
            await message.reject(requeue=requeue)

            # If max retries exceeded, send to dead letter queue
            if not requeue and correlation_id:
                await self._publish_to_dlq(body, str(e))

    async def _update_job_status(
        self,
        correlation_id: UUID,
        status: str,
        result: str | None = None,
        error: str | None = None,
        processing_time_ms: int | None = None,
        cached: bool = False,
    ) -> None:
        """Update job status in cache."""
        status_key = f"job:status:{correlation_id}"
        status_data = {
            "status": status,
            "updated_at": datetime.now(UTC).isoformat(),
            "cached": cached,
        }

        if processing_time_ms is not None:
            status_data["processing_time_ms"] = processing_time_ms

        if error:
            status_data["error"] = error

        await self._cache.set(status_key, json.dumps(status_data), ttl=3600)

        # Store result separately if available
        if result:
            result_key = f"job:result:{correlation_id}"
            await self._cache.set(result_key, result, ttl=3600)

    async def _publish_completion_event(self, tracklist: Tracklist, correlation_id: UUID) -> None:
        """Publish tracklist completion event for downstream processing."""
        if not self._exchange or not self._channel:
            return

        event = {
            "event_type": "tracklist.completed",
            "correlation_id": str(correlation_id),
            "tracklist_id": str(tracklist.id),
            "url": tracklist.url,
            "dj_name": tracklist.dj_name,
            "track_count": len(tracklist.tracks),
            "scraped_at": tracklist.scraped_at.isoformat(),
            "timestamp": datetime.now(UTC).isoformat(),
        }

        message = Message(
            body=json.dumps(event).encode(),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            correlation_id=str(correlation_id),
        )

        await self._exchange.publish(
            message,
            routing_key="tracklist.completed",
        )

        logger.debug(f"Published completion event for {correlation_id}")

    async def _publish_to_dlq(self, original_message: dict[str, Any], error: str) -> None:
        """Publish failed message to dead letter queue."""
        if not self._exchange or not self._channel:
            return

        dlq_message = {
            "original_message": original_message,
            "error": error,
            "failed_at": datetime.now(UTC).isoformat(),
        }

        message = Message(
            body=json.dumps(dlq_message).encode(),
            content_type="application/json",
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )

        # Create DLX exchange if it doesn't exist
        dlx_exchange = await self._channel.declare_exchange(
            f"{self.config.exchange_name}.dlx",
            ExchangeType.TOPIC,
            durable=True,
        )

        await dlx_exchange.publish(
            message,
            routing_key="tracklist.failed",
        )

        logger.info("Published failed message to DLQ")

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
                logger.info("Started consuming tracklist retrieval messages")

                async for message in queue_iter:
                    if not self._running:
                        break

                    # Process message
                    await self.process_tracklist_request(message)

        except asyncio.CancelledError:
            logger.info("Message consumption cancelled")
        except Exception as e:
            logger.error(f"Error in message consumption: {e}")
            raise
        finally:
            await self.disconnect()

    async def stop(self) -> None:
        """Stop the message handler."""
        logger.info("Stopping tracklist message handler")
        self._running = False
        await self.disconnect()
