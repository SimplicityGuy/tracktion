"""API message publisher for submitting tasks to RabbitMQ."""

import json
import logging
from contextlib import suppress
from typing import Any
from uuid import UUID, uuid4

import aio_pika
from aio_pika import ExchangeType, Message

logger = logging.getLogger(__name__)


class APIMessagePublisher:
    """Publisher for API-initiated message queue operations."""

    def __init__(
        self,
        rabbitmq_url: str,
        exchange_name: str = "tracktion.events",
    ) -> None:
        """Initialize the API message publisher.

        Args:
            rabbitmq_url: RabbitMQ connection URL
            exchange_name: Name of the exchange to publish to
        """
        self.rabbitmq_url = rabbitmq_url
        self.exchange_name = exchange_name
        self.connection: aio_pika.abc.AbstractConnection | None = None
        self.channel: aio_pika.abc.AbstractChannel | None = None
        self.exchange: aio_pika.abc.AbstractExchange | None = None

    async def connect(self) -> None:
        """Establish async connection to RabbitMQ with retry logic."""
        if self.connection and not self.connection.is_closed:
            return  # Already connected

        try:
            self.connection = await aio_pika.connect_robust(
                self.rabbitmq_url,
                client_properties={"connection_name": "analysis_api_publisher"},
            )
            self.channel = await self.connection.channel()
            # Enable publisher confirms for reliability - using proper aio_pika pattern
            await self.channel.confirm_delivery()
            self.exchange = await self.channel.declare_exchange(self.exchange_name, ExchangeType.TOPIC, durable=True)
            logger.info(f"Connected to RabbitMQ exchange: {self.exchange_name}")
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            # Clean up partial connections
            await self._cleanup_connections()
            raise

    async def _cleanup_connections(self) -> None:
        """Clean up RabbitMQ connections safely."""
        if self.channel and not self.channel.is_closed:
            with suppress(Exception):  # Ignore cleanup errors
                await self.channel.close()
        if self.connection and not self.connection.is_closed:
            with suppress(Exception):  # Ignore cleanup errors
                await self.connection.close()
        self.channel = None
        self.connection = None
        self.exchange = None

    async def publish_analysis_request(
        self,
        recording_id: UUID,
        file_path: str,
        analysis_types: list[str],
        priority: int = 5,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Publish analysis request to processing queue.

        Args:
            recording_id: UUID of the recording
            file_path: Path to the audio file
            analysis_types: List of analysis types to perform
            priority: Message priority (1-10)
            metadata: Optional metadata

        Returns:
            Correlation ID for tracking the request
        """
        if not self.exchange:
            await self.connect()

        correlation_id = str(uuid4())

        message_data = {
            "recording_id": str(recording_id),
            "file_path": file_path,
            "analysis_types": analysis_types,
            "metadata": metadata or {},
            "correlation_id": correlation_id,
        }

        message = Message(
            body=json.dumps(message_data).encode(),
            priority=priority,
            correlation_id=correlation_id,
            content_type="application/json",
        )

        if self.exchange is not None:
            await self.exchange.publish(message, routing_key="analysis.request")
        else:
            raise RuntimeError("Exchange not initialized")

        logger.info(
            "Published analysis request",
            extra={
                "recording_id": str(recording_id),
                "correlation_id": correlation_id,
                "analysis_types": analysis_types,
            },
        )

        return correlation_id

    async def publish_metadata_extraction(
        self,
        recording_id: UUID,
        extraction_types: list[str] | None = None,
        priority: int = 5,
    ) -> str:
        """Publish metadata extraction request.

        Args:
            recording_id: UUID of the recording
            extraction_types: Types of metadata to extract
            priority: Message priority

        Returns:
            Correlation ID for tracking
        """
        if not self.exchange:
            await self.connect()

        correlation_id = str(uuid4())

        message_data = {
            "recording_id": str(recording_id),
            "extraction_types": extraction_types or ["id3_tags", "audio_analysis"],
            "correlation_id": correlation_id,
        }

        message = Message(
            body=json.dumps(message_data).encode(),
            priority=priority,
            correlation_id=correlation_id,
            content_type="application/json",
        )

        if self.exchange is not None:
            await self.exchange.publish(message, routing_key="metadata.extract")
        else:
            raise RuntimeError("Exchange not initialized")

        logger.info(
            "Published metadata extraction request",
            extra={
                "recording_id": str(recording_id),
                "correlation_id": correlation_id,
            },
        )

        return correlation_id

    async def publish_tracklist_generation(
        self,
        recording_id: UUID,
        source_hint: str | None = None,
        priority: int = 5,
    ) -> str:
        """Publish tracklist generation request.

        Args:
            recording_id: UUID of the recording
            source_hint: Hint for tracklist source
            priority: Message priority

        Returns:
            Correlation ID for tracking
        """
        if not self.exchange:
            await self.connect()

        correlation_id = str(uuid4())

        message_data = {
            "recording_id": str(recording_id),
            "source_hint": source_hint or "auto",
            "correlation_id": correlation_id,
        }

        message = Message(
            body=json.dumps(message_data).encode(),
            priority=priority,
            correlation_id=correlation_id,
            content_type="application/json",
        )

        if self.exchange is not None:
            await self.exchange.publish(message, routing_key="tracklist.generate")
        else:
            raise RuntimeError("Exchange not initialized")

        logger.info(
            "Published tracklist generation request",
            extra={
                "recording_id": str(recording_id),
                "correlation_id": correlation_id,
            },
        )

        return correlation_id

    async def cancel_processing(self, recording_id: UUID) -> str:
        """Publish cancellation request for a recording.

        Args:
            recording_id: UUID of the recording to cancel

        Returns:
            Correlation ID for tracking
        """
        if not self.exchange:
            await self.connect()

        correlation_id = str(uuid4())

        message_data = {
            "recording_id": str(recording_id),
            "action": "cancel",
            "correlation_id": correlation_id,
        }

        message = Message(
            body=json.dumps(message_data).encode(),
            priority=8,  # High priority for cancellations
            correlation_id=correlation_id,
            content_type="application/json",
        )

        if self.exchange is not None:
            await self.exchange.publish(message, routing_key="analysis.cancel")
        else:
            raise RuntimeError("Exchange not initialized")

        logger.info(
            "Published cancellation request",
            extra={
                "recording_id": str(recording_id),
                "correlation_id": correlation_id,
            },
        )

        return correlation_id

    async def close(self) -> None:
        """Close RabbitMQ connections safely."""
        await self._cleanup_connections()
        logger.info("API message publisher connections closed")
