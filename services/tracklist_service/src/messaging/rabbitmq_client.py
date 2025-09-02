"""
RabbitMQ client for CUE generation message handling.
"""

import json
import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import aio_pika
from aio_pika import Message, connect_robust
from aio_pika.abc import (
    AbstractChannel,
    AbstractExchange,
    AbstractIncomingMessage,
    AbstractQueue,
    AbstractRobustConnection,
)
from pydantic import BaseModel

from services.tracklist_service.src.messaging.message_schemas import (
    MESSAGE_ROUTING,
    BaseMessage,
    BatchCueGenerationCompleteMessage,
    BatchCueGenerationMessage,
    CueConversionMessage,
    CueGenerationCompleteMessage,
    CueGenerationMessage,
    CueValidationMessage,
    MessageType,
)

logger = logging.getLogger(__name__)


class RabbitMQConfig(BaseModel):
    """RabbitMQ connection configuration."""

    host: str = "localhost"
    port: int = 5672
    username: str = "guest"
    password: str = "guest"
    virtual_host: str = "/"
    connection_timeout: int = 30
    heartbeat: int = 600
    max_retries: int = 5
    retry_delay: float = 1.0
    prefetch_count: int = 10


class RabbitMQClient:
    """Async RabbitMQ client for CUE generation messaging."""

    def __init__(self, config: RabbitMQConfig):
        """
        Initialize RabbitMQ client.

        Args:
            config: RabbitMQ connection configuration
        """
        self.config = config
        self.connection: AbstractRobustConnection | None = None
        self.channel: AbstractChannel | None = None
        self.exchanges: dict[str, AbstractExchange] = {}
        self.queues: dict[str, AbstractQueue] = {}
        self._closed = False

    async def connect(self) -> None:
        """Establish connection to RabbitMQ."""
        try:
            connection_url = (
                f"amqp://{self.config.username}:{self.config.password}@"
                f"{self.config.host}:{self.config.port}{self.config.virtual_host}"
            )

            self.connection = await connect_robust(
                connection_url,
                timeout=self.config.connection_timeout,
                heartbeat=self.config.heartbeat,
            )

            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=self.config.prefetch_count)

            # Declare exchanges and queues
            await self._setup_infrastructure()

            logger.info("Connected to RabbitMQ successfully")

        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}", exc_info=True)
            raise

    async def disconnect(self) -> None:
        """Close connection to RabbitMQ."""
        try:
            self._closed = True

            if self.channel:
                await self.channel.close()
                self.channel = None

            if self.connection:
                await self.connection.close()
                self.connection = None

            self.exchanges.clear()
            self.queues.clear()

            logger.info("Disconnected from RabbitMQ")

        except Exception as e:
            logger.error(f"Error during RabbitMQ disconnect: {e}", exc_info=True)

    async def _setup_infrastructure(self) -> None:
        """Set up exchanges, queues, and bindings."""
        if not self.channel:
            raise RuntimeError("Channel not initialized")

        # Declare exchanges
        exchanges_to_create = set()
        for routing_config in MESSAGE_ROUTING.values():
            exchanges_to_create.add(routing_config["exchange"])

        for exchange_name in exchanges_to_create:
            exchange = await self.channel.declare_exchange(exchange_name, aio_pika.ExchangeType.DIRECT, durable=True)
            self.exchanges[exchange_name] = exchange
            logger.debug(f"Declared exchange: {exchange_name}")

        # Declare queues and bindings
        for routing_config in MESSAGE_ROUTING.values():
            exchange = self.exchanges[routing_config["exchange"]]

            queue = await self.channel.declare_queue(routing_config["queue"], durable=routing_config["durable"])

            await queue.bind(exchange, routing_config["routing_key"])
            self.queues[routing_config["queue"]] = queue

            logger.debug(
                f"Declared queue {routing_config['queue']} bound to "
                f"{routing_config['exchange']} with key {routing_config['routing_key']}"
            )

    async def publish_message(self, message: BaseMessage, delay_seconds: int | None = None) -> bool:
        """
        Publish a message to the appropriate queue.

        Args:
            message: Message to publish
            delay_seconds: Optional delay before message becomes available

        Returns:
            True if message was published successfully
        """
        try:
            if self._closed or not self.connection or self.connection.is_closed:
                await self.connect()

            routing_config = MESSAGE_ROUTING.get(message.message_type)
            if not routing_config:
                raise ValueError(f"No routing configuration for message type: {message.message_type}")

            exchange = self.exchanges[routing_config["exchange"]]

            # Create message with headers (ensure all values are properly typed)
            headers: dict[str, Any] = {
                "message_type": message.message_type.value,
                "correlation_id": str(message.correlation_id) if message.correlation_id else "",
                "retry_count": str(message.retry_count),
                "priority": str(message.priority),
                "published_at": datetime.now(UTC).isoformat(),
            }

            # Add delay if specified
            if delay_seconds:
                headers["x-delay"] = str(int(delay_seconds * 1000))  # Convert to milliseconds as string

            rabbitmq_message = Message(
                message.to_json().encode("utf-8"),
                headers=headers,
                priority=message.priority,
                message_id=str(message.message_id),
                correlation_id=str(message.correlation_id) if message.correlation_id else None,
                timestamp=message.timestamp,
            )

            await exchange.publish(rabbitmq_message, routing_key=routing_config["routing_key"])

            logger.debug(f"Published message {message.message_id} of type {message.message_type}")
            return True

        except Exception as e:
            logger.error(f"Failed to publish message {message.message_id}: {e}", exc_info=True)
            return False

    async def publish_batch(self, messages: list[BaseMessage]) -> dict[str, Any]:
        """
        Publish multiple messages as a batch.

        Args:
            messages: List of messages to publish

        Returns:
            Batch publication results
        """
        batch_id = uuid4()
        results: dict[str, Any] = {
            "batch_id": str(batch_id),
            "total_messages": len(messages),
            "successful": 0,
            "failed": 0,
            "errors": [],
        }

        try:
            for message in messages:
                success = await self.publish_message(message)
                if success:
                    results["successful"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(f"Failed to publish message {message.message_id}")

            logger.info(f"Batch {batch_id} published: {results['successful']}/{results['total_messages']} successful")

        except Exception as e:
            logger.error(f"Batch publication failed: {e}", exc_info=True)
            results["errors"].append(f"Batch publication error: {e!s}")

        return results

    async def consume_messages(
        self,
        message_type: MessageType,
        handler: Callable[[BaseMessage, AbstractIncomingMessage], Any],
        auto_ack: bool = False,
    ) -> None:
        """
        Start consuming messages of a specific type.

        Args:
            message_type: Type of messages to consume
            handler: Async function to handle messages
            auto_ack: Whether to automatically acknowledge messages
        """
        try:
            routing_config = MESSAGE_ROUTING.get(message_type)
            if not routing_config:
                raise ValueError(f"No routing configuration for message type: {message_type}")

            queue = self.queues[routing_config["queue"]]

            async def message_wrapper(rabbitmq_message: AbstractIncomingMessage) -> None:
                """Wrapper to handle message processing and acknowledgment."""
                try:
                    # Parse message
                    message_data = json.loads(rabbitmq_message.body.decode("utf-8"))

                    # Determine message class based on type
                    message_class = self._get_message_class(message_type)
                    if not message_class:
                        logger.error(f"Unknown message class for type: {message_type}")
                        await rabbitmq_message.reject(requeue=False)
                        return

                    message = message_class.model_validate(message_data)

                    # Call handler
                    await handler(message, rabbitmq_message)

                    # Acknowledge if not auto-ack and not already processed
                    if not auto_ack and not rabbitmq_message.processed:
                        await rabbitmq_message.ack()

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message JSON: {e}")
                    await rabbitmq_message.reject(requeue=False)

                except Exception as e:
                    logger.error(f"Message handler error: {e}", exc_info=True)

                    # Check retry count
                    retry_count = 0
                    if rabbitmq_message.headers and "retry_count" in rabbitmq_message.headers:
                        retry_count_value = rabbitmq_message.headers["retry_count"]
                        if isinstance(retry_count_value, int):
                            retry_count = retry_count_value
                        elif isinstance(retry_count_value, str):
                            try:
                                retry_count = int(retry_count_value)
                            except ValueError:
                                retry_count = 0
                        else:
                            retry_count = 0
                    else:
                        retry_count = 0

                    if retry_count < self.config.max_retries:
                        # Reject and requeue for retry
                        await rabbitmq_message.reject(requeue=True)
                    else:
                        # Max retries reached, reject without requeue
                        logger.error("Max retries reached for message, rejecting")
                        await rabbitmq_message.reject(requeue=False)

            await queue.consume(message_wrapper, no_ack=auto_ack)
            logger.info(f"Started consuming {message_type} messages from queue {routing_config['queue']}")

        except Exception as e:
            logger.error(f"Failed to start message consumption: {e}", exc_info=True)
            raise

    def _get_message_class(self, message_type: MessageType) -> Any | None:
        """Get the appropriate message class for a message type."""

        message_classes = {
            MessageType.CUE_GENERATION: CueGenerationMessage,
            MessageType.CUE_GENERATION_COMPLETE: CueGenerationCompleteMessage,
            MessageType.BATCH_CUE_GENERATION: BatchCueGenerationMessage,
            MessageType.BATCH_CUE_GENERATION_COMPLETE: BatchCueGenerationCompleteMessage,
            MessageType.CUE_VALIDATION: CueValidationMessage,
            MessageType.CUE_CONVERSION: CueConversionMessage,
        }

        return message_classes.get(message_type)

    @asynccontextmanager
    async def connection_context(self) -> AsyncGenerator["RabbitMQClient"]:
        """Context manager for RabbitMQ connection lifecycle."""
        try:
            await self.connect()
            yield self
        finally:
            await self.disconnect()

    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check on RabbitMQ connection.

        Returns:
            Health check results
        """
        health_status = {
            "rabbitmq": "unknown",
            "connection": "unknown",
            "channel": "unknown",
            "exchanges": 0,
            "queues": 0,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        try:
            if self.connection and not self.connection.is_closed:
                health_status["connection"] = "healthy"

                if self.channel and not self.channel.is_closed:
                    health_status["channel"] = "healthy"
                    health_status["exchanges"] = len(self.exchanges)
                    health_status["queues"] = len(self.queues)
                    health_status["rabbitmq"] = "healthy"
                else:
                    health_status["channel"] = "unhealthy"
                    health_status["rabbitmq"] = "degraded"
            else:
                health_status["connection"] = "unhealthy"
                health_status["rabbitmq"] = "unhealthy"

        except Exception as e:
            health_status["rabbitmq"] = "unhealthy"
            health_status["error"] = str(e)

        return health_status


class RabbitMQClientSingleton:
    """Singleton wrapper for RabbitMQClient."""

    _instance: RabbitMQClient | None = None

    @classmethod
    def get_instance(cls) -> RabbitMQClient:
        """Get the singleton RabbitMQClient instance."""
        if cls._instance is None:
            raise RuntimeError("RabbitMQ client not initialized")
        return cls._instance

    @classmethod
    def initialize(cls, config: RabbitMQConfig) -> RabbitMQClient:
        """Initialize the singleton RabbitMQClient instance."""
        cls._instance = RabbitMQClient(config)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (mainly for testing)."""
        cls._instance = None


def get_rabbitmq_client() -> RabbitMQClient:
    """Get the singleton RabbitMQ client instance."""
    return RabbitMQClientSingleton.get_instance()


def initialize_rabbitmq_client(config: RabbitMQConfig) -> RabbitMQClient:
    """Initialize the singleton RabbitMQ client."""
    return RabbitMQClientSingleton.initialize(config)
