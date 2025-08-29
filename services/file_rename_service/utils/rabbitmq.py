"""RabbitMQ connection manager and message handling."""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Any

import aio_pika
from aio_pika import ExchangeType, Message
from aio_pika.abc import (
    AbstractChannel,
    AbstractExchange,
    AbstractIncomingMessage,
    AbstractQueue,
    AbstractRobustConnection,
)

from services.file_rename_service.app.config import settings

logger = logging.getLogger(__name__)


class RabbitMQManager:
    """RabbitMQ connection manager with retry logic and error handling."""

    def __init__(self) -> None:
        """Initialize RabbitMQ manager."""
        self.connection: AbstractRobustConnection | None = None
        self.channel: AbstractChannel | None = None
        self.exchange: AbstractExchange | None = None
        self.queues: dict[str, AbstractQueue] = {}
        self._consumers: dict[str, Any] = {}
        self._is_connected = False

    async def connect(self, max_retries: int = 5, retry_delay: float = 2.0) -> None:
        """Connect to RabbitMQ with retry logic."""
        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting to connect to RabbitMQ (attempt {attempt + 1}/{max_retries})")

                # Create connection
                self.connection = await aio_pika.connect_robust(
                    settings.rabbitmq_url,
                    loop=asyncio.get_event_loop(),
                )

                # Create channel
                self.channel = await self.connection.channel()
                await self.channel.set_qos(prefetch_count=settings.rabbitmq_prefetch_count)

                # Declare exchange
                self.exchange = await self.channel.declare_exchange(
                    settings.rabbitmq_exchange,
                    ExchangeType.TOPIC,
                    durable=True,
                )

                self._is_connected = True
                logger.info("Successfully connected to RabbitMQ")
                return

            except Exception as e:
                logger.error(f"Failed to connect to RabbitMQ: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    raise

    async def disconnect(self) -> None:
        """Disconnect from RabbitMQ."""
        try:
            # Cancel all consumers
            for queue_name in list(self._consumers.keys()):
                await self.stop_consuming(queue_name)

            # Close channel and connection
            if self.channel:
                await self.channel.close()
            if self.connection:
                await self.connection.close()

            self._is_connected = False
            logger.info("Disconnected from RabbitMQ")

        except Exception as e:
            logger.error(f"Error disconnecting from RabbitMQ: {e}")

    async def declare_queue(
        self,
        name: str,
        durable: bool = True,
        auto_delete: bool = False,
        exclusive: bool = False,
    ) -> AbstractQueue:
        """Declare a queue."""
        if not self._is_connected or not self.channel:
            raise RuntimeError("Not connected to RabbitMQ")

        queue = await self.channel.declare_queue(
            name,
            durable=durable,
            auto_delete=auto_delete,
            exclusive=exclusive,
        )
        self.queues[name] = queue
        return queue

    async def bind_queue(self, queue_name: str, routing_key: str) -> None:
        """Bind a queue to the exchange with a routing key."""
        if queue_name not in self.queues:
            await self.declare_queue(queue_name)

        if not self.exchange:
            raise RuntimeError("Exchange not initialized")

        queue = self.queues[queue_name]
        await queue.bind(self.exchange, routing_key=routing_key)
        logger.info(f"Bound queue '{queue_name}' to routing key '{routing_key}'")

    async def publish(
        self,
        routing_key: str,
        message: dict[str, Any],
        persistent: bool = True,
    ) -> None:
        """Publish a message to the exchange."""
        if not self._is_connected or not self.exchange:
            raise RuntimeError("Not connected to RabbitMQ")

        try:
            message_body = json.dumps(message).encode()
            amqp_message = Message(
                body=message_body,
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT if persistent else aio_pika.DeliveryMode.NOT_PERSISTENT,
                content_type="application/json",
            )

            await self.exchange.publish(
                amqp_message,
                routing_key=routing_key,
            )
            logger.debug(f"Published message to '{routing_key}': {message}")

        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            raise

    async def consume(
        self,
        queue_name: str,
        callback: Callable[[dict[str, Any]], Any],
        auto_ack: bool = False,
    ) -> None:
        """Start consuming messages from a queue."""
        if not self._is_connected:
            raise RuntimeError("Not connected to RabbitMQ")

        if queue_name not in self.queues:
            raise ValueError(f"Queue '{queue_name}' not declared")

        queue = self.queues[queue_name]

        async def message_handler(message: AbstractIncomingMessage) -> None:
            """Handle incoming messages."""
            async with message.process():
                try:
                    # Parse message body
                    body = json.loads(message.body.decode())
                    logger.debug(f"Received message from '{queue_name}': {body}")

                    # Call the callback
                    result = callback(body)
                    if asyncio.iscoroutine(result):
                        await result

                    # Acknowledge if not auto-ack
                    if not auto_ack:
                        await message.ack()

                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Reject and requeue the message
                    await message.nack(requeue=True)

        # Start consuming
        consumer_tag = await queue.consume(message_handler, no_ack=auto_ack)
        self._consumers[queue_name] = consumer_tag
        logger.info(f"Started consuming from queue '{queue_name}'")

    async def stop_consuming(self, queue_name: str) -> None:
        """Stop consuming from a queue."""
        if queue_name in self._consumers:
            # Consumer tags are automatically cancelled when closing connection
            # Just remove from tracking
            del self._consumers[queue_name]
            logger.info(f"Stopped consuming from queue '{queue_name}'")

    @property
    def is_connected(self) -> bool:
        """Check if connected to RabbitMQ."""
        return self._is_connected


# Global instance
rabbitmq_manager = RabbitMQManager()


@asynccontextmanager
async def get_rabbitmq_manager() -> AsyncGenerator[RabbitMQManager]:
    """Get RabbitMQ manager context."""
    if not rabbitmq_manager.is_connected:
        await rabbitmq_manager.connect()
    yield rabbitmq_manager


# Message topics
class MessageTopics:
    """RabbitMQ message topics for file rename service."""

    # Request topics
    RENAME_REQUEST = "rename.request"
    PATTERN_ANALYZE = "rename.pattern.analyze"

    # Response topics
    RENAME_RESPONSE = "rename.response"
    PATTERN_RESPONSE = "rename.pattern.response"

    # Feedback topics
    RENAME_FEEDBACK = "rename.feedback"

    # Error topics
    RENAME_ERROR = "rename.error"


# Queue names
class QueueNames:
    """RabbitMQ queue names for file rename service."""

    RENAME_REQUEST_QUEUE = "file_rename.request"
    RENAME_RESPONSE_QUEUE = "file_rename.response"
    RENAME_FEEDBACK_QUEUE = "file_rename.feedback"
    PATTERN_ANALYZE_QUEUE = "file_rename.pattern"
