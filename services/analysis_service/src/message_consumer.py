"""RabbitMQ message consumer for analysis service."""

import json
import logging
import time
from typing import Any, Callable, Dict, Optional

import pika
import pika.exceptions
from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic, BasicProperties

logger = logging.getLogger(__name__)


class MessageConsumer:
    """Handles RabbitMQ message consumption for file analysis."""

    def __init__(
        self,
        rabbitmq_url: str,
        queue_name: str = "analysis_queue",
        exchange_name: str = "tracktion_exchange",
        routing_key: str = "file.analyze",
    ) -> None:
        """Initialize the message consumer.

        Args:
            rabbitmq_url: RabbitMQ connection URL
            queue_name: Name of the queue to consume from
            exchange_name: Name of the exchange
            routing_key: Routing key for message binding
        """
        self.rabbitmq_url = rabbitmq_url
        self.queue_name = queue_name
        self.exchange_name = exchange_name
        self.routing_key = routing_key
        self.connection: Optional[pika.BlockingConnection] = None
        self.channel: Optional[BlockingChannel] = None
        self._retry_count = 0
        self._max_retries = 5
        self._base_delay = 2.0

    def connect(self) -> None:
        """Establish connection to RabbitMQ with retry logic."""
        while self._retry_count < self._max_retries:
            try:
                self.connection = pika.BlockingConnection(pika.URLParameters(self.rabbitmq_url))
                self.channel = self.connection.channel()

                # Declare exchange
                self.channel.exchange_declare(exchange=self.exchange_name, exchange_type="topic", durable=True)

                # Declare queue
                self.channel.queue_declare(queue=self.queue_name, durable=True)

                # Bind queue to exchange
                self.channel.queue_bind(
                    exchange=self.exchange_name, queue=self.queue_name, routing_key=self.routing_key
                )

                # Set QoS
                self.channel.basic_qos(prefetch_count=1)

                logger.info(f"Connected to RabbitMQ queue: {self.queue_name}")
                self._retry_count = 0
                return

            except pika.exceptions.AMQPConnectionError as e:
                self._retry_count += 1
                delay = self._base_delay * (2**self._retry_count)
                logger.warning(f"Connection attempt {self._retry_count} failed: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)

        raise ConnectionError(f"Failed to connect to RabbitMQ after {self._max_retries} attempts")

    def consume(self, callback: Callable[[Dict[str, Any], str], None]) -> None:
        """Start consuming messages from the queue.

        Args:
            callback: Function to process each message
        """
        if not self.channel:
            self.connect()

        def message_callback(
            ch: BlockingChannel, method: Basic.Deliver, properties: BasicProperties, body: bytes
        ) -> None:
            """Process incoming message."""
            correlation_id = properties.correlation_id or "unknown"

            try:
                # Parse message
                message = json.loads(body.decode())
                logger.info(
                    "Received message", extra={"correlation_id": correlation_id, "routing_key": method.routing_key}
                )

                # Process message
                callback(message, correlation_id)

                # Acknowledge message
                ch.basic_ack(delivery_tag=method.delivery_tag)
                logger.info("Message processed successfully", extra={"correlation_id": correlation_id})

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in message: {e}", extra={"correlation_id": correlation_id})
                # Reject message without requeue (bad format)
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

            except Exception as e:
                logger.error(f"Error processing message: {e}", extra={"correlation_id": correlation_id})
                # Requeue message for retry
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

        if self.channel:
            self.channel.basic_consume(queue=self.queue_name, on_message_callback=message_callback, auto_ack=False)

        logger.info("Starting message consumption...")
        try:
            if self.channel:
                self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("Stopping message consumption...")
            self.stop()

    def stop(self) -> None:
        """Stop consuming and close connections."""
        if self.channel:
            self.channel.stop_consuming()
            self.channel.close()
        if self.connection:
            self.connection.close()
        logger.info("Message consumer stopped")

    def __enter__(self) -> "MessageConsumer":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.stop()
