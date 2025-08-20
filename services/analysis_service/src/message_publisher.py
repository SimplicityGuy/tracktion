"""Message publisher with priority support for the analysis service."""

import json
import logging
import os
from typing import Any, Dict, Optional

import pika
from pika.spec import BasicProperties

from .priority_queue import PriorityCalculator, PriorityConfig, add_priority_to_message

logger = logging.getLogger(__name__)


class PriorityMessagePublisher:
    """Publishes messages to RabbitMQ with priority support."""

    def __init__(
        self,
        rabbitmq_url: str,
        exchange_name: str = "tracktion_exchange",
        priority_config: Optional[PriorityConfig] = None,
    ) -> None:
        """Initialize the message publisher.

        Args:
            rabbitmq_url: RabbitMQ connection URL
            exchange_name: Name of the exchange to publish to
            priority_config: Configuration for priority calculation
        """
        self.rabbitmq_url = rabbitmq_url
        self.exchange_name = exchange_name
        self.priority_config = priority_config or PriorityConfig()
        self.priority_calculator = PriorityCalculator(self.priority_config)
        self.connection: Optional[pika.BlockingConnection] = None
        self.channel: Optional[Any] = None

    def connect(self) -> None:
        """Establish connection to RabbitMQ."""
        try:
            self.connection = pika.BlockingConnection(pika.URLParameters(self.rabbitmq_url))
            self.channel = self.connection.channel()

            # Declare exchange
            self.channel.exchange_declare(exchange=self.exchange_name, exchange_type="topic", durable=True)

            logger.info(f"Connected to RabbitMQ exchange: {self.exchange_name}")
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    def publish_analysis_request(
        self,
        file_path: str,
        recording_id: str,
        routing_key: str = "file.analyze",
        correlation_id: Optional[str] = None,
        is_retry: bool = False,
        is_user_request: bool = False,
        custom_priority: Optional[int] = None,
    ) -> bool:
        """Publish an analysis request with calculated priority.

        Args:
            file_path: Path to the audio file
            recording_id: Recording ID for database storage
            routing_key: Routing key for the message
            correlation_id: Correlation ID for tracking
            is_retry: Whether this is a retry attempt
            is_user_request: Whether this was directly requested by a user
            custom_priority: Custom priority override

        Returns:
            True if published successfully, False otherwise
        """
        if not self.channel:
            self.connect()

        try:
            # Get file size if file exists
            file_size_mb = None
            if os.path.exists(file_path):
                file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

            # Build message
            message: Dict[str, Any] = {
                "file_path": file_path,
                "recording_id": recording_id,
                "retry_count": 1 if is_retry else 0,
                "user_request": is_user_request,
            }

            if file_size_mb is not None:
                message["file_size_mb"] = file_size_mb

            # Add priority to message
            message = add_priority_to_message(message, self.priority_calculator, correlation_id=correlation_id)

            # Set message properties
            properties = BasicProperties(
                delivery_mode=2,  # Persistent
                correlation_id=correlation_id,
                priority=message["priority"] if self.priority_config.enable_priority else None,
            )

            # Publish message
            if self.channel is not None:
                self.channel.basic_publish(
                    exchange=self.exchange_name,
                    routing_key=routing_key,
                    body=json.dumps(message),
                    properties=properties,
                )

            logger.info(
                f"Published analysis request for {file_path}",
                extra={
                    "correlation_id": correlation_id,
                    "priority": message["priority"],
                    "routing_key": routing_key,
                },
            )

            return True

        except Exception as e:
            logger.error(f"Failed to publish message: {e}", extra={"correlation_id": correlation_id})
            return False

    def close(self) -> None:
        """Close the connection to RabbitMQ."""
        if self.channel:
            self.channel.close()
        if self.connection:
            self.connection.close()
        logger.info("Message publisher closed")

    def __enter__(self) -> "PriorityMessagePublisher":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
