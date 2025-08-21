"""Message publisher for file discovery events."""

import json
import uuid
from datetime import UTC, datetime
from typing import Any

import pika
import structlog

logger = structlog.get_logger()


class MessagePublisher:
    """Publishes file discovery messages to RabbitMQ."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5672,
        username: str = "guest",
        password: str = "guest",
        exchange: str = "file_events",
        routing_key: str = "file.discovered",
    ) -> None:
        """Initialize the message publisher.

        Args:
            host: RabbitMQ host
            port: RabbitMQ port
            username: RabbitMQ username
            password: RabbitMQ password
            exchange: Exchange name for publishing
            routing_key: Routing key for messages
        """
        self.host = host
        self.port = port
        self.exchange = exchange
        self.routing_key = routing_key
        self.connection: pika.BlockingConnection | None = None
        self.channel: pika.channel.Channel | None = None

        # Setup credentials
        self.credentials = pika.PlainCredentials(username, password)
        self.connection_params = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            credentials=self.credentials,
            heartbeat=30,
            blocked_connection_timeout=300,
        )

    def connect(self) -> None:
        """Connect to RabbitMQ."""
        try:
            self.connection = pika.BlockingConnection(self.connection_params)
            self.channel = self.connection.channel()

            # Declare exchange (idempotent operation)
            self.channel.exchange_declare(exchange=self.exchange, exchange_type="topic", durable=True)

            logger.info("Connected to RabbitMQ", host=self.host, port=self.port, exchange=self.exchange)
        except Exception as e:
            logger.error("Failed to connect to RabbitMQ", host=self.host, port=self.port, error=str(e))
            raise

    def disconnect(self) -> None:
        """Disconnect from RabbitMQ."""
        if self.connection and not self.connection.is_closed:
            try:
                self.connection.close()
                logger.info("Disconnected from RabbitMQ")
            except Exception as e:
                logger.warning("Error disconnecting from RabbitMQ", error=str(e))

    def publish_file_discovered(self, file_info: dict[str, str]) -> bool:
        """Publish a file discovery event.

        Args:
            file_info: Dictionary containing file information

        Returns:
            True if message was published successfully
        """
        if not self.channel or (self.connection and self.connection.is_closed):
            logger.warning("Not connected to RabbitMQ, attempting to reconnect")
            try:
                self.connect()
            except Exception:
                return False

        # Generate correlation ID for tracing
        correlation_id = str(uuid.uuid4())

        # Build message payload
        message = {
            "correlation_id": correlation_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": "file_discovered",
            "file_info": file_info,
            "file_type": self._determine_file_type(file_info.get("extension", "")),
        }

        # Special handling for OGG files
        if file_info.get("extension", "").lower() in [".ogg", ".oga"]:
            message["format_family"] = "ogg_vorbis"
            logger.info(
                "Publishing OGG file discovery event",
                correlation_id=correlation_id,
                file_path=file_info.get("path"),
                extension=file_info.get("extension"),
            )

        try:
            assert self.channel is not None  # For mypy
            self.channel.basic_publish(
                exchange=self.exchange,
                routing_key=self.routing_key,
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Persistent message
                    correlation_id=correlation_id,
                    content_type="application/json",
                ),
            )

            logger.debug(
                "File discovery event published",
                correlation_id=correlation_id,
                file_path=file_info.get("path"),
                routing_key=self.routing_key,
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to publish message",
                correlation_id=correlation_id,
                error=str(e),
                file_path=file_info.get("path"),
            )
            return False

    def _determine_file_type(self, extension: str) -> str:
        """Determine the file type category from extension.

        Args:
            extension: File extension (with or without dot)

        Returns:
            File type category string
        """
        ext = extension.lower()
        if ext in [".mp3"]:
            return "mp3"
        elif ext in [".flac"]:
            return "flac"
        elif ext in [".wav", ".wave"]:
            return "wav"
        elif ext in [".m4a", ".mp4", ".m4b", ".m4p", ".m4v", ".m4r"]:
            return "mp4"
        elif ext in [".ogg", ".oga"]:
            return "ogg"
        else:
            return "unknown"

    def __enter__(self) -> "MessagePublisher":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.disconnect()
