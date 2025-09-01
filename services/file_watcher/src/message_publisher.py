"""Message publisher for file discovery events."""

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pika
import structlog

logger = structlog.get_logger()


@dataclass
class RabbitMQConfig:
    """Configuration for RabbitMQ connection."""

    host: str = "localhost"
    port: int = 5672
    username: str = "guest"
    password: str = "guest"
    exchange: str = "file_events"
    routing_key: str = "file.discovered"


class MessagePublisher:
    """Publishes file discovery messages to RabbitMQ."""

    def __init__(
        self,
        config: RabbitMQConfig | None = None,
        instance_id: str | None = None,
        watched_directory: str | None = None,
    ) -> None:
        """Initialize the message publisher.

        Args:
            config: RabbitMQ configuration object
            instance_id: Unique identifier for this file watcher instance
            watched_directory: Directory being watched by this instance

        """
        self.config = config or RabbitMQConfig()
        self.instance_id = instance_id or "default"
        self.watched_directory = watched_directory or "/data/music"
        self.connection: pika.BlockingConnection | None = None
        self.channel: pika.channel.Channel | None = None

        # Setup credentials
        self.credentials = pika.PlainCredentials(self.config.username, self.config.password)

        # Include instance ID in connection name for RabbitMQ management visibility
        client_properties = {
            "connection_name": f"file_watcher_{self.instance_id}",
            "instance_id": self.instance_id,
            "watched_directory": self.watched_directory,
        }

        self.connection_params = pika.ConnectionParameters(
            host=self.config.host,
            port=self.config.port,
            credentials=self.credentials,
            heartbeat=30,
            blocked_connection_timeout=300,
            client_properties=client_properties,
        )

    def connect(self) -> None:
        """Connect to RabbitMQ."""
        try:
            self.connection = pika.BlockingConnection(self.connection_params)
            self.channel = self.connection.channel()

            # Declare exchange (idempotent operation)
            self.channel.exchange_declare(exchange=self.config.exchange, exchange_type="topic", durable=True)

            logger.info(
                "Connected to RabbitMQ",
                host=self.config.host,
                port=self.config.port,
                exchange=self.config.exchange,
            )
        except Exception as e:
            logger.error(
                "Failed to connect to RabbitMQ",
                host=self.config.host,
                port=self.config.port,
                error=str(e),
            )
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

        # Build message payload with instance metadata
        message = {
            "correlation_id": correlation_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": "file_discovered",
            "file_info": file_info,
            "file_type": self._determine_file_type(file_info.get("extension", "")),
            "instance_id": self.instance_id,
            "watched_directory": self.watched_directory,
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
                exchange=self.config.exchange,
                routing_key=self.config.routing_key,
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
                routing_key=self.config.routing_key,
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
        if ext in [".flac"]:
            return "flac"
        if ext in [".wav", ".wave"]:
            return "wav"
        if ext in [".m4a", ".mp4", ".m4b", ".m4p", ".m4v", ".m4r"]:
            return "mp4"
        if ext in [".ogg", ".oga"]:
            return "ogg"
        return "unknown"

    def _build_message_payload(self, file_info: dict[str, str], event_type: str, correlation_id: str) -> dict[str, Any]:
        """Build the message payload for file events."""
        message = {
            "correlation_id": correlation_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "file_path": file_info.get("path", ""),
            "instance_id": self.instance_id,
            "watched_directory": self.watched_directory,
        }

        # Add old_path for moved/renamed events
        if "old_path" in file_info:
            message["old_path"] = file_info["old_path"]

        # Add hashes for non-deleted events
        if event_type != "deleted":
            if "sha256_hash" in file_info:
                message["sha256_hash"] = file_info["sha256_hash"]
            if "xxh128_hash" in file_info:
                message["xxh128_hash"] = file_info["xxh128_hash"]

        # Add file metadata if available
        if "size_bytes" in file_info:
            message["size_bytes"] = file_info["size_bytes"]
        if "extension" in file_info:
            message["file_type"] = self._determine_file_type(file_info["extension"])

        # Special handling for OGG files
        if file_info.get("extension", "").lower() in [".ogg", ".oga"]:
            message["format_family"] = "ogg_vorbis"

        return message

    def publish_file_event(self, file_info: dict[str, str], event_type: str) -> bool:
        """Publish a file event with specific event type.

        Args:
            file_info: Dictionary containing file information
            event_type: Type of event (created, modified, deleted, moved, renamed)

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
        message = self._build_message_payload(file_info, event_type, correlation_id)

        # Determine routing key based on event type
        routing_key = f"file.{event_type}"

        try:
            assert self.channel is not None  # For mypy
            self.channel.basic_publish(
                exchange=self.config.exchange,
                routing_key=routing_key,
                body=json.dumps(message),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # Persistent message
                    correlation_id=correlation_id,
                    content_type="application/json",
                ),
            )

            logger.debug(
                "File event published",
                event_type=event_type,
                correlation_id=correlation_id,
                file_path=file_info.get("path"),
                routing_key=routing_key,
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to publish message",
                correlation_id=correlation_id,
                event_type=event_type,
                error=str(e),
                file_path=file_info.get("path"),
            )
            return False

    def __enter__(self) -> "MessagePublisher":
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.disconnect()
