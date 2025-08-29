"""Async message publisher for RabbitMQ operations."""

import json
from datetime import UTC, datetime
from typing import Any

import aio_pika
import structlog
from aio_pika import Channel, Connection, ExchangeType

logger = structlog.get_logger()


class AsyncMessagePublisher:
    """Async publisher for sending messages to RabbitMQ."""

    def __init__(self, rabbitmq_url: str, instance_id: str) -> None:
        """Initialize the async message publisher.

        Args:
            rabbitmq_url: RabbitMQ connection URL
            instance_id: Unique identifier for this instance
        """
        self.rabbitmq_url = rabbitmq_url
        self.instance_id = instance_id
        self.connection: Connection | None = None
        self.channel: Channel | None = None
        self.exchange_name = "file_events"
        self.routing_key = "file.discovered"

    async def connect(self) -> None:
        """Establish async connection to RabbitMQ."""
        try:
            self.connection = await aio_pika.connect_robust(
                self.rabbitmq_url,
                connection_class=aio_pika.RobustConnection,
                reconnect_interval=5,
                fail_fast=False,
            )

            self.channel = await self.connection.channel()

            # Set prefetch count for better load distribution
            await self.channel.set_qos(prefetch_count=10)

            # Declare exchange
            await self.channel.declare_exchange(
                self.exchange_name,
                ExchangeType.TOPIC,
                durable=True,
            )

            logger.info(
                "Connected to RabbitMQ",
                exchange=self.exchange_name,
                instance_id=self.instance_id,
            )

        except Exception as e:
            logger.error(
                "Failed to connect to RabbitMQ",
                error=str(e),
                instance_id=self.instance_id,
            )
            raise

    async def disconnect(self) -> None:
        """Close async connection to RabbitMQ."""
        if self.channel:
            await self.channel.close()
        if self.connection:
            await self.connection.close()
        logger.info("Disconnected from RabbitMQ", instance_id=self.instance_id)

    async def publish_file_event(
        self,
        event_type: str,
        file_path: str,
        instance_id: str,
        sha256_hash: str | None = None,
        xxh128_hash: str | None = None,
        old_path: str | None = None,
    ) -> bool:
        """Publish a file event message asynchronously.

        Args:
            event_type: Type of event (created, modified, deleted, moved, renamed)
            file_path: Current path to the file
            instance_id: ID of the watcher instance that detected the event
            sha256_hash: SHA256 hash of the file (None for deleted files)
            xxh128_hash: XXH128 hash of the file (None for deleted files)
            old_path: Previous path (for moved/renamed events)

        Returns:
            True if message was published successfully
        """
        if not self.channel:
            logger.error("Cannot publish - not connected to RabbitMQ")
            return False

        try:
            # Build message payload
            message: dict[str, Any] = {
                "event_type": event_type,
                "file_path": file_path,
                "timestamp": datetime.now(UTC).isoformat(),
                "instance_id": instance_id,
            }

            # Add hashes for non-deleted files
            if event_type != "deleted":
                if sha256_hash:
                    message["sha256_hash"] = sha256_hash
                if xxh128_hash:
                    message["xxh128_hash"] = xxh128_hash

            # Add old path for move/rename events
            if old_path and event_type in ["moved", "renamed"]:
                message["old_path"] = old_path

            # Determine routing key based on event type
            routing_key = f"file.{event_type}"

            # Create message
            aio_message = aio_pika.Message(
                body=json.dumps(message).encode(),
                content_type="application/json",
                delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
            )

            # Get exchange
            exchange = await self.channel.get_exchange(self.exchange_name)

            # Publish message
            await exchange.publish(
                aio_message,
                routing_key=routing_key,
            )

            logger.info(
                f"Published {event_type} event",
                file_path=file_path,
                routing_key=routing_key,
                sha256_hash=sha256_hash,
                instance_id=instance_id,
            )

            return True

        except Exception as e:
            logger.error(
                f"Failed to publish {event_type} event",
                file_path=file_path,
                error=str(e),
                exc_info=True,
            )
            return False

    async def publish_batch(self, events: list[dict[str, Any]]) -> int:
        """Publish multiple events in a batch for better performance.

        Args:
            events: List of event dictionaries to publish

        Returns:
            Number of successfully published events
        """
        if not self.channel:
            logger.error("Cannot publish batch - not connected to RabbitMQ")
            return 0

        published = 0

        try:
            exchange = await self.channel.get_exchange(self.exchange_name)

            for event in events:
                try:
                    # Extract event details
                    event_type = event.get("event_type", "created")
                    routing_key = f"file.{event_type}"

                    # Add timestamp if not present
                    if "timestamp" not in event:
                        event["timestamp"] = datetime.now(UTC).isoformat()

                    # Add instance ID
                    event["instance_id"] = self.instance_id

                    # Create and publish message
                    message = aio_pika.Message(
                        body=json.dumps(event).encode(),
                        content_type="application/json",
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    )

                    await exchange.publish(message, routing_key=routing_key)
                    published += 1

                except Exception as e:
                    logger.error(
                        "Failed to publish event in batch",
                        event=event,
                        error=str(e),
                    )

            logger.info(
                "Published batch of events",
                total=len(events),
                successful=published,
                instance_id=self.instance_id,
            )

        except Exception as e:
            logger.error(
                "Failed to publish batch",
                error=str(e),
                exc_info=True,
            )

        return published
