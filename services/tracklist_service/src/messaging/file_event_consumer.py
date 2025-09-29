"""RabbitMQ consumer for handling file lifecycle events."""

import asyncio
import json
import logging
from typing import Any

from aio_pika import ExchangeType, IncomingMessage
from services.tracklist_service.src.config import get_config
from services.tracklist_service.src.messaging.rabbitmq_client import RabbitMQClient, RabbitMQConfig
from services.tracklist_service.src.services.file_lifecycle_service import FileLifecycleService
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,  # type: ignore[attr-defined]  # SQLAlchemy 2.0 async feature not recognized by mypy type stubs
    create_async_engine,
)

logger = logging.getLogger(__name__)


class FileEventConsumer:
    """Consumer for file lifecycle events from file_watcher service."""

    def __init__(self, rabbitmq_client: RabbitMQClient | None = None):
        """Initialize file event consumer.

        Args:
            rabbitmq_client: RabbitMQ client instance
        """
        # Get config for RabbitMQ
        config = get_config()
        rabbitmq_config = RabbitMQConfig()
        self.rabbitmq_client = rabbitmq_client or RabbitMQClient(rabbitmq_config)
        self.channel: Any | None = None
        self.queue: Any | None = None
        self.consumer_tag: str | None = None

        # Database setup
        db_url = f"postgresql+asyncpg://{config.database.user}:{config.database.password}@{config.database.host}:{config.database.port}/{config.database.name}"
        self.engine = create_async_engine(db_url, echo=False)
        self.SessionLocal = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
        )

        # Configuration
        self.soft_delete_enabled = True  # Default to True

    async def connect(self) -> None:
        """Connect to RabbitMQ and setup consumer."""
        try:
            await self.rabbitmq_client.connect()
            if self.rabbitmq_client.connection:
                self.channel = await self.rabbitmq_client.connection.channel()

            # Set prefetch count for load balancing
            if self.channel:
                await self.channel.set_qos(prefetch_count=10)

                # Declare the file events exchange (should already exist from file_watcher)
                exchange = await self.channel.declare_exchange(
                    "file_events",
                    ExchangeType.TOPIC,
                    durable=True,
                )

                # Declare consumer queue for file lifecycle events
                self.queue = await self.channel.declare_queue(
                    "tracklist.file.lifecycle",
                    durable=True,
                    arguments={
                        "x-dead-letter-exchange": "file_events.dlx",
                        "x-message-ttl": 86400000,  # 24 hours
                    },
                )

                # Bind queue to file event routing keys
                routing_keys = [
                    "file.created",
                    "file.modified",
                    "file.deleted",
                    "file.moved",
                    "file.renamed",
                ]

                for routing_key in routing_keys:
                    if self.queue:
                        await self.queue.bind(exchange, routing_key)

            logger.info("Connected to RabbitMQ for file event consumption")

        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from RabbitMQ."""
        try:
            if self.consumer_tag and self.channel:
                await self.channel.cancel(self.consumer_tag)
                self.consumer_tag = None

            await self.rabbitmq_client.disconnect()
            logger.info("Disconnected from RabbitMQ")

        except Exception as e:
            logger.error(f"Error disconnecting from RabbitMQ: {e}")

    async def start_consuming(self) -> None:
        """Start consuming file event messages."""
        if not self.queue:
            await self.connect()

        try:
            # Start consuming messages
            if self.queue:
                async with self.queue.iterator() as queue_iter:
                    self.consumer_tag = queue_iter.consumer_tag
                    logger.info("Started consuming file events")

                    async for message in queue_iter:
                        async with message.process():
                            await self.process_message(message)

        except asyncio.CancelledError:
            logger.info("File event consumer cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in file event consumer: {e}")
            raise

    async def process_message(self, message: IncomingMessage) -> None:
        """Process a file event message.

        Args:
            message: Incoming RabbitMQ message
        """
        try:
            # Parse message body
            body = json.loads(message.body.decode())
            event_type = body.get("event_type")
            file_path = body.get("file_path")
            old_path = body.get("old_path")  # For moved/renamed events
            sha256_hash = body.get("sha256_hash")
            xxh128_hash = body.get("xxh128_hash")
            file_size = body.get("size_bytes")

            if file_size and isinstance(file_size, str):
                file_size = int(file_size)

            correlation_id = body.get("correlation_id", "unknown")

            logger.info(
                f"Processing file event: {event_type} for {file_path}",
                extra={"correlation_id": correlation_id},
            )

            # Create database session
            async with self.SessionLocal() as session:
                lifecycle_service = FileLifecycleService(session)

                # Handle event based on type
                success = False
                error = None

                if event_type == "created":
                    success, error = await lifecycle_service.handle_file_created(
                        file_path, sha256_hash, xxh128_hash, file_size
                    )

                elif event_type == "modified":
                    success, error = await lifecycle_service.handle_file_modified(
                        file_path, sha256_hash, xxh128_hash, file_size
                    )

                elif event_type == "deleted":
                    success, error = await lifecycle_service.handle_file_deleted(
                        file_path, soft_delete=self.soft_delete_enabled
                    )

                elif event_type == "moved":
                    if old_path:
                        success, error = await lifecycle_service.handle_file_moved(
                            old_path, file_path, sha256_hash, xxh128_hash
                        )
                    else:
                        logger.error(f"Move event missing old_path: {file_path}")
                        error = "Missing old_path for move event"

                elif event_type == "renamed":
                    if old_path:
                        success, error = await lifecycle_service.handle_file_renamed(
                            old_path, file_path, sha256_hash, xxh128_hash
                        )
                    else:
                        logger.error(f"Rename event missing old_path: {file_path}")
                        error = "Missing old_path for rename event"

                else:
                    logger.warning(f"Unknown event type: {event_type}")
                    success = True  # Don't requeue unknown events

                if success:
                    logger.info(
                        f"Successfully processed {event_type} event for {file_path}",
                        extra={"correlation_id": correlation_id},
                    )
                else:
                    logger.error(
                        f"Failed to process {event_type} event for {file_path}: {error}",
                        extra={"correlation_id": correlation_id},
                    )
                    # For critical errors, we might want to send to DLQ
                    # For now, we'll just log and acknowledge

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in message: {e}")
        except Exception as e:
            logger.error(f"Error processing file event message: {e}")
            # Re-raise to let the message be requeued
            raise

    async def cleanup_old_deletes(self, days_old: int = 30) -> None:
        """Periodic task to cleanup old soft-deleted records.

        Args:
            days_old: Number of days after which to permanently delete soft-deleted records
        """
        try:
            async with self.SessionLocal() as session:
                lifecycle_service = FileLifecycleService(session)
                count = await lifecycle_service.cleanup_old_soft_deletes(days_old)
                logger.info(f"Cleaned up {count} old soft-deleted records")
        except Exception as e:
            logger.error(f"Error during cleanup of old soft-deletes: {e}")


async def main() -> None:
    """Main entry point for file event consumer."""
    consumer = FileEventConsumer()

    try:
        await consumer.connect()

        # Start periodic cleanup task
        async def periodic_cleanup() -> None:
            while True:
                await asyncio.sleep(86400)  # Run daily
                await consumer.cleanup_old_deletes(30)

        # Start cleanup task in background
        cleanup_task = asyncio.create_task(periodic_cleanup())

        # Start consuming messages
        await consumer.start_consuming()

    except KeyboardInterrupt:
        logger.info("Shutting down file event consumer")
    finally:
        cleanup_task.cancel()
        await consumer.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
