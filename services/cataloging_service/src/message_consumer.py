"""RabbitMQ message consumer for cataloging service."""

import asyncio
import json
import logging

import aio_pika
from aio_pika import ExchangeType, IncomingMessage
from services.tracklist_service.src.services.file_lifecycle_service import FileLifecycleService
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from .config import get_config

logger = logging.getLogger(__name__)


class CatalogingMessageConsumer:
    """Consumer for file lifecycle events in cataloging service."""

    def __init__(self) -> None:
        """Initialize the message consumer."""
        self.config = get_config()
        self.connection: aio_pika.Connection | None = None
        self.channel: aio_pika.Channel | None = None
        self.queue: aio_pika.Queue | None = None

        # Database setup
        self.engine = create_async_engine(self.config.database.url, echo=False)
        self.SessionLocal = sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def connect(self) -> None:
        """Connect to RabbitMQ and setup consumer."""
        try:
            # Connect to RabbitMQ
            self.connection = await aio_pika.connect_robust(
                self.config.rabbitmq.url,
                client_properties={
                    "connection_name": "cataloging-service",
                },
            )

            # Create channel
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=10)

            # Declare the file events exchange
            exchange = await self.channel.declare_exchange(
                self.config.rabbitmq.exchange,
                ExchangeType.TOPIC,
                durable=True,
            )

            # Declare consumer queue
            self.queue = await self.channel.declare_queue(
                self.config.rabbitmq.queue,
                durable=True,
                arguments={
                    "x-dead-letter-exchange": f"{self.config.rabbitmq.exchange}.dlx",
                    "x-message-ttl": 86400000,  # 24 hours
                },
            )

            # Bind queue to routing keys
            routing_keys = [
                "file.created",
                "file.modified",
                "file.deleted",
                "file.moved",
                "file.renamed",
            ]

            for routing_key in routing_keys:
                await self.queue.bind(exchange, routing_key)

            logger.info("Connected to RabbitMQ for cataloging service")

        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from RabbitMQ."""
        try:
            if self.connection and not self.connection.is_closed:
                await self.connection.close()
            logger.info("Disconnected from RabbitMQ")
        except Exception as e:
            logger.error(f"Error disconnecting from RabbitMQ: {e}")

    async def start_consuming(self) -> None:
        """Start consuming messages."""
        if not self.queue:
            await self.connect()

        try:
            # Start consuming messages
            assert self.queue is not None  # For mypy
            async with self.queue.iterator() as queue_iter:
                logger.info("Started consuming file events for cataloging")

                async for message in queue_iter:
                    async with message.process():
                        await self.process_message(message)

        except asyncio.CancelledError:
            logger.info("Cataloging consumer cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in cataloging consumer: {e}")
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
                f"Cataloging: Processing {event_type} event for {file_path}", extra={"correlation_id": correlation_id}
            )

            # Use the shared FileLifecycleService to handle database operations
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
                        file_path, soft_delete=self.config.service.soft_delete_enabled
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
                        f"Cataloging: Successfully processed {event_type} event for {file_path}",
                        extra={"correlation_id": correlation_id},
                    )
                else:
                    logger.error(
                        f"Cataloging: Failed to process {event_type} event for {file_path}: {error}",
                        extra={"correlation_id": correlation_id},
                    )

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in message: {e}")
        except Exception as e:
            logger.error(f"Error processing cataloging message: {e}")
            # Re-raise to let the message be requeued
            raise

    async def cleanup_old_deletes(self) -> None:
        """Periodic task to cleanup old soft-deleted records."""
        try:
            async with self.SessionLocal() as session:
                lifecycle_service = FileLifecycleService(session)
                count = await lifecycle_service.cleanup_old_soft_deletes(self.config.service.cleanup_interval_days)
                logger.info(f"Cataloging: Cleaned up {count} old soft-deleted records")
        except Exception as e:
            logger.error(f"Error during cleanup of old soft-deletes: {e}")


async def main() -> None:
    """Main entry point for cataloging message consumer."""
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    consumer = CatalogingMessageConsumer()

    try:
        await consumer.connect()

        # Start periodic cleanup task
        async def periodic_cleanup() -> None:
            while True:
                await asyncio.sleep(86400)  # Run daily
                await consumer.cleanup_old_deletes()

        # Start cleanup task in background
        cleanup_task = asyncio.create_task(periodic_cleanup())

        # Start consuming messages
        await consumer.start_consuming()

    except KeyboardInterrupt:
        logger.info("Shutting down cataloging consumer")
    finally:
        cleanup_task.cancel()
        await consumer.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
