"""RabbitMQ message consumer for cataloging service."""

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import aio_pika
from aio_pika import ExchangeType
from sqlalchemy import func, select

if TYPE_CHECKING:
    from aio_pika.abc import AbstractChannel, AbstractIncomingMessage, AbstractQueue, AbstractRobustConnection

from .config import get_config
from .database import get_db_manager
from .repositories import MetadataRepository, RecordingRepository
from .repositories.recording import Recording

logger = logging.getLogger(__name__)


class CatalogingMessageConsumer:
    """Consumer for file lifecycle events in cataloging service."""

    def __init__(self) -> None:
        """Initialize the message consumer."""
        self.config = get_config()
        self.connection: AbstractRobustConnection | None = None
        self.channel: AbstractChannel | None = None
        self.queue: AbstractQueue | None = None
        self.db_manager = get_db_manager()

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
            if self.queue is None:  # Type guard to satisfy mypy
                raise RuntimeError("Queue is not initialized")
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

    async def process_message(self, message: "AbstractIncomingMessage") -> None:
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
                f"Cataloging: Processing {event_type} event for {file_path}",
                extra={"correlation_id": correlation_id},
            )

            # Handle event based on type using repositories
            async with self.db_manager.get_session() as session:
                recording_repo = RecordingRepository(session)
                metadata_repo = MetadataRepository(session)

                if event_type == "created":
                    # Extract file name from path
                    file_name = file_path.split("/")[-1] if "/" in file_path else file_path

                    # Check if file already exists
                    existing = await recording_repo.get_by_file_path(file_path)
                    if not existing:
                        # Create new recording
                        recording: Any = await recording_repo.create(
                            file_path=file_path,
                            file_name=file_name,
                            sha256_hash=sha256_hash,
                            xxh128_hash=xxh128_hash,
                        )

                        # Add metadata if provided
                        metadata = body.get("metadata", {})
                        if metadata:
                            await metadata_repo.bulk_create(recording.id, metadata)
                    else:
                        # Update existing recording
                        if existing.id is None:
                            raise ValueError("Recording ID cannot be None")
                        await recording_repo.update(
                            existing.id,
                            sha256_hash=sha256_hash,
                            xxh128_hash=xxh128_hash,
                        )

                elif event_type == "modified":
                    # For modified, update the hashes
                    file_name = file_path.split("/")[-1] if "/" in file_path else file_path

                    existing = await recording_repo.get_by_file_path(file_path)
                    if existing:
                        if existing.id is None:
                            raise ValueError("Recording ID cannot be None")
                        await recording_repo.update(
                            existing.id,
                            sha256_hash=sha256_hash,
                            xxh128_hash=xxh128_hash,
                        )

                        # Update metadata if provided
                        metadata = body.get("metadata", {})
                        if metadata:
                            if existing.id is None:
                                raise ValueError("Recording ID cannot be None")
                            for key, value in metadata.items():
                                await metadata_repo.upsert(existing.id, key, value)
                    else:
                        # Create new if doesn't exist
                        recording = await recording_repo.create(
                            file_path=file_path,
                            file_name=file_name,
                            sha256_hash=sha256_hash,
                            xxh128_hash=xxh128_hash,
                        )
                        metadata = body.get("metadata", {})
                        if metadata:
                            await metadata_repo.bulk_create(recording.id, metadata)

                elif event_type == "deleted":
                    existing = await recording_repo.get_by_file_path(file_path)
                    if existing:
                        if existing.id is None:
                            raise ValueError("Recording ID cannot be None")
                        await recording_repo.delete(existing.id)

                elif event_type in ["moved", "renamed"]:
                    if old_path:
                        new_name = file_path.split("/")[-1] if "/" in file_path else file_path
                        existing = await recording_repo.get_by_file_path(old_path)
                        if existing:
                            if existing.id is None:
                                raise ValueError("Recording ID cannot be None")
                            await recording_repo.update(
                                existing.id,
                                file_path=file_path,
                                file_name=new_name,
                            )
                    else:
                        logger.error(f"{event_type} event missing old_path: {file_path}")
                        raise ValueError(f"Missing old_path for {event_type} event")

                else:
                    logger.warning(f"Unknown event type: {event_type}")

                logger.info(
                    f"Cataloging: Successfully processed {event_type} event for {file_path}",
                    extra={"correlation_id": correlation_id},
                )

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in message: {e}")
        except Exception as e:
            logger.error(f"Error processing cataloging message: {e}")
            # Re-raise to let the message be requeued
            raise

    async def cleanup_old_deletes(self) -> None:
        """Periodic task to cleanup old records based on configuration."""
        try:
            # Check if cleanup is enabled
            if not self.config.service.soft_delete_enabled:
                logger.debug("Cataloging: Cleanup disabled in configuration")
                return

            # For now, we implement basic cleanup based on age of records
            # In the future, this could be enhanced with proper soft delete support
            cutoff_date = datetime.now(UTC) - timedelta(days=self.config.service.cleanup_interval_days)

            async with self.db_manager.get_session() as session:
                # Count records older than cutoff date for logging
                count_query = select(func.count()).select_from(Recording).where(Recording.created_at < cutoff_date)
                result = await session.execute(count_query)
                old_record_count = result.scalar() or 0

                if old_record_count > 0:
                    logger.info(
                        f"Cataloging: Found {old_record_count} records older than "
                        f"{self.config.service.cleanup_interval_days} days (cutoff: {cutoff_date.isoformat()})"
                    )
                    logger.info("Cataloging: Cleanup logic placeholder - implement deletion policy as needed")
                else:
                    logger.debug("Cataloging: No old records found for cleanup")

        except Exception as e:
            logger.error(f"Error during cleanup of old records: {e}")


async def main() -> None:
    """Main entry point for cataloging message consumer."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

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
