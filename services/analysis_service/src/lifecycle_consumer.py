"""Handle file lifecycle events for the analysis service."""

import json
import logging
import os
import sys

import pika
import pika.exceptions
from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic, BasicProperties

from .audio_cache import AudioCache
from .storage_handler import StorageHandler

logger = logging.getLogger(__name__)


class LifecycleEventConsumer:
    """Handles file lifecycle events for cleanup of analysis data."""

    def __init__(
        self,
        rabbitmq_url: str,
        queue_name: str = "analysis.lifecycle.events",
        exchange_name: str = "file_events",
        routing_key: str = "file.#",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        enable_cache: bool = True,
    ) -> None:
        """Initialize the lifecycle event consumer.

        Args:
            rabbitmq_url: RabbitMQ connection URL
            queue_name: Name of the queue for lifecycle events
            exchange_name: Name of the exchange for file events
            routing_key: Routing key pattern for file events
            redis_host: Redis server hostname
            redis_port: Redis server port
            enable_cache: Whether to enable cache cleanup
        """
        self.rabbitmq_url = rabbitmq_url
        self.queue_name = queue_name
        self.exchange_name = exchange_name
        self.routing_key = routing_key
        self.connection: pika.BlockingConnection | None = None
        self.channel: BlockingChannel | None = None

        # Initialize cache handler for cleanup
        self.cache: AudioCache | None = None
        if enable_cache:
            self.cache = AudioCache(
                redis_host=redis_host,
                redis_port=redis_port,
                enabled=True,
            )

        # Initialize storage handler for Neo4j cleanup
        self.storage_handler: StorageHandler | None = None
        self._init_storage_handler()

    def _init_storage_handler(self) -> None:
        """Initialize storage handler with error handling."""
        try:
            self.storage_handler = StorageHandler()
        except Exception as e:
            logger.warning(f"Could not initialize storage handler: {e}")
            self.storage_handler = None

    def connect(self) -> None:
        """Establish connection to RabbitMQ."""
        try:
            parameters = pika.URLParameters(self.rabbitmq_url)
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()

            # Declare exchange (topic type for routing patterns)
            self.channel.exchange_declare(
                exchange=self.exchange_name,
                exchange_type="topic",
                durable=True,
            )

            # Declare queue
            self.channel.queue_declare(queue=self.queue_name, durable=True)

            # Bind queue to exchange with routing key pattern
            self.channel.queue_bind(
                exchange=self.exchange_name,
                queue=self.queue_name,
                routing_key=self.routing_key,
            )

            logger.info(f"Connected to RabbitMQ for lifecycle events - Queue: {self.queue_name}")

        except pika.exceptions.AMQPConnectionError as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise

    def start_consuming(self) -> None:
        """Start consuming lifecycle event messages."""
        if not self.channel:
            self.connect()

        if not self.channel:
            raise RuntimeError("Failed to establish channel connection")

        # Set up consumer
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self.process_message,
            auto_ack=False,
        )

        logger.info("Starting to consume lifecycle events...")
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("Stopping lifecycle event consumer...")
            self.stop_consuming()

    def stop_consuming(self) -> None:
        """Stop consuming messages and close connections."""
        if self.channel:
            self.channel.stop_consuming()
            self.channel.close()
        if self.connection:
            self.connection.close()
        logger.info("Lifecycle event consumer stopped")

    def process_message(
        self,
        channel: BlockingChannel,
        method: Basic.Deliver,
        properties: BasicProperties,
        body: bytes,
    ) -> None:
        """Process a lifecycle event message.

        Args:
            channel: The channel object
            method: Delivery method containing routing information
            properties: Message properties
            body: Message body as bytes
        """
        try:
            # Parse message
            message = json.loads(body.decode("utf-8"))
            event_type = message.get("event_type", "")
            file_path = message.get("file_path", "")
            correlation_id = message.get("correlation_id", properties.correlation_id)

            logger.info(
                f"Processing lifecycle event: {event_type} for {file_path}",
                extra={"correlation_id": correlation_id},
            )

            # Handle different event types
            if event_type == "deleted":
                self.handle_file_deleted(file_path, correlation_id)
            elif event_type in {"moved", "renamed"}:
                # Both moved and renamed events use old_path field
                old_path = message.get("old_path", "")
                if old_path:
                    self.handle_file_moved(old_path, file_path, correlation_id)
                else:
                    logger.warning(
                        f"Move/rename event missing old_path: {message}",
                        extra={"correlation_id": correlation_id},
                    )

            # Acknowledge message
            channel.basic_ack(delivery_tag=method.delivery_tag)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in message: {e}")
            # Reject message without requeue (dead letter)
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        except Exception as e:
            logger.error(f"Error processing lifecycle event: {e}")
            # Requeue message for retry
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

    def handle_file_deleted(self, file_path: str, correlation_id: str) -> None:
        """Handle file deletion event.

        Args:
            file_path: Path of the deleted file
            correlation_id: Correlation ID for tracking
        """
        logger.info(
            f"Handling file deletion: {file_path}",
            extra={"correlation_id": correlation_id},
        )

        # Clear Redis cache entries
        if self.cache:
            self.clear_cache_entries(file_path, correlation_id)

        # Remove Neo4j analysis data
        self.remove_neo4j_data(file_path, correlation_id)

    def handle_file_moved(self, old_path: str, new_path: str, correlation_id: str) -> None:
        """Handle file move/rename event.

        Args:
            old_path: Original file path
            new_path: New file path
            correlation_id: Correlation ID for tracking
        """
        logger.info(
            f"Handling file move/rename: {old_path} -> {new_path}",
            extra={"correlation_id": correlation_id},
        )

        # Clear old cache entries (new ones will be created on next analysis)
        if self.cache:
            self.clear_cache_entries(old_path, correlation_id)

        # Update Neo4j data with new path
        self.update_neo4j_path(old_path, new_path, correlation_id)

    def handle_file_renamed(self, old_path: str, new_path: str, correlation_id: str) -> None:
        """Handle file rename event (legacy method for compatibility).

        Args:
            old_path: Original file path
            new_path: New file path
            correlation_id: Correlation ID for tracking
        """
        # Just delegate to handle_file_moved since they do the same thing
        self.handle_file_moved(old_path, new_path, correlation_id)

    def clear_cache_entries(self, file_path: str, correlation_id: str) -> None:
        """Clear all cache entries for a file.

        Args:
            file_path: Path of the file
            correlation_id: Correlation ID for tracking
        """
        if not self.cache or not self.cache.redis_client:
            return

        try:
            # Generate file hash to find cache keys
            file_hash = self.cache._generate_file_hash(file_path)
            if not file_hash:
                logger.warning(
                    f"Could not generate hash for {file_path}, skipping cache cleanup",
                    extra={"correlation_id": correlation_id},
                )
                return

            # Build cache keys for all prefixes
            prefixes = [
                self.cache.BPM_PREFIX,
                self.cache.TEMPORAL_PREFIX,
                self.cache.KEY_PREFIX,
                self.cache.MOOD_PREFIX,
            ]

            deleted_count = 0
            for prefix in prefixes:
                cache_key = self.cache._build_cache_key(prefix, file_hash)
                result = self.cache.redis_client.delete(cache_key)
                if result:
                    deleted_count += 1
                    logger.debug(
                        f"Deleted cache key: {cache_key}",
                        extra={"correlation_id": correlation_id},
                    )

            logger.info(
                f"Cleared {deleted_count} cache entries for {file_path}",
                extra={"correlation_id": correlation_id},
            )

        except Exception as e:
            logger.error(
                f"Failed to clear cache entries for {file_path}: {e}",
                extra={"correlation_id": correlation_id},
            )

    def remove_neo4j_data(self, file_path: str, correlation_id: str) -> None:
        """Remove Neo4j analysis data for a file.

        Args:
            file_path: Path of the file
            correlation_id: Correlation ID for tracking
        """
        try:
            if not self.storage_handler or not self.storage_handler.neo4j_repo:
                logger.warning(
                    "Neo4j repository not initialized, skipping cleanup",
                    extra={"correlation_id": correlation_id},
                )
                return

            # Delete recording and all related metadata nodes using file path
            deleted = self.storage_handler.neo4j_repo.delete_recording_by_filepath(file_path)

            if deleted:
                logger.info(
                    f"Removed Neo4j data for {file_path}",
                    extra={"correlation_id": correlation_id},
                )
            else:
                logger.debug(
                    f"No Neo4j data found for {file_path}",
                    extra={"correlation_id": correlation_id},
                )

        except Exception as e:
            logger.error(
                f"Failed to remove Neo4j data for {file_path}: {e}",
                extra={"correlation_id": correlation_id},
            )

    def update_neo4j_path(self, old_path: str, new_path: str, correlation_id: str) -> None:
        """Update Neo4j recording path after move/rename.

        Args:
            old_path: Original file path
            new_path: New file path
            correlation_id: Correlation ID for tracking
        """
        try:
            if not self.storage_handler or not self.storage_handler.neo4j_repo:
                logger.warning(
                    "Neo4j repository not initialized, skipping path update",
                    extra={"correlation_id": correlation_id},
                )
                return

            # Update the file path in Neo4j
            # This would require adding an update method to the Neo4j repository
            # For now, we'll delete the old and let the next analysis recreate
            logger.info(
                f"Updating Neo4j path from {old_path} to {new_path}",
                extra={"correlation_id": correlation_id},
            )

            # Remove old data (new will be created on next analysis)
            deleted = self.storage_handler.neo4j_repo.delete_recording_by_filepath(old_path)

            if deleted:
                logger.info(
                    f"Removed old Neo4j data for {old_path}, new data will be created on next analysis",
                    extra={"correlation_id": correlation_id},
                )

        except Exception as e:
            logger.error(
                f"Failed to update Neo4j path from {old_path} to {new_path}: {e}",
                extra={"correlation_id": correlation_id},
            )


def main() -> None:
    """Main entry point for lifecycle event consumer."""

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Get configuration from environment
    rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))

    # Create and start consumer
    consumer = LifecycleEventConsumer(
        rabbitmq_url=rabbitmq_url,
        redis_host=redis_host,
        redis_port=redis_port,
        enable_cache=True,
    )

    try:
        consumer.connect()
        consumer.start_consuming()
    except KeyboardInterrupt:
        logger.info("Shutting down lifecycle event consumer")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error in lifecycle consumer: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
