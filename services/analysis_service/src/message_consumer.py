"""RabbitMQ message consumer for analysis service."""

import json
import logging
import os
import time
from typing import Any, Callable, Dict, Optional

import pika
import pika.exceptions
from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic, BasicProperties

from .audio_cache import AudioCache
from .bpm_detector import BPMDetector
from .temporal_analyzer import TemporalAnalyzer

logger = logging.getLogger(__name__)


class MessageConsumer:
    """Handles RabbitMQ message consumption for file analysis."""

    def __init__(
        self,
        rabbitmq_url: str,
        queue_name: str = "analysis_queue",
        exchange_name: str = "tracktion_exchange",
        routing_key: str = "file.analyze",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        enable_cache: bool = True,
        enable_temporal_analysis: bool = True,
    ) -> None:
        """Initialize the message consumer.

        Args:
            rabbitmq_url: RabbitMQ connection URL
            queue_name: Name of the queue to consume from
            exchange_name: Name of the exchange
            routing_key: Routing key for message binding
            redis_host: Redis server hostname for caching
            redis_port: Redis server port
            enable_cache: Whether to use Redis caching
            enable_temporal_analysis: Whether to perform temporal BPM analysis
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
        self.enable_temporal_analysis = enable_temporal_analysis

        # Initialize BPM detection components
        self.bpm_detector = BPMDetector()
        self.temporal_analyzer = TemporalAnalyzer() if enable_temporal_analysis else None

        # Initialize cache if enabled
        self.cache: Optional[AudioCache] = None
        if enable_cache:
            try:
                self.cache = AudioCache(redis_host=redis_host, redis_port=redis_port)
            except Exception as e:
                logger.warning(f"Failed to initialize cache: {e}. Processing without cache.")
                self.cache = None

        # Initialize storage handler (will be set by tests or main application)
        self.storage: Optional[Any] = None

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

    def process_audio_file(self, file_path: str, recording_id: str) -> Dict[str, Any]:
        """Process audio file for BPM detection.

        Args:
            file_path: Path to the audio file
            recording_id: Recording ID for database storage

        Returns:
            Processing results including BPM data
        """
        results: Dict[str, Any] = {"recording_id": recording_id, "file_path": file_path}

        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            results["error"] = f"Audio file not found: {file_path}"
            results["bpm_data"] = None
            return results

        # Check cache first
        if self.cache:
            cached_bpm = self.cache.get_bpm_results(file_path)
            if cached_bpm:
                logger.info(f"Using cached BPM results for {file_path}")
                results["bpm_data"] = cached_bpm
                results["from_cache"] = True

                # Check for cached temporal data
                if self.enable_temporal_analysis:
                    cached_temporal = self.cache.get_temporal_results(file_path)
                    if cached_temporal:
                        results["temporal_data"] = cached_temporal

                return results

        # Perform BPM detection
        try:
            logger.info(f"Detecting BPM for {file_path}")
            bpm_results = self.bpm_detector.detect_bpm(file_path)
            results["bpm_data"] = bpm_results
            results["from_cache"] = False

            # Cache BPM results
            if self.cache:
                self.cache.set_bpm_results(
                    file_path,
                    bpm_results,
                    confidence=bpm_results.get("confidence", 0.0),
                    failed=bpm_results.get("error") is not None,
                )

            # Perform temporal analysis if enabled and BPM was successful
            if self.enable_temporal_analysis and self.temporal_analyzer and not bpm_results.get("error"):
                try:
                    logger.info(f"Performing temporal analysis for {file_path}")
                    temporal_results = self.temporal_analyzer.analyze_temporal_bpm(file_path)
                    results["temporal_data"] = temporal_results

                    # Cache temporal results
                    if self.cache:
                        self.cache.set_temporal_results(
                            file_path, temporal_results, stability_score=temporal_results.get("stability_score", 0.0)
                        )
                except Exception as e:
                    logger.error(f"Temporal analysis failed for {file_path}: {e}")
                    results["temporal_data"] = {"error": str(e)}

        except Exception as e:
            logger.error(f"BPM detection failed for {file_path}: {e}")
            results["bpm_data"] = {"error": str(e)}

            # Cache the failure to avoid re-processing immediately
            if self.cache:
                self.cache.set_bpm_results(file_path, {"error": str(e)}, failed=True)

        # Store results in database if we have valid BPM data
        if self.storage and "bpm_data" in results and results["bpm_data"] and "error" not in results["bpm_data"]:
            try:
                from uuid import UUID

                recording_uuid = UUID(recording_id)
                temporal_data = results.get("temporal_data")
                stored = self.storage.store_bpm_data(recording_uuid, results["bpm_data"], temporal_data=temporal_data)
                results["stored"] = stored
            except Exception as e:
                logger.error(f"Failed to store BPM data: {e}")
                results["storage_error"] = str(e)

        return results

    def consume(self, callback: Optional[Callable[[Dict[str, Any], str], None]] = None) -> None:
        """Start consuming messages from the queue.

        Args:
            callback: Optional function to process results after BPM detection
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

                # Extract required fields
                file_path = message.get("file_path")
                recording_id = message.get("recording_id")

                if not file_path or not recording_id:
                    raise ValueError("Message must contain 'file_path' and 'recording_id'")

                # Process audio file for BPM
                results = self.process_audio_file(file_path, recording_id)

                # Call user callback if provided
                if callback:
                    callback(results, correlation_id)

                # Acknowledge message
                ch.basic_ack(delivery_tag=method.delivery_tag)
                logger.info(
                    "Message processed successfully",
                    extra={
                        "correlation_id": correlation_id,
                        "bpm": results.get("bpm_data", {}).get("bpm"),
                        "confidence": results.get("bpm_data", {}).get("confidence"),
                        "from_cache": results.get("from_cache", False),
                    },
                )

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in message: {e}", extra={"correlation_id": correlation_id})
                # Reject message without requeue (bad format)
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

            except FileNotFoundError as e:
                logger.error(f"File not found: {e}", extra={"correlation_id": correlation_id})
                # Don't requeue - file won't magically appear
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)

            except ValueError as e:
                logger.error(f"Invalid message format: {e}", extra={"correlation_id": correlation_id})
                # Don't requeue - message format is wrong
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
