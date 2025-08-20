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
from .batch_processor import BatchConfig, BatchProcessor
from .bpm_detector import BPMDetector
from .key_detector import KeyDetector
from .model_manager import ModelManager
from .mood_analyzer import MoodAnalyzer
from .priority_queue import PriorityCalculator, PriorityConfig, setup_priority_queue
from .progress_tracker import ProgressTracker
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
        enable_key_detection: bool = True,
        enable_mood_analysis: bool = True,
        models_dir: Optional[str] = None,
        auto_download_models: bool = True,
        priority_config: Optional[PriorityConfig] = None,
        batch_config: Optional[BatchConfig] = None,
        enable_batch_processing: bool = False,
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
            enable_key_detection: Whether to perform musical key detection
            enable_mood_analysis: Whether to perform mood and genre analysis
            models_dir: Directory for TensorFlow models
            auto_download_models: Whether to auto-download missing models
            priority_config: Configuration for priority queue behavior
            batch_config: Configuration for batch processing
            enable_batch_processing: Whether to enable batch processing mode
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
        self.enable_key_detection = enable_key_detection
        self.enable_mood_analysis = enable_mood_analysis

        # Initialize priority calculator
        self.priority_config = priority_config or PriorityConfig()
        self.priority_calculator = PriorityCalculator(self.priority_config)

        # Initialize batch processor if enabled
        self.enable_batch_processing = enable_batch_processing
        self.batch_processor: Optional[BatchProcessor] = None
        if enable_batch_processing:
            batch_config = batch_config or BatchConfig.from_env()
            self.batch_processor = BatchProcessor(
                process_func=self.process_audio_file,
                config=batch_config,
            )
            logger.info(f"Batch processing enabled with {batch_config.default_workers} workers")

        # Initialize BPM detection components
        self.bpm_detector = BPMDetector()
        self.temporal_analyzer = TemporalAnalyzer() if enable_temporal_analysis else None

        # Initialize key detection
        self.key_detector = KeyDetector() if enable_key_detection else None

        # Initialize mood analysis with model manager
        self.model_manager: Optional[ModelManager] = None
        self.mood_analyzer: Optional[MoodAnalyzer] = None
        if enable_mood_analysis:
            try:
                self.model_manager = ModelManager(
                    models_dir=models_dir, auto_download=auto_download_models, lazy_load=True
                )
                self.mood_analyzer = MoodAnalyzer(model_manager=self.model_manager)
                logger.info("Initialized mood analyzer with model manager")
            except Exception as e:
                logger.warning(f"Failed to initialize mood analyzer: {e}")
                self.mood_analyzer = None

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

        # Initialize progress tracker
        self.progress_tracker: Optional[ProgressTracker] = None
        try:
            self.progress_tracker = ProgressTracker(
                redis_host=redis_host,
                redis_port=redis_port,
                key_prefix="analysis:progress",
            )
            logger.info("Initialized progress tracker")
        except Exception as e:
            logger.warning(f"Failed to initialize progress tracker: {e}. Processing without progress tracking.")
            self.progress_tracker = None

    def connect(self) -> None:
        """Establish connection to RabbitMQ with retry logic."""
        while self._retry_count < self._max_retries:
            try:
                self.connection = pika.BlockingConnection(pika.URLParameters(self.rabbitmq_url))
                self.channel = self.connection.channel()

                # Declare exchange
                self.channel.exchange_declare(exchange=self.exchange_name, exchange_type="topic", durable=True)

                # Declare priority queue if priority is enabled
                if self.priority_config.enable_priority:
                    setup_priority_queue(self.channel, self.queue_name, max_priority=self.priority_config.max_priority)
                else:
                    # Declare regular queue
                    self.channel.queue_declare(queue=self.queue_name, durable=True)

                # Bind queue to exchange
                self.channel.queue_bind(
                    exchange=self.exchange_name, queue=self.queue_name, routing_key=self.routing_key
                )

                # Set QoS - higher prefetch for batch processing
                prefetch = 10 if self.enable_batch_processing else 1
                self.channel.basic_qos(prefetch_count=prefetch)

                logger.info(f"Connected to RabbitMQ queue: {self.queue_name}")
                self._retry_count = 0
                return

            except pika.exceptions.AMQPConnectionError as e:
                self._retry_count += 1
                delay = self._base_delay * (2**self._retry_count)
                logger.warning(f"Connection attempt {self._retry_count} failed: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)

        raise ConnectionError(f"Failed to connect to RabbitMQ after {self._max_retries} attempts")

    def process_audio_file(
        self, file_path: str, recording_id: str, correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process audio file for all analysis features.

        Args:
            file_path: Path to the audio file
            recording_id: Recording ID for database storage
            correlation_id: Optional correlation ID for tracking

        Returns:
            Processing results including BPM, key, and mood data
        """
        results: Dict[str, Any] = {"recording_id": recording_id, "file_path": file_path}

        # Track processing start
        if self.progress_tracker and correlation_id:
            self.progress_tracker.track_file_started(correlation_id, "Starting analysis")

        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"Audio file not found: {file_path}")
            results["error"] = f"Audio file not found: {file_path}"
            results["bpm_data"] = None
            # Track failure
            if self.progress_tracker and correlation_id:
                self.progress_tracker.track_file_completed(
                    correlation_id, success=False, error_message=f"Audio file not found: {file_path}"
                )
            return results

        # Check cache first for all results
        cache_hit = False
        if self.cache:
            cached_bpm = self.cache.get_bpm_results(file_path)
            if cached_bpm:
                logger.info(f"Using cached BPM results for {file_path}")
                results["bpm_data"] = cached_bpm
                cache_hit = True

                # Check for cached temporal data
                if self.enable_temporal_analysis:
                    cached_temporal = self.cache.get_temporal_results(file_path)
                    if cached_temporal:
                        results["temporal_data"] = cached_temporal

                # Check for cached key data
                if self.enable_key_detection:
                    cached_key = self.cache.get_key_results(file_path)
                    if cached_key:
                        results["key_data"] = cached_key

                # Check for cached mood data
                if self.enable_mood_analysis:
                    cached_mood = self.cache.get_mood_results(file_path)
                    if cached_mood:
                        results["mood_data"] = cached_mood

        # Perform BPM detection if not cached
        if not cache_hit:
            try:
                logger.info(f"Detecting BPM for {file_path}")
                # Update progress
                if self.progress_tracker and correlation_id:
                    self.progress_tracker.update_progress(correlation_id, 20.0, "Detecting BPM")

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
                        # Update progress
                        if self.progress_tracker and correlation_id:
                            self.progress_tracker.update_progress(correlation_id, 40.0, "Analyzing temporal BPM")

                        temporal_results = self.temporal_analyzer.analyze_temporal_bpm(file_path)
                        results["temporal_data"] = temporal_results

                        # Cache temporal results
                        if self.cache:
                            self.cache.set_temporal_results(
                                file_path,
                                temporal_results,
                                stability_score=temporal_results.get("stability_score", 0.0),
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
        else:
            results["from_cache"] = True

        # Perform key detection if enabled and not cached
        if self.enable_key_detection and self.key_detector and "key_data" not in results:
            try:
                logger.info(f"Detecting musical key for {file_path}")
                # Update progress
                if self.progress_tracker and correlation_id:
                    self.progress_tracker.update_progress(correlation_id, 60.0, "Detecting musical key")
                key_result = self.key_detector.detect_key(file_path)
                if key_result:
                    results["key_data"] = {
                        "key": key_result.key,
                        "scale": key_result.scale,
                        "confidence": key_result.confidence,
                        "agreement": key_result.agreement,
                        "needs_review": key_result.needs_review,
                    }
                    if key_result.alternative_key:
                        results["key_data"]["alternative"] = {
                            "key": key_result.alternative_key,
                            "scale": key_result.alternative_scale,
                        }

                    # Cache the key detection results
                    if self.cache:
                        self.cache.set_key_results(file_path, results["key_data"], confidence=key_result.confidence)
                else:
                    results["key_data"] = {"error": "Key detection failed"}
            except Exception as e:
                logger.error(f"Key detection failed for {file_path}: {e}")
                results["key_data"] = {"error": str(e)}

        # Perform mood analysis if enabled and not cached
        if self.enable_mood_analysis and self.mood_analyzer and "mood_data" not in results:
            try:
                logger.info(f"Analyzing mood and genre for {file_path}")
                # Update progress
                if self.progress_tracker and correlation_id:
                    self.progress_tracker.update_progress(correlation_id, 80.0, "Analyzing mood and genre")
                mood_result = self.mood_analyzer.analyze_mood(file_path)
                if mood_result:
                    results["mood_data"] = {
                        "mood_scores": mood_result.mood_scores,
                        "primary_genre": mood_result.primary_genre,
                        "genre_confidence": mood_result.genre_confidence,
                        "genres": mood_result.genres[:3],  # Top 3 genres
                        "danceability": mood_result.danceability,
                        "energy": mood_result.energy,
                        "valence": mood_result.valence,
                        "arousal": mood_result.arousal,
                        "voice_instrumental": mood_result.voice_instrumental,
                        "overall_confidence": mood_result.overall_confidence,
                        "needs_review": mood_result.needs_review,
                    }

                    # Cache the mood analysis results
                    if self.cache:
                        self.cache.set_mood_results(
                            file_path, results["mood_data"], confidence=mood_result.overall_confidence
                        )
                else:
                    results["mood_data"] = {"error": "Mood analysis failed"}
            except Exception as e:
                logger.error(f"Mood analysis failed for {file_path}: {e}")
                results["mood_data"] = {"error": str(e)}

        # Store all results in database
        if self.storage:
            try:
                from uuid import UUID

                recording_uuid = UUID(recording_id)

                # Store BPM data if available
                if "bpm_data" in results and results["bpm_data"] and "error" not in results["bpm_data"]:
                    temporal_data = results.get("temporal_data")
                    stored = self.storage.store_bpm_data(
                        recording_uuid, results["bpm_data"], temporal_data=temporal_data
                    )
                    results["bpm_stored"] = stored

                # Store key data if available
                if "key_data" in results and results["key_data"] and "error" not in results["key_data"]:
                    stored = self.storage.store_key_data(recording_uuid, results["key_data"])
                    results["key_stored"] = stored

                # Store mood data if available
                if "mood_data" in results and results["mood_data"] and "error" not in results["mood_data"]:
                    stored = self.storage.store_mood_data(recording_uuid, results["mood_data"])
                    results["mood_stored"] = stored

            except Exception as e:
                logger.error(f"Failed to store analysis data: {e}")
                results["storage_error"] = str(e)

        # Track completion
        if self.progress_tracker and correlation_id:
            # Check if any critical errors occurred
            has_error = (
                results.get("error")
                or (results.get("bpm_data") and results["bpm_data"].get("error"))
                or results.get("storage_error")
            )
            if has_error:
                error_msg = results.get("error") or results.get("storage_error") or "Processing failed"
                self.progress_tracker.track_file_completed(correlation_id, success=False, error_message=error_msg)
            else:
                self.progress_tracker.track_file_completed(correlation_id, success=True)

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

                # Extract priority if present
                message_priority = message.get("priority", self.priority_config.default_priority)

                logger.info(
                    "Received message",
                    extra={
                        "correlation_id": correlation_id,
                        "routing_key": method.routing_key,
                        "priority": message_priority,
                    },
                )

                # Extract required fields
                file_path = message.get("file_path")
                recording_id = message.get("recording_id")

                if not file_path or not recording_id:
                    raise ValueError("Message must contain 'file_path' and 'recording_id'")

                # Track file as queued
                if self.progress_tracker:
                    self.progress_tracker.track_file_queued(file_path, recording_id, correlation_id)

                # Process in batch mode or immediately
                if self.enable_batch_processing and self.batch_processor:
                    # Add to batch for processing
                    self.batch_processor.add_to_batch(file_path, recording_id, correlation_id)
                    results = {"status": "queued_for_batch", "file_path": file_path}
                else:
                    # Process immediately
                    results = self.process_audio_file(file_path, recording_id, correlation_id)

                # Call user callback if provided
                if callback:
                    callback(results, correlation_id)

                # Acknowledge message
                ch.basic_ack(delivery_tag=method.delivery_tag)
                # Prepare log extras
                log_extras = {
                    "correlation_id": correlation_id,
                    "from_cache": results.get("from_cache", False),
                }

                # Add BPM data to log
                if "bpm_data" in results and results["bpm_data"]:
                    log_extras["bpm"] = results["bpm_data"].get("bpm")
                    log_extras["bpm_confidence"] = results["bpm_data"].get("confidence")

                # Add key data to log
                if "key_data" in results and results["key_data"] and "error" not in results["key_data"]:
                    log_extras["key"] = f"{results['key_data']['key']} {results['key_data']['scale']}"
                    log_extras["key_confidence"] = results["key_data"].get("confidence")

                # Add mood data to log
                if "mood_data" in results and results["mood_data"] and "error" not in results["mood_data"]:
                    log_extras["genre"] = results["mood_data"].get("primary_genre")
                    log_extras["danceability"] = results["mood_data"].get("danceability")

                logger.info("Message processed successfully", extra=log_extras)

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
        # Flush batch processor if enabled
        if self.batch_processor:
            logger.info("Flushing batch processor before stopping")
            self.batch_processor.flush_batch()
            self.batch_processor.shutdown(wait=True)

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
