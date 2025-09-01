"""
Unit tests for message consumer module.

Tests message processing, caching, error handling, and RabbitMQ integration.
"""

import json
import tempfile
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import pika
import pika.exceptions
import pytest

from services.analysis_service.src.message_consumer import MessageConsumer


class TestMessageConsumer:
    """Test suite for MessageConsumer class."""

    @pytest.fixture
    def mock_cache(self):
        """Create mock cache instance."""
        cache = Mock()
        cache.get_bpm_results.return_value = None
        cache.get_temporal_results.return_value = None
        cache.set_bpm_results.return_value = True
        cache.set_temporal_results.return_value = True
        return cache

    @pytest.fixture
    def mock_bpm_detector(self):
        """Create mock BPM detector."""
        detector = Mock()
        detector.detect_bpm.return_value = {
            "bpm": 120.0,
            "confidence": 0.95,
            "algorithm": "primary",
            "needs_review": False,
            "beats": [0.5, 1.0, 1.5],
        }
        return detector

    @pytest.fixture
    def mock_temporal_analyzer(self):
        """Create mock temporal analyzer."""
        analyzer = Mock()
        analyzer.analyze_temporal_bpm.return_value = {
            "average_bpm": 120.0,
            "start_bpm": 118.0,
            "end_bpm": 122.0,
            "stability_score": 0.92,
            "is_variable_tempo": False,
            "tempo_changes": [],
        }
        return analyzer

    @pytest.fixture
    def mock_storage(self):
        """Create mock storage handler."""
        storage = Mock()
        storage.store_bpm_data.return_value = True
        return storage

    @pytest.fixture
    def consumer(self, mock_cache, mock_bpm_detector, mock_temporal_analyzer):
        """Create MessageConsumer with mocked dependencies."""
        with (
            patch("services.analysis_service.src.message_consumer.AudioCache") as mock_cache_class,
            patch("services.analysis_service.src.message_consumer.BPMDetector") as mock_detector_class,
            patch("services.analysis_service.src.message_consumer.TemporalAnalyzer") as mock_analyzer_class,
        ):
            mock_cache_class.return_value = mock_cache
            mock_detector_class.return_value = mock_bpm_detector
            mock_analyzer_class.return_value = mock_temporal_analyzer

            consumer = MessageConsumer(
                rabbitmq_url="amqp://guest:guest@localhost:5672/",
                queue_name="test_queue",
                enable_cache=True,
                enable_temporal_analysis=True,
            )

            # Set the mocked components directly
            consumer.cache = mock_cache
            consumer.bpm_detector = mock_bpm_detector
            consumer.temporal_analyzer = mock_temporal_analyzer

            return consumer

    def test_initialization(self):
        """Test MessageConsumer initialization."""
        consumer = MessageConsumer(
            rabbitmq_url="amqp://test:test@localhost:5672/",
            queue_name="test_queue",
            exchange_name="test_exchange",
            routing_key="test.route",
            enable_cache=False,
            enable_temporal_analysis=False,
        )

        assert consumer.rabbitmq_url == "amqp://test:test@localhost:5672/"
        assert consumer.queue_name == "test_queue"
        assert consumer.exchange_name == "test_exchange"
        assert consumer.routing_key == "test.route"
        assert consumer.cache is None
        assert consumer.temporal_analyzer is None
        assert consumer.connection is None
        assert consumer.channel is None

    @patch("services.analysis_service.src.message_consumer.pika.BlockingConnection")
    def test_connect_success(self, mock_connection_class, consumer):
        """Test successful RabbitMQ connection."""
        mock_connection = Mock()
        mock_channel = Mock()
        mock_connection.channel.return_value = mock_channel
        mock_connection_class.return_value = mock_connection

        consumer.connect()

        assert consumer.connection == mock_connection
        assert consumer.channel == mock_channel
        mock_channel.exchange_declare.assert_called_once()
        mock_channel.queue_declare.assert_called_once()
        mock_channel.queue_bind.assert_called_once()
        mock_channel.basic_qos.assert_called_once_with(prefetch_count=1)

    @patch("services.analysis_service.src.message_consumer.pika.BlockingConnection")
    @patch("services.analysis_service.src.message_consumer.time.sleep")
    def test_connect_retry(self, mock_sleep, mock_connection_class, consumer):
        """Test connection retry on failure."""
        # Fail twice, then succeed
        mock_connection_class.side_effect = [
            pika.exceptions.AMQPConnectionError("Connection failed"),
            pika.exceptions.AMQPConnectionError("Connection failed"),
            Mock(channel=Mock(return_value=Mock())),
        ]

        consumer.connect()

        assert mock_connection_class.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("services.analysis_service.src.message_consumer.pika.BlockingConnection")
    @patch("services.analysis_service.src.message_consumer.time.sleep")
    def test_connect_max_retries_exceeded(self, mock_sleep, mock_connection_class, consumer):
        """Test connection failure after max retries."""
        mock_connection_class.side_effect = pika.exceptions.AMQPConnectionError("Connection failed")

        with pytest.raises(ConnectionError, match="Failed to connect to RabbitMQ"):
            consumer.connect()

        assert mock_connection_class.call_count == 5  # max_retries

    def test_process_audio_file_success(self, consumer, mock_storage):
        """Test successful audio file processing."""
        consumer.storage = mock_storage

        # Create temporary test file
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            recording_id = str(uuid.uuid4())
            result = consumer.process_audio_file(tmp_path, recording_id)

            assert result["recording_id"] == recording_id
            assert result["file_path"] == tmp_path
            assert result["bpm_data"]["bpm"] == 120.0
            assert result["bpm_data"]["confidence"] == 0.95
            assert result["temporal_data"]["average_bpm"] == 120.0
            assert result["temporal_data"]["stability_score"] == 0.92
            assert result["from_cache"] is False
            assert result.get("stored") is True

            # Verify detector and analyzer were called
            consumer.bpm_detector.detect_bpm.assert_called_once_with(tmp_path)
            consumer.temporal_analyzer.analyze_temporal_bpm.assert_called_once_with(tmp_path)

            # Verify caching
            consumer.cache.set_bpm_results.assert_called_once()
            consumer.cache.set_temporal_results.assert_called_once()

            # Verify storage
            mock_storage.store_bpm_data.assert_called_once()

        finally:
            Path(tmp_path).unlink()

    def test_process_audio_file_cached(self, consumer):
        """Test processing with cached results."""
        cached_bpm = {"bpm": 128.0, "confidence": 0.98, "algorithm": "cached"}
        cached_temporal = {"average_bpm": 128.0, "stability_score": 0.95}

        consumer.cache.get_bpm_results.return_value = cached_bpm
        consumer.cache.get_temporal_results.return_value = cached_temporal

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            recording_id = str(uuid.uuid4())
            result = consumer.process_audio_file(tmp_path, recording_id)

            assert result["bpm_data"] == cached_bpm
            assert result["temporal_data"] == cached_temporal
            assert result["from_cache"] is True

            # Detector should not be called when using cache
            consumer.bpm_detector.detect_bpm.assert_not_called()
            consumer.temporal_analyzer.analyze_temporal_bpm.assert_not_called()

        finally:
            Path(tmp_path).unlink()

    def test_process_audio_file_not_found(self, consumer):
        """Test processing non-existent file."""
        recording_id = str(uuid.uuid4())
        result = consumer.process_audio_file("/nonexistent/file.mp3", recording_id)

        assert "error" in result
        assert "Audio file not found" in result["error"]
        assert result["bpm_data"] is None

        # Should not call detector for missing files
        consumer.bpm_detector.detect_bpm.assert_not_called()

    def test_process_audio_file_bpm_failure(self, consumer):
        """Test handling of BPM detection failure."""
        consumer.bpm_detector.detect_bpm.side_effect = Exception("BPM detection error")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            recording_id = str(uuid.uuid4())
            result = consumer.process_audio_file(tmp_path, recording_id)

            assert "error" in result["bpm_data"]
            assert "BPM detection error" in result["bpm_data"]["error"]
            assert "temporal_data" not in result  # Should not run temporal analysis

            # Should cache the failure
            consumer.cache.set_bpm_results.assert_called_once()
            call_args = consumer.cache.set_bpm_results.call_args
            assert call_args[1]["failed"] is True

        finally:
            Path(tmp_path).unlink()

    def test_process_audio_file_temporal_failure(self, consumer):
        """Test handling of temporal analysis failure."""
        consumer.temporal_analyzer.analyze_temporal_bpm.side_effect = Exception("Temporal error")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            recording_id = str(uuid.uuid4())
            result = consumer.process_audio_file(tmp_path, recording_id)

            # BPM should succeed
            assert result["bpm_data"]["bpm"] == 120.0

            # Temporal should have error
            assert "error" in result["temporal_data"]
            assert "Temporal error" in result["temporal_data"]["error"]

            # BPM results should still be cached
            consumer.cache.set_bpm_results.assert_called_once()

        finally:
            Path(tmp_path).unlink()

    def test_process_audio_file_no_cache(self):
        """Test processing without cache."""
        with (
            patch("services.analysis_service.src.message_consumer.BPMDetector") as mock_detector_class,
            patch("services.analysis_service.src.message_consumer.TemporalAnalyzer") as mock_analyzer_class,
        ):
            mock_detector = Mock()
            mock_detector.detect_bpm.return_value = {
                "bpm": 120.0,
                "confidence": 0.95,
            }
            mock_detector_class.return_value = mock_detector

            mock_analyzer = Mock()
            mock_analyzer.analyze_temporal_bpm.return_value = {"average_bpm": 120.0}
            mock_analyzer_class.return_value = mock_analyzer

            consumer = MessageConsumer(
                rabbitmq_url="amqp://localhost",
                enable_cache=False,
                enable_temporal_analysis=True,
            )
            consumer.bpm_detector = mock_detector
            consumer.temporal_analyzer = mock_analyzer

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                result = consumer.process_audio_file(tmp_path, str(uuid.uuid4()))

                assert result["bpm_data"]["bpm"] == 120.0
                assert consumer.cache is None

            finally:
                Path(tmp_path).unlink()

    def test_process_audio_file_storage_failure(self, consumer, mock_storage):
        """Test handling of storage failure."""
        consumer.storage = mock_storage
        mock_storage.store_bpm_data.side_effect = Exception("Storage error")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            recording_id = str(uuid.uuid4())
            result = consumer.process_audio_file(tmp_path, recording_id)

            # Processing should succeed
            assert result["bpm_data"]["bpm"] == 120.0

            # Storage error should be captured
            assert "storage_error" in result
            assert "Storage error" in result["storage_error"]

        finally:
            Path(tmp_path).unlink()

    @patch("services.analysis_service.src.message_consumer.pika.BlockingConnection")
    def test_consume_message_success(self, mock_connection_class, consumer):
        """Test successful message consumption."""
        mock_channel = Mock()
        consumer.channel = mock_channel

        # Create test message
        message = {"file_path": "/test/audio.mp3", "recording_id": str(uuid.uuid4())}

        # Mock process_audio_file
        consumer.process_audio_file = Mock(return_value={"bpm_data": {"bpm": 120.0}, "from_cache": False})

        # Track callback
        callback_called = False

        def test_callback(results, correlation_id):
            nonlocal callback_called
            callback_called = True
            assert results["bpm_data"]["bpm"] == 120.0

        # Mock start_consuming to not block
        mock_channel.start_consuming = Mock()

        # Start consuming
        consumer.consume(test_callback)

        # Get the registered callback
        call_args = mock_channel.basic_consume.call_args
        assert call_args is not None
        # Handle both keyword and positional arguments
        if "on_message_callback" in call_args.kwargs:
            registered_callback = call_args.kwargs["on_message_callback"]
        else:
            registered_callback = call_args[1].get("on_message_callback")

        # Manually trigger message callback
        mock_method = Mock()
        mock_method.delivery_tag = 1
        mock_method.routing_key = "test.route"

        mock_properties = Mock()
        mock_properties.correlation_id = "test-correlation-id"

        # Call it with our test message
        registered_callback(mock_channel, mock_method, mock_properties, json.dumps(message).encode())

        # Verify processing
        consumer.process_audio_file.assert_called_once_with(message["file_path"], message["recording_id"])

        # Verify acknowledgment
        mock_channel.basic_ack.assert_called_once_with(delivery_tag=1)

        # Verify callback was called
        assert callback_called

    @patch("services.analysis_service.src.message_consumer.pika.BlockingConnection")
    def test_consume_message_invalid_json(self, mock_connection_class, consumer):
        """Test handling of invalid JSON message."""
        mock_channel = Mock()
        consumer.channel = mock_channel

        # Start consuming
        consumer.consume()

        # Get the registered callback
        call_args = mock_channel.basic_consume.call_args
        registered_callback = call_args[1]["on_message_callback"]

        # Send invalid JSON
        mock_method = Mock()
        mock_method.delivery_tag = 1

        registered_callback(mock_channel, mock_method, Mock(correlation_id="test"), b"invalid json")

        # Should reject message without requeue
        mock_channel.basic_nack.assert_called_once_with(delivery_tag=1, requeue=False)

    @patch("services.analysis_service.src.message_consumer.pika.BlockingConnection")
    def test_consume_message_missing_fields(self, mock_connection_class, consumer):
        """Test handling of message with missing required fields."""
        mock_channel = Mock()
        consumer.channel = mock_channel

        # Start consuming
        consumer.consume()

        # Get the registered callback
        call_args = mock_channel.basic_consume.call_args
        registered_callback = call_args[1]["on_message_callback"]

        # Send message without required fields
        message = {"file_path": "/test/audio.mp3"}  # Missing recording_id

        mock_method = Mock()
        mock_method.delivery_tag = 1

        registered_callback(
            mock_channel,
            mock_method,
            Mock(correlation_id="test"),
            json.dumps(message).encode(),
        )

        # Should reject message without requeue (ValueError is not retryable)
        mock_channel.basic_nack.assert_called_once_with(delivery_tag=1, requeue=False)

    def test_consume_without_connection(self, consumer):
        """Test consume automatically connects if not connected."""
        with patch.object(consumer, "connect") as mock_connect:
            mock_channel = Mock()
            consumer.channel = None  # Not connected

            # After connect, set channel
            def set_channel():
                consumer.channel = mock_channel

            mock_connect.side_effect = set_channel

            consumer.consume()

            mock_connect.assert_called_once()

    def test_initialization_with_config_object(self):
        """Test initialization using component configs."""
        # Test that MessageConsumer can be initialized with various configurations
        with (
            patch("services.analysis_service.src.message_consumer.AudioCache"),
            patch("services.analysis_service.src.message_consumer.BPMDetector"),
            patch("services.analysis_service.src.message_consumer.TemporalAnalyzer"),
        ):
            # Test with explicit parameters
            consumer = MessageConsumer(
                rabbitmq_url="amqp://guest:guest@rabbitmq:5672/",
                queue_name="analysis_queue",
                exchange_name="tracktion_exchange",
                routing_key="file.analyze",
                redis_host="redis",
                redis_port=6379,
                enable_cache=True,
                enable_temporal_analysis=True,
            )

            assert consumer.rabbitmq_url == "amqp://guest:guest@rabbitmq:5672/"
            assert consumer.queue_name == "analysis_queue"
            assert consumer.exchange_name == "tracktion_exchange"
            assert consumer.routing_key == "file.analyze"
