"""
Unit tests for BPM integration in message consumer.

Tests the integration of BPM detection with RabbitMQ message processing.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from services.analysis_service.src.message_consumer import MessageConsumer


class TestMessageConsumerBPMIntegration:
    """Test suite for BPM detection integration in MessageConsumer."""

    def setup_method(self):
        """Set up test fixtures."""
        # Mock RabbitMQ connection
        with patch("services.analysis_service.src.message_consumer.pika.BlockingConnection"):
            self.consumer = MessageConsumer(
                rabbitmq_url="amqp://test",
                redis_host="localhost",
                redis_port=6379,
                enable_cache=True,
                enable_temporal_analysis=True,
            )

        # Mock the BPM detector and temporal analyzer
        self.consumer.bpm_detector = MagicMock()
        self.consumer.temporal_analyzer = MagicMock()
        self.consumer.cache = MagicMock()

    def test_initialization_with_bpm_components(self):
        """Test that BPM components are initialized."""
        with patch("services.analysis_service.src.message_consumer.BPMDetector") as mock_detector:
            with patch("services.analysis_service.src.message_consumer.TemporalAnalyzer") as mock_analyzer:
                with patch("services.analysis_service.src.message_consumer.AudioCache") as mock_cache:
                    MessageConsumer(rabbitmq_url="amqp://test", enable_cache=True, enable_temporal_analysis=True)

                    mock_detector.assert_called_once()
                    mock_analyzer.assert_called_once()
                    mock_cache.assert_called_once_with(redis_host="localhost", redis_port=6379)

    def test_initialization_without_cache(self):
        """Test initialization with cache disabled."""
        with patch("services.analysis_service.src.message_consumer.BPMDetector"):
            with patch("services.analysis_service.src.message_consumer.TemporalAnalyzer"):
                consumer = MessageConsumer(
                    rabbitmq_url="amqp://test", enable_cache=False, enable_temporal_analysis=True
                )

                assert consumer.cache is None

    def test_initialization_without_temporal(self):
        """Test initialization with temporal analysis disabled."""
        with patch("services.analysis_service.src.message_consumer.BPMDetector"):
            with patch("services.analysis_service.src.message_consumer.AudioCache"):
                consumer = MessageConsumer(
                    rabbitmq_url="amqp://test", enable_cache=True, enable_temporal_analysis=False
                )

                assert consumer.temporal_analyzer is None

    @patch("os.path.exists")
    def test_process_audio_file_with_cache_hit(self, mock_exists):
        """Test processing with cached results."""
        mock_exists.return_value = True

        # Setup cache hit
        cached_bpm = {"bpm": 128.0, "confidence": 0.95, "algorithm_version": "1.0"}
        cached_temporal = {"average_bpm": 128.0, "stability_score": 0.92}
        self.consumer.cache.get_bpm_results.return_value = cached_bpm
        self.consumer.cache.get_temporal_results.return_value = cached_temporal

        result = self.consumer.process_audio_file("/audio/test.mp3", "rec-123")

        assert result["recording_id"] == "rec-123"
        assert result["file_path"] == "/audio/test.mp3"
        assert result["bpm_data"] == cached_bpm
        assert result["temporal_data"] == cached_temporal
        assert result["from_cache"] is True

        # Verify BPM detector was not called (cache hit)
        self.consumer.bpm_detector.detect_bpm.assert_not_called()

    @patch("os.path.exists")
    def test_process_audio_file_with_cache_miss(self, mock_exists):
        """Test processing without cached results."""
        mock_exists.return_value = True

        # Setup cache miss
        self.consumer.cache.get_bpm_results.return_value = None

        # Setup BPM detection results
        bpm_results = {"bpm": 120.0, "confidence": 0.88, "algorithm": "RhythmExtractor2013"}
        temporal_results = {"average_bpm": 120.0, "start_bpm": 118.0, "end_bpm": 122.0, "stability_score": 0.85}

        self.consumer.bpm_detector.detect_bpm.return_value = bpm_results
        self.consumer.temporal_analyzer.analyze_temporal_bpm.return_value = temporal_results

        result = self.consumer.process_audio_file("/audio/test.mp3", "rec-123")

        assert result["recording_id"] == "rec-123"
        assert result["bpm_data"] == bpm_results
        assert result["temporal_data"] == temporal_results
        assert result["from_cache"] is False

        # Verify cache was updated
        self.consumer.cache.set_bpm_results.assert_called_once_with(
            "/audio/test.mp3", bpm_results, confidence=0.88, failed=False
        )
        self.consumer.cache.set_temporal_results.assert_called_once_with(
            "/audio/test.mp3", temporal_results, stability_score=0.85
        )

    @patch("os.path.exists")
    def test_process_audio_file_not_found(self, mock_exists):
        """Test processing with non-existent file."""
        mock_exists.return_value = False

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            self.consumer.process_audio_file("/audio/missing.mp3", "rec-123")

    @patch("os.path.exists")
    def test_process_audio_file_bpm_detection_error(self, mock_exists):
        """Test handling of BPM detection errors."""
        mock_exists.return_value = True
        self.consumer.cache.get_bpm_results.return_value = None

        # Simulate BPM detection error
        self.consumer.bpm_detector.detect_bpm.side_effect = Exception("Audio corrupt")

        result = self.consumer.process_audio_file("/audio/corrupt.mp3", "rec-123")

        assert result["bpm_data"]["error"] == "Audio corrupt"
        assert "temporal_data" not in result  # Temporal analysis skipped due to error

        # Verify error was cached
        self.consumer.cache.set_bpm_results.assert_called_once()
        call_args = self.consumer.cache.set_bpm_results.call_args
        assert call_args[1]["failed"] is True

    @patch("os.path.exists")
    def test_process_audio_file_temporal_error(self, mock_exists):
        """Test handling of temporal analysis errors."""
        mock_exists.return_value = True
        self.consumer.cache.get_bpm_results.return_value = None

        # BPM succeeds
        bpm_results = {"bpm": 120.0, "confidence": 0.9}
        self.consumer.bpm_detector.detect_bpm.return_value = bpm_results

        # Temporal analysis fails
        self.consumer.temporal_analyzer.analyze_temporal_bpm.side_effect = Exception("Memory error")

        result = self.consumer.process_audio_file("/audio/test.mp3", "rec-123")

        assert result["bpm_data"] == bpm_results
        assert result["temporal_data"]["error"] == "Memory error"

    @patch("os.path.exists")
    def test_process_audio_file_without_cache(self, mock_exists):
        """Test processing when cache is disabled."""
        mock_exists.return_value = True
        self.consumer.cache = None  # Disable cache

        bpm_results = {"bpm": 128.0, "confidence": 0.95}
        self.consumer.bpm_detector.detect_bpm.return_value = bpm_results

        result = self.consumer.process_audio_file("/audio/test.mp3", "rec-123")

        assert result["bpm_data"] == bpm_results
        assert result["from_cache"] is False

    @patch("os.path.exists")
    def test_process_audio_file_without_temporal(self, mock_exists):
        """Test processing when temporal analysis is disabled."""
        mock_exists.return_value = True
        self.consumer.enable_temporal_analysis = False
        self.consumer.temporal_analyzer = None
        self.consumer.cache.get_bpm_results.return_value = None

        bpm_results = {"bpm": 128.0, "confidence": 0.95}
        self.consumer.bpm_detector.detect_bpm.return_value = bpm_results

        result = self.consumer.process_audio_file("/audio/test.mp3", "rec-123")

        assert result["bpm_data"] == bpm_results
        assert "temporal_data" not in result

    def test_message_consume_valid_message(self):
        """Test consuming a valid message."""
        # Setup channel mock
        mock_channel = MagicMock()
        self.consumer.channel = mock_channel

        # Create test message
        message = {"file_path": "/audio/test.mp3", "recording_id": "rec-123"}
        body = json.dumps(message).encode()

        # Mock properties
        properties = MagicMock()
        properties.correlation_id = "corr-123"

        # Mock method
        method = MagicMock()
        method.routing_key = "file.analyze"
        method.delivery_tag = 1

        # Mock process_audio_file
        with patch.object(self.consumer, "process_audio_file") as mock_process:
            mock_process.return_value = {
                "recording_id": "rec-123",
                "bpm_data": {"bpm": 128.0, "confidence": 0.9},
                "from_cache": False,
            }

            # Call the internal message callback
            callback = None

            def capture_callback(queue, on_message_callback, auto_ack):
                nonlocal callback
                callback = on_message_callback

            mock_channel.basic_consume.side_effect = capture_callback
            self.consumer.consume()

            # Trigger the callback
            callback(mock_channel, method, properties, body)

            # Verify processing
            mock_process.assert_called_once_with("/audio/test.mp3", "rec-123")
            mock_channel.basic_ack.assert_called_once_with(delivery_tag=1)

    def test_message_consume_invalid_json(self):
        """Test consuming an invalid JSON message."""
        mock_channel = MagicMock()
        self.consumer.channel = mock_channel

        # Invalid JSON
        body = b"not json"

        properties = MagicMock()
        properties.correlation_id = "corr-123"

        method = MagicMock()
        method.delivery_tag = 1

        # Setup callback capture
        callback = None

        def capture_callback(queue, on_message_callback, auto_ack):
            nonlocal callback
            callback = on_message_callback

        mock_channel.basic_consume.side_effect = capture_callback
        self.consumer.consume()

        # Trigger the callback
        callback(mock_channel, method, properties, body)

        # Verify message was rejected without requeue
        mock_channel.basic_nack.assert_called_once_with(delivery_tag=1, requeue=False)

    def test_message_consume_missing_fields(self):
        """Test consuming a message with missing required fields."""
        mock_channel = MagicMock()
        self.consumer.channel = mock_channel

        # Message missing recording_id
        message = {"file_path": "/audio/test.mp3"}
        body = json.dumps(message).encode()

        properties = MagicMock()
        properties.correlation_id = "corr-123"

        method = MagicMock()
        method.delivery_tag = 1

        # Setup callback capture
        callback = None

        def capture_callback(queue, on_message_callback, auto_ack):
            nonlocal callback
            callback = on_message_callback

        mock_channel.basic_consume.side_effect = capture_callback
        self.consumer.consume()

        # Trigger the callback
        callback(mock_channel, method, properties, body)

        # Verify message was rejected without requeue (bad format)
        mock_channel.basic_nack.assert_called_once_with(delivery_tag=1, requeue=False)

    def test_message_consume_processing_error(self):
        """Test handling of processing errors that should be retried."""
        mock_channel = MagicMock()
        self.consumer.channel = mock_channel

        message = {"file_path": "/audio/test.mp3", "recording_id": "rec-123"}
        body = json.dumps(message).encode()

        properties = MagicMock()
        properties.correlation_id = "corr-123"

        method = MagicMock()
        method.delivery_tag = 1

        # Mock process_audio_file to raise an exception
        with patch.object(self.consumer, "process_audio_file") as mock_process:
            mock_process.side_effect = Exception("Temporary error")

            # Setup callback capture
            callback = None

            def capture_callback(queue, on_message_callback, auto_ack):
                nonlocal callback
                callback = on_message_callback

            mock_channel.basic_consume.side_effect = capture_callback
            self.consumer.consume()

            # Trigger the callback
            callback(mock_channel, method, properties, body)

            # Verify message was requeued for retry
            mock_channel.basic_nack.assert_called_once_with(delivery_tag=1, requeue=True)

    def test_message_consume_with_user_callback(self):
        """Test consuming messages with a user-provided callback."""
        mock_channel = MagicMock()
        self.consumer.channel = mock_channel

        message = {"file_path": "/audio/test.mp3", "recording_id": "rec-123"}
        body = json.dumps(message).encode()

        properties = MagicMock()
        properties.correlation_id = "corr-123"

        method = MagicMock()
        method.delivery_tag = 1

        # Mock process_audio_file
        with patch.object(self.consumer, "process_audio_file") as mock_process:
            results = {
                "recording_id": "rec-123",
                "bpm_data": {"bpm": 128.0, "confidence": 0.9},
                "from_cache": False,
            }
            mock_process.return_value = results

            # User callback
            user_callback = MagicMock()

            # Setup callback capture
            internal_callback = None

            def capture_callback(queue, on_message_callback, auto_ack):
                nonlocal internal_callback
                internal_callback = on_message_callback

            mock_channel.basic_consume.side_effect = capture_callback
            self.consumer.consume(callback=user_callback)

            # Trigger the callback
            internal_callback(mock_channel, method, properties, body)

            # Verify user callback was called
            user_callback.assert_called_once_with(results, "corr-123")
            mock_channel.basic_ack.assert_called_once()


class TestMessageConsumerCacheFailure:
    """Test cache initialization failures."""

    def test_cache_initialization_failure(self):
        """Test graceful handling of cache initialization failure."""
        with patch("services.analysis_service.src.message_consumer.BPMDetector"):
            with patch("services.analysis_service.src.message_consumer.TemporalAnalyzer"):
                with patch("services.analysis_service.src.message_consumer.AudioCache") as mock_cache:
                    # Simulate cache initialization failure
                    mock_cache.side_effect = Exception("Redis connection failed")

                    consumer = MessageConsumer(rabbitmq_url="amqp://test", enable_cache=True)

                    # Should continue without cache
                    assert consumer.cache is None
