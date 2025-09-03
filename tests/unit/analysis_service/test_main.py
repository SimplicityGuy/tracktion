"""Tests for main.py module in analysis_service."""

import os
import signal
import sys
from pathlib import Path
from unittest import mock
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

# Add the service module to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "services" / "analysis_service" / "src"))

from config import BPMConfig
from exceptions import InvalidAudioFileError, MetadataExtractionError, RetryableError
from main import AnalysisService, main


class TestAnalysisServiceInitialization:
    """Test AnalysisService initialization."""

    def test_init_sets_default_values(self):
        """Test that __init__ sets proper default values."""
        service = AnalysisService()

        # Check initial state
        assert service.running is False
        assert service.consumer is None
        assert service.extractor is None
        assert service.storage is None
        assert service.rename_integration is None
        assert service.bpm_detector is None
        assert service.key_detector is None
        assert service.messaging_service is None
        assert service._shutdown_requested is False

        # Check default configuration values
        assert service.rabbitmq_url == "amqp://guest:guest@localhost:5672/"
        assert service.queue_name == "analysis_queue"
        assert service.exchange_name == "tracktion_exchange"
        assert service.routing_key == "file.analyze"
        assert service.max_retries == 3
        assert service.retry_delay == 5.0
        assert service.enable_audio_analysis is True

    def test_init_respects_environment_variables(self):
        """Test that initialization respects environment variables."""
        env_vars = {
            "RABBITMQ_URL": "amqp://custom:pass@rabbit:5672/vhost",
            "ANALYSIS_QUEUE": "custom_analysis_queue",
            "EXCHANGE_NAME": "custom_exchange",
            "ANALYSIS_ROUTING_KEY": "custom.routing.key",
            "MAX_RETRIES": "5",
            "RETRY_DELAY": "10.0",
            "ENABLE_AUDIO_ANALYSIS": "false",
        }

        with patch.dict(os.environ, env_vars):
            service = AnalysisService()

            assert service.rabbitmq_url == "amqp://custom:pass@rabbit:5672/vhost"
            assert service.queue_name == "custom_analysis_queue"
            assert service.exchange_name == "custom_exchange"
            assert service.routing_key == "custom.routing.key"
            assert service.max_retries == 5
            assert service.retry_delay == 10.0
            assert service.enable_audio_analysis is False


class TestAnalysisServiceComponentInitialization:
    """Test service component initialization."""

    @patch("main.StorageHandler")
    @patch("main.MetadataExtractor")
    @patch("main.BPMDetector")
    @patch("main.KeyDetector")
    @patch("main.MessageConsumer")
    @patch("main.DatabaseManager")
    @patch("main.RenameProposalRepository")
    @patch("main.RecordingRepository")
    @patch("main.MetadataRepository")
    @patch("main.FileRenameProposalIntegration")
    @patch("main.FileRenameProposalConfig")
    @patch("main.signal")
    def test_initialize_success(
        self,
        mock_signal,
        mock_config,
        mock_integration,
        mock_metadata_repo,
        mock_recording_repo,
        mock_proposal_repo,
        mock_db_manager,
        mock_message_consumer,
        mock_key_detector,
        mock_bpm_detector,
        mock_metadata_extractor,
        mock_storage_handler,
    ):
        """Test successful initialization of all components."""
        # Setup mocks
        mock_config.from_env.return_value = Mock()
        service = AnalysisService()

        # Call initialize
        service.initialize()

        # Verify all components were created
        mock_metadata_extractor.assert_called_once()
        mock_storage_handler.assert_called_once()
        mock_bpm_detector.assert_called_once()
        mock_key_detector.assert_called_once()
        mock_message_consumer.assert_called_once()

        # Verify signal handlers were set up
        mock_signal.signal.assert_called()

    @patch("main.StorageHandler")
    @patch("main.MetadataExtractor")
    @patch("main.MessageConsumer")
    @patch("main.signal")
    def test_initialize_minimal_success(
        self,
        mock_signal,
        mock_message_consumer,
        mock_metadata_extractor,
        mock_storage_handler,
    ):
        """Test successful initialization with minimal components."""
        service = AnalysisService()
        service.enable_audio_analysis = False  # Disable audio analysis

        # Call initialize
        service.initialize()

        # Verify core components were created
        mock_metadata_extractor.assert_called_once()
        mock_storage_handler.assert_called_once()
        mock_message_consumer.assert_called_once()

        # Verify audio analysis components were not initialized
        assert service.bpm_detector is None
        assert service.key_detector is None

    @patch("main.MetadataExtractor")
    def test_initialize_metadata_extractor_failure(self, mock_metadata_extractor):
        """Test initialization failure when MetadataExtractor fails."""
        mock_metadata_extractor.side_effect = Exception("Failed to initialize extractor")

        service = AnalysisService()

        with pytest.raises(Exception, match="Failed to initialize extractor"):
            service.initialize()

    @patch("main.StorageHandler")
    @patch("main.MetadataExtractor")
    @patch("main.BPMDetector")
    def test_initialize_audio_analysis_failure(
        self,
        mock_bpm_detector,
        mock_metadata_extractor,
        mock_storage_handler,
    ):
        """Test initialization continues when audio analysis fails."""
        mock_bpm_detector.side_effect = Exception("BPM detector failed")

        service = AnalysisService()
        service.enable_audio_analysis = True

        # Should not raise exception
        with patch("main.MessageConsumer"), patch("main.signal"):
            service.initialize()

        # Audio analysis should be disabled
        assert service.enable_audio_analysis is False
        assert service.bpm_detector is None


class TestAnalysisServiceMessageProcessing:
    """Test message processing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = AnalysisService()
        self.service.extractor = Mock()
        self.service.storage = Mock()
        self.correlation_id = "test-correlation-123"

    @pytest.mark.asyncio
    async def test_process_message_success(self):
        """Test successful message processing."""
        message = {
            "recording_id": str(uuid4()),
            "file_path": "/test/file.mp3",
        }

        # Mock successful processing
        self.service._process_file = Mock()
        self.service._send_notification = AsyncMock()

        await self.service.process_message(message, self.correlation_id)

        # Verify processing was called
        self.service._process_file.assert_called_once()

        # Verify success notification was sent
        self.service._send_notification.assert_called_once_with(
            recording_id=mock.ANY,
            status="completed",
            correlation_id=self.correlation_id,
        )

    @pytest.mark.asyncio
    async def test_process_message_missing_required_fields(self):
        """Test message processing with missing required fields."""
        invalid_messages = [
            {},  # Missing both fields
            {"recording_id": str(uuid4())},  # Missing file_path
            {"file_path": "/test/file.mp3"},  # Missing recording_id
            {"recording_id": "", "file_path": "/test/file.mp3"},  # Empty recording_id
            {"recording_id": str(uuid4()), "file_path": ""},  # Empty file_path
        ]

        self.service._process_file = Mock()

        for message in invalid_messages:
            await self.service.process_message(message, self.correlation_id)
            # Processing should not be called for invalid messages
            self.service._process_file.assert_not_called()
            self.service._process_file.reset_mock()

    @pytest.mark.asyncio
    async def test_process_message_invalid_uuid(self):
        """Test message processing with invalid UUID."""
        message = {
            "recording_id": "invalid-uuid",
            "file_path": "/test/file.mp3",
        }

        self.service._process_file = Mock()

        await self.service.process_message(message, self.correlation_id)

        # Processing should not be called for invalid UUID
        self.service._process_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_message_retryable_error(self):
        """Test message processing with retryable error."""
        message = {
            "recording_id": str(uuid4()),
            "file_path": "/test/file.mp3",
        }

        self.service._process_file = Mock(side_effect=RetryableError("Temporary failure"))

        # Test that the retryable error is handled (for now, just verify no crash)
        await self.service.process_message(message, self.correlation_id)

        # Verify that _process_file was called
        self.service._process_file.assert_called_once()


class TestAnalysisServiceFileProcessing:
    """Test file processing functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = AnalysisService()
        self.service.extractor = Mock()
        self.service.storage = Mock()
        self.service.enable_audio_analysis = False  # Disable for simplicity
        self.recording_id = uuid4()
        self.file_path = "/test/audio.mp3"
        self.correlation_id = "test-correlation-123"

    def test_process_file_success(self):
        """Test successful file processing."""
        metadata = {"title": "Test Song", "artist": "Test Artist"}
        self.service.extractor.extract.return_value = metadata

        self.service._process_file(self.recording_id, self.file_path, self.correlation_id)

        # Verify metadata extraction
        self.service.extractor.extract.assert_called_once_with(self.file_path)

        # Verify metadata storage
        self.service.storage.store_metadata.assert_called_once_with(self.recording_id, metadata, self.correlation_id)

        # Verify status update
        self.service.storage.update_recording_status.assert_called_once_with(
            self.recording_id, "processed", None, self.correlation_id
        )

    def test_process_file_with_audio_analysis(self):
        """Test file processing with audio analysis enabled."""
        self.service.enable_audio_analysis = True
        self.service.bmp_detector = Mock()
        self.service.key_detector = Mock()

        # Configure mocks
        self.service.bpm_detector.config = BPMConfig()
        self.service._is_audio_format_supported = Mock(return_value=True)
        self.service._perform_audio_analysis = Mock()

        metadata = {"title": "Test Song", "artist": "Test Artist"}
        self.service.extractor.extract.return_value = metadata

        self.service._process_file(self.recording_id, self.file_path, self.correlation_id)

        # Verify audio analysis was performed
        self.service._perform_audio_analysis.assert_called_once()

    def test_process_file_invalid_audio_file_error(self):
        """Test file processing with InvalidAudioFileError."""
        self.service.extractor.extract.side_effect = InvalidAudioFileError("Invalid file")

        with pytest.raises(InvalidAudioFileError):
            self.service._process_file(self.recording_id, self.file_path, self.correlation_id)

        # Verify status was updated to invalid
        self.service.storage.update_recording_status.assert_called_once_with(
            self.recording_id, "invalid", "Invalid file", self.correlation_id
        )

    def test_process_file_retry_with_success(self):
        """Test file processing retry mechanism with eventual success."""
        metadata = {"title": "Test Song", "artist": "Test Artist"}

        # First call fails, second succeeds
        self.service.extractor.extract.side_effect = [
            MetadataExtractionError("Temporary failure"),
            metadata,
        ]

        self.service._process_file(self.recording_id, self.file_path, self.correlation_id)

        # Verify retry occurred
        assert self.service.extractor.extract.call_count == 2

        # Verify final success
        self.service.storage.store_metadata.assert_called_once_with(self.recording_id, metadata, self.correlation_id)

    def test_process_file_max_retries_exceeded(self):
        """Test file processing when max retries are exceeded."""
        error = MetadataExtractionError("Persistent failure")
        self.service.extractor.extract.side_effect = error
        self.service.max_retries = 2

        with pytest.raises(MetadataExtractionError):
            self.service._process_file(self.recording_id, self.file_path, self.correlation_id)

        # Verify all retries were attempted
        assert self.service.extractor.extract.call_count == 3  # Initial + 2 retries

        # Verify status was updated to failed
        self.service.storage.update_recording_status.assert_called_with(
            self.recording_id, "failed", "Persistent failure", self.correlation_id
        )


class TestAnalysisServiceAudioAnalysis:
    """Test audio analysis functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = AnalysisService()
        self.service.bpm_detector = Mock()
        self.service.key_detector = Mock()
        self.file_path = "/test/audio.mp3"
        self.correlation_id = "test-correlation-123"
        self.recording_id = uuid4()

    def test_is_audio_format_supported_with_detector_config(self):
        """Test audio format support check with detector config."""
        config = BPMConfig()
        self.service.bpm_detector.config = config

        # Test supported format
        assert self.service._is_audio_format_supported("/test/file.mp3") is True

        # Test unsupported format
        assert self.service._is_audio_format_supported("/test/file.txt") is False

    def test_is_audio_format_supported_fallback(self):
        """Test audio format support check with fallback logic."""
        # Don't set config on detector to trigger fallback

        # Test supported format
        assert self.service._is_audio_format_supported("/test/file.mp3") is True

        # Test unsupported format
        assert self.service._is_audio_format_supported("/test/file.txt") is False

    def test_is_audio_format_supported_no_detector(self):
        """Test audio format support check with no detector."""
        self.service.bmp_detector = None

        assert self.service._is_audio_format_supported("/test/file.mp3") is False

    def test_perform_audio_analysis_bmp_success(self):
        """Test audio analysis with successful BPM detection."""
        metadata = {}
        bpm_result = {
            "bpm": 128.5,
            "confidence": 0.85,
            "algorithm": "percival",
            "needs_review": False,
        }

        self.service.bmp_detector.detect_bpm.return_value = bpm_result
        self.service.key_detector.detect_key.return_value = None

        self.service._perform_audio_analysis(self.file_path, metadata, self.correlation_id, self.recording_id)

        # Verify BMP results were added to metadata
        assert metadata["bpm"] == "128.5"
        assert metadata["bmp_confidence"] == "0.85"
        assert metadata["bpm_algorithm"] == "percival"
        assert "bpm_needs_review" not in metadata

    def test_perform_audio_analysis_key_success(self):
        """Test audio analysis with successful key detection."""
        metadata = {}
        key_result = Mock()
        key_result.key = "C"
        key_result.scale = "major"
        key_result.confidence = 0.75
        key_result.alternative_key = "Am"
        key_result.alternative_scale = "minor"
        key_result.needs_review = True

        self.service.bpm_detector.detect_bpm.side_effect = Exception("BPM failed")
        self.service.key_detector.detect_key.return_value = key_result

        self.service._perform_audio_analysis(self.file_path, metadata, self.correlation_id, self.recording_id)

        # Verify key results were added to metadata
        assert metadata["key"] == "C major"
        assert metadata["key_confidence"] == "0.75"
        assert metadata["key_alternative"] == "Am minor"
        assert metadata["key_needs_review"] == "true"

    def test_perform_audio_analysis_failures(self):
        """Test audio analysis with failures."""
        metadata = {}

        self.service.bpm_detector.detect_bpm.side_effect = Exception("BPM failed")
        self.service.key_detector.detect_key.side_effect = Exception("Key failed")

        # Should not raise exception
        self.service._perform_audio_analysis(self.file_path, metadata, self.correlation_id, self.recording_id)

        # Only analysis version should be added
        assert metadata["audio_analysis_version"] == "1.0"
        assert len(metadata) == 1


class TestAnalysisServiceLifecycle:
    """Test service lifecycle management."""

    def test_health_check_healthy(self):
        """Test health check when service is healthy."""
        service = AnalysisService()
        service.running = True
        service.consumer = Mock()
        service.consumer.connection = Mock()
        service.consumer.connection.is_closed = False
        service.storage = Mock()

        health = service.health_check()

        assert health["service"] == "analysis_service"
        assert health["status"] == "healthy"
        assert health["healthy"] is True
        assert health["components"]["rabbitmq"]["status"] == "connected"
        assert health["components"]["storage"]["status"] == "initialized"

    def test_health_check_unhealthy(self):
        """Test health check when service is unhealthy."""
        service = AnalysisService()
        service.running = False
        service.consumer = None
        service.storage = None

        health = service.health_check()

        assert health["status"] == "not_running"
        assert health["healthy"] is False
        assert health["components"]["rabbitmq"]["status"] == "not_initialized"
        assert health["components"]["storage"]["status"] == "not_initialized"

    @patch("main.signal")
    def test_handle_shutdown(self, mock_signal):
        """Test signal handler for shutdown."""
        service = AnalysisService()
        service.shutdown = Mock()

        # Simulate signal
        service._handle_shutdown(signal.SIGTERM, None)

        assert service._shutdown_requested is True
        service.shutdown.assert_called_once()

    def test_shutdown_graceful(self):
        """Test graceful shutdown."""
        service = AnalysisService()
        service.running = True
        service.consumer = Mock()
        service.storage = Mock()

        service.shutdown()

        assert service.running is False
        service.consumer.stop.assert_called_once()
        service.storage.close.assert_called_once()

    def test_shutdown_already_shutdown(self):
        """Test shutdown when already shut down."""
        service = AnalysisService()
        service.running = False
        service.consumer = Mock()
        service.storage = Mock()

        service.shutdown()

        # Should not call stop/close methods
        service.consumer.stop.assert_not_called()
        service.storage.close.assert_not_called()

    def test_shutdown_with_exceptions(self):
        """Test shutdown continues despite exceptions."""
        service = AnalysisService()
        service.running = True
        service.consumer = Mock()
        service.consumer.stop.side_effect = Exception("Stop failed")
        service.storage = Mock()
        service.storage.close.side_effect = Exception("Close failed")

        # Should not raise exception
        service.shutdown()

        assert service.running is False


class TestMainFunction:
    """Test main entry point."""

    @patch("main.AnalysisService")
    def test_main_creates_and_runs_service(self, mock_service_class):
        """Test that main creates and runs the service."""
        mock_service = Mock()
        mock_service_class.return_value = mock_service

        main()

        mock_service_class.assert_called_once()
        mock_service.run.assert_called_once()


class TestAnalysisServiceRun:
    """Test the service run method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = AnalysisService()

    @patch("main.asyncio")
    def test_run_success(self, mock_asyncio):
        """Test successful run."""
        self.service.initialize = Mock()
        self.service.consumer = Mock()
        self.service.shutdown = Mock()

        # Mock sync_process_message wrapper
        def mock_consume(callback):
            # Simulate successful processing
            pass

        self.service.consumer.consume = Mock(side_effect=mock_consume)

        self.service.run()

        # Verify initialization and consumption
        self.service.initialize.assert_called_once()
        assert self.service.running is True
        self.service.consumer.consume.assert_called_once()

    def test_run_keyboard_interrupt(self):
        """Test run handles KeyboardInterrupt."""
        self.service.initialize = Mock()
        self.service.consumer = Mock()
        self.service.consumer.consume.side_effect = KeyboardInterrupt()
        self.service.shutdown = Mock()

        self.service.run()

        # Should handle interrupt gracefully
        self.service.shutdown.assert_called_once()

    def test_run_generic_exception(self):
        """Test run handles generic exceptions."""
        self.service.initialize = Mock(side_effect=Exception("Initialization failed"))
        self.service.shutdown = Mock()

        self.service.run()

        # Should handle exception and shutdown
        self.service.shutdown.assert_called_once()
