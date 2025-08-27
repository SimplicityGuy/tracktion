"""
Tests for RabbitMQ message handling in tracklist service.

Test suite covering message publishing, consuming, job processing,
and message queue integration.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
from datetime import datetime, timezone

from services.tracklist_service.src.messaging.import_handler import (
    ImportJobMessage, ImportResultMessage, ImportMessageHandler
)
from services.tracklist_service.src.models.tracklist import ImportTracklistRequest
from services.tracklist_service.src.workers.import_worker import ImportWorker


class TestImportJobMessage:
    """Test import job message serialization."""
    
    def test_job_message_creation(self):
        """Test creating import job message."""
        correlation_id = str(uuid4())
        audio_file_id = uuid4()
        
        request = ImportTracklistRequest(
            url="https://1001tracklists.com/tracklist/12345/test-set",
            audio_file_id=audio_file_id,
            force_refresh=False,
            cue_format="standard"
        )
        
        job_message = ImportJobMessage(
            correlation_id=correlation_id,
            request=request,
            created_at=datetime.now(timezone.utc).isoformat(),
            retry_count=0,
            priority=5
        )
        
        assert job_message.correlation_id == correlation_id
        assert job_message.request == request
        assert job_message.retry_count == 0
        assert job_message.priority == 5
    
    def test_job_message_serialization(self):
        """Test job message to/from dict conversion."""
        correlation_id = str(uuid4())
        audio_file_id = uuid4()
        
        request = ImportTracklistRequest(
            url="https://1001tracklists.com/tracklist/12345/test-set",
            audio_file_id=audio_file_id,
            force_refresh=True,
            cue_format="traktor"
        )
        
        original_message = ImportJobMessage(
            correlation_id=correlation_id,
            request=request,
            created_at=datetime.now(timezone.utc).isoformat(),
            retry_count=2,
            priority=3
        )
        
        # Convert to dict and back
        message_dict = original_message.to_dict()
        reconstructed_message = ImportJobMessage.from_dict(message_dict)
        
        assert reconstructed_message.correlation_id == original_message.correlation_id
        assert reconstructed_message.request.url == original_message.request.url
        assert reconstructed_message.request.audio_file_id == original_message.request.audio_file_id
        assert reconstructed_message.request.force_refresh == original_message.request.force_refresh
        assert reconstructed_message.request.cue_format == original_message.request.cue_format
        assert reconstructed_message.retry_count == original_message.retry_count
        assert reconstructed_message.priority == original_message.priority


class TestImportResultMessage:
    """Test import result message."""
    
    def test_success_result_message(self):
        """Test successful result message."""
        correlation_id = str(uuid4())
        tracklist_id = str(uuid4())
        
        result_message = ImportResultMessage(
            correlation_id=correlation_id,
            success=True,
            tracklist_id=tracklist_id,
            processing_time_ms=5000
        )
        
        assert result_message.correlation_id == correlation_id
        assert result_message.success is True
        assert result_message.tracklist_id == tracklist_id
        assert result_message.processing_time_ms == 5000
        assert result_message.error is None
    
    def test_failure_result_message(self):
        """Test failure result message."""
        correlation_id = str(uuid4())
        
        result_message = ImportResultMessage(
            correlation_id=correlation_id,
            success=False,
            error="Import failed: Invalid URL",
            processing_time_ms=1000
        )
        
        assert result_message.correlation_id == correlation_id
        assert result_message.success is False
        assert result_message.error == "Import failed: Invalid URL"
        assert result_message.tracklist_id is None
    
    def test_result_message_dict_conversion(self):
        """Test result message dictionary conversion."""
        correlation_id = str(uuid4())
        
        result_message = ImportResultMessage(
            correlation_id=correlation_id,
            success=True,
            tracklist_id=str(uuid4()),
            processing_time_ms=3000
        )
        
        result_dict = result_message.to_dict()
        
        assert result_dict["correlation_id"] == correlation_id
        assert result_dict["success"] is True
        assert "tracklist_id" in result_dict
        assert result_dict["processing_time_ms"] == 3000
        assert "completed_at" in result_dict


class TestImportMessageHandler:
    """Test import message handler functionality."""
    
    @pytest.fixture
    def message_handler(self):
        """Create message handler for testing."""
        return ImportMessageHandler(connection_url="amqp://guest:guest@localhost:5672/")
    
    @pytest.mark.asyncio
    async def test_message_handler_initialization(self, message_handler):
        """Test message handler initialization."""
        assert message_handler.connection is None
        assert message_handler.channel is None
        assert message_handler.exchange is None
        
        assert message_handler.import_queue == "tracklist_import_queue"
        assert message_handler.import_result_queue == "tracklist_import_result_queue"
        assert message_handler.import_retry_queue == "tracklist_import_retry_queue"
        assert message_handler.import_dlq == "tracklist_import_dlq"
    
    @pytest.mark.asyncio
    @patch('aio_pika.connect_robust')
    async def test_message_handler_connection(self, mock_connect, message_handler):
        """Test message handler connection setup."""
        # Mock connection objects
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_exchange = AsyncMock()
        
        mock_connect.return_value = mock_connection
        mock_connection.channel.return_value = mock_channel
        mock_channel.declare_exchange.return_value = mock_exchange
        
        # Mock queue declarations
        mock_queue = AsyncMock()
        mock_channel.declare_queue.return_value = mock_queue
        
        await message_handler.connect()
        
        # Verify connection setup
        mock_connect.assert_called_once()
        mock_connection.channel.assert_called_once()
        mock_channel.set_qos.assert_called_once_with(prefetch_count=1)
        mock_channel.declare_exchange.assert_called_once()
        
        # Verify queues were declared
        assert mock_channel.declare_queue.call_count == 4  # 4 queues
        assert mock_queue.bind.call_count == 3  # 3 queues bind to exchange
    
    @pytest.mark.asyncio
    @patch('aio_pika.connect_robust')
    async def test_publish_import_job(self, mock_connect, message_handler):
        """Test publishing import job message."""
        # Setup mocks
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_exchange = AsyncMock()
        
        mock_connect.return_value = mock_connection
        mock_connection.channel.return_value = mock_channel
        mock_channel.declare_exchange.return_value = mock_exchange
        mock_channel.declare_queue.return_value = AsyncMock()
        
        await message_handler.connect()
        
        # Create test job message
        correlation_id = str(uuid4())
        request = ImportTracklistRequest(
            url="https://1001tracklists.com/tracklist/12345/test-set",
            audio_file_id=uuid4(),
            force_refresh=False,
            cue_format="standard"
        )
        
        job_message = ImportJobMessage(
            correlation_id=correlation_id,
            request=request,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        
        # Publish job
        result = await message_handler.publish_import_job(job_message)
        
        assert result is True
        mock_exchange.publish.assert_called_once()
        
        # Verify message properties
        publish_call = mock_exchange.publish.call_args
        message = publish_call[0][0]  # First positional argument
        routing_key = publish_call[1]["routing_key"]
        
        assert routing_key == "tracklist.import"
        assert message.correlation_id == correlation_id
    
    @pytest.mark.asyncio
    @patch('aio_pika.connect_robust')
    async def test_publish_import_result(self, mock_connect, message_handler):
        """Test publishing import result message."""
        # Setup mocks
        mock_connection = AsyncMock()
        mock_channel = AsyncMock()
        mock_exchange = AsyncMock()
        
        mock_connect.return_value = mock_connection
        mock_connection.channel.return_value = mock_channel
        mock_channel.declare_exchange.return_value = mock_exchange
        mock_channel.declare_queue.return_value = AsyncMock()
        
        await message_handler.connect()
        
        # Create test result message
        correlation_id = str(uuid4())
        result_message = ImportResultMessage(
            correlation_id=correlation_id,
            success=True,
            tracklist_id=str(uuid4()),
            processing_time_ms=5000
        )
        
        # Publish result
        result = await message_handler.publish_import_result(result_message)
        
        assert result is True
        mock_exchange.publish.assert_called_once()
        
        # Verify routing key
        publish_call = mock_exchange.publish.call_args
        routing_key = publish_call[1]["routing_key"]
        assert routing_key == "tracklist.import.result"
    
    @pytest.mark.asyncio
    async def test_ping_without_connection(self, message_handler):
        """Test ping when not connected."""
        # Should try to connect
        with patch.object(message_handler, 'connect', new_callable=AsyncMock) as mock_connect:
            result = await message_handler.ping()
            mock_connect.assert_called_once()
            assert result is True
    
    def test_handler_registration(self, message_handler):
        """Test registering import handlers."""
        async def test_handler(job_message):
            pass
        
        assert len(message_handler.import_handlers) == 0
        
        message_handler.register_import_handler(test_handler)
        
        assert len(message_handler.import_handlers) == 1
        assert message_handler.import_handlers[0] == test_handler


class TestImportWorker:
    """Test import worker functionality."""
    
    @pytest.fixture
    def import_worker(self):
        """Create import worker for testing."""
        return ImportWorker()
    
    def test_worker_initialization(self, import_worker):
        """Test worker initialization."""
        assert import_worker.import_service is not None
        assert import_worker.matching_service is not None
        assert import_worker.timing_service is not None
        assert import_worker.cue_integration_service is not None
        
        assert import_worker.is_running is False
        assert import_worker.processed_count == 0
        assert import_worker.error_count == 0
    
    def test_worker_stats(self, import_worker):
        """Test worker statistics."""
        stats = import_worker.get_stats()
        
        assert stats["is_running"] is False
        assert stats["processed_count"] == 0
        assert stats["error_count"] == 0
        assert stats["success_rate"] == 0
        
        # Simulate some processing
        import_worker.processed_count = 10
        import_worker.error_count = 2
        
        stats = import_worker.get_stats()
        assert stats["processed_count"] == 10
        assert stats["error_count"] == 2
        assert stats["success_rate"] == 80.0  # (10-2)/10 * 100
    
    @pytest.mark.asyncio
    async def test_process_import_job_success(self, import_worker):
        """Test successful import job processing."""
        # Create test job message
        correlation_id = str(uuid4())
        audio_file_id = uuid4()
        
        request = ImportTracklistRequest(
            url="https://1001tracklists.com/tracklist/12345/test-set",
            audio_file_id=audio_file_id,
            force_refresh=False,
            cue_format="standard"
        )
        
        job_message = ImportJobMessage(
            correlation_id=correlation_id,
            request=request,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        
        # Mock all service calls to return successful results
        with patch.object(import_worker.import_service, 'import_tracklist') as mock_import, \
             patch.object(import_worker.matching_service, 'match_tracklist_to_audio') as mock_match, \
             patch.object(import_worker.timing_service, 'adjust_track_timings') as mock_timing, \
             patch.object(import_worker.cue_integration_service, 'generate_cue_file') as mock_cue, \
             patch('services.tracklist_service.src.workers.import_worker.get_db_context') as mock_db, \
             patch('services.tracklist_service.src.workers.import_worker.import_message_handler') as mock_handler:
            
            # Setup mock returns
            from services.tracklist_service.src.models.tracklist import Tracklist, TrackEntry
            from datetime import timedelta
            
            # Mock tracklist
            mock_tracklist = Tracklist(
                id=uuid4(),
                audio_file_id=audio_file_id,
                source="1001tracklists",
                tracks=[
                    TrackEntry(
                        position=1,
                        start_time=timedelta(minutes=1),
                        end_time=timedelta(minutes=4),
                        artist="Test Artist",
                        title="Test Track"
                    )
                ]
            )
            
            mock_import.return_value = mock_tracklist
            
            # Mock matching result
            mock_matching_result = MagicMock()
            mock_matching_result.confidence_score = 0.95
            mock_matching_result.metadata = {"duration_seconds": 3600}
            mock_match.return_value = mock_matching_result
            
            mock_timing.return_value = mock_tracklist
            
            # Mock CUE result
            mock_cue_result = MagicMock()
            mock_cue_result.success = True
            mock_cue_result.cue_file_id = uuid4()
            mock_cue_result.cue_file_path = "/path/to/test.cue"
            mock_cue.return_value = mock_cue_result
            
            # Mock database
            mock_db_session = AsyncMock()
            mock_db.return_value.__aenter__.return_value = mock_db_session
            
            # Mock message handler
            mock_handler.publish_import_result = AsyncMock(return_value=True)
            
            # Process the job
            await import_worker.process_import_job(job_message)
            
            # Verify all services were called
            mock_import.assert_called_once()
            mock_match.assert_called_once()
            mock_timing.assert_called_once()
            mock_cue.assert_called_once()
            mock_db_session.add.assert_called_once()
            mock_db_session.commit.assert_called_once()
            mock_handler.publish_import_result.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_import_job_failure(self, import_worker):
        """Test import job processing with failure."""
        from services.tracklist_service.src.exceptions import ImportError
        
        # Create test job message
        correlation_id = str(uuid4())
        audio_file_id = uuid4()
        
        request = ImportTracklistRequest(
            url="https://1001tracklists.com/tracklist/12345/test-set",
            audio_file_id=audio_file_id,
            force_refresh=False,
            cue_format="standard"
        )
        
        job_message = ImportJobMessage(
            correlation_id=correlation_id,
            request=request,
            created_at=datetime.now(timezone.utc).isoformat()
        )
        
        # Mock import service to fail
        with patch.object(import_worker.import_service, 'import_tracklist') as mock_import, \
             patch('services.tracklist_service.src.workers.import_worker.import_message_handler') as mock_handler:
            
            mock_import.side_effect = ImportError("Import failed", url=request.url)
            mock_handler.publish_import_result = AsyncMock(return_value=True)
            
            # Process the job (should not raise)
            await import_worker.process_import_job(job_message)
            
            # Verify error result was published
            mock_handler.publish_import_result.assert_called_once()
            
            # Check the published result
            call_args = mock_handler.publish_import_result.call_args[0][0]
            assert call_args.success is False
            assert "Import failed" in call_args.error
            assert call_args.correlation_id == correlation_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])