"""Tests for notification system integration (Task 8)."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from services.analysis_service.src.main import AnalysisService


class TestNotificationIntegration:
    """Test notification system integration in analysis service."""

    @pytest.fixture
    def analysis_service(self):
        """Create analysis service instance."""
        service = AnalysisService()
        service.messaging_service = AsyncMock()
        return service

    @pytest.mark.asyncio
    async def test_send_notification_completed(self, analysis_service):
        """Test sending notification for completed analysis."""
        recording_id = uuid4()
        correlation_id = str(uuid4())
        metadata = {"duration_ms": 180000, "tracks_found": 10}

        # Test
        await analysis_service.send_notification(
            recording_id=recording_id, status="completed", correlation_id=correlation_id, metadata=metadata
        )

        # Verify
        analysis_service.messaging_service.publish_message.assert_called_once()
        call_args = analysis_service.messaging_service.publish_message.call_args

        assert call_args.kwargs["exchange_name"] == "notifications"
        assert call_args.kwargs["routing_key"] == "analysis_completed"
        assert call_args.kwargs["correlation_id"] == correlation_id

        message = call_args.kwargs["message"]
        assert message["recording_id"] == str(recording_id)
        assert message["status"] == "completed"
        assert message["service"] == "analysis_service"
        assert message["metadata"] == metadata
        assert "Analysis completed for recording" in message["message"]

    @pytest.mark.asyncio
    async def test_send_notification_failed(self, analysis_service):
        """Test sending notification for failed analysis."""
        recording_id = uuid4()
        correlation_id = str(uuid4())
        metadata = {"error": "File not found", "retry_count": 3}

        # Test
        await analysis_service.send_notification(
            recording_id=recording_id, status="failed", correlation_id=correlation_id, metadata=metadata
        )

        # Verify
        analysis_service.messaging_service.publish_message.assert_called_once()
        call_args = analysis_service.messaging_service.publish_message.call_args

        assert call_args.kwargs["routing_key"] == "analysis_failed"

        message = call_args.kwargs["message"]
        assert message["status"] == "failed"
        assert message["alert_type"] == "error"
        assert "Analysis failed for recording" in message["message"]
        assert message["metadata"]["error"] == "File not found"

    @pytest.mark.asyncio
    async def test_send_notification_processing(self, analysis_service):
        """Test sending notification for processing status."""
        recording_id = uuid4()
        correlation_id = str(uuid4())

        # Test
        await analysis_service.send_notification(
            recording_id=recording_id, status="processing", correlation_id=correlation_id
        )

        # Verify
        call_args = analysis_service.messaging_service.publish_message.call_args
        assert call_args.kwargs["routing_key"] == "analysis_started"

        message = call_args.kwargs["message"]
        assert "Analysis started for recording" in message["message"]

    @pytest.mark.asyncio
    async def test_send_notification_custom_status(self, analysis_service):
        """Test sending notification with custom status."""
        recording_id = uuid4()
        correlation_id = str(uuid4())

        # Test
        await analysis_service.send_notification(
            recording_id=recording_id, status="pending_review", correlation_id=correlation_id
        )

        # Verify
        call_args = analysis_service.messaging_service.publish_message.call_args
        assert call_args.kwargs["routing_key"] == "analysis_status_update"

        message = call_args.kwargs["message"]
        assert "Analysis status update" in message["message"]
        assert "pending_review" in message["message"]

    @pytest.mark.asyncio
    async def test_send_notification_no_messaging_service(self, analysis_service):
        """Test notification when messaging service is not available."""
        analysis_service.messaging_service = None
        recording_id = uuid4()
        correlation_id = str(uuid4())

        # Test - should not raise exception
        await analysis_service.send_notification(
            recording_id=recording_id, status="completed", correlation_id=correlation_id
        )

        # Verify - no exception raised, just logged

    @pytest.mark.asyncio
    async def test_send_notification_publish_error(self, analysis_service):
        """Test notification when publish fails."""
        recording_id = uuid4()
        correlation_id = str(uuid4())

        # Make publish_message raise an exception
        analysis_service.messaging_service.publish_message.side_effect = Exception("Connection failed")

        # Test - should not raise exception
        await analysis_service.send_notification(
            recording_id=recording_id, status="completed", correlation_id=correlation_id
        )

        # Verify - exception caught and logged
        analysis_service.messaging_service.publish_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_notification_includes_timestamp(self, analysis_service):
        """Test that notification includes timestamp."""
        recording_id = uuid4()
        correlation_id = str(uuid4())

        with patch("services.analysis_service.src.main.datetime") as mock_datetime:
            mock_now = datetime(2025, 1, 2, 12, 0, 0, tzinfo=UTC)
            mock_datetime.now.return_value = mock_now

            # Test
            await analysis_service.send_notification(
                recording_id=recording_id, status="completed", correlation_id=correlation_id
            )

            # Verify
            call_args = analysis_service.messaging_service.publish_message.call_args
            message = call_args.kwargs["message"]
            assert message["timestamp"] == mock_now.isoformat()

    @pytest.mark.asyncio
    async def test_send_notification_without_metadata(self, analysis_service):
        """Test sending notification without metadata."""
        recording_id = uuid4()
        correlation_id = str(uuid4())

        # Test
        await analysis_service.send_notification(
            recording_id=recording_id, status="completed", correlation_id=correlation_id, metadata=None
        )

        # Verify
        call_args = analysis_service.messaging_service.publish_message.call_args
        message = call_args.kwargs["message"]
        assert "metadata" not in message or message.get("metadata") is None


class TestNotificationMessageFormat:
    """Test notification message format and structure."""

    @pytest.fixture
    def analysis_service(self):
        """Create analysis service instance."""
        service = AnalysisService()
        service.messaging_service = AsyncMock()
        return service

    @pytest.mark.asyncio
    async def test_notification_message_structure(self, analysis_service):
        """Test that notification messages have correct structure."""
        recording_id = uuid4()
        correlation_id = str(uuid4())
        metadata = {"key": "value"}

        # Test
        await analysis_service.send_notification(
            recording_id=recording_id, status="completed", correlation_id=correlation_id, metadata=metadata
        )

        # Verify message structure
        call_args = analysis_service.messaging_service.publish_message.call_args
        message = call_args.kwargs["message"]

        # Required fields
        assert "recording_id" in message
        assert "status" in message
        assert "correlation_id" in message
        assert "timestamp" in message
        assert "service" in message
        assert "message" in message

        # Field types
        assert isinstance(message["recording_id"], str)
        assert isinstance(message["status"], str)
        assert isinstance(message["correlation_id"], str)
        assert isinstance(message["timestamp"], str)
        assert isinstance(message["service"], str)
        assert isinstance(message["message"], str)

        # Service identification
        assert message["service"] == "analysis_service"

    @pytest.mark.asyncio
    async def test_notification_routing_keys(self, analysis_service):
        """Test that correct routing keys are used for different statuses."""
        recording_id = uuid4()
        correlation_id = str(uuid4())

        status_to_routing_key = {
            "completed": "analysis_completed",
            "failed": "analysis_failed",
            "processing": "analysis_started",
            "custom_status": "analysis_status_update",
        }

        for status, expected_routing_key in status_to_routing_key.items():
            # Reset mock
            analysis_service.messaging_service.reset_mock()

            # Test
            await analysis_service.send_notification(
                recording_id=recording_id, status=status, correlation_id=correlation_id
            )

            # Verify
            call_args = analysis_service.messaging_service.publish_message.call_args
            assert call_args.kwargs["routing_key"] == expected_routing_key
