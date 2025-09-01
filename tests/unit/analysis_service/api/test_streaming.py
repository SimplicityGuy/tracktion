"""Tests for streaming endpoints."""

import json

from fastapi import status
from fastapi.testclient import TestClient

from services.analysis_service.src.api.app import app


class TestStreamingEndpoints:
    """Test streaming response endpoints."""

    def test_audio_streaming(self):
        """Test audio file streaming."""
        client = TestClient(app)
        recording_id = "test-recording-123"

        with client as c:
            response = c.get(f"/v1/streaming/audio/{recording_id}")

            assert response.status_code == status.HTTP_200_OK
            assert response.headers["content-type"] == "audio/wav"
            assert "X-Recording-ID" in response.headers

            # Read streaming content
            content = b""
            for chunk in response.iter_bytes(chunk_size=1024):
                content += chunk

            assert len(content) > 0

    def test_partial_content_streaming(self):
        """Test partial content with range request."""
        client = TestClient(app)
        recording_id = "test-recording-456"

        response = client.get(
            f"/v1/streaming/audio/{recording_id}",
            params={"start_byte": 0, "end_byte": 1024},
        )

        assert response.status_code == status.HTTP_206_PARTIAL_CONTENT
        assert "Accept-Ranges" in response.headers

    def test_sse_events(self):
        """Test Server-Sent Events streaming."""
        client = TestClient(app)
        recording_id = "test-recording-789"

        # SSE endpoints return event stream
        response = client.get(
            f"/v1/streaming/events/{recording_id}",
            headers={"Accept": "text/event-stream"},
        )

        assert response.status_code == status.HTTP_200_OK
        # Note: TestClient doesn't fully support SSE,
        # but we can verify the endpoint exists

    def test_batch_processing_stream(self):
        """Test batch processing NDJSON stream."""
        client = TestClient(app)

        recording_ids = ["rec1", "rec2", "rec3"]

        response = client.post("/v1/streaming/batch-process", json=recording_ids, params={"batch_size": 2})

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "application/x-ndjson"
        assert response.headers["X-Total-Count"] == "3"
        assert response.headers["X-Batch-Size"] == "2"

        # Parse NDJSON response
        lines = response.text.strip().split("\n")
        for line in lines:
            if line:
                data = json.loads(line)
                assert "recording_id" in data
                assert "status" in data

    def test_log_streaming(self):
        """Test log streaming."""
        client = TestClient(app)
        recording_id = "test-recording-logs"

        response = client.get(f"/v1/streaming/logs/{recording_id}")

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        assert "X-Recording-ID" in response.headers

        # Check log content
        logs = response.text
        assert "[INFO]" in logs
        assert "[DEBUG]" in logs

    def test_log_follow_mode(self):
        """Test log streaming with follow mode."""
        client = TestClient(app)
        recording_id = "test-recording-follow"

        response = client.get(f"/v1/streaming/logs/{recording_id}", params={"follow": True})

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["X-Follow"] == "True"
