"""Tests for WebSocket support."""

import json

import pytest
from fastapi.testclient import TestClient

from services.analysis_service.src.api.app import app
from services.analysis_service.src.api.websocket import manager


class TestWebSocketConnection:
    """Test WebSocket connection management."""

    def test_websocket_connection(self):
        """Test basic WebSocket connection."""
        client = TestClient(app)

        with client.websocket_connect("/v1/ws") as websocket:
            # Should receive welcome message
            data = websocket.receive_text()
            message = json.loads(data)

            assert message["type"] == "welcome"
            assert "client_id" in message
            assert message["message"] == "Connected to Analysis Service WebSocket"

    def test_websocket_ping_pong(self):
        """Test WebSocket ping/pong."""
        client = TestClient(app)

        with client.websocket_connect("/v1/ws") as websocket:
            # Receive welcome
            websocket.receive_text()

            # Send ping
            websocket.send_text(json.dumps({"type": "ping"}))

            # Should receive pong
            data = websocket.receive_text()
            message = json.loads(data)

            assert message["type"] == "pong"
            assert "timestamp" in message

    def test_websocket_subscription(self):
        """Test WebSocket subscription to recording updates."""
        client = TestClient(app)

        with client.websocket_connect("/v1/ws") as websocket:
            # Receive welcome
            websocket.receive_text()

            # Subscribe to recording
            recording_id = "test-recording-123"
            websocket.send_text(json.dumps({"type": "subscribe", "recording_id": recording_id}))

            # Should receive subscription confirmation
            data = websocket.receive_text()
            message = json.loads(data)

            assert message["type"] == "subscribed"
            assert message["recording_id"] == recording_id

    def test_websocket_unsubscription(self):
        """Test WebSocket unsubscription."""
        client = TestClient(app)

        with client.websocket_connect("/v1/ws") as websocket:
            # Receive welcome
            websocket.receive_text()

            # Subscribe first
            recording_id = "test-recording-456"
            websocket.send_text(json.dumps({"type": "subscribe", "recording_id": recording_id}))
            websocket.receive_text()  # subscription confirmation

            # Unsubscribe
            websocket.send_text(json.dumps({"type": "unsubscribe", "recording_id": recording_id}))

            # Should receive unsubscription confirmation
            data = websocket.receive_text()
            message = json.loads(data)

            assert message["type"] == "unsubscribed"
            assert message["recording_id"] == recording_id

    def test_websocket_status(self):
        """Test WebSocket status request."""
        client = TestClient(app)

        with client.websocket_connect("/v1/ws?client_id=test-client") as websocket:
            # Receive welcome
            websocket.receive_text()

            # Request status
            websocket.send_text(json.dumps({"type": "get_status"}))

            # Should receive status
            data = websocket.receive_text()
            message = json.loads(data)

            assert message["type"] == "status"
            assert message["client_id"] == "test-client"
            assert "subscriptions" in message
            assert "connected_at" in message
            assert "total_connections" in message


class TestConnectionManager:
    """Test ConnectionManager functionality."""

    @pytest.mark.asyncio
    async def test_manager_subscription_tracking(self):
        """Test that manager tracks subscriptions correctly."""
        # Reset manager state
        manager.active_connections.clear()
        manager.subscriptions.clear()
        manager.client_metadata.clear()

        client_id = "test-client-1"
        recording_id = "recording-1"

        # Subscribe client
        manager.subscribe(client_id, recording_id)

        assert recording_id in manager.subscriptions
        assert client_id in manager.subscriptions[recording_id]
        assert manager.get_subscription_count(recording_id) == 1

        # Unsubscribe
        manager.unsubscribe(client_id, recording_id)

        assert recording_id not in manager.subscriptions
        assert manager.get_subscription_count(recording_id) == 0

    def test_manager_connection_count(self):
        """Test connection counting."""
        # Reset manager state
        manager.active_connections.clear()

        assert manager.get_connection_count() == 0

        # Simulate connections
        manager.active_connections["client1"] = None
        manager.active_connections["client2"] = None

        assert manager.get_connection_count() == 2
