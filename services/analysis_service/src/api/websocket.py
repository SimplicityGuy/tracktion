"""WebSocket implementation for real-time updates."""

import asyncio
import json
from typing import Any, Dict, Set

from fastapi import WebSocket
from fastapi.websockets import WebSocketState

from ..structured_logging import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for broadcasting updates."""

    def __init__(self) -> None:
        """Initialize connection manager."""
        # Active connections by client ID
        self.active_connections: Dict[str, WebSocket] = {}
        # Subscriptions: recording_id -> set of client_ids
        self.subscriptions: Dict[str, Set[str]] = {}
        # Client metadata
        self.client_metadata: Dict[str, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """Accept and register a new WebSocket connection.

        Args:
            websocket: WebSocket connection
            client_id: Unique client identifier
        """
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.client_metadata[client_id] = {"connected_at": asyncio.get_event_loop().time(), "subscriptions": set()}
        logger.info("WebSocket client connected", extra={"client_id": client_id})

    def disconnect(self, client_id: str) -> None:
        """Remove a WebSocket connection.

        Args:
            client_id: Client identifier to disconnect
        """
        if client_id in self.active_connections:
            del self.active_connections[client_id]

            # Remove subscriptions
            for recording_id in list(self.subscriptions.keys()):
                if client_id in self.subscriptions[recording_id]:
                    self.subscriptions[recording_id].remove(client_id)
                    if not self.subscriptions[recording_id]:
                        del self.subscriptions[recording_id]

            # Clean up metadata
            if client_id in self.client_metadata:
                del self.client_metadata[client_id]

            logger.info("WebSocket client disconnected", extra={"client_id": client_id})

    async def send_personal_message(self, message: str, client_id: str) -> None:
        """Send a message to a specific client.

        Args:
            message: Message to send
            client_id: Target client ID
        """
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    logger.error(
                        "Failed to send message to client",
                        extra={"client_id": client_id, "error": str(e)},
                    )
                    self.disconnect(client_id)

    async def send_json(self, data: Dict[str, Any], client_id: str) -> None:
        """Send JSON data to a specific client.

        Args:
            data: Data to send as JSON
            client_id: Target client ID
        """
        await self.send_personal_message(json.dumps(data), client_id)

    async def broadcast(self, message: str) -> None:
        """Broadcast a message to all connected clients.

        Args:
            message: Message to broadcast
        """
        disconnected_clients = []

        for client_id, websocket in self.active_connections.items():
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.send_text(message)
                except Exception as e:
                    logger.error(
                        "Failed to broadcast to client",
                        extra={"client_id": client_id, "error": str(e)},
                    )
                    disconnected_clients.append(client_id)
            else:
                disconnected_clients.append(client_id)

        # Clean up disconnected clients
        for client_id in disconnected_clients:
            self.disconnect(client_id)

    async def broadcast_json(self, data: Dict[str, Any]) -> None:
        """Broadcast JSON data to all connected clients.

        Args:
            data: Data to broadcast as JSON
        """
        await self.broadcast(json.dumps(data))

    def subscribe(self, client_id: str, recording_id: str) -> None:
        """Subscribe a client to recording updates.

        Args:
            client_id: Client to subscribe
            recording_id: Recording to subscribe to
        """
        if recording_id not in self.subscriptions:
            self.subscriptions[recording_id] = set()

        self.subscriptions[recording_id].add(client_id)

        if client_id in self.client_metadata:
            self.client_metadata[client_id]["subscriptions"].add(recording_id)

        logger.info(
            "Client subscribed to recording",
            extra={"client_id": client_id, "recording_id": recording_id},
        )

    def unsubscribe(self, client_id: str, recording_id: str) -> None:
        """Unsubscribe a client from recording updates.

        Args:
            client_id: Client to unsubscribe
            recording_id: Recording to unsubscribe from
        """
        if recording_id in self.subscriptions:
            self.subscriptions[recording_id].discard(client_id)
            if not self.subscriptions[recording_id]:
                del self.subscriptions[recording_id]

        if client_id in self.client_metadata:
            self.client_metadata[client_id]["subscriptions"].discard(recording_id)

        logger.info(
            "Client unsubscribed from recording",
            extra={"client_id": client_id, "recording_id": recording_id},
        )

    async def broadcast_to_recording(self, recording_id: str, data: Dict[str, Any]) -> None:
        """Broadcast update to all clients subscribed to a recording.

        Args:
            recording_id: Recording ID to broadcast to
            data: Data to broadcast
        """
        if recording_id in self.subscriptions:
            message = json.dumps({"recording_id": recording_id, "data": data})

            disconnected_clients = []

            for client_id in self.subscriptions[recording_id]:
                if client_id in self.active_connections:
                    websocket = self.active_connections[client_id]
                    if websocket.client_state == WebSocketState.CONNECTED:
                        try:
                            await websocket.send_text(message)
                        except Exception as e:
                            logger.error(
                                "Failed to send to subscriber",
                                extra={
                                    "client_id": client_id,
                                    "recording_id": recording_id,
                                    "error": str(e),
                                },
                            )
                            disconnected_clients.append(client_id)
                    else:
                        disconnected_clients.append(client_id)

            # Clean up disconnected clients
            for client_id in disconnected_clients:
                self.disconnect(client_id)

    def get_connection_count(self) -> int:
        """Get the number of active connections.

        Returns:
            Number of active connections
        """
        return len(self.active_connections)

    def get_subscription_count(self, recording_id: str) -> int:
        """Get the number of subscribers for a recording.

        Args:
            recording_id: Recording ID to check

        Returns:
            Number of subscribers
        """
        return len(self.subscriptions.get(recording_id, set()))


# Global connection manager instance
manager = ConnectionManager()
