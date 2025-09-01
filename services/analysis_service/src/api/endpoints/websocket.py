"""WebSocket endpoints for real-time updates."""

import asyncio
import json
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from services.analysis_service.src.api.websocket import manager
from services.analysis_service.src.structured_logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["websocket"])


@router.websocket("/v1/ws")
async def websocket_endpoint(websocket: WebSocket, client_id: str | None = Query(None)) -> None:
    """Main WebSocket endpoint for real-time updates.

    Args:
        websocket: WebSocket connection
        client_id: Optional client ID (generated if not provided)
    """
    # Generate client ID if not provided
    if not client_id:
        client_id = str(uuid4())

    await manager.connect(websocket, client_id)

    try:
        # Send welcome message
        await manager.send_json(
            {
                "type": "welcome",
                "client_id": client_id,
                "message": "Connected to Analysis Service WebSocket",
            },
            client_id,
        )

        # Handle incoming messages
        while True:
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                await handle_websocket_message(client_id, message)
            except json.JSONDecodeError:
                await manager.send_json({"type": "error", "message": "Invalid JSON format"}, client_id)
            except Exception as e:
                logger.error(
                    "Error handling WebSocket message",
                    extra={"client_id": client_id, "error": str(e)},
                )
                await manager.send_json({"type": "error", "message": "Error processing message"}, client_id)

    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info("WebSocket disconnected", extra={"client_id": client_id})
    except Exception as e:
        logger.error("WebSocket error", extra={"client_id": client_id, "error": str(e)})
        manager.disconnect(client_id)


async def handle_websocket_message(client_id: str, message: dict[str, Any]) -> None:
    """Handle incoming WebSocket messages.

    Args:
        client_id: Client identifier
        message: Received message
    """
    msg_type = message.get("type")

    if msg_type == "ping":
        # Respond to ping with pong
        await manager.send_json({"type": "pong", "timestamp": asyncio.get_event_loop().time()}, client_id)

    elif msg_type == "subscribe":
        # Subscribe to recording updates
        recording_id = message.get("recording_id")
        if recording_id:
            manager.subscribe(client_id, recording_id)
            await manager.send_json(
                {
                    "type": "subscribed",
                    "recording_id": recording_id,
                    "message": f"Subscribed to updates for recording {recording_id}",
                },
                client_id,
            )
        else:
            await manager.send_json(
                {"type": "error", "message": "Missing recording_id for subscription"},
                client_id,
            )

    elif msg_type == "unsubscribe":
        # Unsubscribe from recording updates
        recording_id = message.get("recording_id")
        if recording_id:
            manager.unsubscribe(client_id, recording_id)
            await manager.send_json(
                {
                    "type": "unsubscribed",
                    "recording_id": recording_id,
                    "message": f"Unsubscribed from updates for recording {recording_id}",
                },
                client_id,
            )
        else:
            await manager.send_json(
                {"type": "error", "message": "Missing recording_id for unsubscription"},
                client_id,
            )

    elif msg_type == "get_status":
        # Get connection status
        metadata = manager.client_metadata.get(client_id, {})
        await manager.send_json(
            {
                "type": "status",
                "client_id": client_id,
                "subscriptions": list(metadata.get("subscriptions", set())),
                "connected_at": metadata.get("connected_at"),
                "total_connections": manager.get_connection_count(),
            },
            client_id,
        )

    else:
        # Unknown message type
        await manager.send_json({"type": "error", "message": f"Unknown message type: {msg_type}"}, client_id)


# Event broadcasting functions (to be called from other parts of the application)
async def broadcast_progress_update(
    recording_id: str, progress: float, status: str, message: str | None = None
) -> None:
    """Broadcast progress update for a recording.

    Args:
        recording_id: Recording being processed
        progress: Progress percentage (0-1)
        status: Current status
        message: Optional status message
    """
    await manager.broadcast_to_recording(
        recording_id,
        {
            "type": "progress",
            "progress": progress,
            "status": status,
            "message": message,
            "timestamp": asyncio.get_event_loop().time(),
        },
    )


async def broadcast_analysis_complete(recording_id: str, results: dict[str, Any]) -> None:
    """Broadcast analysis completion for a recording.

    Args:
        recording_id: Recording that completed
        results: Analysis results
    """
    await manager.broadcast_to_recording(
        recording_id,
        {
            "type": "analysis_complete",
            "results": results,
            "timestamp": asyncio.get_event_loop().time(),
        },
    )


async def broadcast_error(recording_id: str, error: str, details: dict[str, Any] | None = None) -> None:
    """Broadcast error for a recording.

    Args:
        recording_id: Recording with error
        error: Error message
        details: Optional error details
    """
    await manager.broadcast_to_recording(
        recording_id,
        {
            "type": "error",
            "error": error,
            "details": details,
            "timestamp": asyncio.get_event_loop().time(),
        },
    )
