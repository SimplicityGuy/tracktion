"""Streaming endpoints for Analysis Service."""

import asyncio
import json
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from services.analysis_service.src.repositories import AsyncRecordingRepository
from services.analysis_service.src.structured_logging import get_logger
from shared.core_types.src.async_database import AsyncDatabaseManager

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/streaming", tags=["streaming"])

# Initialize database components
db_manager = AsyncDatabaseManager()
recording_repo = AsyncRecordingRepository(db_manager)


async def generate_audio_chunks(
    file_path: str,
    chunk_size: int = 8192,
    start_byte: int | None = None,
    end_byte: int | None = None,
) -> AsyncGenerator[bytes]:
    """Generate audio file chunks for streaming.

    Args:
        file_path: Path to audio file
        chunk_size: Size of each chunk in bytes
        start_byte: Optional start byte for range requests
        end_byte: Optional end byte for range requests

    Yields:
        Audio file chunks
    """
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    try:
        file_size = Path(file_path).stat().st_size

        # Set range boundaries
        start = start_byte or 0
        end = min(end_byte or file_size - 1, file_size - 1)

        # Validate range
        if start >= file_size or start > end:
            raise ValueError(f"Invalid range: {start}-{end} for file size {file_size}")

        with Path(file_path).open("rb") as f:
            f.seek(start)
            bytes_to_read = end - start + 1
            bytes_read = 0

            while bytes_read < bytes_to_read:
                chunk_size_to_read = min(chunk_size, bytes_to_read - bytes_read)
                chunk = f.read(chunk_size_to_read)

                if not chunk:
                    break

                bytes_read += len(chunk)
                yield chunk

                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.001)

    except Exception as e:
        logger.error(f"Error streaming file {file_path}: {e}")
        raise


@router.get("/audio/{recording_id}")
async def stream_audio(
    request: Request,
    recording_id: str,
    chunk_size: int = Query(8192, description="Chunk size in bytes"),
    start_byte: int | None = Query(None, description="Start byte for range request"),
    end_byte: int | None = Query(None, description="End byte for range request"),
) -> StreamingResponse:
    """Stream audio file in chunks.

    Args:
        request: FastAPI request object for range headers
        recording_id: Recording ID to stream
        chunk_size: Size of each chunk
        start_byte: Optional start byte for partial content
        end_byte: Optional end byte for partial content

    Returns:
        Streaming audio response
    """
    # Get recording from database
    try:
        recording_uuid = UUID(recording_id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid recording ID format") from e

    recording = await recording_repo.get_by_id(recording_uuid)
    if not recording:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Recording not found: {recording_id}")

    file_path = recording.file_path

    # Verify file exists
    if not Path(file_path).exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Audio file not found: {file_path}")

    # Handle Range requests from headers if not in query params
    range_header = request.headers.get("range")
    if range_header and not start_byte and not end_byte:
        try:
            # Parse "bytes=start-end" header
            range_match = range_header.replace("bytes=", "")
            if "-" in range_match:
                start_str, end_str = range_match.split("-", 1)
                start_byte = int(start_str) if start_str else None
                end_byte = int(end_str) if end_str else None
        except (ValueError, AttributeError):
            # Invalid range header, ignore
            pass

    # Get file info
    file_path_obj = Path(file_path)
    file_size = file_path_obj.stat().st_size
    file_ext = file_path_obj.suffix.lower()

    # Determine media type based on file extension
    media_type_map = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
    }
    media_type = media_type_map.get(file_ext, "audio/octet-stream")

    # Set range boundaries
    start = start_byte or 0
    end = min(end_byte or file_size - 1, file_size - 1)
    content_length = end - start + 1

    # Create headers for audio streaming
    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "public, max-age=3600",
        "X-Recording-ID": recording_id,
        "Content-Length": str(content_length),
    }

    # Add range headers if partial content requested
    is_partial = start_byte is not None or end_byte is not None or range_header

    if is_partial:
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        status_code = status.HTTP_206_PARTIAL_CONTENT
    else:
        status_code = status.HTTP_200_OK

    try:
        return StreamingResponse(
            generate_audio_chunks(file_path, chunk_size, start, end),
            media_type=media_type,
            headers=headers,
            status_code=status_code,
        )
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e)) from e


async def generate_analysis_events(
    recording_id: str,
) -> AsyncGenerator[dict[str, Any]]:
    """Generate Server-Sent Events for analysis progress.

    Args:
        recording_id: Recording being analyzed

    Yields:
        SSE events with analysis progress
    """
    # Simulate analysis progress
    stages = [
        ("loading", 0.1, "Loading audio file"),
        ("preprocessing", 0.2, "Preprocessing audio"),
        ("bpm_detection", 0.4, "Detecting BPM"),
        ("key_detection", 0.6, "Detecting key"),
        ("mood_analysis", 0.8, "Analyzing mood"),
        ("complete", 1.0, "Analysis complete"),
    ]

    for stage, progress, message in stages:
        await asyncio.sleep(1)  # Simulate processing time

        event_data = {
            "recording_id": recording_id,
            "stage": stage,
            "progress": progress,
            "message": message,
            "timestamp": asyncio.get_event_loop().time(),
        }

        # Yield SSE event
        yield {"event": "progress", "data": json.dumps(event_data)}

    # Send completion event
    yield {
        "event": "complete",
        "data": json.dumps(
            {
                "recording_id": recording_id,
                "results": {"bpm": 128.5, "key": "Am", "mood": "energetic"},
            }
        ),
    }


@router.get("/events/{recording_id}")
async def stream_analysis_events(recording_id: str) -> EventSourceResponse:
    """Stream analysis progress using Server-Sent Events.

    Args:
        recording_id: Recording ID to monitor

    Returns:
        SSE stream of analysis events
    """
    logger.info("Starting SSE stream for analysis", extra={"recording_id": recording_id})

    return EventSourceResponse(generate_analysis_events(recording_id))


async def batch_process_generator(recording_ids: list[str], batch_size: int = 5) -> AsyncGenerator[str]:
    """Generate batch processing results as NDJSON stream.

    Args:
        recording_ids: List of recordings to process
        batch_size: Number of recordings per batch

    Yields:
        NDJSON lines with processing results
    """
    for i in range(0, len(recording_ids), batch_size):
        batch = recording_ids[i : i + batch_size]

        # Process batch (simulated)
        for recording_id in batch:
            await asyncio.sleep(0.5)  # Simulate processing

            result = {
                "recording_id": recording_id,
                "status": "processed",
                "timestamp": asyncio.get_event_loop().time(),
            }

            # Yield as NDJSON line
            yield json.dumps(result) + "\n"


@router.post("/batch-process")
async def stream_batch_processing(
    recording_ids: list[str],
    batch_size: int = Query(5, description="Batch size for processing"),
) -> StreamingResponse:
    """Stream batch processing results as NDJSON.

    Args:
        recording_ids: List of recording IDs to process
        batch_size: Number of recordings per batch

    Returns:
        Streaming NDJSON response with processing results
    """
    if not recording_ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No recording IDs provided")

    logger.info(
        "Starting batch processing stream",
        extra={"count": len(recording_ids), "batch_size": batch_size},
    )

    return StreamingResponse(
        batch_process_generator(recording_ids, batch_size),
        media_type="application/x-ndjson",
        headers={
            "X-Total-Count": str(len(recording_ids)),
            "X-Batch-Size": str(batch_size),
        },
    )


async def generate_log_stream(recording_id: str, follow: bool = False) -> AsyncGenerator[str]:
    """Generate log stream for a recording's processing.

    Args:
        recording_id: Recording to get logs for
        follow: Whether to follow new logs

    Yields:
        Log lines
    """
    # Simulate log streaming
    logs = [
        f"[INFO] Starting analysis for {recording_id}",
        "[DEBUG] Loading audio file from storage",
        "[DEBUG] Audio loaded: 44100Hz, 2 channels, 180.5s",
        "[INFO] Running BPM detection",
        "[DEBUG] BPM detected: 128.5 (confidence: 0.95)",
        "[INFO] Running key detection",
        "[DEBUG] Key detected: Am (confidence: 0.88)",
        "[INFO] Analysis complete",
    ]

    for log in logs:
        await asyncio.sleep(0.3)
        yield log + "\n"

    if follow:
        # Continue streaming new logs
        max_streaming_updates = 5  # Maximum number of streaming updates for demo
        streaming_update_delay = 2  # Delay between streaming updates in seconds

        counter = 0
        while counter < max_streaming_updates:
            await asyncio.sleep(streaming_update_delay)
            yield f"[DEBUG] Monitoring... (update {counter})\n"
            counter += 1


@router.get("/logs/{recording_id}")
async def stream_logs(
    recording_id: str, follow: bool = Query(False, description="Follow log output")
) -> StreamingResponse:
    """Stream processing logs for a recording.

    Args:
        recording_id: Recording ID to get logs for
        follow: Whether to continue following new logs

    Returns:
        Streaming text response with logs
    """
    return StreamingResponse(
        generate_log_stream(recording_id, follow),
        media_type="text/plain",
        headers={"X-Recording-ID": recording_id, "X-Follow": str(follow)},
    )
