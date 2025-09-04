"""Streaming endpoints for Analysis Service."""

import asyncio
import json
from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from services.analysis_service.src.async_progress_tracker import AsyncProgressTracker
from services.analysis_service.src.repositories import AsyncAnalysisResultRepository, AsyncRecordingRepository
from services.analysis_service.src.structured_logging import get_logger
from shared.core_types.src.async_database import AsyncDatabaseManager
from shared.core_types.src.models import AnalysisResult

logger = get_logger(__name__)

router = APIRouter(prefix="/v1/streaming", tags=["streaming"])

# Initialize database components
db_manager = AsyncDatabaseManager()
recording_repo = AsyncRecordingRepository(db_manager)
analysis_result_repo = AsyncAnalysisResultRepository(db_manager)

# Initialize progress tracker
progress_tracker = AsyncProgressTracker(
    redis_url=None,  # Use in-memory tracking for now
    enable_websocket=False,  # SSE instead of WebSocket
    update_interval_seconds=0.5,  # More frequent updates for better UX
)

# Initialize progress tracker lazily on first use
# This will be set when the app starts up with an event loop
_tracker_task: asyncio.Task | None = None


async def ensure_tracker_initialized() -> None:
    """Ensure the progress tracker is initialized."""
    global _tracker_task  # noqa: PLW0603 - Global needed for singleton initialization
    if _tracker_task is None:
        _tracker_task = asyncio.create_task(progress_tracker.initialize())


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
    if not file_path or not Path(file_path).exists():
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
    if not file_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File path is None")
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
        if not file_path:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File path is None")
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
    """Generate Server-Sent Events for real analysis progress.

    Args:
        recording_id: Recording being analyzed

    Yields:
        SSE events with analysis progress from actual job status
    """
    try:
        recording_uuid = UUID(recording_id)
    except ValueError:
        # Invalid UUID format, send error and return
        yield {
            "event": "error",
            "data": json.dumps(
                {
                    "recording_id": recording_id,
                    "error": "Invalid recording ID format",
                    "timestamp": asyncio.get_event_loop().time(),
                }
            ),
        }
        return

    # Verify recording exists
    recording = await recording_repo.get_by_id(recording_uuid)
    if not recording:
        yield {
            "event": "error",
            "data": json.dumps(
                {
                    "recording_id": recording_id,
                    "error": "Recording not found",
                    "timestamp": asyncio.get_event_loop().time(),
                }
            ),
        }
        return

    # Track analysis progress using real database queries
    last_status = None
    last_progress = 0.0
    analysis_complete = False
    max_iterations = 300  # Maximum 5 minutes of polling (300 * 1s)
    iteration = 0

    # Send initial status
    yield {
        "event": "started",
        "data": json.dumps(
            {
                "recording_id": recording_id,
                "message": "Starting analysis monitoring",
                "timestamp": asyncio.get_event_loop().time(),
            }
        ),
    }

    while not analysis_complete and iteration < max_iterations:
        iteration += 1

        # Query current recording status
        current_recording = await recording_repo.get_by_id(recording_uuid)
        if not current_recording:
            break

        current_status = current_recording.processing_status

        # Query analysis results to determine progress
        analysis_results = await analysis_result_repo.get_by_recording_id(recording_uuid)

        # Calculate progress based on analysis results
        total_analyses = 4  # bpm, key, mood, energy (typical analysis types)
        completed_analyses = len([r for r in analysis_results if r.status == "completed"])
        in_progress_analyses = len([r for r in analysis_results if r.status == "processing"])
        # Count failed analyses (tracked for potential future use)
        _ = len([r for r in analysis_results if r.status == "failed"])

        # Calculate progress percentage
        if current_status == "completed":
            current_progress = 100.0
            analysis_complete = True
        elif current_status == "failed":
            current_progress = 0.0
            analysis_complete = True
        elif current_status == "processing":
            # Base progress on completed + partial credit for in-progress
            current_progress = min(95.0, (completed_analyses + (in_progress_analyses * 0.5)) / total_analyses * 100)
        else:
            # pending or other status
            current_progress = 5.0

        # Determine current stage based on analysis results
        current_stage = "initializing"
        stage_message = "Initializing analysis"

        if analysis_results:

            def get_analysis_created_at(result: AnalysisResult) -> datetime:
                return cast("datetime", result.created_at)

            latest_result = max(analysis_results, key=get_analysis_created_at)
            if latest_result.analysis_type == "bpm":
                current_stage = "bpm_detection"
                stage_message = f"Detecting BPM ({latest_result.status})"
            elif latest_result.analysis_type == "key":
                current_stage = "key_detection"
                stage_message = f"Detecting key ({latest_result.status})"
            elif latest_result.analysis_type == "mood":
                current_stage = "mood_analysis"
                stage_message = f"Analyzing mood ({latest_result.status})"
            elif latest_result.analysis_type == "energy":
                current_stage = "energy_analysis"
                stage_message = f"Analyzing energy ({latest_result.status})"

        # Only send updates when status or progress changes significantly
        progress_changed = abs(current_progress - last_progress) >= 1.0
        status_changed = current_status != last_status

        if status_changed or progress_changed or analysis_complete:
            event_data = {
                "recording_id": recording_id,
                "stage": current_stage,
                "progress": current_progress,
                "message": stage_message,
                "status": current_status,
                "completed_analyses": completed_analyses,
                "total_analyses": total_analyses,
                "timestamp": asyncio.get_event_loop().time(),
            }

            # Add error details if analysis failed
            if current_status == "failed" and current_recording.processing_error:
                event_data["error"] = current_recording.processing_error

            yield {"event": "progress", "data": json.dumps(event_data)}

            last_status = current_status
            last_progress = current_progress

        # Handle completion
        if analysis_complete:
            if current_status == "completed":
                # Get final results
                results = await analysis_result_repo.get_completed_results_for_recording(recording_uuid)

                yield {
                    "event": "complete",
                    "data": json.dumps(
                        {
                            "recording_id": recording_id,
                            "results": results,
                            "message": "Analysis completed successfully",
                            "timestamp": asyncio.get_event_loop().time(),
                        }
                    ),
                }
            elif current_status == "failed":
                yield {
                    "event": "failed",
                    "data": json.dumps(
                        {
                            "recording_id": recording_id,
                            "error": current_recording.processing_error or "Analysis failed",
                            "message": "Analysis failed",
                            "timestamp": asyncio.get_event_loop().time(),
                        }
                    ),
                }
            break

        # Wait before next poll (1 second interval)
        await asyncio.sleep(1.0)

    # Handle timeout case
    if iteration >= max_iterations and not analysis_complete:
        yield {
            "event": "timeout",
            "data": json.dumps(
                {
                    "recording_id": recording_id,
                    "error": "Analysis monitoring timed out",
                    "message": "Analysis is taking longer than expected",
                    "timestamp": asyncio.get_event_loop().time(),
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
    logger.info("Starting SSE stream for real analysis progress", extra={"recording_id": recording_id})

    return EventSourceResponse(
        generate_analysis_events(recording_id),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Recording-ID": recording_id,
        },
    )


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
    """Generate log stream for a recording's processing based on actual analysis status.

    Args:
        recording_id: Recording to get logs for
        follow: Whether to follow new logs

    Yields:
        Log lines from actual analysis progress
    """
    try:
        recording_uuid = UUID(recording_id)
    except ValueError:
        yield f"[ERROR] Invalid recording ID format: {recording_id}\n"
        return

    # Verify recording exists
    recording = await recording_repo.get_by_id(recording_uuid)
    if not recording:
        yield f"[ERROR] Recording not found: {recording_id}\n"
        return

    yield f"[INFO] Starting log streaming for recording {recording_id}\n"
    yield f"[INFO] File: {recording.file_path}\n"
    yield f"[INFO] Status: {recording.processing_status}\n"

    # Get initial analysis results
    analysis_results = await analysis_result_repo.get_by_recording_id(recording_uuid)

    # Show historical analysis results
    if analysis_results:

        def get_result_created_at(result: AnalysisResult) -> datetime:
            return cast("datetime", result.created_at)

        for result in sorted(analysis_results, key=get_result_created_at):
            created_time = result.created_at.strftime("%H:%M:%S") if result.created_at else "unknown"
            status_display = result.status.upper() if result.status else "UNKNOWN"
            yield f"[{created_time}] [{status_display}] {result.analysis_type} analysis"
            if result.status == "completed":
                if result.result_data and result.confidence_score:
                    yield f" - Result: {result.result_data} (confidence: {result.confidence_score:.2f})"
                elif result.processing_time_ms:
                    yield f" - Processing time: {result.processing_time_ms}ms"
            elif result.status == "failed" and result.error_message:
                yield f" - Error: {result.error_message}"
            yield "\n"

    if not follow:
        yield "[INFO] Log streaming complete (static mode)\n"
        return

    # Follow mode: monitor for changes
    yield "[INFO] Following analysis progress...\n"

    last_seen_results = len(analysis_results)
    last_status = recording.processing_status
    poll_count = 0
    max_polls = 180  # 3 minutes maximum

    while poll_count < max_polls:
        await asyncio.sleep(1.0)
        poll_count += 1

        # Check for recording status changes
        current_recording = await recording_repo.get_by_id(recording_uuid)
        if not current_recording:
            yield "[ERROR] Recording disappeared during monitoring\n"
            break

        # Log status changes
        if current_recording.processing_status != last_status:
            yield (
                f"[{poll_count:03d}s] [STATUS] Changed from '{last_status}' "
                f"to '{current_recording.processing_status}'\n"
            )
            last_status = current_recording.processing_status

            # Check for completion or failure
            if current_recording.processing_status == "completed":
                yield f"[{poll_count:03d}s] [SUCCESS] Analysis completed successfully\n"
                break
            elif current_recording.processing_status == "failed":
                error_msg = current_recording.processing_error or "Unknown error"
                yield f"[{poll_count:03d}s] [FAILED] Analysis failed: {error_msg}\n"
                break

        # Check for new analysis results
        current_results = await analysis_result_repo.get_by_recording_id(recording_uuid)
        if len(current_results) > last_seen_results:
            # New results found
            def get_current_result_created_at(result: AnalysisResult) -> datetime:
                return cast("datetime", result.created_at)

            new_results = sorted(current_results, key=get_current_result_created_at)[last_seen_results:]
            for result in new_results:
                yield f"[{poll_count:03d}s] [NEW] {result.analysis_type} analysis: {result.status}\n"
                if result.status == "completed" and result.processing_time_ms:
                    yield f"[{poll_count:03d}s] [PERF] Processing time: {result.processing_time_ms}ms\n"
            last_seen_results = len(current_results)

        # Periodic heartbeat
        if poll_count % 10 == 0:
            yield f"[{poll_count:03d}s] [HEARTBEAT] Monitoring active (status: {last_status})\n"

    if poll_count >= max_polls:
        yield f"[{poll_count:03d}s] [TIMEOUT] Monitoring timeout reached\n"


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
