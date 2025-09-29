"""
CUE Generation API endpoints.

Provides REST endpoints for generating CUE files from tracklist data
with support for multiple formats, validation, and batch processing.
"""

import hashlib
import logging
import tempfile
import time
from collections.abc import Generator
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from services.tracklist_service.src.database import get_db_session as db_session_getter
from services.tracklist_service.src.messaging.message_schemas import (
    BatchCueGenerationMessage,
    CueGenerationMessage,
    MessageType,
)
from services.tracklist_service.src.messaging.rabbitmq_client import get_rabbitmq_client
from services.tracklist_service.src.models.cue_file import (
    BatchCueGenerationResponse,
    BatchGenerateCueRequest,
    CueFileDB,
    CueFormat,
    CueGenerationResponse,
    GenerateCueRequest,
)
from services.tracklist_service.src.models.tracklist import TrackEntry, Tracklist
from services.tracklist_service.src.repository.cue_file_repository import CueFileRepository
from services.tracklist_service.src.services.audio_validation_service import AudioValidationService
from services.tracklist_service.src.services.cache_service import CacheConfig, CacheService
from services.tracklist_service.src.services.cue_generation_service import CueGenerationService
from services.tracklist_service.src.services.storage_service import StorageConfig, StorageService
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from shared.core_types.src.async_database import AsyncDatabaseManager
from shared.core_types.src.async_repositories import AsyncTracklistRepository
from shared.core_types.src.database import DatabaseManager
from shared.core_types.src.models import Tracklist as TracklistModel
from shared.core_types.src.repositories import JobRepository

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/cue", tags=["cue"])

# Initialize services
storage_config = StorageConfig(
    primary="filesystem",
    filesystem={
        "base_path": "/tmp/cue_files/",  # Use tmp for testing
        "structure": "{year}/{month}/{audio_file_id}/{format}.cue",
        "permissions": "644",
    },
    backup=True,
    max_versions=5,
)
storage_service = StorageService(storage_config)

# Initialize cache service
cache_config = CacheConfig(
    redis_host="localhost",
    redis_port=6379,
    default_ttl=3600,
    cue_content_ttl=7200,
    enable_compression=True,
    cache_warming_enabled=True,
    popular_formats=["standard", "cdj", "traktor"],
)
cache_service = CacheService(cache_config)

# Initialize database managers - async for repositories, sync for services
async_db_manager = AsyncDatabaseManager()
sync_db_manager = DatabaseManager()

# Initialize CUE generation service with sync database manager
cue_generation_service = CueGenerationService(storage_service, cache_service, sync_db_manager)

# Initialize tracklist repository with async database manager
tracklist_repo = AsyncTracklistRepository(async_db_manager)
audio_validation_service = AudioValidationService()

# Initialize job repository for API endpoints with sync database manager
job_repo = JobRepository(sync_db_manager)


# Database dependency using proper database connection
def get_db_session() -> Generator[Session]:
    """Get database session for dependency injection."""
    # Using sync generator as FastAPI handles it properly
    yield from db_session_getter()


# Repository dependency
async def get_cue_file_repository(
    session: AsyncSession,  # Will be injected via Depends in route
) -> CueFileRepository:
    """Get CUE file repository."""
    return CueFileRepository(session)


async def get_tracklist_by_id(tracklist_id: UUID) -> Tracklist:
    """
    Helper function to retrieve tracklist by ID.
    """
    # First, try to get by exact UUID
    async with async_db_manager.get_db_session() as session:
        result = await session.execute(select(TracklistModel).where(TracklistModel.id == tracklist_id))
        tracklist_db = result.scalar_one_or_none()

        if not tracklist_db:
            raise HTTPException(
                status_code=404,
                detail=f"Tracklist {tracklist_id} not found",
            )

        # Convert DB model to API Tracklist model
        tracks_data = tracklist_db.tracks if isinstance(tracklist_db.tracks, list) else []
        tracks_list = [TrackEntry.from_dict(t) for t in tracks_data] if tracks_data else []

        return Tracklist(
            id=tracklist_db.id,
            audio_file_id=tracklist_db.audio_file_id,
            source=tracklist_db.source,
            created_at=tracklist_db.created_at,
            updated_at=tracklist_db.updated_at,
            tracks=tracks_list,
            cue_file_id=tracklist_db.cue_file_id,
            confidence_score=tracklist_db.confidence_score or 1.0,
            draft_version=tracklist_db.draft_version,
            is_draft=tracklist_db.is_draft or False,
            parent_tracklist_id=tracklist_db.parent_tracklist_id,
            default_cue_format=tracklist_db.default_cue_format,
        )


@router.post("/generate", response_model=CueGenerationResponse)
async def generate_cue_file(
    request: GenerateCueRequest,
    tracklist_data: Tracklist,
    background_tasks: BackgroundTasks,
    async_processing: bool = Query(False, description="Process generation asynchronously"),
) -> CueGenerationResponse:
    """
    Generate a CUE file from tracklist data.

    Args:
        request: CUE generation configuration
        tracklist_data: Source tracklist
        background_tasks: FastAPI background tasks
        async_processing: Whether to process asynchronously

    Returns:
        CueGenerationResponse with generation results or job ID
    """
    start_time = time.time()
    correlation_id = uuid4()

    try:
        # Validate tracklist for CUE generation
        validation_result = await cue_generation_service.validate_tracklist_for_cue(tracklist_data, request.format)

        if not validation_result.valid:
            return CueGenerationResponse(
                success=False,
                job_id=correlation_id,
                cue_file_id=None,
                file_path=None,
                validation_report=validation_result,
                error=f"Tracklist validation failed: {validation_result.error}",
                processing_time_ms=None,
            )

        # Handle async processing
        if async_processing:
            background_tasks.add_task(process_cue_generation_async, tracklist_data, request, correlation_id)

            return CueGenerationResponse(
                success=True,
                job_id=correlation_id,
                cue_file_id=None,  # Will be available when job completes
                file_path=None,
                validation_report=validation_result,
                error=None,
                processing_time_ms=None,
            )

        # Process synchronously
        response = await cue_generation_service.generate_cue_file(tracklist_data, request)
        response.processing_time_ms = int((time.time() - start_time) * 1000)

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CUE generation failed: {e}", exc_info=True)

        processing_time = int((time.time() - start_time) * 1000)
        return CueGenerationResponse(
            success=False,
            job_id=correlation_id,
            cue_file_id=None,
            file_path=None,
            validation_report=None,
            error=f"CUE generation failed: {e!s}",
            processing_time_ms=processing_time,
        )


@router.post("/generate/batch", response_model=BatchCueGenerationResponse)
async def generate_multiple_cue_files(
    request: BatchGenerateCueRequest,
    tracklist_data: Tracklist,
    background_tasks: BackgroundTasks,
    async_processing: bool = Query(False, description="Process generation asynchronously"),
) -> BatchCueGenerationResponse:
    """
    Generate CUE files in multiple formats from a single tracklist.

    Args:
        request: Batch generation configuration
        tracklist_data: Source tracklist
        background_tasks: FastAPI background tasks
        async_processing: Whether to process asynchronously

    Returns:
        BatchCueGenerationResponse with results for each format
    """
    start_time = time.time()

    try:
        # Handle async processing
        if async_processing:
            correlation_id = uuid4()
            background_tasks.add_task(
                process_batch_cue_generation_async,
                tracklist_data,
                request,
                correlation_id,
            )

            return BatchCueGenerationResponse(
                success=True,
                results=[],
                total_files=len(request.formats),
                successful_files=0,
                failed_files=0,
                processing_time_ms=int((time.time() - start_time) * 1000),
            )

        # Process synchronously
        response = await cue_generation_service.generate_multiple_formats(tracklist_data, request)
        response.processing_time_ms = int((time.time() - start_time) * 1000)

        return response

    except Exception as e:
        logger.error(f"Batch CUE generation failed: {e}", exc_info=True)

        processing_time = int((time.time() - start_time) * 1000)
        return BatchCueGenerationResponse(
            success=False,
            results=[],
            total_files=len(request.formats),
            successful_files=0,
            failed_files=len(request.formats),
            processing_time_ms=processing_time,
        )


@router.post("/generate/{tracklist_id}", response_model=CueGenerationResponse)
async def generate_cue_for_tracklist(
    request: GenerateCueRequest,
    background_tasks: BackgroundTasks,
    tracklist_id: UUID,  # Path parameter automatically parsed
    async_processing: bool = False,  # Query parameter with default
) -> CueGenerationResponse:
    """
    Generate a CUE file for an existing tracklist by ID.

    Args:
        tracklist_id: Target tracklist ID
        request: CUE generation configuration
        background_tasks: FastAPI background tasks
        async_processing: Whether to process asynchronously

    Returns:
        CueGenerationResponse with generation results
    """
    try:
        # Retrieve tracklist from database
        tracklist = await get_tracklist_by_id(tracklist_id)

        # Generate CUE file
        result: CueGenerationResponse = await generate_cue_file(request, tracklist, background_tasks, async_processing)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CUE generation by ID failed: {e}", exc_info=True)
        return CueGenerationResponse(
            success=False,
            job_id=uuid4(),
            cue_file_id=None,
            file_path=None,
            validation_report=None,
            error=f"Failed to generate CUE for tracklist {tracklist_id}: {e!s}",
            processing_time_ms=None,
        )


@router.get("/formats", response_model=list[str])
async def get_supported_formats() -> list[str]:
    """
    Get list of supported CUE formats.

    Returns:
        List of supported format names
    """
    try:
        formats = cue_generation_service.get_supported_formats()
        return [f.value if hasattr(f, "value") else str(f) for f in formats]
    except Exception as e:
        logger.error(f"Failed to get supported formats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve supported formats: {e!s}") from e


@router.get("/formats/{format}/capabilities", response_model=dict[str, Any])
async def get_format_capabilities(
    format: str,  # Path parameter automatically parsed
) -> dict[str, Any]:
    """
    Get capabilities and limitations for a specific CUE format.

    Args:
        format: CUE format name

    Returns:
        Dictionary of format capabilities
    """
    try:
        # Validate format
        try:
            cue_format = CueFormat(format)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Unsupported CUE format: {format}") from e

        return cue_generation_service.get_format_capabilities(cue_format)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get format capabilities: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve capabilities for format {format}: {e!s}",
        ) from e


@router.get("/formats/conversion-preview", response_model=list[str])
async def get_conversion_preview(
    source_format: str,
    target_format: str,
) -> list[str]:
    """
    Preview potential data loss for format conversion.

    Args:
        source_format: Source format name
        target_format: Target format name

    Returns:
        List of warnings about potential data loss
    """
    try:
        # Validate formats
        try:
            source = CueFormat(source_format)
            target = CueFormat(target_format)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid CUE format: {e!s}") from e

        return cue_generation_service.get_conversion_preview(source, target)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversion preview: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to preview conversion from {source_format} to {target_format}: {e!s}",
        ) from e


@router.get("/jobs/{job_id}/status", response_model=dict[str, Any])
async def get_generation_job_status(
    job_id: UUID,  # Path parameter automatically parsed
) -> dict[str, Any]:
    """
    Get the status of a CUE generation job.

    Args:
        job_id: Generation job ID

    Returns:
        JSON response with job status and results
    """
    try:
        # Check if job repository is available
        if not job_repo:
            return {
                "job_id": str(job_id),
                "status": "unknown",
                "message": "Job tracking not configured",
                "timestamp": datetime.now(UTC).isoformat(),
            }

        # Get job from repository
        job = job_repo.get_by_id(job_id)
        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found",
            )

        # Return job status
        return {
            "job_id": str(job.id),
            "status": job.status,
            "job_type": job.job_type,
            "service_name": job.service_name,
            "progress": job.progress,
            "total_items": job.total_items,
            "result": job.result,
            "error_message": job.error_message,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve status for job {job_id}: {e!s}",
        ) from e


@router.get("/download/{cue_file_id}")
async def download_cue_file(
    cue_file_id: UUID,  # Path parameter automatically parsed
) -> FileResponse:
    """
    Download a generated CUE file.

    Args:
        cue_file_id: CUE file ID

    Returns:
        File download response
    """
    try:
        # Get CUE file from database first
        async with async_db_manager.get_db_session() as session:
            cue_file_repo = CueFileRepository(session)
            cue_file = await cue_file_repo.get_cue_file_by_id(cue_file_id)
            if not cue_file:
                raise HTTPException(status_code=404, detail=f"CUE file {cue_file_id} not found")

            # Retrieve file content from storage
            success, content, error = storage_service.retrieve_cue_file(str(cue_file.file_path))
            if not success:
                logger.error(f"Failed to retrieve CUE file content: {error}")
                raise HTTPException(status_code=500, detail=f"Failed to retrieve file content: {error}")

            # Create temporary file for download
            with tempfile.NamedTemporaryFile(mode="w", suffix=".cue", delete=False) as tmp_file:
                if content:
                    tmp_file.write(content)
                tmp_path = tmp_file.name

            # Generate appropriate filename
            filename = f"{cue_file.format}_{str(cue_file_id)[:8]}_v{cue_file.version}.cue"

            return FileResponse(
                path=tmp_path,
                filename=filename,
                media_type="application/x-cue",
                headers={
                    "Content-Disposition": f"attachment; filename={filename}",
                    "Cache-Control": "no-cache",
                    "X-File-Version": str(cue_file.version),
                    "X-File-Format": str(cue_file.format),
                },
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download CUE file: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download CUE file {cue_file_id}: {e!s}",
        ) from e


async def process_cue_generation_async(tracklist: Tracklist, request: GenerateCueRequest, correlation_id: UUID) -> None:
    """
    Process CUE generation asynchronously via RabbitMQ.

    Args:
        tracklist: Source tracklist
        request: Generation request
        correlation_id: Job correlation ID
    """
    try:
        # Create RabbitMQ message
        job_id = uuid4()
        message = CueGenerationMessage(
            message_id=uuid4(),
            message_type=MessageType.CUE_GENERATION,
            correlation_id=correlation_id,
            retry_count=0,
            priority=7,  # High priority for API requests
            tracklist_id=tracklist.id,
            format=request.format.value,
            options=request.options,
            validate_audio=request.validate_audio,
            audio_file_path=request.audio_file_path,
            job_id=job_id,
            requested_by="api_user",
        )

        # Publish to RabbitMQ
        try:
            rabbitmq_client = get_rabbitmq_client()
            success = await rabbitmq_client.publish_message(message)
            if success:
                logger.info(f"Published async CUE generation request {job_id} to RabbitMQ")
            else:
                logger.error(f"Failed to publish CUE generation request {job_id} to RabbitMQ")
        except RuntimeError:
            # RabbitMQ not available, fall back to direct processing
            logger.warning("RabbitMQ not available, processing CUE generation directly")
            response = await cue_generation_service.generate_cue_file(tracklist, request)
            logger.info(f"Direct CUE generation {correlation_id}: {'success' if response.success else 'failed'}")

    except Exception as e:
        logger.error(f"Async CUE generation {correlation_id} failed: {e}", exc_info=True)


async def process_batch_cue_generation_async(
    tracklist: Tracklist, request: BatchGenerateCueRequest, correlation_id: UUID
) -> None:
    """
    Process batch CUE generation asynchronously via RabbitMQ.

    Args:
        tracklist: Source tracklist
        request: Batch generation request
        correlation_id: Job correlation ID
    """
    try:
        # Create RabbitMQ message
        batch_job_id = uuid4()
        message = BatchCueGenerationMessage(
            message_id=uuid4(),
            message_type=MessageType.BATCH_CUE_GENERATION,
            correlation_id=correlation_id,
            retry_count=0,
            priority=6,  # Slightly lower priority for batch operations
            tracklist_id=tracklist.id,
            formats=[fmt.value for fmt in request.formats],
            options=request.options,
            validate_audio=request.validate_audio,
            audio_file_path=request.audio_file_path,
            batch_job_id=batch_job_id,
            requested_by="api_user",
        )

        # Publish to RabbitMQ
        try:
            rabbitmq_client = get_rabbitmq_client()
            success = await rabbitmq_client.publish_message(message)
            if success:
                logger.info(f"Published async batch CUE generation request {batch_job_id} to RabbitMQ")
            else:
                logger.error(f"Failed to publish batch CUE generation request {batch_job_id} to RabbitMQ")
        except RuntimeError:
            # RabbitMQ not available, fall back to direct processing
            logger.warning("RabbitMQ not available, processing batch CUE generation directly")
            response = await cue_generation_service.generate_multiple_formats(tracklist, request)
            logger.info(
                f"Direct batch CUE generation {correlation_id}: "
                f"{response.successful_files}/{response.total_files} successful"
            )

    except Exception as e:
        logger.error(f"Async batch CUE generation {correlation_id} failed: {e}", exc_info=True)


# CUE File Management Endpoints


@router.get("/files/{cue_file_id}")
async def get_cue_file_info(
    cue_file_id: UUID,  # Path parameter automatically parsed
    repository: CueFileRepository,  # Will be injected via Depends in route
) -> dict[str, Any]:
    """
    Get information about a specific CUE file.

    Args:
        cue_file_id: CUE file ID
        repository: CUE file repository

    Returns:
        CUE file information including metadata and versions
    """
    try:
        # Get CUE file from database
        cue_file = await repository.get_cue_file_by_id(cue_file_id)
        if not cue_file:
            raise HTTPException(status_code=404, detail=f"CUE file {cue_file_id} not found")

        # Get storage information
        file_info = storage_service.get_file_info(str(cue_file.file_path))

        return {
            "id": str(cue_file.id),
            "tracklist_id": str(cue_file.tracklist_id),
            "format": cue_file.format,
            "file_path": cue_file.file_path,
            "file_size": cue_file.file_size,
            "checksum": cue_file.checksum,
            "created_at": cue_file.created_at.isoformat(),
            "updated_at": cue_file.updated_at.isoformat(),
            "version": cue_file.version,
            "is_active": cue_file.is_active,
            "metadata": cue_file.format_metadata or {},
            "storage_info": file_info,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get CUE file info: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve CUE file {cue_file_id}: {e!s}",
        ) from e


@router.get("/files/{cue_file_id}/download")
async def download_cue_file_by_id(
    cue_file_id: UUID,  # Path parameter automatically parsed
    repository: CueFileRepository,  # Will be injected via Depends in route
) -> FileResponse:
    """
    Download a CUE file by its ID.

    Args:
        cue_file_id: CUE file ID
        repository: CUE file repository

    Returns:
        File download response with proper headers
    """
    try:
        # Get CUE file from database
        cue_file = await repository.get_cue_file_by_id(cue_file_id)
        if not cue_file:
            raise HTTPException(status_code=404, detail=f"CUE file {cue_file_id} not found")

        # Retrieve file content from storage
        success, content, error = storage_service.retrieve_cue_file(str(cue_file.file_path))
        if not success:
            logger.error(f"Failed to retrieve CUE file content: {error}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve file content: {error}")

        # Create temporary file for download
        with tempfile.NamedTemporaryFile(mode="w", suffix=".cue", delete=False) as tmp_file:
            if content:
                tmp_file.write(content)
            tmp_path = tmp_file.name

        # Generate appropriate filename
        filename = f"{cue_file.format}_{str(cue_file_id)[:8]}_v{cue_file.version}.cue"

        return FileResponse(
            path=tmp_path,
            filename=filename,
            media_type="application/x-cue",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Cache-Control": "no-cache",
                "X-File-Version": str(cue_file.version),
                "X-File-Format": str(cue_file.format),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download CUE file: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download CUE file {cue_file_id}: {e!s}",
        ) from e


@router.post("/files/{cue_file_id}/regenerate")
async def regenerate_cue_file(
    cue_file_id: UUID,  # Path parameter automatically parsed
    repository: CueFileRepository,  # Will be injected via Depends in route
    options: dict[str, Any] | None = None,
) -> CueGenerationResponse:
    """
    Regenerate a CUE file with updated options.

    Args:
        cue_file_id: CUE file ID to regenerate
        options: New generation options
        repository: CUE file repository

    Returns:
        CueGenerationResponse with regeneration results
    """
    try:
        # Get existing CUE file
        cue_file = await repository.get_cue_file_by_id(cue_file_id)
        if not cue_file:
            return CueGenerationResponse(
                success=False,
                job_id=uuid4(),
                cue_file_id=None,
                file_path=None,
                validation_report=None,
                error=f"CUE file {cue_file_id} not found",
                processing_time_ms=None,
            )

        # Get tracklist to regenerate from
        tracklist = await get_tracklist_by_id(UUID(str(cue_file.tracklist_id)))

        # Use the regeneration service with new options
        response = await cue_generation_service.regenerate_cue_file(cue_file_id, tracklist, options)

        logger.info(f"Regenerated CUE file {cue_file_id}: {'success' if response.success else 'failed'}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to regenerate CUE file: {e}", exc_info=True)
        return CueGenerationResponse(
            success=False,
            job_id=uuid4(),
            cue_file_id=None,
            file_path=None,
            validation_report=None,
            error=f"Regeneration failed: {e!s}",
            processing_time_ms=None,
        )


@router.delete("/files/{cue_file_id}")
async def delete_cue_file(
    cue_file_id: UUID,  # Path parameter automatically parsed
    repository: CueFileRepository,  # Will be injected via Depends in route
    soft_delete: bool = True,
) -> dict[str, Any]:
    """
    Delete a CUE file (soft delete by default).

    Args:
        cue_file_id: CUE file ID to delete
        soft_delete: Whether to soft delete (True) or permanently delete (False)
        repository: CUE file repository

    Returns:
        Deletion status and details
    """
    try:
        # Check if file exists
        cue_file = await repository.get_cue_file_by_id(cue_file_id)
        if not cue_file:
            raise HTTPException(status_code=404, detail=f"CUE file {cue_file_id} not found")

        # Perform deletion based on type
        if soft_delete:
            success = await repository.soft_delete_cue_file(cue_file_id)
        else:
            # Also delete from storage for hard delete
            storage_service.delete_cue_file(str(cue_file.file_path))
            success = await repository.hard_delete_cue_file(cue_file_id)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete CUE file")

        deletion_result = {
            "success": True,
            "cue_file_id": str(cue_file_id),
            "deletion_type": "soft" if soft_delete else "hard",
            "deleted_at": datetime.now(UTC).isoformat(),
            "message": f"CUE file {'soft deleted' if soft_delete else 'permanently deleted'} successfully",
        }

        logger.info(f"{'Soft' if soft_delete else 'Hard'} deleted CUE file {cue_file_id}")
        return deletion_result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete CUE file: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete CUE file {cue_file_id}: {e!s}") from e


@router.get("/files/{cue_file_id}/versions")
async def get_cue_file_versions(
    cue_file_id: UUID,  # Path parameter automatically parsed
    repository: CueFileRepository,  # Will be injected via Depends in route
) -> list[dict[str, Any]]:
    """
    Get all versions of a CUE file.

    Args:
        cue_file_id: CUE file ID
        repository: CUE file repository

    Returns:
        List of file versions with metadata
    """
    try:
        # Get all versions for this CUE file
        versions = await repository.get_file_versions(cue_file_id)

        if not versions:
            raise HTTPException(
                status_code=404,
                detail=f"CUE file {cue_file_id} not found or has no versions",
            )

        # Convert to response format
        version_data = []
        for version in versions:
            # Get storage info if available
            storage_info = storage_service.get_file_info(str(version.file_path))

            version_data.append(
                {
                    "id": str(version.id),
                    "version": version.version,
                    "file_path": version.file_path,
                    "size": version.file_size,
                    "checksum": version.checksum,
                    "created_at": version.created_at.isoformat(),
                    "updated_at": version.updated_at.isoformat(),
                    "is_current": version.is_active,
                    "format": version.format,
                    "metadata": version.format_metadata or {},
                    "storage_info": storage_info,
                }
            )

        logger.debug(f"Retrieved {len(version_data)} versions for CUE file {cue_file_id}")
        return version_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get CUE file versions: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve versions for CUE file {cue_file_id}: {e!s}",
        ) from e


@router.get("/files")
async def list_cue_files(
    repository: CueFileRepository,  # Will be injected via Depends in route
    tracklist_id: UUID | None = None,
    format: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict[str, Any]:
    """
    List CUE files with optional filtering and pagination.

    Args:
        tracklist_id: Optional filter by tracklist ID
        format: Optional filter by CUE format
        limit: Number of files to return (1-100)
        offset: Number of files to skip
        repository: CUE file repository

    Returns:
        List of CUE files with pagination metadata
    """
    try:
        # Validate format if provided
        if format:
            try:
                CueFormat(format)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid CUE format: {format}") from e

        # Get files from repository
        cue_files = await repository.list_cue_files(
            tracklist_id=tracklist_id,
            cue_format=format,
            limit=limit,
            offset=offset,
        )

        # Get total count for pagination
        total_count = await repository.count_cue_files(
            tracklist_id=tracklist_id,
            cue_format=format,
        )

        # Convert to response format
        file_data = []
        for cue_file in cue_files:
            # Get storage info if available
            storage_info = storage_service.get_file_info(str(cue_file.file_path))

            file_data.append(
                {
                    "id": str(cue_file.id),
                    "tracklist_id": str(cue_file.tracklist_id),
                    "format": cue_file.format,
                    "file_path": cue_file.file_path,
                    "file_size": cue_file.file_size,
                    "checksum": cue_file.checksum,
                    "created_at": cue_file.created_at.isoformat(),
                    "updated_at": cue_file.updated_at.isoformat(),
                    "version": cue_file.version,
                    "is_active": cue_file.is_active,
                    "metadata": cue_file.format_metadata or {},
                    "storage_info": storage_info,
                }
            )

        has_more = (offset + len(file_data)) < total_count

        response = {
            "files": file_data,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "total": total_count,
                "has_more": has_more,
                "returned": len(file_data),
            },
            "filters": {
                "tracklist_id": str(tracklist_id) if tracklist_id else None,
                "format": format,
            },
        }

        logger.debug(f"Listed {len(file_data)} CUE files (total: {total_count})")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list CUE files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list CUE files: {e!s}") from e


# Validation Endpoints


@router.post("/files/{cue_file_id}/validate")
async def validate_cue_file(
    cue_file_id: UUID,  # Path parameter automatically parsed
    repository: CueFileRepository,  # Will be injected via Depends in route
    audio_file_path: str | None = None,
    validation_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Validate a CUE file against audio file and format specifications.

    Args:
        cue_file_id: CUE file ID to validate
        audio_file_path: Optional path to audio file for duration validation
        validation_options: Optional validation configuration
        repository: CUE file repository

    Returns:
        Detailed validation report with errors, warnings, and metadata
    """
    try:
        start_time = time.time()

        # Get CUE file from database
        cue_file = await repository.get_cue_file_by_id(cue_file_id)
        if not cue_file:
            raise HTTPException(status_code=404, detail=f"CUE file {cue_file_id} not found")

        # Retrieve CUE file content from storage
        success, content, error = storage_service.retrieve_cue_file(str(cue_file.file_path))
        if not success or content is None:
            logger.error(f"Failed to retrieve CUE file content for validation: {error}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve file content: {error}")

        # Initialize validation report
        validation_report: dict[str, Any] = {
            "cue_file_id": str(cue_file_id),
            "format": cue_file.format,
            "file_path": cue_file.file_path,
            "validation_timestamp": datetime.now(UTC).isoformat(),
            "valid": True,
            "errors": [],
            "warnings": [],
            "metadata": {
                "file_size": cue_file.file_size,
                "checksum": cue_file.checksum,
                "version": cue_file.version,
            },
        }

        # Validate CUE content using CUE integration service
        try:
            if cue_generation_service.cue_integration:
                format_validation = cue_generation_service.cue_integration.validate_cue_content(
                    content, CueFormat(cue_file.format), validation_options or {}
                )
            else:
                # Fallback validation if cue_integration is not available
                format_validation = cue_generation_service.validate_cue_content(
                    content, CueFormat(cue_file.format), validation_options or {}
                )

            validation_report["valid"] = format_validation.valid
            if format_validation.error:
                validation_report["errors"].append(
                    {
                        "type": "format_error",
                        "message": format_validation.error,
                        "severity": "error",
                    }
                )

            validation_report["warnings"].extend(
                [
                    {
                        "type": "format_warning",
                        "message": warning,
                        "severity": "warning",
                    }
                    for warning in format_validation.warnings
                ]
            )

            # Add format-specific metadata
            if format_validation.metadata:
                validation_report["metadata"].update(format_validation.metadata)

        except Exception as e:
            validation_report["valid"] = False
            validation_report["errors"].append(
                {
                    "type": "validation_error",
                    "message": f"CUE content validation failed: {e!s}",
                    "severity": "error",
                }
            )

        # Validate against audio file if provided
        if audio_file_path:
            try:
                # Get tracklist for audio validation
                tracklist = await get_tracklist_by_id(UUID(str(cue_file.tracklist_id)))

                # Validate timing against audio duration (now using real audio duration detection)
                tolerance = 1.0  # Default tolerance
                audio_validation = await audio_validation_service.validate_audio_duration(
                    audio_file_path, tracklist, tolerance
                )

                if not audio_validation.valid:
                    validation_report["valid"] = False
                    validation_report["errors"].append(
                        {
                            "type": "audio_timing_error",
                            "message": audio_validation.error or "Audio timing validation failed",
                            "severity": "error",
                        }
                    )

                validation_report["warnings"].extend(
                    [
                        {
                            "type": "audio_timing_warning",
                            "message": warning,
                            "severity": "warning",
                        }
                        for warning in audio_validation.warnings
                    ]
                )

                # Add audio validation metadata
                validation_report["metadata"]["audio_duration"] = audio_validation.audio_duration
                validation_report["metadata"]["tracklist_duration"] = audio_validation.tracklist_duration

            except Exception as e:
                validation_report["warnings"].append(
                    {
                        "type": "audio_validation_warning",
                        "message": f"Audio validation failed: {e!s}",
                        "severity": "warning",
                    }
                )

        # Add processing time
        processing_time = (time.time() - start_time) * 1000
        validation_report["processing_time_ms"] = round(processing_time, 2)

        # Add actionable recommendations
        recommendations = []
        if not validation_report["valid"]:
            if any(err["type"] == "format_error" for err in validation_report["errors"]):
                recommendations.append("Fix CUE format issues by regenerating the file")
            if any(err["type"] == "audio_timing_error" for err in validation_report["errors"]):
                recommendations.append("Check tracklist timing against actual audio duration")

        if validation_report["warnings"]:
            recommendations.append("Review warnings for potential quality improvements")

        validation_report["recommendations"] = recommendations

        logger.info(
            f"Validated CUE file {cue_file_id}: {'valid' if validation_report['valid'] else 'invalid'} "
            f"({len(validation_report['errors'])} errors, {len(validation_report['warnings'])} warnings)"
        )

        return validation_report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate CUE file: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to validate CUE file {cue_file_id}: {e!s}",
        ) from e


@router.post("/validate")
async def validate_tracklist_for_cue(
    tracklist_id: UUID,
    format: str,
    audio_file_path: str | None = None,
    validation_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Validate a tracklist for CUE file generation in a specific format.

    Args:
        tracklist_id: Tracklist ID to validate
        format: Target CUE format for validation
        audio_file_path: Optional path to audio file for duration validation
        validation_options: Optional validation configuration

    Returns:
        Detailed validation report for tracklist compatibility
    """
    try:
        start_time = time.time()

        # Validate format
        try:
            cue_format = CueFormat(format)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Unsupported CUE format: {format}") from e

        # Get tracklist
        tracklist = await get_tracklist_by_id(tracklist_id)

        # Validate tracklist for CUE generation
        validation_result = await cue_generation_service.validate_tracklist_for_cue(tracklist, cue_format.value)

        # Initialize comprehensive validation report
        validation_report: dict[str, Any] = {
            "tracklist_id": str(tracklist_id),
            "target_format": format,
            "validation_timestamp": datetime.now(UTC).isoformat(),
            "valid": validation_result.valid,
            "errors": [],
            "warnings": [],
            "metadata": validation_result.metadata or {},
        }

        # Convert validation result to detailed report
        if validation_result.error:
            validation_report["errors"].append(
                {
                    "type": "tracklist_error",
                    "message": validation_result.error,
                    "severity": "error",
                }
            )

        validation_report["warnings"].extend(
            [
                {"type": "tracklist_warning", "message": warning, "severity": "warning"}
                for warning in validation_result.warnings
            ]
        )

        # Validate against audio file if provided
        if audio_file_path and validation_result.valid:
            try:
                audio_validation = await audio_validation_service.validate_audio_duration(audio_file_path, tracklist)

                if not audio_validation.valid:
                    validation_report["valid"] = False
                    validation_report["errors"].append(
                        {
                            "type": "audio_timing_error",
                            "message": audio_validation.error or "Audio timing validation failed",
                            "severity": "error",
                        }
                    )

                validation_report["warnings"].extend(
                    [
                        {
                            "type": "audio_timing_warning",
                            "message": warning,
                            "severity": "warning",
                        }
                        for warning in audio_validation.warnings
                    ]
                )

                # Add audio validation metadata
                validation_report["metadata"]["audio_duration"] = audio_validation.audio_duration
                validation_report["metadata"]["tracklist_duration"] = audio_validation.tracklist_duration

            except Exception as e:
                validation_report["warnings"].append(
                    {
                        "type": "audio_validation_warning",
                        "message": f"Audio validation failed: {e!s}",
                        "severity": "warning",
                    }
                )

        # Add processing time
        processing_time = (time.time() - start_time) * 1000
        validation_report["processing_time_ms"] = round(processing_time, 2)

        # Add actionable recommendations
        recommendations = []
        if not validation_report["valid"]:
            if any(err["type"] == "tracklist_error" for err in validation_report["errors"]):
                recommendations.append("Fix tracklist data issues before CUE generation")
            if any(err["type"] == "audio_timing_error" for err in validation_report["errors"]):
                recommendations.append("Verify tracklist timing matches audio file duration")
        else:
            recommendations.append(f"Tracklist is ready for {format} CUE file generation")

        if validation_report["warnings"]:
            recommendations.append("Review warnings for optimal CUE file quality")

        validation_report["recommendations"] = recommendations

        logger.info(
            f"Validated tracklist {tracklist_id} for {format}: "
            f"{'valid' if validation_report['valid'] else 'invalid'} "
            f"({len(validation_report['errors'])} errors, {len(validation_report['warnings'])} warnings)"
        )

        return validation_report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to validate tracklist: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to validate tracklist {tracklist_id}: {e!s}",
        ) from e


# Format Conversion Endpoints


@router.post("/files/{cue_file_id}/convert")
async def convert_cue_file(
    cue_file_id: UUID,  # Path parameter automatically parsed
    repository: CueFileRepository,  # Will be injected via Depends in route
    target_format: str,
    preserve_metadata: bool = True,
    conversion_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Convert a CUE file to a different format.

    Args:
        cue_file_id: Source CUE file ID
        target_format: Target CUE format (standard, cdj, traktor, serato, rekordbox, kodi)
        preserve_metadata: Whether to preserve metadata during conversion
        conversion_options: Optional conversion configuration
        repository: CUE file repository

    Returns:
        Conversion response with new CUE file information and conversion warnings
    """
    try:
        start_time = time.time()

        # Validate target format
        try:
            target_cue_format = CueFormat(target_format)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported target CUE format: {target_format}",
            ) from e

        # Get source CUE file from database
        source_cue_file = await repository.get_cue_file_by_id(cue_file_id)
        if not source_cue_file:
            raise HTTPException(status_code=404, detail=f"CUE file {cue_file_id} not found")

        # Check if conversion is needed
        if source_cue_file.format == target_format:
            return {
                "success": True,
                "message": f"CUE file is already in {target_format} format",
                "cue_file_id": str(cue_file_id),
                "source_format": source_cue_file.format,
                "target_format": target_format,
                "conversion_needed": False,
                "warnings": [],
                "processing_time_ms": 0,
            }

        # Retrieve source CUE file content
        success, content, error = storage_service.retrieve_cue_file(str(source_cue_file.file_path))
        if not success:
            logger.error(f"Failed to retrieve source CUE file content: {error}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve source file: {error}")

        # Get conversion preview warnings
        conversion_warnings = cue_generation_service.get_conversion_preview(
            CueFormat(source_cue_file.format), target_cue_format
        )

        # Initialize conversion report
        conversion_report: dict[str, Any] = {
            "success": True,
            "cue_file_id": None,
            "source_cue_file_id": str(cue_file_id),
            "source_format": source_cue_file.format,
            "target_format": target_format,
            "conversion_needed": True,
            "warnings": [],
            "errors": [],
            "metadata": {
                "preserve_metadata": preserve_metadata,
                "conversion_options": conversion_options or {},
            },
        }

        # Add conversion preview warnings
        conversion_report["warnings"].extend(
            [
                {
                    "type": "conversion_warning",
                    "message": warning,
                    "severity": "warning",
                }
                for warning in conversion_warnings
            ]
        )

        try:
            # Perform conversion using CUE integration service
            if not cue_generation_service.cue_integration:
                raise HTTPException(status_code=503, detail="CUE integration service not available")

            conversion_result = cue_generation_service.cue_integration.convert_cue_format(
                content,
                CueFormat(source_cue_file.format),
                target_cue_format,
                preserve_metadata,
                conversion_options or {},
            )

            if not conversion_result.success:
                conversion_report["success"] = False
                conversion_report["errors"].append(
                    {
                        "type": "conversion_error",
                        "message": conversion_result.error or "Conversion failed",
                        "severity": "error",
                    }
                )

                processing_time = (time.time() - start_time) * 1000
                conversion_report["processing_time_ms"] = round(processing_time, 2)

                return conversion_report

            # Store converted CUE file
            converted_content = conversion_result.converted_content

            # Generate file path for converted file
            file_path = f"cue_files/{source_cue_file.tracklist_id}_{target_format}.cue"
            success, stored_path, error = storage_service.store_cue_file(
                file_path,
                converted_content,
                {
                    "converted_from": source_cue_file.format,
                    "original_cue_file_id": str(cue_file_id),
                    "preserve_metadata": preserve_metadata,
                },
            )

            if not success:
                conversion_report["success"] = False
                conversion_report["errors"].append(
                    {
                        "type": "storage_error",
                        "message": f"Failed to store converted file: {error}",
                        "severity": "error",
                    }
                )

                processing_time = (time.time() - start_time) * 1000
                conversion_report["processing_time_ms"] = round(processing_time, 2)

                return conversion_report

            # Create database record for converted CUE file
            converted_cue_file = CueFileDB(
                tracklist_id=source_cue_file.tracklist_id,
                file_path=stored_path or file_path,
                format=target_format,
                file_size=len(converted_content.encode("utf-8")),
                checksum=hashlib.sha256(converted_content.encode("utf-8")).hexdigest(),
                version=1,
                is_active=True,
                format_metadata={
                    "converted_from": source_cue_file.format,
                    "original_cue_file_id": str(cue_file_id),
                    "preserve_metadata": preserve_metadata,
                    "conversion_timestamp": datetime.now(UTC).isoformat(),
                },
            )

            # Save to database
            created_cue_file = await repository.create_cue_file(converted_cue_file)

            conversion_report["cue_file_id"] = str(created_cue_file.id)
            conversion_report["file_path"] = str(created_cue_file.file_path)
            conversion_report["file_size"] = int(created_cue_file.file_size)
            conversion_report["checksum"] = str(created_cue_file.checksum)
            conversion_report["version"] = int(created_cue_file.version)

            # Add conversion-specific warnings from the result
            if conversion_result.warnings:
                conversion_report["warnings"].extend(
                    [
                        {
                            "type": "format_conversion_warning",
                            "message": warning,
                            "severity": "warning",
                        }
                        for warning in conversion_result.warnings
                    ]
                )

        except Exception as e:
            conversion_report["success"] = False
            conversion_report["errors"].append(
                {
                    "type": "conversion_error",
                    "message": f"Conversion failed: {e!s}",
                    "severity": "error",
                }
            )

        # Add processing time
        processing_time = (time.time() - start_time) * 1000
        conversion_report["processing_time_ms"] = round(processing_time, 2)

        # Add recommendations
        recommendations = []
        if conversion_report["success"]:
            recommendations.append(f"Successfully converted from {source_cue_file.format} to {target_format}")
            if conversion_report["warnings"]:
                recommendations.append("Review warnings for potential data loss during conversion")
        else:
            recommendations.append("Fix conversion errors before retrying")

        conversion_report["recommendations"] = recommendations

        logger.info(
            f"Converted CUE file {cue_file_id} from {source_cue_file.format} to {target_format}: "
            f"{'success' if conversion_report['success'] else 'failed'} "
            f"({len(conversion_report.get('warnings', []))} warnings)"
        )

        return conversion_report

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to convert CUE file: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to convert CUE file {cue_file_id}: {e!s}",
        ) from e


# Cache Management Endpoints


@router.get("/cache/stats")
async def get_cache_stats() -> dict[str, Any]:
    """
    Get cache statistics and performance metrics.

    Returns:
        Cache statistics including hit rates, sizes, and configuration
    """
    try:
        return await cache_service.get_cache_stats()

    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve cache statistics: {e!s}") from e


@router.post("/cache/warm")
async def warm_cache(
    tracklist_ids: list[UUID],
    formats: list[str] | None = None,
) -> dict[str, Any]:
    """
    Warm cache for specified tracklist/format combinations.

    Args:
        tracklist_ids: List of tracklist IDs to pre-generate and cache
        formats: List of CUE formats to cache (defaults to popular formats)

    Returns:
        Cache warming results and statistics
    """
    try:
        # Validate formats if provided
        if formats:
            valid_formats = [
                "standard",
                "cdj",
                "traktor",
                "serato",
                "rekordbox",
                "kodi",
            ]
            for fmt in formats:
                if fmt not in valid_formats:
                    raise HTTPException(status_code=400, detail=f"Invalid format: {fmt}")

        return await cache_service.warm_cache(tracklist_ids, formats)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cache warming failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to warm cache: {e!s}") from e


@router.delete("/cache/invalidate/{tracklist_id}")
async def invalidate_tracklist_cache(
    tracklist_id: UUID,  # Path parameter automatically parsed
) -> dict[str, Any]:
    """
    Invalidate all cached CUE content for a specific tracklist.

    Args:
        tracklist_id: Tracklist ID to invalidate from cache

    Returns:
        Invalidation results
    """
    try:
        invalidated_count = await cue_generation_service.invalidate_tracklist_cache(tracklist_id)

        return {
            "success": True,
            "tracklist_id": str(tracklist_id),
            "invalidated_entries": invalidated_count,
            "message": f"Invalidated {invalidated_count} cache entries for tracklist {tracklist_id}",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Cache invalidation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to invalidate cache for tracklist {tracklist_id}: {e!s}",
        ) from e


@router.delete("/cache/clear")
async def clear_cache(
    pattern: str | None = None,
) -> dict[str, Any]:
    """
    Clear cache entries, optionally matching a specific pattern.

    Args:
        pattern: Optional pattern to match keys (Redis patterns supported)

    Returns:
        Cache clearing results
    """
    try:
        cleared_count = await cache_service.clear_cache(pattern)

        return {
            "success": True,
            "cleared_entries": cleared_count,
            "pattern": pattern or "all",
            "message": f"Cleared {cleared_count} cache entries",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Cache clearing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {e!s}") from e


@router.get("/health")
async def cue_health_check() -> JSONResponse:
    """
    Health check endpoint for the CUE generation API.

    Returns:
        JSON response with service health status
    """
    health_status: dict[str, Any] = {
        "service": "cue_generation_api",
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "components": {},
    }

    # Check CUE generation service
    try:
        # Test service instantiation and format listing
        formats = cue_generation_service.get_supported_formats()
        health_status["components"]["cue_service"] = "healthy"
        health_status["components"]["supported_formats"] = len(formats)
    except Exception as e:
        health_status["components"]["cue_service"] = f"unhealthy: {e!s}"
        health_status["status"] = "unhealthy"

    # Check storage service
    try:
        # Test storage backend availability
        # For filesystem, check if we can create a test file
        test_result = storage_service.primary_backend.store("test", "health_check.cue")
        if test_result.success:
            storage_service.primary_backend.delete("health_check.cue")
        health_status["components"]["storage"] = "healthy"
    except Exception as e:
        health_status["components"]["storage"] = f"unhealthy: {e!s}"
        if health_status["status"] == "healthy":
            health_status["status"] = "degraded"

    # Check audio validation service
    try:
        AudioValidationService()
        health_status["components"]["audio_validation"] = "healthy"
    except Exception as e:
        health_status["components"]["audio_validation"] = f"unhealthy: {e!s}"
        if health_status["status"] == "healthy":
            health_status["status"] = "degraded"

    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)
