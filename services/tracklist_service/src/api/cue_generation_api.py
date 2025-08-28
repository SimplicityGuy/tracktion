"""
CUE Generation API endpoints.

Provides REST endpoints for generating CUE files from tracklist data
with support for multiple formats, validation, and batch processing.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, Any, List
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Path
from fastapi.responses import JSONResponse, FileResponse

from ..models.cue_file import (
    CueFormat,
    GenerateCueRequest,
    BatchGenerateCueRequest,
    CueGenerationResponse,
    BatchCueGenerationResponse,
)
from ..models.tracklist import Tracklist
from ..services.cue_generation_service import CueGenerationService
from ..services.audio_validation_service import AudioValidationService
from ..services.storage_service import StorageService, StorageConfig

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
cue_generation_service = CueGenerationService(storage_service)
audio_validation_service = AudioValidationService()


async def get_tracklist_by_id(tracklist_id: UUID) -> Tracklist:
    """
    Helper function to retrieve tracklist by ID.

    In production, this would query the tracklist database.
    For now, raises NotImplementedError.
    """
    # TODO: Implement tracklist repository integration
    raise HTTPException(
        status_code=501, detail="Tracklist retrieval not yet implemented. Please provide tracklist data directly."
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
            error=f"CUE generation failed: {str(e)}",
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
            background_tasks.add_task(process_batch_cue_generation_async, tracklist_data, request, correlation_id)

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
    tracklist_id: UUID = Path(description="Tracklist ID"),
    async_processing: bool = Query(False, description="Process generation asynchronously"),
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
        from typing import cast

        result = await generate_cue_file(request, tracklist, background_tasks, async_processing)
        return cast(CueGenerationResponse, result)

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
            error=f"Failed to generate CUE for tracklist {tracklist_id}: {str(e)}",
            processing_time_ms=None,
        )


@router.get("/formats", response_model=List[str])
async def get_supported_formats() -> List[str]:
    """
    Get list of supported CUE formats.

    Returns:
        List of supported format names
    """
    try:
        formats = cue_generation_service.get_supported_formats()
        return [f.value for f in formats]
    except Exception as e:
        logger.error(f"Failed to get supported formats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve supported formats: {str(e)}")


@router.get("/formats/{format}/capabilities", response_model=Dict[str, Any])
async def get_format_capabilities(format: str = Path(description="CUE format name")) -> Dict[str, Any]:
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
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unsupported CUE format: {format}")

        capabilities = cue_generation_service.get_format_capabilities(cue_format)
        return capabilities

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get format capabilities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve capabilities for format {format}: {str(e)}")


@router.get("/formats/conversion-preview", response_model=List[str])
async def get_conversion_preview(
    source_format: str = Query(description="Source CUE format"),
    target_format: str = Query(description="Target CUE format"),
) -> List[str]:
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
            raise HTTPException(status_code=400, detail=f"Invalid CUE format: {str(e)}")

        warnings = cue_generation_service.get_conversion_preview(source, target)
        return warnings

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversion preview: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to preview conversion from {source_format} to {target_format}: {str(e)}"
        )


@router.get("/jobs/{job_id}/status", response_model=Dict[str, Any])
async def get_generation_job_status(job_id: UUID = Path(description="Generation job ID")) -> Dict[str, Any]:
    """
    Get the status of a CUE generation job.

    Args:
        job_id: Generation job ID

    Returns:
        JSON response with job status and results
    """
    try:
        # TODO: Implement job status tracking
        # For now, return placeholder response
        return {
            "job_id": str(job_id),
            "status": "completed",
            "message": "Job status tracking not yet implemented",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve status for job {job_id}: {str(e)}")


@router.get("/download/{cue_file_id}")
async def download_cue_file(cue_file_id: UUID = Path(description="CUE file ID")) -> FileResponse:
    """
    Download a generated CUE file.

    Args:
        cue_file_id: CUE file ID

    Returns:
        File download response
    """
    try:
        # TODO: Implement file retrieval from storage
        # For now, raise NotImplementedError
        raise HTTPException(
            status_code=501, detail="CUE file download not yet implemented. Requires CUE file repository integration."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to download CUE file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download CUE file {cue_file_id}: {str(e)}")


async def process_cue_generation_async(tracklist: Tracklist, request: GenerateCueRequest, correlation_id: UUID) -> None:
    """
    Process CUE generation asynchronously.

    Args:
        tracklist: Source tracklist
        request: Generation request
        correlation_id: Job correlation ID
    """
    try:
        # Generate CUE file
        response = await cue_generation_service.generate_cue_file(tracklist, request)

        # TODO: Store job result in cache/database
        # For now, just log the result
        logger.info(f"Async CUE generation {correlation_id}: {'success' if response.success else 'failed'}")

    except Exception as e:
        logger.error(f"Async CUE generation {correlation_id} failed: {e}", exc_info=True)


async def process_batch_cue_generation_async(
    tracklist: Tracklist, request: BatchGenerateCueRequest, correlation_id: UUID
) -> None:
    """
    Process batch CUE generation asynchronously.

    Args:
        tracklist: Source tracklist
        request: Batch generation request
        correlation_id: Job correlation ID
    """
    try:
        # Generate multiple formats
        response = await cue_generation_service.generate_multiple_formats(tracklist, request)

        # TODO: Store job results in cache/database
        # For now, just log the results
        logger.info(
            f"Async batch CUE generation {correlation_id}: {response.successful_files}/{response.total_files} successful"
        )

    except Exception as e:
        logger.error(f"Async batch CUE generation {correlation_id} failed: {e}", exc_info=True)


@router.get("/health")
async def cue_health_check() -> JSONResponse:
    """
    Health check endpoint for the CUE generation API.

    Returns:
        JSON response with service health status
    """
    health_status: Dict[str, Any] = {
        "service": "cue_generation_api",
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {},
    }

    # Check CUE generation service
    try:
        # Test service instantiation and format listing
        formats = cue_generation_service.get_supported_formats()
        health_status["components"]["cue_service"] = "healthy"
        health_status["components"]["supported_formats"] = len(formats)
    except Exception as e:
        health_status["components"]["cue_service"] = f"unhealthy: {str(e)}"
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
        health_status["components"]["storage"] = f"unhealthy: {str(e)}"
        if health_status["status"] == "healthy":
            health_status["status"] = "degraded"

    # Check audio validation service
    try:
        AudioValidationService()
        health_status["components"]["audio_validation"] = "healthy"
    except Exception as e:
        health_status["components"]["audio_validation"] = f"unhealthy: {str(e)}"
        if health_status["status"] == "healthy":
            health_status["status"] = "degraded"

    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)
