"""
CUE Generation Service - High-level orchestration for CUE file generation.

This service manages the complete CUE file generation workflow including:
- Tracklist to CUE conversion
- File storage and versioning
- Validation and quality checks
- Background job processing
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import UUID, uuid4

from ..models.cue_file import (
    CueFile,
    CueGenerationJob,
    CueFormat,
    CueGenerationStatus,
    ValidationResult,
    GenerateCueRequest,
    CueGenerationResponse,
    BatchGenerateCueRequest,
    BatchCueGenerationResponse,
)
from ..models.tracklist import Tracklist
from .cue_integration import CueIntegrationService
from .storage_service import StorageService


class CueGenerationService:
    """High-level service for orchestrating CUE file generation workflows."""

    def __init__(self, storage_service: StorageService):
        self.cue_integration = CueIntegrationService()
        self.storage_service = storage_service

    async def generate_cue_file(
        self, tracklist: Tracklist, generation_request: GenerateCueRequest
    ) -> CueGenerationResponse:
        """
        Generate a CUE file from tracklist data.

        Args:
            tracklist: Source tracklist
            generation_request: Generation configuration

        Returns:
            Generation response with file data or error details
        """
        try:
            # Create generation job for tracking
            audio_filename = (
                Path(generation_request.audio_file_path).name if generation_request.audio_file_path else "audio.wav"
            )
            job = CueGenerationJob(
                id=uuid4(),
                tracklist_id=tracklist.id,
                format=generation_request.format,
                options=generation_request.options or {},
                status=CueGenerationStatus.PENDING,
                started_at=None,
                completed_at=None,
                cue_file_id=None,
                error_message=None,
                validation_report=None,
            )

            # Generate CUE content
            success, content, error_msg = self.cue_integration.generate_cue_content(
                tracklist=tracklist,
                cue_format=generation_request.format,
                audio_filename=audio_filename,
                options=job.options,
            )

            if not success:
                job.status = CueGenerationStatus.FAILED
                job.error_message = error_msg
                return CueGenerationResponse(
                    success=False,
                    job_id=job.id,
                    cue_file_id=None,
                    file_path=None,
                    validation_report=None,
                    error=error_msg,
                    processing_time_ms=None,
                )

            # Validate generated content if requested
            validation_result = None
            if generation_request.validate_audio:
                validation_result = self.cue_integration.validate_cue_content(
                    content=content, cue_format=generation_request.format
                )

                if not validation_result.valid:
                    job.status = CueGenerationStatus.FAILED
                    job.error_message = f"Validation failed: {validation_result.error}"
                    return CueGenerationResponse(
                        success=False,
                        job_id=job.id,
                        cue_file_id=None,
                        file_path=None,
                        validation_report=validation_result,
                        error=job.error_message,
                        processing_time_ms=None,
                    )

            # Store CUE file (always store for now)
            storage_result = self.storage_service.store_cue_file(
                content=content,
                audio_file_id=tracklist.audio_file_id,
                cue_format=generation_request.format.value,
                metadata={
                    "tracklist_id": str(tracklist.id),
                    "generation_options": job.options,
                    "audio_filename": audio_filename,
                },
            )

            if not storage_result.success:
                job.status = CueGenerationStatus.FAILED
                job.error_message = f"Storage failed: {storage_result.error}"
                return CueGenerationResponse(
                    success=False,
                    job_id=job.id,
                    cue_file_id=None,
                    file_path=None,
                    validation_report=None,
                    error=job.error_message,
                    processing_time_ms=None,
                )

            # Create CUE file record
            import hashlib

            content_bytes = content.encode("utf-8")
            cue_file = CueFile(
                id=uuid4(),
                tracklist_id=tracklist.id,
                file_path=storage_result.file_path or "",
                format=generation_request.format,
                file_size=len(content_bytes),
                checksum=hashlib.sha256(content_bytes).hexdigest(),
                metadata=job.options,
            )

            # Update job status
            job.status = CueGenerationStatus.COMPLETED
            job.completed_at = datetime.now(timezone.utc)
            job.cue_file_id = cue_file.id

            return CueGenerationResponse(
                success=True,
                job_id=job.id,
                cue_file_id=cue_file.id,
                file_path=storage_result.file_path,
                validation_report=validation_result,
                error=None,
                processing_time_ms=None,
            )

        except Exception as e:
            return CueGenerationResponse(
                success=False,
                job_id=job.id if "job" in locals() else uuid4(),
                cue_file_id=None,
                file_path=None,
                validation_report=None,
                error=f"Generation failed: {str(e)}",
                processing_time_ms=None,
            )

    async def generate_multiple_formats(
        self, tracklist: Tracklist, bulk_request: BatchGenerateCueRequest
    ) -> BatchCueGenerationResponse:
        """
        Generate CUE files in multiple formats from a single tracklist.

        Args:
            tracklist: Source tracklist
            bulk_request: Bulk generation configuration

        Returns:
            Bulk generation response with results for each format
        """
        results = []

        # Process each format request
        for cue_format in bulk_request.formats:
            generation_request = GenerateCueRequest(
                format=cue_format,
                options=bulk_request.options,
                validate_audio=bulk_request.validate_audio,
                audio_file_path=bulk_request.audio_file_path,
            )

            # Generate for this format
            result = await self.generate_cue_file(tracklist, generation_request)
            results.append(result)

        # Calculate summary statistics
        successful_count = sum(1 for r in results if r.success)
        failed_count = len(results) - successful_count

        return BatchCueGenerationResponse(
            success=failed_count == 0,
            results=results,
            total_files=len(bulk_request.formats),
            successful_files=successful_count,
            failed_files=failed_count,
            processing_time_ms=None,
        )

    async def regenerate_cue_file(
        self, cue_file_id: UUID, tracklist: Tracklist, new_options: Optional[Dict] = None
    ) -> CueGenerationResponse:
        """
        Regenerate an existing CUE file with updated options.

        Args:
            cue_file_id: ID of CUE file to regenerate
            tracklist: Source tracklist (in case it was updated)
            new_options: Updated generation options

        Returns:
            Generation response for the updated file
        """
        # TODO: Implement when we have CUE file repository
        # For now, this is a placeholder for the API
        raise NotImplementedError("Regeneration requires CUE file repository")

    def get_format_capabilities(self, cue_format: CueFormat) -> Dict[str, Any]:
        """
        Get capabilities and limitations for a specific CUE format.

        Args:
            cue_format: CUE format to query

        Returns:
            Dictionary of format capabilities
        """
        return self.cue_integration.get_format_capabilities(cue_format)

    def get_conversion_preview(self, source_format: CueFormat, target_format: CueFormat) -> List[str]:
        """
        Preview potential data loss for format conversion.

        Args:
            source_format: Source format
            target_format: Target format

        Returns:
            List of warnings about potential data loss
        """
        return self.cue_integration.get_conversion_warnings(source_format, target_format)

    def get_supported_formats(self) -> List[CueFormat]:
        """Get list of all supported CUE formats."""
        return self.cue_integration.get_supported_formats()

    async def validate_tracklist_for_cue(self, tracklist: Tracklist, cue_format: CueFormat) -> ValidationResult:
        """
        Validate that a tracklist is suitable for CUE generation.

        Args:
            tracklist: Tracklist to validate
            cue_format: Target CUE format

        Returns:
            Validation result with warnings and recommendations
        """
        warnings = []
        metadata: Dict[str, Any] = {}

        # Check track count limits
        format_caps = self.get_format_capabilities(cue_format)
        max_tracks = format_caps.get("max_tracks", 99)

        if len(tracklist.tracks) > max_tracks:
            warnings.append(
                f"Track count ({len(tracklist.tracks)}) exceeds {cue_format.value} format limit ({max_tracks})"
            )

        # Check for missing required fields
        missing_fields = []
        for track in tracklist.tracks:
            if not track.artist:
                missing_fields.append(f"Track {track.position}: missing artist")
            if not track.title:
                missing_fields.append(f"Track {track.position}: missing title")
            if not track.start_time:
                missing_fields.append(f"Track {track.position}: missing start time")

        if missing_fields:
            warnings.extend(missing_fields)

        # Check for potential timing issues
        for i, track in enumerate(tracklist.tracks[:-1]):
            next_track = tracklist.tracks[i + 1]
            if track.end_time and track.end_time > next_track.start_time:
                warnings.append(f"Track {track.position} end time overlaps with track {next_track.position} start time")

        metadata["track_count"] = len(tracklist.tracks)
        metadata["format_capabilities"] = format_caps

        return ValidationResult(
            valid=len(missing_fields) == 0,  # Only invalid if required fields are missing
            error=None,
            warnings=warnings,
            audio_duration=None,
            tracklist_duration=None,
            metadata=metadata,
        )
