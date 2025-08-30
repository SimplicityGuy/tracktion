"""
CUE Generation Service - High-level orchestration for CUE file generation.

This service manages the complete CUE file generation workflow including:
- Tracklist to CUE conversion
- File storage and versioning
- Validation and quality checks
- Background job processing
"""

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

# Import request/response models from models.cue_file to avoid duplication
from ..models.cue_file import (
    GenerateCueRequest as ModelGenerateCueRequest,
    BatchGenerateCueRequest as ModelBatchGenerateCueRequest,
    CueGenerationResponse as ModelCueGenerationResponse,
    BatchCueGenerationResponse as ModelBatchCueGenerationResponse,
    CueFormat as ModelCueFormat,  # Import the Enum from models
)

# Import time utilities
from services.tracklist_service.src.utils.time_utils import parse_cue_time

# Import CUE handler components from Epic 5
try:
    from services.analysis_service.src.cue_handler import (  # type: ignore[import-not-found,attr-defined]
        CDJGenerator,
        CueConverter,
        CueFormat as ImportedCueFormat,
        CueParser,
        CueValidator,
        KodiGenerator,
        RekordboxGenerator,
        SeratoGenerator,
        StandardGenerator,
        TraktorGenerator,
    )

    CUE_HANDLER_AVAILABLE = True
    CueFormat = ImportedCueFormat  # type: ignore[misc,no-redef,assignment]
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("CUE handler not available, using placeholder implementation")
    CUE_HANDLER_AVAILABLE = False
    CueFormat = ModelCueFormat  # type: ignore[misc,assignment]


logger = logging.getLogger(__name__)

# Re-export with original names for backward compatibility
GenerateCueRequest = ModelGenerateCueRequest
BatchGenerateCueRequest = ModelBatchGenerateCueRequest
CueGenerationResponse = ModelCueGenerationResponse
BatchCueGenerationResponse = ModelBatchCueGenerationResponse


class CueGenerationService:
    """Service for generating CUE files from tracklists."""

    def __init__(self, storage_service: Any, cache_service: Any = None):
        """
        Initialize CUE generation service.

        Args:
            storage_service: Storage service for file operations
            cache_service: Optional cache service
        """
        self.storage_service = storage_service
        self.cache_service = cache_service
        self.generators: Dict[str, Any] = {}
        self.validator: Optional[Any] = None
        self.converter: Optional[Any] = None
        self.parser: Optional[Any] = None
        self.cue_integration: Optional[Any] = None  # Placeholder for CUE integration service

        # Initialize format-specific generators if CUE handler is available
        if CUE_HANDLER_AVAILABLE:
            self.generators = {
                "standard": StandardGenerator(),
                "cdj": CDJGenerator(),
                "traktor": TraktorGenerator(),
                "serato": SeratoGenerator(),
                "rekordbox": RekordboxGenerator(),
                "kodi": KodiGenerator(),
            }
            self.validator = CueValidator()
            self.converter = CueConverter()
            self.parser = CueParser()
            # Initialize cue_integration with necessary methods
            self.cue_integration = self  # Use self as integration service

    async def generate_cue_file(self, tracklist: Any, request: GenerateCueRequest) -> CueGenerationResponse:
        """
        Generate a single CUE file from a tracklist.

        Args:
            tracklist: Tracklist data
            request: Generation request parameters

        Returns:
            CUE generation response with file details
        """
        start_time = datetime.now(timezone.utc)

        try:
            if not CUE_HANDLER_AVAILABLE:
                # Fallback implementation
                cue_content = self._generate_placeholder_cue(tracklist, request.format)
            else:
                # Transform tracklist to CUE data structure
                cue_data = self._transform_tracklist_to_cue(tracklist, request.format)

                # Get appropriate generator
                generator = self.generators.get(request.format.lower())
                if not generator:
                    return CueGenerationResponse(
                        success=False,
                        job_id=UUID("00000000-0000-0000-0000-000000000000"),
                        cue_file_id=None,
                        file_path=None,
                        validation_report=None,
                        error=f"Unsupported format: {request.format}",
                        processing_time_ms=0,
                    )

                # Generate CUE content
                cue_content = generator.generate(cue_data, **request.options)

            # Validate if requested
            validation_report = None
            if request.validate_audio and request.audio_file_path:
                validation_report = await self._validate_against_audio(cue_content, request.audio_file_path, tracklist)
                if validation_report and not validation_report.get("valid", False):
                    logger.warning(f"CUE validation failed: {validation_report}")

            # Store file if requested
            cue_file_id = None
            file_path = None
            if request.store_file:
                cue_file_id, file_path = await self._store_cue_file(
                    tracklist.id if hasattr(tracklist, "id") else uuid4(),
                    request.format,
                    cue_content,
                    validation_report,
                )

            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            return CueGenerationResponse(
                success=True,
                job_id=UUID("00000000-0000-0000-0000-000000000000"),  # TODO: implement job tracking
                cue_file_id=cue_file_id,
                file_path=file_path,
                validation_report=validation_report,
                error=None,
                processing_time_ms=processing_time,
            )

        except Exception as e:
            logger.error(f"Failed to generate CUE file: {e}", exc_info=True)
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            return CueGenerationResponse(
                success=False,
                job_id=UUID("00000000-0000-0000-0000-000000000000"),
                cue_file_id=None,
                file_path=None,
                validation_report=None,
                error=str(e),
                processing_time_ms=int(processing_time),
            )

    async def generate_multiple_formats(
        self, tracklist: Any, request: BatchGenerateCueRequest
    ) -> BatchCueGenerationResponse:
        """
        Generate CUE files in multiple formats.

        Args:
            tracklist: Tracklist data
            request: Batch generation request

        Returns:
            Batch generation response with all results
        """
        start_time = datetime.now(timezone.utc)
        results = []
        successful_count = 0
        failed_count = 0

        for format_type in request.formats:
            single_request = GenerateCueRequest(
                format=format_type,
                options=request.options,
                validate_audio=request.validate_audio,
                audio_file_path=request.audio_file_path,
                store_file=request.store_files,
            )

            response = await self.generate_cue_file(tracklist, single_request)
            results.append(response)

            if response.success:
                successful_count += 1
            else:
                failed_count += 1

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        return BatchCueGenerationResponse(
            success=failed_count == 0,
            total_files=len(request.formats),
            successful_files=successful_count,
            failed_files=failed_count,
            results=results,
            processing_time_ms=processing_time,
        )

    def _transform_tracklist_to_cue(self, tracklist: Any, format_type: str) -> Dict[str, Any]:
        """
        Transform tracklist data to CUE data structure.

        Args:
            tracklist: Tracklist model
            format_type: Target CUE format

        Returns:
            CUE data structure for generator
        """
        cue_data: Dict[str, Any] = {
            "title": getattr(tracklist, "title", "Untitled Mix"),
            "performer": getattr(tracklist, "artist", "Unknown Artist"),
            "file": getattr(tracklist, "audio_file_path", "audio.wav"),
            "tracks": [],
            "metadata": {
                "date": tracklist.created_at.strftime("%Y-%m-%d") if hasattr(tracklist, "created_at") else None,
                "genre": getattr(tracklist, "genre", None),
                "source": getattr(tracklist, "source", "tracklist_service"),
                "tracklist_id": str(tracklist.id) if hasattr(tracklist, "id") else None,
            },
        }

        # Transform tracks
        tracks_list: List[Dict[str, Any]] = []
        if hasattr(tracklist, "tracks"):
            for idx, track in enumerate(tracklist.tracks, 1):
                track_data: Dict[str, Any] = {
                    "number": idx,
                    "title": getattr(track, "title", f"Track {idx}"),
                    "performer": getattr(track, "artist", cue_data["performer"]),
                    "start_time": self._format_time(getattr(track, "start_time", "00:00:00")),
                    "end_time": self._format_time(track.end_time)
                    if hasattr(track, "end_time") and track.end_time
                    else None,
                    "metadata": {},
                }

                # Add format-specific metadata
                metadata: Dict[str, Any] = track_data["metadata"]
                if format_type.lower() in ["traktor", "serato", "rekordbox"]:
                    metadata["bpm"] = getattr(track, "bpm", None)
                    metadata["key"] = getattr(track, "key", None)

                if format_type.lower() == "cdj":
                    metadata["hot_cues"] = getattr(track, "hot_cues", [])
                    metadata["memory_cues"] = getattr(track, "memory_cues", [])

                tracks_list.append(track_data)

        cue_data["tracks"] = tracks_list
        return cue_data

    def _format_time(self, time_value: Any) -> str:
        """
        Format time value to CUE time format (MM:SS:FF).

        Args:
            time_value: Time value (datetime, timedelta, or string)

        Returns:
            Formatted time string
        """
        if isinstance(time_value, timedelta):
            total_seconds = int(time_value.total_seconds())
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            frames = 0  # Default to 0 frames
            return f"{minutes:02d}:{seconds:02d}:{frames:02d}"
        elif isinstance(time_value, datetime):
            # Assume it's a time offset from start
            return "00:00:00"
        elif isinstance(time_value, str):
            # Assume it's already formatted
            return time_value
        else:
            return "00:00:00"

    def _generate_placeholder_cue(self, tracklist: Any, format_type: str) -> str:
        """
        Generate placeholder CUE content when CUE handler is not available.

        Args:
            tracklist: Tracklist data
            format_type: Target format

        Returns:
            CUE file content string
        """
        lines = []

        # Add header
        lines.append("REM GENERATED BY Tracklist Service")
        lines.append(f"REM FORMAT {format_type.upper()}")
        lines.append(f"REM DATE {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")

        title = getattr(tracklist, "title", "Untitled Mix")
        performer = getattr(tracklist, "artist", "Unknown Artist")
        lines.append(f'TITLE "{title}"')
        lines.append(f'PERFORMER "{performer}"')

        # Add file reference
        audio_file = getattr(tracklist, "audio_file_path", "audio.wav")
        lines.append(f'FILE "{audio_file}" WAVE')

        # Add tracks
        if hasattr(tracklist, "tracks"):
            for idx, track in enumerate(tracklist.tracks, 1):
                lines.append(f"  TRACK {idx:02d} AUDIO")

                track_title = getattr(track, "title", f"Track {idx}")
                track_artist = getattr(track, "artist", performer)
                lines.append(f'    TITLE "{track_title}"')
                lines.append(f'    PERFORMER "{track_artist}"')

                # Format start time
                start_time = getattr(track, "start_time", None)
                if start_time:
                    formatted_time = self._format_time(start_time)
                else:
                    # Calculate based on previous track
                    formatted_time = f"{(idx - 1) * 5:02d}:00:00"  # 5 minutes per track default
                lines.append(f"    INDEX 01 {formatted_time}")

        return "\n".join(lines)

    async def _validate_against_audio(self, cue_content: str, audio_file_path: str, tracklist: Any) -> Dict[str, Any]:
        """
        Validate CUE content against audio file.

        Args:
            cue_content: Generated CUE content
            audio_file_path: Path to audio file
            tracklist: Original tracklist

        Returns:
            Validation report
        """
        try:
            if not CUE_HANDLER_AVAILABLE or not self.validator:
                # Basic validation without CUE handler
                return {
                    "valid": True,
                    "errors": [],
                    "warnings": [{"field": "validator", "message": "CUE handler not available, basic validation only"}],
                    "metadata": {
                        "validated_at": datetime.now(timezone.utc).isoformat(),
                    },
                }

            # Parse the generated CUE content
            if not self.parser:
                return {
                    "valid": False,
                    "errors": [{"field": "parser", "message": "CUE parser not available"}],
                    "warnings": [],
                    "metadata": {},
                }
            parsed_cue = self.parser.parse(cue_content)

            # Run validation
            validation_result = self.validator.validate(parsed_cue, strict=True)  # type: ignore[call-arg, arg-type]

            # Check audio duration if file exists
            audio_path = Path(audio_file_path)
            if audio_path.exists():
                # Get audio duration (placeholder - would use actual audio library)
                audio_duration_seconds = 3600  # Placeholder: 1 hour

                # Check if last track exceeds audio duration
                if parsed_cue.tracks:  # type: ignore[attr-defined]
                    last_track = parsed_cue.tracks[-1]  # type: ignore[attr-defined]
                    last_track_time = self._parse_time_to_seconds(last_track.indices[0].time)
                    if last_track_time > audio_duration_seconds:
                        validation_result.add_error(  # type: ignore[attr-defined]
                            "audio_duration",
                            f"Last track starts after audio ends ({last_track_time}s > {audio_duration_seconds}s)",
                        )

            return {
                "valid": validation_result.is_valid,
                "errors": [{"field": e.field, "message": e.message} for e in validation_result.errors],  # type: ignore[attr-defined]
                "warnings": [{"field": w.field, "message": w.message} for w in validation_result.warnings],  # type: ignore[attr-defined]
                "metadata": {
                    "track_count": len(parsed_cue.tracks),  # type: ignore[attr-defined]
                    "audio_file": audio_file_path,
                    "validated_at": datetime.now(timezone.utc).isoformat(),
                },
            }

        except Exception as e:
            logger.error(f"Validation failed: {e}", exc_info=True)
            return {
                "valid": False,
                "errors": [{"field": "validation", "message": str(e)}],
                "warnings": [],
                "metadata": {},
            }

    def _parse_time_to_seconds(self, time_str: str) -> float:
        """Parse CUE time format to seconds."""
        # Use centralized utility function
        return parse_cue_time(time_str)

    async def _store_cue_file(
        self, tracklist_id: UUID, format_name: str, content: str, validation_report: Optional[Dict[str, Any]]
    ) -> Tuple[UUID, str]:
        """
        Store CUE file in storage and database.

        Args:
            tracklist_id: Tracklist ID
            format_name: CUE format name
            content: CUE file content
            validation_report: Optional validation report

        Returns:
            Tuple of (cue_file_id, file_path)
        """
        # Generate file path
        file_path = self._generate_file_path(tracklist_id, format_name)

        # Store file content
        success, stored_path, error = self.storage_service.store_cue_file(file_path, content)
        if not success:
            raise Exception(f"Failed to store CUE file: {error}")

        # Return generated ID and path (database record would be created by repository)
        cue_file_id = uuid4()
        return cue_file_id, stored_path

    def _generate_file_path(self, tracklist_id: UUID, format_name: str) -> str:
        """Generate file path for CUE file storage."""
        now = datetime.now(timezone.utc)
        return f"{now.year}/{now.month:02d}/{tracklist_id}/{format_name}.cue"

    async def validate_tracklist_for_cue(self, tracklist: Any, format_type: Optional[str] = None) -> Any:
        """
        Validate if tracklist is suitable for CUE generation.

        Args:
            tracklist: Tracklist to validate
            format_type: Optional format type for format-specific validation

        Returns:
            ValidationResult-like object with valid, error, and warnings attributes
        """

        class ValidationResult:
            def __init__(self) -> None:
                self.valid = True
                self.error: Optional[str] = None
                self.warnings: List[str] = []
                self.errors: List[str] = []
                self.metadata: Dict[str, Any] = {}

        result = ValidationResult()
        errors = []

        # Check required fields
        if not tracklist:
            errors.append("Tracklist is required")
            result.valid = False
            result.error = "Tracklist is required"
            result.errors = errors
            return result

    def get_format_capabilities(self, format_type: str) -> Dict[str, Any]:
        """
        Get capabilities and features for a specific CUE format.

        Args:
            format_type: CUE format type

        Returns:
            Dictionary of format capabilities
        """
        capabilities = {
            "standard": {
                "supports_multiple_files": True,
                "supports_pregap": True,
                "supports_postgap": True,
                "supports_isrc": True,
                "supports_catalog": True,
                "max_tracks": 99,
                "timing_precision": "frames",
                "metadata_fields": ["title", "performer", "songwriter", "rem"],
            },
            "cdj": {
                "supports_multiple_files": False,
                "supports_pregap": False,
                "supports_postgap": False,
                "supports_isrc": False,
                "supports_catalog": False,
                "max_tracks": 999,
                "timing_precision": "milliseconds",
                "metadata_fields": ["title", "artist", "bpm", "key"],
            },
            "traktor": {
                "supports_multiple_files": False,
                "supports_pregap": False,
                "supports_postgap": False,
                "supports_isrc": False,
                "supports_catalog": False,
                "max_tracks": 999,
                "timing_precision": "milliseconds",
                "metadata_fields": ["title", "artist", "bpm", "key", "rating"],
            },
            "serato": {
                "supports_multiple_files": False,
                "supports_pregap": False,
                "supports_postgap": False,
                "supports_isrc": False,
                "supports_catalog": False,
                "max_tracks": 999,
                "timing_precision": "milliseconds",
                "metadata_fields": ["title", "artist", "bpm", "key", "color"],
            },
            "rekordbox": {
                "supports_multiple_files": False,
                "supports_pregap": False,
                "supports_postgap": False,
                "supports_isrc": False,
                "supports_catalog": False,
                "max_tracks": 999,
                "timing_precision": "milliseconds",
                "metadata_fields": ["title", "artist", "bpm", "key", "color", "rating"],
            },
            "kodi": {
                "supports_multiple_files": True,
                "supports_pregap": True,
                "supports_postgap": False,
                "supports_isrc": False,
                "supports_catalog": False,
                "max_tracks": 999,
                "timing_precision": "seconds",
                "metadata_fields": ["title", "artist", "album", "year", "genre"],
            },
        }

        return capabilities.get(
            format_type.lower(),
            {"error": f"Unknown format: {format_type}", "supported_formats": list(capabilities.keys())},
        )

    async def convert_cue_format(
        self, cue_content: str, source_format: str, target_format: str
    ) -> CueGenerationResponse:
        """
        Convert CUE file from one format to another.

        Args:
            cue_content: Source CUE content
            source_format: Source format
            target_format: Target format

        Returns:
            Generation response with converted content
        """
        try:
            if not CUE_HANDLER_AVAILABLE or not self.converter:
                return CueGenerationResponse(
                    success=False,
                    job_id=UUID("00000000-0000-0000-0000-000000000000"),
                    cue_file_id=None,
                    file_path=None,
                    validation_report=None,
                    error="CUE handler not available for format conversion",
                )

            # Parse source content
            if not self.parser:
                return CueGenerationResponse(
                    success=False,
                    job_id=UUID("00000000-0000-0000-0000-000000000000"),
                    cue_file_id=None,
                    file_path=None,
                    validation_report=None,
                    error="CUE parser not available for format conversion",
                )
            parsed_cue = self.parser.parse(cue_content)

            # Get format enums
            source_fmt = getattr(CueFormat, source_format.upper(), None)
            target_fmt = getattr(CueFormat, target_format.upper(), None)

            if not source_fmt or not target_fmt:
                return CueGenerationResponse(
                    success=False,
                    job_id=UUID("00000000-0000-0000-0000-000000000000"),
                    cue_file_id=None,
                    file_path=None,
                    validation_report=None,
                    error=f"Invalid format: {source_format} or {target_format}",
                )

            # Convert to target format
            conversion_result = self.converter.convert(parsed_cue, source_fmt, target_fmt)  # type: ignore[arg-type]

            if conversion_result.success:
                return CueGenerationResponse(
                    success=True,
                    job_id=UUID("00000000-0000-0000-0000-000000000000"),
                    cue_file_id=None,
                    file_path=None,
                    validation_report=None,  # TODO: properly convert validation report
                    error=None,
                )
            else:
                return CueGenerationResponse(
                    success=False,
                    job_id=UUID("00000000-0000-0000-0000-000000000000"),
                    cue_file_id=None,
                    file_path=None,
                    validation_report=None,
                    error="Conversion failed",
                )

        except Exception as e:
            logger.error(f"Format conversion failed: {e}", exc_info=True)
            return CueGenerationResponse(
                success=False,
                job_id=UUID("00000000-0000-0000-0000-000000000000"),
                cue_file_id=None,
                file_path=None,
                validation_report=None,
                error=str(e),
            )

    def get_supported_formats(self) -> List[Any]:
        """
        Get list of supported CUE formats.

        Returns:
            List of supported CueFormat enum values
        """
        # Return instances if it's an enum, otherwise return empty list
        if hasattr(CueFormat, "__members__"):
            return list(CueFormat.__members__.values())  # type: ignore
        return []

    async def invalidate_tracklist_cache(self, tracklist_id: UUID) -> int:
        """
        Invalidate cache for a specific tracklist.

        Args:
            tracklist_id: ID of the tracklist to invalidate

        Returns:
            Number of cache entries invalidated
        """
        if not self.cache_service:
            return 0

        # Invalidate all formats for this tracklist
        invalidated_count = 0
        if hasattr(CueFormat, "__members__"):
            for format_name in CueFormat.__members__.values():  # type: ignore
                cache_key = f"cue:{tracklist_id}:{format_name.value}"  # type: ignore
                if hasattr(self.cache_service, "delete"):
                    await self.cache_service.delete(cache_key)
                    invalidated_count += 1

        return invalidated_count

    def get_conversion_preview(
        self,
        source_format: Any,
        target_format: Any,  # Accept both CueFormat types
    ) -> List[str]:
        """
        Get preview of potential issues when converting between formats.

        Args:
            source_format: Source CUE format
            target_format: Target CUE format

        Returns:
            List of warning messages about potential data loss or issues
        """
        warnings = []

        # Check if formats are the same
        if source_format == target_format:
            warnings.append("Source and target formats are the same")
            return warnings

        # Convert to string values for comparison if needed
        if hasattr(source_format, "value"):
            source_str = source_format.value
        else:
            source_str = str(source_format).lower()

        if hasattr(target_format, "value"):
            target_str = target_format.value
        else:
            target_str = str(target_format).lower()

        # Format-specific conversion warnings
        conversion_warnings = {
            ("standard", "cdj"): [
                "CDJ format may not support all metadata fields",
                "Some timing precision may be lost",
            ],
            ("standard", "traktor"): [
                "Traktor-specific cue points will be generated",
                "BPM information may need manual adjustment",
            ],
            ("rekordbox", "standard"): [
                "Rekordbox-specific metadata will be lost",
                "Hot cue colors will not be preserved",
            ],
            ("serato", "standard"): [
                "Serato-specific cue point types will be simplified",
                "Loop information may be lost",
            ],
        }

        # Get specific warnings for this conversion
        key = (source_str, target_str)
        if key in conversion_warnings:
            warnings.extend(conversion_warnings[key])
        else:
            # Generic warning for unsupported conversions
            warnings.append(f"Converting from {source_str} to {target_str} may result in data loss")

        return warnings

    def validate_cue_content(self, content: str, format_type: Any, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate CUE file content.

        Args:
            content: CUE file content to validate
            format_type: CUE format type
            options: Validation options

        Returns:
            Validation result dictionary
        """
        errors: List[str] = []
        warnings: List[str] = []
        result: Dict[str, Any] = {
            "valid": True,
            "errors": errors,
            "warnings": warnings,
            "metadata": {
                "format": str(format_type),
                "content_length": len(content),
                "line_count": content.count("\n") + 1,
            },
        }

        # Basic validation checks
        if not content:
            result["valid"] = False
            errors.append("CUE content is empty")
            return result

        # Check for required CUE file markers
        if "TITLE" not in content and "title" not in content.lower():
            warnings.append("CUE file may be missing TITLE field")

        if "FILE" not in content and "file" not in content.lower():
            warnings.append("CUE file may be missing FILE declaration")

        if "TRACK" not in content and "track" not in content.lower():
            result["valid"] = False
            errors.append("CUE file has no TRACK entries")

        return result

    def convert_cue_format_sync(
        self, content: str, source_format: Any, target_format: Any, preserve_metadata: bool, options: Dict[str, Any]
    ) -> Any:
        """
        Convert CUE content between formats (synchronous version for cue_integration compatibility).

        Args:
            content: Source CUE content
            source_format: Source format
            target_format: Target format
            preserve_metadata: Whether to preserve metadata
            options: Conversion options

        Returns:
            Conversion result object
        """

        class ConversionResult:
            def __init__(self) -> None:
                self.success = True
                self.output = content  # Default to original content
                self.warnings: List[str] = []
                self.data_loss: List[str] = []
                self.error: Optional[str] = None

        result = ConversionResult()

        # Add conversion warnings
        preview_warnings = self.get_conversion_preview(source_format, target_format)
        result.warnings = preview_warnings

        # For now, just return the original content with warnings
        # Actual conversion would happen here if CUE handler was available
        if not CUE_HANDLER_AVAILABLE:
            result.warnings.append("CUE handler not available, returning original content")

        return result

    async def regenerate_cue_file(
        self, tracklist: Any, cue_file: Any, options: Optional[Dict[str, Any]] = None
    ) -> CueGenerationResponse:
        """
        Regenerate a CUE file for an existing tracklist and CUE file record.

        Args:
            tracklist: Tracklist data
            cue_file: Existing CUE file record
            options: Optional regeneration options

        Returns:
            CueGenerationResponse with regenerated file details
        """
        # Create a request object for regeneration
        request = GenerateCueRequest(
            format=cue_file.format if hasattr(cue_file, "format") else "standard",
            options=options or {},
            validate_audio=False,
            audio_file_path=None,
        )

        # Use the regular generate method
        return await self.generate_cue_file(tracklist, request)
