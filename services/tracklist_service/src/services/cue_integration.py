"""
CUE Handler integration service for tracklist service.

This module provides a wrapper around the analysis service's CUE handler
to integrate CUE file generation and validation with tracklist data.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from ..models.tracklist import Tracklist
from ..models.cue_file import CueFormat, ValidationResult
from ..utils.time_utils import timedelta_to_milliseconds

# Add the analysis service to the path so we can import from it
analysis_service_path = Path(__file__).parent.parent.parent.parent / "analysis_service" / "src"
sys.path.insert(0, str(analysis_service_path))

try:
    from cue_handler import (
        CueGenerator,
        CueFormat as CueHandlerFormat,
        CueTrack,
        CueFile,
        CueDisc,
        get_generator,
        CueValidator,
        CueConverter,
        ConversionMode,
    )
    from cue_handler.models import CueTime
except ImportError as e:
    raise ImportError(f"Could not import CUE handler components: {e}")


class CueFormatMapper:
    """Maps between tracklist service and CUE handler format enums."""

    FORMAT_MAPPING = {
        CueFormat.STANDARD: CueHandlerFormat.STANDARD,
        CueFormat.CDJ: CueHandlerFormat.CDJ,
        CueFormat.TRAKTOR: CueHandlerFormat.TRAKTOR,
        CueFormat.SERATO: CueHandlerFormat.SERATO,
        CueFormat.REKORDBOX: CueHandlerFormat.REKORDBOX,
        CueFormat.KODI: CueHandlerFormat.KODI,
    }

    REVERSE_MAPPING = {v: k for k, v in FORMAT_MAPPING.items()}

    @classmethod
    def to_cue_handler_format(cls, format_val: CueFormat) -> CueHandlerFormat:
        """Convert tracklist service format to CUE handler format."""
        if format_val not in cls.FORMAT_MAPPING:
            raise ValueError(f"Unsupported format: {format_val}")
        return cls.FORMAT_MAPPING[format_val]

    @classmethod
    def from_cue_handler_format(cls, format_val: CueHandlerFormat) -> CueFormat:
        """Convert CUE handler format to tracklist service format."""
        if format_val not in cls.REVERSE_MAPPING:
            raise ValueError(f"Unknown CUE handler format: {format_val}")
        return cls.REVERSE_MAPPING[format_val]


class TracklistToCueMapper:
    """Maps tracklist data to CUE handler data structures."""

    # Note: timedelta_to_milliseconds has been moved to utils.time_utils
    # and is imported at the top of this file

    @staticmethod
    def milliseconds_to_cue_time(ms: int) -> CueTime:
        """Convert milliseconds to CUE time format."""
        total_seconds = ms // 1000
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        frames = int((ms % 1000) * 75 / 1000)  # Convert milliseconds to frames (75fps)

        return CueTime(minutes=minutes, seconds=seconds, frames=frames)

    @classmethod
    def tracklist_to_cue_tracks(cls, tracklist: Tracklist) -> List[CueTrack]:
        """Convert tracklist tracks to CUE tracks."""
        cue_tracks = []

        for track in tracklist.tracks:
            start_ms = timedelta_to_milliseconds(track.start_time)

            # Build performer and title
            performer = track.artist
            title = track.title
            if track.remix:
                title = f"{title} ({track.remix})"

            # Create CUE track
            cue_track = CueTrack(number=track.position, title=title, performer=performer, start_time_ms=start_ms)

            # Add track to indices (INDEX 01 is the main index)
            cue_track.indices[1] = start_ms

            # Add REM fields for additional metadata
            if track.label:
                cue_track.rem_fields["LABEL"] = track.label
            if track.catalog_track_id:
                cue_track.rem_fields["CATALOG_ID"] = str(track.catalog_track_id)
            if track.confidence < 1.0:
                cue_track.rem_fields["CONFIDENCE"] = f"{track.confidence:.2f}"
            if track.transition_type:
                cue_track.rem_fields["TRANSITION"] = track.transition_type
            if track.is_manual_entry:
                cue_track.rem_fields["MANUAL_ENTRY"] = "true"

            cue_tracks.append(cue_track)

        return cue_tracks


class CueIntegrationService:
    """Main service for integrating CUE handler with tracklist service."""

    def __init__(self) -> None:
        self.generator = CueGenerator()
        self.validator = CueValidator()
        self.converter = CueConverter()
        self.format_mapper = CueFormatMapper()
        self.tracklist_mapper = TracklistToCueMapper()

    def generate_cue_content(
        self,
        tracklist: Tracklist,
        cue_format: CueFormat,
        audio_filename: str = "audio.wav",
        options: Optional[Dict] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Generate CUE file content from tracklist.

        Args:
            tracklist: Source tracklist
            cue_format: Target CUE format
            audio_filename: Name of audio file to reference
            options: Generation options

        Returns:
            Tuple of (success, content, error_message)
        """
        try:
            # Convert format
            handler_format = self.format_mapper.to_cue_handler_format(cue_format)

            # Create CUE tracks from tracklist
            cue_tracks = self.tracklist_mapper.tracklist_to_cue_tracks(tracklist)

            # Create CUE file reference
            cue_file = CueFile(filename=audio_filename, file_type="WAVE", tracks=cue_tracks)

            # Create CUE disc with metadata
            cue_disc = CueDisc(
                title=f"Tracklist {tracklist.id}",
                performer="DJ Mix",
                rem_fields={
                    "TRACKLIST_ID": str(tracklist.id),
                    "SOURCE": tracklist.source,
                    "CONFIDENCE_SCORE": f"{tracklist.confidence_score:.2f}",
                    "CREATED_AT": tracklist.created_at.isoformat(),
                },
            )

            if tracklist.is_draft:
                cue_disc.rem_fields["DRAFT"] = "true"
                if tracklist.draft_version:
                    cue_disc.rem_fields["DRAFT_VERSION"] = str(tracklist.draft_version)

            # Get appropriate generator for format
            format_generator = get_generator(handler_format)

            # Apply options if provided
            if options:
                self._apply_generation_options(format_generator, options)

            # Generate CUE content with correct API signature
            content = format_generator.generate(cue_disc, [cue_file])

            return True, content, None

        except Exception as e:
            return False, "", f"CUE generation failed: {str(e)}"

    def validate_cue_content(
        self, content: str, cue_format: Optional[CueFormat] = None, audio_duration_seconds: Optional[float] = None
    ) -> ValidationResult:
        """
        Validate CUE file content.

        Args:
            content: CUE file content to validate
            cue_format: Expected format (optional)
            audio_duration_seconds: Audio duration for validation (optional)

        Returns:
            ValidationResult with validation details
        """
        try:
            # Use CUE handler validator
            cue_validation = self.validator.validate(content)

            # Convert to tracklist service ValidationResult
            validation_result = ValidationResult(
                valid=cue_validation.is_valid,
                error=None,
                warnings=[issue.message for issue in cue_validation.issues if issue.severity.name == "WARNING"],
                audio_duration=None,
                tracklist_duration=None,
                metadata={
                    "total_issues": len(cue_validation.issues),
                    "errors": [issue.message for issue in cue_validation.issues if issue.severity.name == "ERROR"],
                },
            )

            # Add error message if validation failed
            if not cue_validation.is_valid:
                error_messages = [issue.message for issue in cue_validation.issues if issue.severity.name == "ERROR"]
                validation_result.error = "; ".join(error_messages)

            # Add audio duration validation if provided
            if audio_duration_seconds is not None:
                validation_result.audio_duration = audio_duration_seconds
                # TODO: Add logic to extract tracklist duration from content
                # and compare with audio duration

            return validation_result

        except Exception as e:
            return ValidationResult(
                valid=False, error=f"Validation failed: {str(e)}", audio_duration=None, tracklist_duration=None
            )

    def convert_cue_format(
        self, content: str, source_format: CueFormat, target_format: CueFormat, preserve_metadata: bool = True
    ) -> Tuple[bool, str, List[str], Optional[str]]:
        """
        Convert CUE content between formats.

        Args:
            content: Source CUE content
            source_format: Source format
            target_format: Target format
            preserve_metadata: Whether to preserve metadata

        Returns:
            Tuple of (success, converted_content, warnings, error_message)
        """
        try:
            # Convert formats
            handler_source = self.format_mapper.to_cue_handler_format(source_format)
            handler_target = self.format_mapper.to_cue_handler_format(target_format)

            # Set conversion mode
            mode = ConversionMode.PRESERVE_METADATA if preserve_metadata else ConversionMode.LOSSY

            # Perform conversion
            report = self.converter.convert(content, handler_source, handler_target, mode)

            # Extract results
            success = report.success
            converted_content = report.converted_content or ""
            warnings = [change.description for change in report.changes if change.is_warning]
            error_message = report.error if not success else None

            return success, converted_content, warnings, error_message

        except Exception as e:
            return False, "", [], f"Conversion failed: {str(e)}"

    def get_format_capabilities(self, cue_format: CueFormat) -> Dict[str, Any]:
        """
        Get capabilities for a specific CUE format.

        Args:
            cue_format: CUE format to query

        Returns:
            Dictionary of format capabilities
        """
        try:
            from cue_handler.format_mappings import get_format_capabilities

            handler_format = self.format_mapper.to_cue_handler_format(cue_format)
            capabilities = get_format_capabilities(handler_format)

            return {
                "max_tracks": capabilities.get("max_tracks", 99),
                "supports_isrc": capabilities.get("supports_isrc", False),
                "supports_flags": capabilities.get("supports_flags", False),
                "supports_rem": capabilities.get("supports_rem", False),
                "supports_pregap": capabilities.get("supports_pregap", False),
                "supports_postgap": capabilities.get("supports_postgap", False),
                "encoding": capabilities.get("encoding", "UTF-8"),
                "line_ending": capabilities.get("line_ending", "CRLF"),
            }

        except Exception as e:
            return {"error": f"Could not get capabilities: {str(e)}"}

    def get_conversion_warnings(self, source_format: CueFormat, target_format: CueFormat) -> List[str]:
        """
        Get potential warnings for format conversion.

        Args:
            source_format: Source format
            target_format: Target format

        Returns:
            List of warning messages
        """
        try:
            from cue_handler.format_mappings import get_lossy_warnings

            handler_source = self.format_mapper.to_cue_handler_format(source_format)
            handler_target = self.format_mapper.to_cue_handler_format(target_format)

            warnings = get_lossy_warnings(handler_source, handler_target)
            return warnings or []

        except Exception as e:
            return [f"Could not get conversion warnings: {str(e)}"]

    def _apply_generation_options(self, generator: Any, options: Dict) -> None:
        """Apply generation options to generator."""
        # This is a placeholder for applying format-specific options
        # In a real implementation, you would configure the generator
        # based on the options dictionary
        pass

    def get_supported_formats(self) -> List[CueFormat]:
        """Get list of supported CUE formats."""
        return list(CueFormat)

    def extract_metadata_from_content(self, content: str) -> Dict[str, Any]:
        """
        Extract metadata from CUE content.

        Args:
            content: CUE file content

        Returns:
            Dictionary of extracted metadata
        """
        try:
            # TODO: Implement metadata extraction using CUE parser
            # This would parse the CUE content and extract track information,
            # disc information, and REM fields

            # For now, return basic metadata
            lines = content.split("\n")
            track_count = sum(1 for line in lines if line.strip().startswith("TRACK"))

            return {
                "track_count": track_count,
                "has_file_reference": any("FILE " in line for line in lines),
                "has_rem_fields": any("REM " in line for line in lines),
                "line_count": len(lines),
            }

        except Exception as e:
            return {"error": f"Could not extract metadata: {str(e)}"}


# Create a singleton instance for use across the service
cue_integration = CueIntegrationService()
