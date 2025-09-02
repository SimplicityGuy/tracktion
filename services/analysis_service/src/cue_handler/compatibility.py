"""Compatibility checking for CUE format conversions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .format_mappings import FORMAT_CAPABILITIES, get_format_capabilities, get_lossy_warnings
from .generator import CueFormat


class CompatibilityLevel(Enum):
    """Compatibility level between formats."""

    FULL = "full"  # No data loss
    HIGH = "high"  # Minor data loss or transformations
    MEDIUM = "medium"  # Some features lost but core data preserved
    LOW = "low"  # Significant feature loss
    INCOMPATIBLE = "incompatible"  # Conversion not recommended


@dataclass
class CompatibilityIssue:
    """Represents a single compatibility issue."""

    feature: str
    severity: CompatibilityLevel
    description: str
    workaround: str | None = None


@dataclass
class CompatibilityReport:
    """Report of format compatibility check."""

    source_format: CueFormat
    target_format: CueFormat
    compatibility_level: CompatibilityLevel
    issues: list[CompatibilityIssue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    can_convert: bool = True
    metadata_preservation_estimate: float = 100.0

    def add_issue(self, issue: CompatibilityIssue) -> None:
        """Add a compatibility issue to the report."""
        self.issues.append(issue)
        self._update_compatibility_level()

    def add_warning(self, warning: str) -> None:
        """Add a warning to the report."""
        self.warnings.append(warning)

    def add_suggestion(self, suggestion: str) -> None:
        """Add a suggestion to the report."""
        self.suggestions.append(suggestion)

    def _update_compatibility_level(self) -> None:
        """Update overall compatibility level based on issues."""
        if not self.issues:
            self.compatibility_level = CompatibilityLevel.FULL
            return

        # Find the most severe issue
        severity_order = [
            CompatibilityLevel.INCOMPATIBLE,
            CompatibilityLevel.LOW,
            CompatibilityLevel.MEDIUM,
            CompatibilityLevel.HIGH,
            CompatibilityLevel.FULL,
        ]

        for severity in severity_order:
            if any(issue.severity == severity for issue in self.issues):
                self.compatibility_level = severity
                break

        # Update conversion feasibility
        if self.compatibility_level == CompatibilityLevel.INCOMPATIBLE:
            self.can_convert = False

    def estimate_metadata_preservation(self) -> None:
        """Estimate percentage of metadata that will be preserved."""
        if self.compatibility_level == CompatibilityLevel.FULL:
            self.metadata_preservation_estimate = 100.0
        elif self.compatibility_level == CompatibilityLevel.HIGH:
            self.metadata_preservation_estimate = 90.0
        elif self.compatibility_level == CompatibilityLevel.MEDIUM:
            self.metadata_preservation_estimate = 70.0
        elif self.compatibility_level == CompatibilityLevel.LOW:
            self.metadata_preservation_estimate = 50.0
        else:  # INCOMPATIBLE
            self.metadata_preservation_estimate = 0.0

        # Adjust based on specific issues
        critical_features = ["multi_file", "pregap_postgap", "isrc_support"]
        critical_count = sum(
            1
            for issue in self.issues
            if issue.feature in critical_features
            and issue.severity in [CompatibilityLevel.LOW, CompatibilityLevel.INCOMPATIBLE]
        )

        if critical_count > 0:
            self.metadata_preservation_estimate *= 1 - 0.1 * critical_count


class CompatibilityChecker:
    """Check compatibility between CUE formats."""

    def __init__(self) -> None:
        """Initialize the compatibility checker."""
        self.format_capabilities = FORMAT_CAPABILITIES

    def check_compatibility(
        self,
        source_format: CueFormat,
        target_format: CueFormat,
        cue_sheet: Any | None = None,
    ) -> CompatibilityReport:
        """Check compatibility between two formats.

        Args:
            source_format: Source format
            target_format: Target format
            cue_sheet: Optional CueSheet object for content-specific checks

        Returns:
            CompatibilityReport with detailed compatibility information
        """
        report = CompatibilityReport(
            source_format=source_format,
            target_format=target_format,
            compatibility_level=CompatibilityLevel.FULL,
        )

        # Get format capabilities
        source_caps = get_format_capabilities(source_format)
        target_caps = get_format_capabilities(target_format)

        if not source_caps or not target_caps:
            report.add_issue(
                CompatibilityIssue(
                    feature="format",
                    severity=CompatibilityLevel.INCOMPATIBLE,
                    description="Unknown format capabilities",
                )
            )
            return report

        # Check basic compatibility
        self._check_track_limit(source_caps, target_caps, cue_sheet, report)
        self._check_file_support(source_caps, target_caps, cue_sheet, report)
        self._check_command_support(source_caps, target_caps, report)
        self._check_metadata_support(source_caps, target_caps, report)
        self._check_encoding_support(source_caps, target_caps, report)

        # Add lossy conversion warnings
        warnings = get_lossy_warnings(source_format, target_format)
        for warning in warnings:
            report.add_warning(warning)

        # Add format-specific suggestions
        self._add_conversion_suggestions(source_format, target_format, report)

        # Estimate metadata preservation
        report.estimate_metadata_preservation()

        return report

    def _check_track_limit(
        self,
        source_caps: dict[str, Any],
        target_caps: dict[str, Any],
        cue_sheet: Any | None,
        report: CompatibilityReport,
    ) -> None:
        """Check track count compatibility."""
        target_max = target_caps.get("max_tracks")

        if target_max is None:
            return  # No limit

        # Check actual track count if cue_sheet provided
        if cue_sheet and hasattr(cue_sheet, "tracks"):
            try:
                track_count = len(cue_sheet.tracks)
                if track_count > target_max:
                    report.add_issue(
                        CompatibilityIssue(
                            feature="track_count",
                            severity=CompatibilityLevel.LOW,
                            description=f"Source has {track_count} tracks, target supports max {target_max}",
                            workaround=f"Only first {target_max} tracks will be converted",
                        )
                    )
            except (TypeError, AttributeError):
                # Handle mock objects or invalid tracks attribute
                pass

        # Check format level compatibility
        source_max = source_caps.get("max_tracks")
        if source_max is not None and source_max > target_max:
            # Potential issue based on format capabilities
            report.add_warning(f"Target format supports fewer tracks (max {target_max})")

    def _check_file_support(
        self,
        source_caps: dict[str, Any],
        target_caps: dict[str, Any],
        cue_sheet: Any | None,
        report: CompatibilityReport,
    ) -> None:
        """Check multi-file support compatibility."""
        source_multi = source_caps.get("multi_file", False)
        target_multi = target_caps.get("multi_file", False)

        if source_multi and not target_multi:
            # Check if actually using multi-file
            if cue_sheet and hasattr(cue_sheet, "files"):
                try:
                    if len(cue_sheet.files) > 1:
                        report.add_issue(
                            CompatibilityIssue(
                                feature="multi_file",
                                severity=CompatibilityLevel.MEDIUM,
                                description="Multiple FILE references will be consolidated",
                                workaround="All tracks will reference the first FILE",
                            )
                        )
                except (TypeError, AttributeError):
                    # Handle mock objects or invalid files attribute
                    pass
            else:
                report.add_warning("Target format doesn't support multiple FILE references")

    def _check_command_support(
        self,
        source_caps: dict[str, Any],
        target_caps: dict[str, Any],
        report: CompatibilityReport,
    ) -> None:
        """Check command support compatibility."""
        # Check PREGAP/POSTGAP
        if source_caps.get("pregap_postgap") and not target_caps.get("pregap_postgap"):
            report.add_issue(
                CompatibilityIssue(
                    feature="pregap_postgap",
                    severity=CompatibilityLevel.MEDIUM,
                    description="PREGAP/POSTGAP commands will be removed",
                    workaround="Gap timing will be lost",
                )
            )

        # Check FLAGS
        source_flags = source_caps.get("flags")
        target_flags = target_caps.get("flags")

        if source_flags == "all" and target_flags == "limited":
            report.add_warning("Some FLAGS may not be supported in target format")
        elif source_flags and not target_flags:
            report.add_issue(
                CompatibilityIssue(
                    feature="flags",
                    severity=CompatibilityLevel.HIGH,
                    description="FLAGS commands will be removed",
                    workaround="Track flags will be lost",
                )
            )

        # Check ISRC
        if source_caps.get("isrc_support") and not target_caps.get("isrc_support"):
            report.add_issue(
                CompatibilityIssue(
                    feature="isrc_support",
                    severity=CompatibilityLevel.HIGH,
                    description="ISRC codes not supported in target format",
                    workaround="ISRC codes will be stored as REM if possible",
                )
            )

    def _check_metadata_support(
        self,
        source_caps: dict[str, Any],
        target_caps: dict[str, Any],
        report: CompatibilityReport,
    ) -> None:
        """Check metadata support compatibility."""
        source_rem = source_caps.get("rem_fields", "none")
        target_rem = target_caps.get("rem_fields", "none")

        # Check REM field support
        if source_rem == "all" and target_rem == "limited":
            report.add_warning("Some REM fields may be lost in conversion")
        elif source_rem in ["all", "extended"] and target_rem == "none":
            report.add_issue(
                CompatibilityIssue(
                    feature="rem_fields",
                    severity=CompatibilityLevel.MEDIUM,
                    description="REM comments not supported in target format",
                    workaround="All REM metadata will be lost",
                )
            )

        # Check advanced features
        if source_caps.get("beat_grid") and not target_caps.get("beat_grid"):
            report.add_warning("Beat grid information will be stored as REM or lost")

        if source_caps.get("loop_points") and not target_caps.get("loop_points"):
            report.add_warning("Loop points will be stored as REM or lost")

        if source_caps.get("color_coding") and not target_caps.get("color_coding"):
            report.add_warning("Color coding information will be lost")

    def _check_encoding_support(
        self,
        source_caps: dict[str, Any],
        target_caps: dict[str, Any],
        report: CompatibilityReport,
    ) -> None:
        """Check character encoding compatibility."""
        source_limit = source_caps.get("char_limit", 80)
        target_limit = target_caps.get("char_limit", 80)

        if source_limit > target_limit:
            report.add_warning(f"Long text fields may be truncated (limit: {target_limit} chars)")

    def _add_conversion_suggestions(
        self,
        source_format: CueFormat,
        target_format: CueFormat,
        report: CompatibilityReport,
    ) -> None:
        """Add format-specific conversion suggestions."""
        # Standard to DJ software
        if source_format == CueFormat.STANDARD and target_format in [
            CueFormat.TRAKTOR,
            CueFormat.SERATO,
            CueFormat.REKORDBOX,
        ]:
            report.add_suggestion(
                "Consider adding BPM and KEY information before conversion for better DJ software integration"
            )
            report.add_suggestion("INDEX points will be converted to cue points/memory cues")

        # DJ software to Standard
        if (
            source_format in [CueFormat.TRAKTOR, CueFormat.SERATO, CueFormat.REKORDBOX]
            and target_format == CueFormat.STANDARD
        ):
            report.add_suggestion("Advanced DJ features (beat grid, loops, hot cues) will be stored as REM comments")
            report.add_suggestion("Consider using a DJ-specific format for preserving all features")

        # Between DJ software
        if (
            source_format in [CueFormat.TRAKTOR, CueFormat.SERATO, CueFormat.REKORDBOX]
            and target_format in [CueFormat.TRAKTOR, CueFormat.SERATO, CueFormat.REKORDBOX]
            and source_format != target_format
        ):
            report.add_suggestion("Cue point colors and types may need manual adjustment after conversion")
            report.add_suggestion("Beat grid and BPM information should transfer but may need re-analysis")

        # To CDJ
        if target_format == CueFormat.CDJ:
            report.add_suggestion("CDJ format is optimized for Pioneer CDJ hardware compatibility")
            report.add_suggestion("Consider validating the output on target CDJ hardware")

        # To Kodi
        if target_format == CueFormat.KODI:
            report.add_suggestion("Kodi format preserves most metadata - good for archival purposes")
            report.add_suggestion("NFO companion files can be generated for additional metadata")

    def get_compatibility_matrix(
        self,
    ) -> dict[tuple[CueFormat, CueFormat], CompatibilityLevel]:
        """Get a matrix of compatibility levels between all format pairs.

        Returns:
            Dictionary mapping (source, target) format pairs to compatibility levels
        """
        matrix = {}
        formats = list(CueFormat)

        for source in formats:
            for target in formats:
                if source == target:
                    matrix[(source, target)] = CompatibilityLevel.FULL
                else:
                    # Simple heuristic based on capabilities
                    report = self.check_compatibility(source, target)
                    matrix[(source, target)] = report.compatibility_level

        return matrix

    def recommend_conversion_path(self, source_format: CueFormat, target_format: CueFormat) -> list[CueFormat]:
        """Recommend an optimal conversion path if direct conversion is lossy.

        Args:
            source_format: Starting format
            target_format: Desired end format

        Returns:
            List of formats representing the conversion path
        """
        # Direct conversion is usually best
        direct_compat = self.check_compatibility(source_format, target_format)

        if direct_compat.compatibility_level in [
            CompatibilityLevel.FULL,
            CompatibilityLevel.HIGH,
        ]:
            return [source_format, target_format]

        # Check if going through STANDARD format is better
        to_standard = self.check_compatibility(source_format, CueFormat.STANDARD)
        from_standard = self.check_compatibility(CueFormat.STANDARD, target_format)

        if to_standard.compatibility_level == CompatibilityLevel.FULL and from_standard.compatibility_level in [
            CompatibilityLevel.FULL,
            CompatibilityLevel.HIGH,
        ]:
            return [source_format, CueFormat.STANDARD, target_format]

        # For DJ formats, sometimes going through another DJ format is better
        dj_formats = [CueFormat.TRAKTOR, CueFormat.SERATO, CueFormat.REKORDBOX]

        if source_format in dj_formats and target_format in dj_formats:
            # Try intermediate DJ formats
            for intermediate in dj_formats:
                if intermediate not in (source_format, target_format):
                    path1 = self.check_compatibility(source_format, intermediate)
                    path2 = self.check_compatibility(intermediate, target_format)

                    if path1.compatibility_level in [
                        CompatibilityLevel.FULL,
                        CompatibilityLevel.HIGH,
                    ] and path2.compatibility_level in [
                        CompatibilityLevel.FULL,
                        CompatibilityLevel.HIGH,
                    ]:
                        return [source_format, intermediate, target_format]

        # Default to direct conversion
        return [source_format, target_format]
