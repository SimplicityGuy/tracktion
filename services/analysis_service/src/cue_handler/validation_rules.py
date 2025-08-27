"""Validation rule implementations for CUE files.

Each rule validates a specific aspect of CUE file correctness.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional
import logging

from .models import CueSheet

logger = logging.getLogger(__name__)


class Severity(Enum):
    """Validation issue severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""

    severity: Severity
    line_number: Optional[int]
    category: str
    message: str
    suggestion: Optional[str] = None


class ValidationRule(ABC):
    """Base class for all validation rules."""

    @abstractmethod
    def validate(self, cue_sheet: CueSheet, cue_path: Path, cue_content: str) -> List[ValidationIssue]:
        """Validate the CUE sheet and return any issues found."""
        pass

    def _get_line_number(self, content: str, search_text: str) -> Optional[int]:
        """Find the line number containing the search text."""
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            if search_text in line:
                return i
        return None


class SyntaxValidator(ValidationRule):
    """Validates CUE command syntax and structure."""

    def validate(self, cue_sheet: CueSheet, cue_path: Path, cue_content: str) -> List[ValidationIssue]:
        """Validate syntax rules."""
        issues = []

        # Check for required commands
        if not cue_sheet.files:
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    line_number=None,
                    category="syntax",
                    message="No FILE command found in CUE sheet",
                    suggestion="Add a FILE command with the audio file path and format",
                )
            )

        # Check each track
        track_numbers = set()
        for file_ref in cue_sheet.files:
            for track in file_ref.tracks:
                # Check for duplicate track numbers
                if track.number in track_numbers:
                    line_num = self._get_line_number(cue_content, f"TRACK {track.number:02d}")
                    issues.append(
                        ValidationIssue(
                            severity=Severity.ERROR,
                            line_number=line_num,
                            category="syntax",
                            message=f"Duplicate track number: {track.number:02d}",
                            suggestion="Ensure track numbers are sequential and unique",
                        )
                    )
                track_numbers.add(track.number)

                # Check for required INDEX 01
                if 1 not in track.indices:
                    line_num = self._get_line_number(cue_content, f"TRACK {track.number:02d}")
                    issues.append(
                        ValidationIssue(
                            severity=Severity.ERROR,
                            line_number=line_num,
                            category="syntax",
                            message=f"Track {track.number:02d} missing required INDEX 01",
                            suggestion="Add INDEX 01 with the track start time",
                        )
                    )

                # Check character limits
                if track.title and len(track.title) > 80:
                    line_num = self._get_line_number(cue_content, f'TITLE "{track.title}"')
                    issues.append(
                        ValidationIssue(
                            severity=Severity.WARNING,
                            line_number=line_num,
                            category="syntax",
                            message=f"Track {track.number:02d} TITLE exceeds 80 character limit ({len(track.title)} chars)",
                            suggestion=f"Truncate to: {track.title[:77]}...",
                        )
                    )

                if track.performer and len(track.performer) > 80:
                    line_num = self._get_line_number(cue_content, f'PERFORMER "{track.performer}"')
                    issues.append(
                        ValidationIssue(
                            severity=Severity.WARNING,
                            line_number=line_num,
                            category="syntax",
                            message=f"Track {track.number:02d} PERFORMER exceeds 80 character limit ({len(track.performer)} chars)",
                            suggestion=f"Truncate to: {track.performer[:77]}...",
                        )
                    )

                # Validate ISRC format (12 characters, alphanumeric)
                if track.isrc:
                    if not re.match(r"^[A-Z]{2}[A-Z0-9]{3}\d{2}\d{5}$", track.isrc):
                        line_num = self._get_line_number(cue_content, f"ISRC {track.isrc}")
                        issues.append(
                            ValidationIssue(
                                severity=Severity.WARNING,
                                line_number=line_num,
                                category="syntax",
                                message=f"Invalid ISRC format: {track.isrc}",
                                suggestion="Format as XX-XXX-YY-NNNNN (country[2]-registrant[3]-year[2]-code[5])",
                            )
                        )

                # Validate FLAGS values
                valid_flags = {"DCP", "4CH", "PRE", "SCMS"}
                if track.flags:
                    invalid_flags = set(track.flags) - valid_flags
                    if invalid_flags:
                        line_num = self._get_line_number(cue_content, f"FLAGS {' '.join(track.flags)}")
                        issues.append(
                            ValidationIssue(
                                severity=Severity.WARNING,
                                line_number=line_num,
                                category="syntax",
                                message=f"Invalid FLAGS values: {', '.join(invalid_flags)}",
                                suggestion=f"Use only: {', '.join(valid_flags)}",
                            )
                        )

        # Validate CATALOG format (13 digits UPC/EAN)
        if cue_sheet.catalog:
            if not re.match(r"^\d{13}$", cue_sheet.catalog):
                line_num = self._get_line_number(cue_content, f"CATALOG {cue_sheet.catalog}")
                issues.append(
                    ValidationIssue(
                        severity=Severity.WARNING,
                        line_number=line_num,
                        category="syntax",
                        message=f"Invalid CATALOG format: {cue_sheet.catalog}",
                        suggestion="Use 13 digits for UPC/EAN barcode",
                    )
                )

        # Check disc-level character limits
        if cue_sheet.title and len(cue_sheet.title) > 80:
            line_num = self._get_line_number(cue_content, f'TITLE "{cue_sheet.title}"')
            issues.append(
                ValidationIssue(
                    severity=Severity.WARNING,
                    line_number=line_num,
                    category="syntax",
                    message=f"Disc TITLE exceeds 80 character limit ({len(cue_sheet.title)} chars)",
                    suggestion=f"Truncate to: {cue_sheet.title[:77]}...",
                )
            )

        if cue_sheet.performer and len(cue_sheet.performer) > 80:
            line_num = self._get_line_number(cue_content, f'PERFORMER "{cue_sheet.performer}"')
            issues.append(
                ValidationIssue(
                    severity=Severity.WARNING,
                    line_number=line_num,
                    category="syntax",
                    message=f"Disc PERFORMER exceeds 80 character limit ({len(cue_sheet.performer)} chars)",
                    suggestion=f"Truncate to: {cue_sheet.performer[:77]}...",
                )
            )

        return issues


class CommandOrderValidator(ValidationRule):
    """Validates proper command ordering in CUE file."""

    def validate(self, cue_sheet: CueSheet, cue_path: Path, cue_content: str) -> List[ValidationIssue]:
        """Validate command order rules."""
        issues = []
        lines = [
            line.strip()
            for line in cue_content.split("\n")
            if line.strip() and not line.strip().startswith(("REM", ";", "//"))
        ]

        # Check CATALOG appears before FILE
        catalog_idx = None
        first_file_idx = None

        for i, line in enumerate(lines):
            if line.startswith("CATALOG"):
                catalog_idx = i
            elif line.startswith("FILE") and first_file_idx is None:
                first_file_idx = i

        if catalog_idx is not None and first_file_idx is not None and catalog_idx > first_file_idx:
            line_num = self._get_line_number(cue_content, lines[catalog_idx])
            issues.append(
                ValidationIssue(
                    severity=Severity.ERROR,
                    line_number=line_num,
                    category="order",
                    message="CATALOG must appear before FILE command",
                    suggestion="Move CATALOG to the beginning of the CUE file",
                )
            )

        # Check track number sequence
        for file_ref in cue_sheet.files:
            expected_track = 1
            for track in file_ref.tracks:
                if track.number != expected_track:
                    line_num = self._get_line_number(cue_content, f"TRACK {track.number:02d}")
                    issues.append(
                        ValidationIssue(
                            severity=Severity.WARNING,
                            line_number=line_num,
                            category="order",
                            message=f"Track {track.number:02d} out of sequence (expected {expected_track:02d})",
                            suggestion="Renumber tracks to be sequential starting from 01",
                        )
                    )
                expected_track = track.number + 1

            # Check INDEX order within tracks
            for track in file_ref.tracks:
                # Sort indices by number
                sorted_indices = sorted(track.indices.items())
                prev_time = None

                for idx_num, idx_time in sorted_indices:
                    # Check INDEX 00 before INDEX 01
                    if idx_num == 1 and 0 in track.indices:
                        idx00_time = track.indices[0]
                        if idx00_time >= idx_time:
                            line_num = self._get_line_number(cue_content, "INDEX 00")
                            issues.append(
                                ValidationIssue(
                                    severity=Severity.ERROR,
                                    line_number=line_num,
                                    category="order",
                                    message=f"Track {track.number:02d} INDEX 00 must be before INDEX 01",
                                    suggestion="Ensure pregap INDEX 00 time is less than track start INDEX 01",
                                )
                            )

                    # Check increasing times
                    if prev_time and idx_time <= prev_time:
                        line_num = self._get_line_number(cue_content, f"INDEX {idx_num:02d}")
                        issues.append(
                            ValidationIssue(
                                severity=Severity.ERROR,
                                line_number=line_num,
                                category="order",
                                message=f"Track {track.number:02d} INDEX {idx_num:02d} time not increasing",
                                suggestion="INDEX times must be cumulative and increasing",
                            )
                        )
                    prev_time = idx_time

        return issues


class FileReferenceValidator(ValidationRule):
    """Validates audio file references."""

    def validate(self, cue_sheet: CueSheet, cue_path: Path, cue_content: str) -> List[ValidationIssue]:
        """Validate file references."""
        issues = []
        cue_dir = cue_path.parent

        for file_ref in cue_sheet.files:
            # Check if file exists
            audio_path = cue_dir / file_ref.filename

            if not audio_path.exists():
                # Try absolute path
                audio_path_abs = Path(file_ref.filename)
                if not audio_path_abs.exists():
                    line_num = self._get_line_number(cue_content, file_ref.filename)
                    issues.append(
                        ValidationIssue(
                            severity=Severity.ERROR,
                            line_number=line_num,
                            category="file",
                            message=f"Referenced audio file not found: {file_ref.filename}",
                            suggestion=f"Check file exists at: {audio_path.resolve()}",
                        )
                    )
                    continue
                else:
                    audio_path = audio_path_abs
                    # Warn about absolute paths
                    line_num = self._get_line_number(cue_content, file_ref.filename)
                    issues.append(
                        ValidationIssue(
                            severity=Severity.WARNING,
                            line_number=line_num,
                            category="file",
                            message=f"Using absolute path: {file_ref.filename}",
                            suggestion=f"Consider using relative path: {audio_path.name}",
                        )
                    )

            # Check file permissions
            if audio_path.exists() and not audio_path.is_file():
                line_num = self._get_line_number(cue_content, file_ref.filename)
                issues.append(
                    ValidationIssue(
                        severity=Severity.ERROR,
                        line_number=line_num,
                        category="file",
                        message=f"Path is not a file: {file_ref.filename}",
                        suggestion="Ensure the path points to a valid audio file",
                    )
                )
            elif audio_path.exists():
                try:
                    with open(audio_path, "rb") as f:
                        f.read(1)
                except PermissionError:
                    line_num = self._get_line_number(cue_content, file_ref.filename)
                    issues.append(
                        ValidationIssue(
                            severity=Severity.ERROR,
                            line_number=line_num,
                            category="file",
                            message=f"Cannot read audio file (permission denied): {file_ref.filename}",
                            suggestion="Check file permissions",
                        )
                    )
                except Exception as e:
                    line_num = self._get_line_number(cue_content, file_ref.filename)
                    issues.append(
                        ValidationIssue(
                            severity=Severity.WARNING,
                            line_number=line_num,
                            category="file",
                            message=f"Error accessing audio file: {str(e)}",
                            suggestion="Check file integrity",
                        )
                    )

            # Validate file format matches declaration
            if audio_path.exists():
                ext = audio_path.suffix.lower()
                file_type = file_ref.file_type.lower()

                format_map = {
                    ".wav": "wave",
                    ".mp3": "mp3",
                    ".flac": "wave",  # FLAC often uses WAVE type for compatibility
                    ".aiff": "aiff",
                    ".aif": "aiff",
                }

                expected_type = format_map.get(ext)
                if expected_type and file_type != expected_type:
                    # Special case: FLAC can use WAVE
                    if not (ext == ".flac" and file_type == "wave"):
                        line_num = self._get_line_number(
                            cue_content, f'FILE "{file_ref.filename}" {file_ref.file_type}'
                        )
                        issues.append(
                            ValidationIssue(
                                severity=Severity.WARNING,
                                line_number=line_num,
                                category="file",
                                message=f"File extension {ext} doesn't match declared type {file_ref.file_type}",
                                suggestion=f"Consider using type {expected_type.upper()}",
                            )
                        )

        # Check for multi-file CUE
        if len(cue_sheet.files) > 1:
            issues.append(
                ValidationIssue(
                    severity=Severity.INFO,
                    line_number=None,
                    category="file",
                    message=f"Multi-file CUE detected ({len(cue_sheet.files)} files)",
                    suggestion="Some software may have limited support for multi-file CUEs",
                )
            )

        return issues


class TimestampValidator(ValidationRule):
    """Validates timestamp format and consistency."""

    def validate(self, cue_sheet: CueSheet, cue_path: Path, cue_content: str) -> List[ValidationIssue]:
        """Validate timestamp rules."""
        issues = []

        for file_ref in cue_sheet.files:
            prev_track_end = None

            for i, track in enumerate(file_ref.tracks):
                # Get track start time
                if 1 not in track.indices:
                    continue

                track_start = track.indices[1]

                # Check for overlapping with previous track
                if prev_track_end and track_start < prev_track_end:
                    line_num = self._get_line_number(cue_content, f"TRACK {track.number:02d}")
                    issues.append(
                        ValidationIssue(
                            severity=Severity.ERROR,
                            line_number=line_num,
                            category="timing",
                            message=f"Track {track.number:02d} overlaps with previous track",
                            suggestion=f"Adjust start time to be after {prev_track_end}",
                        )
                    )

                # Check for negative gaps
                if prev_track_end and track_start > prev_track_end:
                    gap_frames = track_start.to_frames() - prev_track_end.to_frames()
                    gap_seconds = gap_frames / 75.0

                    if gap_seconds > 2.0:  # More than 2 seconds gap
                        line_num = self._get_line_number(cue_content, f"TRACK {track.number:02d}")
                        issues.append(
                            ValidationIssue(
                                severity=Severity.INFO,
                                line_number=line_num,
                                category="timing",
                                message=f"Gap of {gap_seconds:.1f}s before track {track.number:02d}",
                                suggestion="Consider using INDEX 00 for pregap if intentional",
                            )
                        )

                # Validate frame values (0-74)
                for idx_num, idx_time in track.indices.items():
                    if idx_time.frames < 0 or idx_time.frames > 74:
                        line_num = self._get_line_number(cue_content, f"INDEX {idx_num:02d}")
                        issues.append(
                            ValidationIssue(
                                severity=Severity.ERROR,
                                line_number=line_num,
                                category="timing",
                                message=f"Invalid frame value {idx_time.frames} (must be 0-74)",
                                suggestion="Frames represent 1/75th of a second (0-74)",
                            )
                        )

                # Determine track end for next iteration
                # Use next track's INDEX 00 if available, otherwise use its INDEX 01
                if i < len(file_ref.tracks) - 1:
                    next_track = file_ref.tracks[i + 1]
                    if 0 in next_track.indices:
                        prev_track_end = next_track.indices[0]
                    elif 1 in next_track.indices:
                        prev_track_end = next_track.indices[1]

                # Validate pregap/postgap timing
                if track.pregap:
                    # PREGAP is a duration, not a timestamp
                    if 0 in track.indices and 1 in track.indices:
                        actual_pregap_frames = track.indices[1].to_frames() - track.indices[0].to_frames()
                        expected_pregap_frames = track.pregap.to_frames()  # pregap is a CueTime

                        if abs(actual_pregap_frames - expected_pregap_frames) > 5:  # Allow small tolerance
                            line_num = self._get_line_number(cue_content, "PREGAP")
                            issues.append(
                                ValidationIssue(
                                    severity=Severity.WARNING,
                                    line_number=line_num,
                                    category="timing",
                                    message="PREGAP duration doesn't match INDEX positions",
                                    suggestion="Verify pregap duration matches actual index gap",
                                )
                            )

        return issues


class CrossReferenceValidator(ValidationRule):
    """Validates cross-references and internal consistency."""

    def validate(self, cue_sheet: CueSheet, cue_path: Path, cue_content: str) -> List[ValidationIssue]:
        """Validate cross-reference rules."""
        issues = []

        # Count actual tracks
        total_tracks = sum(len(f.tracks) for f in cue_sheet.files)

        # Check if track count matches content
        lines = cue_content.split("\n")
        track_commands = [line for line in lines if line.strip().startswith("TRACK")]
        if len(track_commands) != total_tracks:
            issues.append(
                ValidationIssue(
                    severity=Severity.WARNING,
                    line_number=None,
                    category="consistency",
                    message=f"Track count mismatch: {len(track_commands)} TRACK commands but {total_tracks} parsed tracks",
                    suggestion="Check for parsing errors or malformed track definitions",
                )
            )

        # Validate multi-file timing continuity
        if len(cue_sheet.files) > 1:
            prev_file_last_time = None

            for i, file_ref in enumerate(cue_sheet.files):
                if file_ref.tracks:
                    first_track = file_ref.tracks[0]

                    if 1 in first_track.indices and prev_file_last_time:
                        # First track of new file should start at 00:00:00 typically
                        if first_track.indices[1].to_frames() != 0:
                            line_num = self._get_line_number(cue_content, f'FILE "{file_ref.filename}"')
                            issues.append(
                                ValidationIssue(
                                    severity=Severity.INFO,
                                    line_number=line_num,
                                    category="consistency",
                                    message=f"File {i + 1} first track doesn't start at 00:00:00",
                                    suggestion="Multi-file CUEs typically reset timing for each file",
                                )
                            )

                    # Track last time of this file
                    if file_ref.tracks:
                        last_track = file_ref.tracks[-1]
                        if last_track.indices:
                            # Get the maximum time from all indices
                            max_time = max(time.to_frames() for time in last_track.indices.values())
                            # Find the actual CueTime object with that max time
                            for time in last_track.indices.values():
                                if time.to_frames() == max_time:
                                    prev_file_last_time = time
                                    break

        # Check for orphaned REM fields
        rem_fields = cue_sheet.rem_fields
        known_rem_fields = {
            "GENRE",
            "DATE",
            "DISCID",
            "COMMENT",
            "DISCNUMBER",
            "COMPOSER",
            "REPLAYGAIN_ALBUM_GAIN",
            "REPLAYGAIN_ALBUM_PEAK",
            "REPLAYGAIN_TRACK_GAIN",
            "REPLAYGAIN_TRACK_PEAK",
        }

        for key in rem_fields:
            if key.upper() not in known_rem_fields:
                line_num = self._get_line_number(cue_content, f"REM {key}")
                issues.append(
                    ValidationIssue(
                        severity=Severity.INFO,
                        line_number=line_num,
                        category="consistency",
                        message=f"Non-standard REM field: {key}",
                        suggestion="Consider using standard REM fields for better compatibility",
                    )
                )

        # Check POSTGAP doesn't overlap next track
        for file_ref in cue_sheet.files:
            for i, track in enumerate(file_ref.tracks[:-1]):  # Skip last track
                if track.postgap:
                    next_track = file_ref.tracks[i + 1]

                    if 1 in next_track.indices and 1 in track.indices:
                        # Calculate when postgap would end
                        postgap_end_frames = track.indices[1].to_frames() + track.postgap.to_frames()

                        if postgap_end_frames > next_track.indices[1].to_frames():
                            line_num = self._get_line_number(cue_content, "POSTGAP")
                            issues.append(
                                ValidationIssue(
                                    severity=Severity.ERROR,
                                    line_number=line_num,
                                    category="consistency",
                                    message=f"Track {track.number:02d} POSTGAP overlaps with next track",
                                    suggestion="Reduce POSTGAP duration or adjust track timing",
                                )
                            )

        return issues


class CompatibilityValidator(ValidationRule):
    """Validates compatibility with various DJ software."""

    def validate(self, cue_sheet: CueSheet, cue_path: Path, cue_content: str) -> List[ValidationIssue]:
        """Validate compatibility rules."""
        issues = []

        # CDJ compatibility checks
        cdj_issues = []

        # CDJ: Limited character support
        if cue_sheet.title and len(cue_sheet.title) > 60:
            cdj_issues.append("Disc title exceeds CDJ 60 character display limit")

        for file_ref in cue_sheet.files:
            for track in file_ref.tracks:
                if track.title and len(track.title) > 60:
                    cdj_issues.append(f"Track {track.number:02d} title exceeds CDJ 60 character display limit")
                if track.performer and len(track.performer) > 60:
                    cdj_issues.append(f"Track {track.number:02d} performer exceeds CDJ 60 character display limit")

        if cdj_issues:
            issues.append(
                ValidationIssue(
                    severity=Severity.INFO,
                    line_number=None,
                    category="compatibility",
                    message=f"CDJ compatibility: {'; '.join(cdj_issues[:3])}",
                    suggestion="Consider shorter titles/performers for CDJ display",
                )
            )

        # Traktor compatibility checks
        if len(cue_sheet.files) > 1:
            issues.append(
                ValidationIssue(
                    severity=Severity.INFO,
                    line_number=None,
                    category="compatibility",
                    message="Traktor: Limited support for multi-file CUE sheets",
                    suggestion="Consider using single-file CUE for better Traktor compatibility",
                )
            )

        # Serato compatibility checks
        for file_ref in cue_sheet.files:
            for track in file_ref.tracks:
                # Serato prefers INDEX 01 only
                if len(track.indices) > 2:
                    line_num = self._get_line_number(cue_content, f"TRACK {track.number:02d}")
                    issues.append(
                        ValidationIssue(
                            severity=Severity.INFO,
                            line_number=line_num,
                            category="compatibility",
                            message=f"Serato: Track {track.number:02d} has multiple indices (Serato uses INDEX 01 only)",
                            suggestion="Serato will ignore additional INDEX points",
                        )
                    )

        # Rekordbox compatibility checks
        if not cue_sheet.title:
            issues.append(
                ValidationIssue(
                    severity=Severity.INFO,
                    line_number=None,
                    category="compatibility",
                    message="Rekordbox: No disc title found",
                    suggestion="Add a disc-level TITLE for better Rekordbox display",
                )
            )

        # Kodi compatibility checks
        # Kodi handles most standard CUE formats well
        if cue_sheet.catalog and not re.match(r"^\d{13}$", cue_sheet.catalog):
            issues.append(
                ValidationIssue(
                    severity=Severity.INFO,
                    line_number=None,
                    category="compatibility",
                    message="Kodi: Non-standard CATALOG format",
                    suggestion="Kodi expects 13-digit UPC/EAN for CATALOG",
                )
            )

        return issues
