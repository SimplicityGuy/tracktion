"""Audio file analysis for CUE validation.

Analyzes audio files to extract duration and verify against CUE timing.
"""

import logging
from pathlib import Path

from .models import CueSheet
from .validation_rules import Severity, ValidationIssue

try:
    from mutagen import File

    HAS_MUTAGEN = True
except ImportError:
    HAS_MUTAGEN = False

try:
    from pydub import AudioSegment

    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """Analyzes audio files for CUE validation."""

    def analyze_durations(self, cue_sheet: CueSheet, cue_path: Path) -> tuple[float | None, float | None]:
        """Analyze and compare audio vs CUE durations.

        Args:
            cue_sheet: Parsed CUE sheet
            cue_path: Path to the CUE file

        Returns:
            Tuple of (audio_duration_ms, cue_duration_ms)
        """
        audio_duration = self._get_total_audio_duration(cue_sheet, cue_path)
        cue_duration = self._get_cue_duration(cue_sheet)

        return audio_duration, cue_duration

    def _get_total_audio_duration(self, cue_sheet: CueSheet, cue_path: Path) -> float | None:
        """Get total duration of all referenced audio files.

        Args:
            cue_sheet: Parsed CUE sheet
            cue_path: Path to the CUE file

        Returns:
            Total duration in milliseconds, or None if cannot determine
        """
        total_duration = 0.0
        cue_dir = cue_path.parent

        for file_ref in cue_sheet.files:
            audio_path = cue_dir / file_ref.filename

            # Try absolute path if relative doesn't exist
            if not audio_path.exists():
                audio_path = Path(file_ref.filename)

            if audio_path.exists():
                duration = self._get_audio_file_duration(str(audio_path))
                if duration:
                    total_duration += duration
                else:
                    # If we can't get duration for any file, return None
                    return None
            else:
                # File doesn't exist, can't determine duration
                return None

        return total_duration if total_duration > 0 else None

    def _get_audio_file_duration(self, file_path: str) -> float | None:
        """Get audio file duration in milliseconds.

        Uses mutagen for metadata extraction, falls back to pydub
        for direct audio analysis if needed.

        Args:
            file_path: Path to audio file

        Returns:
            Duration in milliseconds, or None if cannot determine
        """
        if HAS_MUTAGEN:
            try:
                audio = File(file_path)
                if audio and audio.info:
                    return float(audio.info.length) * 1000  # Convert to ms
            except Exception as e:
                logger.warning(f"mutagen failed for {file_path}: {e}")
        else:
            logger.warning("mutagen not installed, trying pydub")

        if HAS_PYDUB:
            try:
                # Determine format from extension
                ext = Path(file_path).suffix.lower()[1:]  # Remove the dot
                format_map = {
                    "wav": "wav",
                    "mp3": "mp3",
                    "flac": "flac",
                    "ogg": "ogg",
                    "aiff": "aiff",
                    "aif": "aiff",
                }

                audio_format = format_map.get(ext, ext)
                audio = AudioSegment.from_file(file_path, format=audio_format)
                return len(audio)  # Already in ms
            except Exception as e:
                logger.warning(f"pydub failed for {file_path}: {e}")
        else:
            logger.warning("pydub not installed, cannot determine audio duration")

        return None

    def _get_cue_duration(self, cue_sheet: CueSheet) -> float | None:
        """Get total duration from CUE sheet timing.

        Args:
            cue_sheet: Parsed CUE sheet

        Returns:
            Duration in milliseconds based on last track's end time
        """
        max_time_frames = 0

        for file_ref in cue_sheet.files:
            for track in file_ref.tracks:
                # Find the latest time in this track
                for idx_time in track.indices.values():
                    time_frames = idx_time.to_frames()
                    max_time_frames = max(max_time_frames, time_frames)

                # Account for postgap
                if track.postgap:
                    # postgap is a CueTime
                    postgap_frames = track.postgap.to_frames()
                    # Find INDEX 01 to add postgap to
                    if 1 in track.indices:
                        postgap_end = track.indices[1].to_frames() + postgap_frames
                        max_time_frames = max(max_time_frames, postgap_end)

        if max_time_frames > 0:
            # Convert frames to milliseconds (75 frames per second)
            return (max_time_frames / 75.0) * 1000

        return None

    def validate_track_bounds(self, cue_sheet: CueSheet, cue_path: Path) -> list[ValidationIssue]:
        """Validate that track times are within audio file bounds.

        Args:
            cue_sheet: Parsed CUE sheet
            cue_path: Path to the CUE file

        Returns:
            List of validation issues
        """
        issues = []
        cue_dir = cue_path.parent

        for file_ref in cue_sheet.files:
            audio_path = cue_dir / file_ref.filename

            # Try absolute path if relative doesn't exist
            if not audio_path.exists():
                audio_path = Path(file_ref.filename)

            if audio_path.exists():
                audio_duration = self._get_audio_file_duration(str(audio_path))

                if audio_duration:
                    audio_duration_frames = int(audio_duration * 75 / 1000)

                    for track in file_ref.tracks:
                        # Check each index point
                        for idx_num, idx_time in track.indices.items():
                            idx_frames = idx_time.to_frames()

                            if idx_frames > audio_duration_frames:
                                issues.append(
                                    ValidationIssue(
                                        severity=Severity.ERROR,
                                        line_number=0,  # Audio validation doesn't have line numbers
                                        category="Audio Bounds",
                                        message=(
                                            f"Track {track.number:02d} INDEX {idx_num:02d} at {idx_time} exceeds "
                                            f"audio duration ({audio_duration / 1000:.1f}s)"
                                        ),
                                        suggestion="Adjust track timing to fit within audio file duration",
                                    )
                                )

        return issues
