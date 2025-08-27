"""Audio file analysis for CUE validation.

Analyzes audio files to extract duration and verify against CUE timing.
"""

from pathlib import Path
from typing import Optional, Tuple
import logging

from .models import CueSheet

logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """Analyzes audio files for CUE validation."""

    def analyze_durations(self, cue_sheet: CueSheet, cue_path: Path) -> Tuple[Optional[float], Optional[float]]:
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

    def _get_total_audio_duration(self, cue_sheet: CueSheet, cue_path: Path) -> Optional[float]:
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

    def _get_audio_file_duration(self, file_path: str) -> Optional[float]:
        """Get audio file duration in milliseconds.

        Uses mutagen for metadata extraction, falls back to pydub
        for direct audio analysis if needed.

        Args:
            file_path: Path to audio file

        Returns:
            Duration in milliseconds, or None if cannot determine
        """
        try:
            from mutagen import File

            audio = File(file_path)
            if audio and audio.info:
                return float(audio.info.length) * 1000  # Convert to ms
        except ImportError:
            logger.warning("mutagen not installed, trying pydub")
        except Exception as e:
            logger.warning(f"mutagen failed for {file_path}: {e}")

        try:
            from pydub import AudioSegment

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
        except ImportError:
            logger.warning("pydub not installed, cannot determine audio duration")
        except Exception as e:
            logger.warning(f"pydub failed for {file_path}: {e}")

        return None

    def _get_cue_duration(self, cue_sheet: CueSheet) -> Optional[float]:
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
                    if time_frames > max_time_frames:
                        max_time_frames = time_frames

                # Account for postgap
                if track.postgap:
                    # postgap is a CueTime
                    postgap_frames = track.postgap.to_frames()
                    # Find INDEX 01 to add postgap to
                    if 1 in track.indices:
                        postgap_end = track.indices[1].to_frames() + postgap_frames
                        if postgap_end > max_time_frames:
                            max_time_frames = postgap_end

        if max_time_frames > 0:
            # Convert frames to milliseconds (75 frames per second)
            return (max_time_frames / 75.0) * 1000

        return None

    def validate_track_bounds(self, cue_sheet: CueSheet, cue_path: Path) -> list:
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
                                    {
                                        "track": track.number,
                                        "index": idx_num,
                                        "time": str(idx_time),
                                        "audio_duration": audio_duration / 1000,
                                        "message": f"Track {track.number:02d} INDEX {idx_num:02d} at {idx_time} exceeds audio duration ({audio_duration / 1000:.1f}s)",
                                    }
                                )

        return issues
