"""
Audio Duration Validation Service for CUE file generation.

This service validates that CUE file track timings are consistent with
the actual audio file duration and detects potential timing issues.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from ..models.cue_file import ValidationResult
from ..models.tracklist import Tracklist


class AudioValidationService:
    """Service for validating audio file compatibility with tracklist timings."""

    def __init__(self) -> None:
        """Initialize audio validation service."""
        # For now, this is a placeholder implementation
        # In production, you would integrate with audio analysis libraries
        # like librosa, pydub, or ffmpeg for actual audio duration detection
        pass

    async def validate_audio_duration(
        self, audio_file_path: str, tracklist: Tracklist, tolerance_seconds: float = 2.0
    ) -> ValidationResult:
        """
        Validate tracklist timing against actual audio file duration.

        Args:
            audio_file_path: Path to audio file
            tracklist: Tracklist to validate
            tolerance_seconds: Acceptable difference in seconds

        Returns:
            Validation result with timing analysis
        """
        warnings = []
        metadata: Dict[str, Any] = {}

        try:
            # Get actual audio duration
            actual_duration = await self.get_audio_duration(audio_file_path)
            if actual_duration is None:
                return ValidationResult(
                    valid=False,
                    error=f"Could not determine duration of audio file: {audio_file_path}",
                    audio_duration=None,
                    tracklist_duration=None,
                )

            # Calculate tracklist duration from last track end time
            if not tracklist.tracks:
                return ValidationResult(
                    valid=False,
                    error="Tracklist has no tracks to validate",
                    audio_duration=None,
                    tracklist_duration=None,
                )

            # Find the last track with end time
            last_track_end = None
            for track in sorted(tracklist.tracks, key=lambda t: t.position, reverse=True):
                if track.end_time:
                    last_track_end = track.end_time
                    break

            if last_track_end is None:
                warnings.append("No track end times found - cannot validate total duration")
            else:
                tracklist_duration = float(last_track_end.total_seconds())
                duration_diff = abs(actual_duration - tracklist_duration)

                if duration_diff > tolerance_seconds:
                    warnings.append(
                        f"Duration mismatch: audio={actual_duration:.1f}s, "
                        f"tracklist={tracklist_duration:.1f}s, diff={duration_diff:.1f}s"
                    )

                metadata["actual_duration_seconds"] = actual_duration
                metadata["tracklist_duration_seconds"] = tracklist_duration
                metadata["duration_difference_seconds"] = duration_diff
                metadata["tolerance_seconds"] = tolerance_seconds

            # Validate individual track timings
            track_warnings = await self.validate_track_timings(tracklist, actual_duration, tolerance_seconds)
            warnings.extend(track_warnings)

            # Validate track sequence
            sequence_warnings = self.validate_track_sequence(tracklist)
            warnings.extend(sequence_warnings)

            metadata["track_count"] = len(tracklist.tracks)
            metadata["validation_type"] = "audio_duration"

            return ValidationResult(
                valid=len([w for w in warnings if "Duration mismatch" in w or "beyond audio" in w]) == 0,
                error=None,
                warnings=warnings,
                audio_duration=metadata.get("actual_duration_seconds"),
                tracklist_duration=metadata.get("tracklist_duration_seconds"),
                metadata=metadata,
            )

        except Exception as e:
            return ValidationResult(
                valid=False, error=f"Audio validation failed: {str(e)}", audio_duration=None, tracklist_duration=None
            )

    async def get_audio_duration(self, audio_file_path: str) -> Optional[float]:
        """
        Get the duration of an audio file in seconds.

        Args:
            audio_file_path: Path to audio file

        Returns:
            Duration in seconds, or None if cannot be determined
        """
        try:
            # Check if file exists
            if not Path(audio_file_path).exists():
                return None

            # TODO: Implement actual audio duration detection
            # This is a placeholder implementation
            # In production, you would use:
            # - librosa.get_duration(filename=audio_file_path)
            # - or subprocess call to ffprobe
            # - or pydub.AudioSegment.from_file(audio_file_path).duration_seconds

            # For now, return a mock duration based on file extension
            # This allows testing without actual audio libraries
            if audio_file_path.endswith((".mp3", ".wav", ".flac")):
                return 420.0  # Mock 7 minute duration
            else:
                return None

        except Exception:
            return None

    async def validate_track_timings(
        self, tracklist: Tracklist, audio_duration_seconds: float, tolerance_seconds: float = 1.0
    ) -> List[str]:
        """
        Validate individual track timing consistency.

        Args:
            tracklist: Tracklist to validate
            audio_duration_seconds: Total audio duration
            tolerance_seconds: Tolerance for timing issues

        Returns:
            List of timing warning messages
        """
        warnings = []

        for track in tracklist.tracks:
            # Check if track start time is reasonable
            if track.start_time:
                start_seconds = track.start_time.total_seconds()

                # Track starts after audio ends
                if start_seconds > audio_duration_seconds + tolerance_seconds:
                    warnings.append(
                        f"Track {track.position} starts at {start_seconds:.1f}s, "
                        f"beyond audio duration {audio_duration_seconds:.1f}s"
                    )

                # Track starts at negative time
                if start_seconds < 0:
                    warnings.append(f"Track {track.position} has negative start time")

            # Check if track end time is reasonable
            if track.end_time:
                end_seconds = track.end_time.total_seconds()

                # Track ends after audio with significant margin
                if end_seconds > audio_duration_seconds + tolerance_seconds:
                    warnings.append(
                        f"Track {track.position} ends at {end_seconds:.1f}s, "
                        f"beyond audio duration {audio_duration_seconds:.1f}s"
                    )

            # Check track duration is reasonable
            if track.start_time and track.end_time:
                track_duration = (track.end_time - track.start_time).total_seconds()

                if track_duration < 5:  # Very short track
                    warnings.append(f"Track {track.position} is very short ({track_duration:.1f}s)")
                elif track_duration > 900:  # Very long track (>15 minutes)
                    warnings.append(f"Track {track.position} is very long ({track_duration:.1f}s)")

        return warnings

    def validate_track_sequence(self, tracklist: Tracklist) -> List[str]:
        """
        Validate that tracks are in correct sequence without gaps or overlaps.

        Args:
            tracklist: Tracklist to validate

        Returns:
            List of sequencing warning messages
        """
        warnings = []

        sorted_tracks = sorted(tracklist.tracks, key=lambda t: t.position)

        for i in range(len(sorted_tracks) - 1):
            current_track = sorted_tracks[i]
            next_track = sorted_tracks[i + 1]

            # Check if we have timing information
            if current_track.end_time and next_track.start_time:
                current_end = current_track.end_time.total_seconds()
                next_start = next_track.start_time.total_seconds()

                # Gap between tracks
                if next_start > current_end + 1.0:  # More than 1 second gap
                    gap = next_start - current_end
                    warnings.append(
                        f"Gap of {gap:.1f}s between track {current_track.position} and track {next_track.position}"
                    )

                # Overlap between tracks
                elif next_start < current_end - 0.1:  # More than 100ms overlap
                    overlap = current_end - next_start
                    warnings.append(
                        f"Overlap of {overlap:.1f}s between track {current_track.position} "
                        f"and track {next_track.position}"
                    )

        return warnings

    async def suggest_timing_corrections(self, tracklist: Tracklist, audio_file_path: str) -> Dict[str, Any]:
        """
        Suggest corrections for timing issues in tracklist.

        Args:
            tracklist: Tracklist with potential timing issues
            audio_file_path: Path to audio file

        Returns:
            Dictionary with correction suggestions
        """
        suggestions: Dict[str, Any] = {"corrections": [], "metadata": {}}

        try:
            audio_duration = await self.get_audio_duration(audio_file_path)
            if audio_duration is None:
                return suggestions

            # Suggest corrections for tracks that extend beyond audio
            for track in tracklist.tracks:
                if track.end_time:
                    end_seconds = track.end_time.total_seconds()
                    if end_seconds > audio_duration:
                        suggestions["corrections"].append(
                            {
                                "track_position": track.position,
                                "issue": "extends_beyond_audio",
                                "current_end_time": end_seconds,
                                "suggested_end_time": float(audio_duration),
                                "reason": f"Track ends {end_seconds - audio_duration:.1f}s after audio",
                            }
                        )

            # Suggest filling gaps
            sorted_tracks = sorted(tracklist.tracks, key=lambda t: t.position)
            for i in range(len(sorted_tracks) - 1):
                current = sorted_tracks[i]
                next_track = sorted_tracks[i + 1]

                if current.end_time and next_track.start_time:
                    gap = next_track.start_time.total_seconds() - current.end_time.total_seconds()
                    if gap > 1.0:  # Significant gap
                        suggestions["corrections"].append(
                            {
                                "track_position": current.position,
                                "issue": "gap_after_track",
                                "gap_duration": gap,
                                "suggested_action": "extend_track_or_add_transition",
                            }
                        )

            suggestions["metadata"]["audio_duration"] = audio_duration
            suggestions["metadata"]["correction_count"] = len(suggestions["corrections"])

        except Exception as e:
            suggestions["error"] = str(e)

        return suggestions

    def estimate_track_durations(self, tracklist: Tracklist, audio_duration_seconds: float) -> List[Dict[str, Any]]:
        """
        Estimate track durations when end times are missing.

        Args:
            tracklist: Tracklist with potentially missing end times
            audio_duration_seconds: Total audio duration

        Returns:
            List of duration estimates for each track
        """
        estimates = []
        sorted_tracks = sorted(tracklist.tracks, key=lambda t: t.position)

        for i, track in enumerate(sorted_tracks):
            estimate: Dict[str, Union[int, float, str]] = {
                "track_position": track.position,
                "start_time": float(track.start_time.total_seconds()) if track.start_time else 0.0,
            }

            # If we have an end time, use it
            if track.end_time:
                estimate["end_time"] = float(track.end_time.total_seconds())
                estimate["duration"] = float(estimate["end_time"]) - float(estimate["start_time"])
                estimate["method"] = "provided"
            else:
                # Estimate based on next track start or audio end
                if i < len(sorted_tracks) - 1:
                    next_track = sorted_tracks[i + 1]
                    if next_track.start_time:
                        estimate["end_time"] = float(next_track.start_time.total_seconds())
                        estimate["method"] = "next_track_start"
                    else:
                        # Distribute remaining time
                        remaining_tracks = len(sorted_tracks) - i
                        remaining_time = audio_duration_seconds - float(estimate["start_time"])
                        avg_duration = remaining_time / remaining_tracks
                        estimate["end_time"] = float(estimate["start_time"]) + avg_duration
                        estimate["method"] = "average_remaining"
                else:
                    # Last track - use audio end
                    estimate["end_time"] = audio_duration_seconds
                    estimate["method"] = "audio_end"

                estimate["duration"] = float(estimate["end_time"]) - float(estimate["start_time"])

            estimates.append(estimate)

        return estimates
