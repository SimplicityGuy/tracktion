"""
Timing adjustment service for aligning tracklist timestamps.

This service handles timing adjustments and validations for
track timestamps when importing from 1001tracklists and for manual tracklist creation.
"""

import logging
from datetime import timedelta
from typing import Any

from services.tracklist_service.src.models.tracklist import TrackEntry
from services.tracklist_service.src.utils.time_utils import parse_time_string

logger = logging.getLogger(__name__)


class TimingService:
    """Service for adjusting and validating track timings."""

    def __init__(self) -> None:
        """Initialize the timing service."""
        self.min_track_duration = timedelta(seconds=30)  # Minimum 30 seconds per track
        self.max_track_duration = timedelta(minutes=20)  # Maximum 20 minutes per track
        self._calculation_cache: dict[str, Any] = {}  # Cache for expensive calculations

    def adjust_track_timings(
        self,
        tracks: list[TrackEntry],
        audio_duration: timedelta | None = None,
        offset: timedelta | None = None,
    ) -> list[TrackEntry]:
        """
        Adjust track timings to align with audio duration.

        Args:
            tracks: List of track entries with initial timings
            audio_duration: Total duration of the audio file (optional)
            offset: Optional offset to apply to all timestamps

        Returns:
            List of tracks with adjusted timings
        """
        if not tracks:
            return tracks

        # Apply offset if provided
        if offset:
            tracks = self._apply_offset(tracks, offset)

        # Validate and fix timing issues
        tracks = self._fix_timing_gaps(tracks)

        # Only ensure within duration if audio_duration is provided
        if audio_duration:
            tracks = self._ensure_within_duration(tracks, audio_duration)

        return self._validate_track_durations(tracks)

    def parse_timing_format(self, timing_str: str) -> timedelta:
        """
        Parse various timing formats from 1001tracklists.

        Args:
            timing_str: Timing string in various formats

        Returns:
            timedelta object

        Supported formats:
            - MM:SS (e.g., "5:30")
            - HH:MM:SS (e.g., "1:05:30")
            - M:SS (e.g., "5:30")
            - H:MM:SS (e.g., "1:05:30")
            - Decimal minutes (e.g., "5.5" for 5 minutes 30 seconds)
        """
        # Delegate to the centralized utility function
        result = parse_time_string(timing_str)
        if result == timedelta(0) and timing_str:
            logger.warning(f"Failed to parse timing: {timing_str}")
        return result  # type: ignore[no-any-return]  # parse_time_string returns timedelta but typed as Any

    def calculate_offset_from_start(
        self, first_track_time: timedelta, mix_start_time: timedelta = timedelta(0)
    ) -> timedelta:
        """
        Calculate offset for mix that doesn't start at 00:00.

        Args:
            first_track_time: Timestamp of first track in tracklist
            mix_start_time: Actual start time of mix in audio file

        Returns:
            Offset to apply to all timestamps
        """
        return mix_start_time - first_track_time

    def validate_timing_consistency(
        self, tracks: list[TrackEntry], audio_duration: timedelta
    ) -> tuple[bool, list[str]]:
        """
        Validate timing consistency and identify issues.

        Args:
            tracks: List of track entries
            audio_duration: Total duration of audio file

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        if not tracks:
            return True, []

        # Check for overlaps
        for i in range(len(tracks) - 1):
            current = tracks[i]
            next_track = tracks[i + 1]

            if current.end_time and next_track.start_time and current.end_time > next_track.start_time:
                overlap = current.end_time - next_track.start_time
                issues.append(
                    f"Track {current.position} overlaps with track {next_track.position} "
                    f"by {overlap.total_seconds():.1f} seconds"
                )

        # Check if tracks exceed audio duration
        last_track = tracks[-1]
        if last_track.end_time:
            if last_track.end_time > audio_duration:
                excess = last_track.end_time - audio_duration
                issues.append(f"Last track ends {excess.total_seconds():.1f} seconds after audio duration")
        elif last_track.start_time > audio_duration:
            excess = last_track.start_time - audio_duration
            issues.append(f"Last track starts {excess.total_seconds():.1f} seconds after audio duration")

        # Check for negative timestamps
        for track in tracks:
            if track.start_time < timedelta(0):
                issues.append(f"Track {track.position} has negative start time")
            if track.end_time and track.end_time < timedelta(0):
                issues.append(f"Track {track.position} has negative end time")

        # Check for very short tracks
        for track in tracks:
            if track.end_time:
                duration = track.end_time - track.start_time
                if duration < self.min_track_duration:
                    issues.append(f"Track {track.position} is too short ({duration.total_seconds():.1f} seconds)")

        return len(issues) == 0, issues

    def _apply_offset(self, tracks: list[TrackEntry], offset: timedelta) -> list[TrackEntry]:
        """
        Apply offset to all track timestamps.

        Args:
            tracks: List of track entries
            offset: Offset to apply

        Returns:
            Tracks with adjusted timestamps
        """
        for track in tracks:
            track.start_time = track.start_time + offset
            if track.end_time:
                track.end_time = track.end_time + offset

        return tracks

    def _fix_timing_gaps(self, tracks: list[TrackEntry]) -> list[TrackEntry]:
        """
        Fix gaps and overlaps in track timings.

        Args:
            tracks: List of track entries

        Returns:
            Tracks with fixed timings
        """
        if len(tracks) < 2:
            return tracks

        for i in range(len(tracks) - 1):
            current = tracks[i]
            next_track = tracks[i + 1]

            # If current track has no end time, set it to next track's start
            if not current.end_time and next_track.start_time:
                current.end_time = next_track.start_time

            # Fix overlaps
            elif current.end_time and next_track.start_time and current.end_time > next_track.start_time:
                # Split the difference
                midpoint = current.start_time + (next_track.start_time - current.start_time) / 2
                current.end_time = midpoint
                next_track.start_time = midpoint

        return tracks

    def _ensure_within_duration(self, tracks: list[TrackEntry], audio_duration: timedelta) -> list[TrackEntry]:
        """
        Ensure all tracks are within audio duration.

        Args:
            tracks: List of track entries
            audio_duration: Total audio duration

        Returns:
            Tracks adjusted to fit within duration
        """
        if not tracks:
            return tracks

        # If last track extends beyond duration, truncate it
        last_track = tracks[-1]
        if last_track.end_time and last_track.end_time > audio_duration:
            last_track.end_time = audio_duration
        elif not last_track.end_time and last_track.start_time < audio_duration:
            # Only set end time if track starts before duration ends
            last_track.end_time = audio_duration

        # Check if all tracks need scaling
        if last_track.start_time > audio_duration:
            # Scale all timestamps proportionally
            scale_factor = audio_duration.total_seconds() / last_track.start_time.total_seconds()
            scale_factor = min(scale_factor, 1.0)  # Don't expand, only compress

            for track in tracks:
                track.start_time = timedelta(seconds=track.start_time.total_seconds() * scale_factor)
                if track.end_time:
                    track.end_time = timedelta(seconds=track.end_time.total_seconds() * scale_factor)

        return tracks

    def _validate_track_durations(self, tracks: list[TrackEntry]) -> list[TrackEntry]:
        """
        Validate and fix individual track durations.

        Args:
            tracks: List of track entries

        Returns:
            Tracks with validated durations
        """
        for i, track in enumerate(tracks):
            if track.end_time:
                duration = track.end_time - track.start_time

                # Fix very short tracks
                if duration < self.min_track_duration:
                    # Extend to minimum duration if possible
                    if i < len(tracks) - 1:
                        next_track = tracks[i + 1]
                        max_end = next_track.start_time
                        track.end_time = min(track.start_time + self.min_track_duration, max_end)

                # Fix very long tracks
                elif duration > self.max_track_duration:
                    track.end_time = track.start_time + self.max_track_duration

        return tracks

    # Additional methods for manual tracklist timing adjustment

    def detect_timing_conflicts(
        self,
        track: TrackEntry,
        other_tracks: list[TrackEntry],
    ) -> list[dict[str, Any]]:
        """
        Detect timing conflicts between a track and other tracks.

        Args:
            track: Track to check.
            other_tracks: Other tracks to check against.

        Returns:
            List of conflict dictionaries with details.
        """
        conflicts = []

        track_end = track.end_time if track.end_time else track.start_time + timedelta(seconds=1)

        for other in other_tracks:
            if other.position == track.position:
                continue

            other_end = other.end_time if other.end_time else other.start_time + timedelta(seconds=1)

            # Check for overlap: start1 < end2 AND start2 < end1
            if track.start_time < other_end and other.start_time < track_end:
                overlap_start = max(track.start_time, other.start_time)
                overlap_end = min(track_end, other_end)
                overlap_duration = (overlap_end - overlap_start).total_seconds()

                conflict = {
                    "type": "overlap",
                    "track_position": track.position,
                    "conflicting_position": other.position,
                    "overlap_start": overlap_start,
                    "overlap_end": overlap_end,
                    "overlap_duration": overlap_duration,
                    "severity": "high" if overlap_duration > 5 else "medium",
                }
                conflicts.append(conflict)

        return conflicts

    def detect_all_timing_conflicts(
        self,
        tracks: list[TrackEntry],
    ) -> list[tuple[TrackEntry, TrackEntry, str]]:
        """
        Detect all timing conflicts across a list of tracks.

        Args:
            tracks: List of tracks to check.

        Returns:
            List of tuples containing (track1, track2, reason).
        """
        conflicts = []

        for i, track1 in enumerate(tracks):
            track1_end = track1.end_time if track1.end_time else track1.start_time + timedelta(seconds=1)

            for track2 in tracks[i + 1 :]:
                track2_end = track2.end_time if track2.end_time else track2.start_time + timedelta(seconds=1)

                # Check for overlap
                if track1.start_time < track2_end and track2.start_time < track1_end:
                    overlap_start = max(track1.start_time, track2.start_time)
                    overlap_end = min(track1_end, track2_end)
                    overlap_duration = (overlap_end - overlap_start).total_seconds()

                    reason = f"Tracks overlap for {overlap_duration:.1f} seconds"
                    conflicts.append((track1, track2, reason))

        return conflicts

    def auto_calculate_end_times(
        self,
        tracks: list[TrackEntry],
        audio_duration: timedelta | None = None,
        default_gap: timedelta = timedelta(seconds=0),
    ) -> list[TrackEntry]:
        """
        Automatically calculate end times based on next track's start time.

        Args:
            tracks: List of tracks to process.
            audio_duration: Total audio duration (optional).
            default_gap: Default gap between tracks (default: 0 seconds).

        Returns:
            List of tracks with calculated end times.
        """
        if not tracks:
            return tracks

        # Sort tracks by position
        sorted_tracks = sorted(tracks, key=lambda t: t.position)

        for i in range(len(sorted_tracks)):
            track = sorted_tracks[i]

            # If track already has an end time, skip
            if track.end_time:
                continue

            # For all tracks except the last, set end time based on next track
            if i < len(sorted_tracks) - 1:
                next_track = sorted_tracks[i + 1]
                track.end_time = next_track.start_time - default_gap
            # For the last track, use audio duration if available
            elif audio_duration:
                track.end_time = audio_duration
            else:
                # Default to 3 minutes after start if no other info
                track.end_time = track.start_time + timedelta(minutes=3)

        return sorted_tracks

    def shift_tracks_after_position(
        self,
        tracks: list[TrackEntry],
        position: int,
        shift_amount: timedelta,
    ) -> list[TrackEntry]:
        """
        Shift all tracks after a given position by a specified amount.

        Args:
            tracks: List of all tracks.
            position: Position after which to shift tracks.
            shift_amount: Amount to shift by (can be negative).

        Returns:
            List of shifted tracks.
        """
        shifted_tracks = []

        for track in tracks:
            if track.position > position:
                track.start_time += shift_amount
                if track.end_time:
                    track.end_time += shift_amount
                shifted_tracks.append(track)

        return shifted_tracks

    def normalize_track_positions(
        self,
        tracks: list[TrackEntry],
    ) -> list[TrackEntry]:
        """
        Normalize track positions to ensure they are sequential starting from 1.

        Args:
            tracks: List of tracks to normalize.

        Returns:
            List of tracks with normalized positions.
        """
        if not tracks:
            return tracks

        # Sort by current position
        sorted_tracks = sorted(tracks, key=lambda t: t.position)

        # Reassign positions sequentially
        for i, track in enumerate(sorted_tracks, start=1):
            track.position = i

        return sorted_tracks

    def calculate_total_duration(
        self,
        tracks: list[TrackEntry],
    ) -> timedelta:
        """
        Calculate total duration of a tracklist.

        Args:
            tracks: List of tracks.

        Returns:
            Total duration.
        """
        if not tracks:
            return timedelta(0)

        # Find the last track by position
        last_track = max(tracks, key=lambda t: t.position)

        if last_track.end_time:
            return last_track.end_time  # type: ignore[no-any-return]  # TrackEntry.end_time returns timedelta but typed as Any
        # Estimate based on start time + default duration
        return last_track.start_time + timedelta(minutes=3)  # type: ignore[no-any-return]  # TrackEntry.start_time returns timedelta but typed as Any

    def suggest_timing_adjustments(
        self,
        tracks: list[TrackEntry],
        target_duration: timedelta | None = None,
    ) -> list[dict[str, Any]]:
        """
        Suggest timing adjustments to fix conflicts and improve flow.

        Args:
            tracks: List of tracks to analyze.
            target_duration: Target total duration (optional).

        Returns:
            List of adjustment suggestions.
        """
        suggestions: list[dict[str, Any]] = []

        if not tracks:
            return suggestions

        # Sort tracks by position
        sorted_tracks = sorted(tracks, key=lambda t: t.position)

        # Check for overlaps and suggest fixes
        for i in range(len(sorted_tracks) - 1):
            track = sorted_tracks[i]
            next_track = sorted_tracks[i + 1]

            track_end = track.end_time if track.end_time else track.start_time + timedelta(minutes=3)

            if track_end > next_track.start_time:
                # Overlap detected
                overlap = (track_end - next_track.start_time).total_seconds()

                suggestion = {
                    "type": "fix_overlap",
                    "track_position": track.position,
                    "next_position": next_track.position,
                    "current_overlap": overlap,
                    "suggested_action": "adjust_end_time",
                    "suggested_end": next_track.start_time - timedelta(seconds=0.5),
                    "priority": "high",
                }
                suggestions.append(suggestion)
            elif track_end < next_track.start_time - timedelta(seconds=10):
                # Large gap detected
                gap = (next_track.start_time - track_end).total_seconds()

                if gap > 10:
                    suggestion = {
                        "type": "large_gap",
                        "track_position": track.position,
                        "next_position": next_track.position,
                        "gap_duration": gap,
                        "suggested_action": "extend_or_shift",
                        "priority": "low",
                    }
                    suggestions.append(suggestion)

        # Check total duration if target provided
        if target_duration and sorted_tracks:
            last_track = sorted_tracks[-1]
            last_end = last_track.end_time if last_track.end_time else last_track.start_time + timedelta(minutes=3)

            if last_end > target_duration:
                excess = (last_end - target_duration).total_seconds()
                suggestion = {
                    "type": "exceeds_duration",
                    "current_duration": last_end.total_seconds(),
                    "target_duration": target_duration.total_seconds(),
                    "excess_duration": excess,
                    "suggested_action": "trim_or_remove_tracks",
                    "priority": "medium",
                }
                suggestions.append(suggestion)

        return suggestions

    def batch_validate_timings(
        self,
        tracklists: list[list[TrackEntry]],
        audio_durations: list[timedelta] | None = None,
    ) -> list[tuple[bool, list[str]]]:
        """Validate timings for multiple tracklists efficiently.

        Args:
            tracklists: List of track lists to validate.
            audio_durations: Optional list of corresponding audio durations.

        Returns:
            List of validation results (is_valid, issues) for each tracklist.
        """
        results = []
        durations: list[timedelta | None] = list(audio_durations) if audio_durations is not None else []
        # Extend with None values if needed
        while len(durations) < len(tracklists):
            durations.append(None)

        for tracks, duration in zip(tracklists, durations, strict=False):
            # Use cached validation if available
            cache_key = f"validate_{len(tracks)}_{duration}"
            if cache_key in self._calculation_cache:
                results.append(self._calculation_cache[cache_key])
            else:
                if duration:
                    result = self.validate_timing_consistency(tracks, duration)
                else:
                    # Basic validation without duration
                    issues = []
                    for i in range(len(tracks) - 1):
                        end_time = tracks[i].end_time
                        start_time = tracks[i + 1].start_time
                        if end_time is not None and start_time is not None and end_time > start_time:
                            issues.append(f"Track {i + 1} overlaps with track {i + 2}")
                    result = (len(issues) == 0, issues)

                self._calculation_cache[cache_key] = result
                results.append(result)

        return results

    def optimize_timing_layout(
        self,
        tracks: list[TrackEntry],
        target_duration: timedelta,
    ) -> list[TrackEntry]:
        """Optimize track timing layout for smooth transitions.

        Args:
            tracks: List of tracks to optimize.
            target_duration: Target total duration.

        Returns:
            Optimized track list with adjusted timings.
        """
        if not tracks:
            return tracks

        # Calculate optimal track duration
        avg_duration = target_duration / len(tracks)

        # Adjust each track proportionally
        optimized = []
        current_time = timedelta(0)

        for i, track in enumerate(tracks):
            # Set start time
            track.start_time = current_time

            # Calculate end time based on optimal duration
            if i < len(tracks) - 1:
                # Not the last track
                track.end_time = current_time + avg_duration
                current_time = track.end_time
            else:
                # Last track fills remaining time
                track.end_time = target_duration

            optimized.append(track)

        return optimized
