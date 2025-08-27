"""
Timing adjustment service for aligning tracklist timestamps.

This service handles timing adjustments and validations for
track timestamps when importing from 1001tracklists.
"""

import logging
from datetime import timedelta
from typing import List, Optional, Tuple

from ..models.tracklist import TrackEntry

logger = logging.getLogger(__name__)


class TimingService:
    """Service for adjusting and validating track timings."""
    
    def __init__(self) -> None:
        """Initialize the timing service."""
        self.min_track_duration = timedelta(seconds=30)  # Minimum 30 seconds per track
        self.max_track_duration = timedelta(minutes=20)  # Maximum 20 minutes per track
    
    def adjust_track_timings(
        self,
        tracks: List[TrackEntry],
        audio_duration: Optional[timedelta] = None,
        offset: Optional[timedelta] = None
    ) -> List[TrackEntry]:
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
        
        tracks = self._validate_track_durations(tracks)
        
        return tracks
    
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
        if not timing_str:
            return timedelta(0)
        
        timing_str = timing_str.strip()
        
        # Try decimal minutes format
        if '.' in timing_str and ':' not in timing_str:
            try:
                minutes = float(timing_str)
                return timedelta(minutes=minutes)
            except ValueError:
                pass
        
        # Try time format (MM:SS or HH:MM:SS)
        if ':' in timing_str:
            parts = timing_str.split(':')
            
            try:
                if len(parts) == 2:
                    # MM:SS or M:SS format
                    minutes = int(parts[0])
                    seconds = int(parts[1])
                    return timedelta(minutes=minutes, seconds=seconds)
                elif len(parts) == 3:
                    # HH:MM:SS or H:MM:SS format
                    hours = int(parts[0])
                    minutes = int(parts[1])
                    seconds = int(parts[2])
                    return timedelta(hours=hours, minutes=minutes, seconds=seconds)
            except ValueError:
                logger.warning(f"Failed to parse timing: {timing_str}")
        
        return timedelta(0)
    
    def calculate_offset_from_start(
        self,
        first_track_time: timedelta,
        mix_start_time: timedelta = timedelta(0)
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
        self,
        tracks: List[TrackEntry],
        audio_duration: timedelta
    ) -> Tuple[bool, List[str]]:
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
            
            if current.end_time and next_track.start_time:
                if current.end_time > next_track.start_time:
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
                issues.append(
                    f"Last track ends {excess.total_seconds():.1f} seconds "
                    f"after audio duration"
                )
        elif last_track.start_time > audio_duration:
            excess = last_track.start_time - audio_duration
            issues.append(
                f"Last track starts {excess.total_seconds():.1f} seconds "
                f"after audio duration"
            )
        
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
                    issues.append(
                        f"Track {track.position} is too short "
                        f"({duration.total_seconds():.1f} seconds)"
                    )
        
        return len(issues) == 0, issues
    
    def _apply_offset(
        self,
        tracks: List[TrackEntry],
        offset: timedelta
    ) -> List[TrackEntry]:
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
    
    def _fix_timing_gaps(self, tracks: List[TrackEntry]) -> List[TrackEntry]:
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
            elif current.end_time and next_track.start_time:
                if current.end_time > next_track.start_time:
                    # Split the difference
                    midpoint = current.start_time + (next_track.start_time - current.start_time) / 2
                    current.end_time = midpoint
                    next_track.start_time = midpoint
        
        return tracks
    
    def _ensure_within_duration(
        self,
        tracks: List[TrackEntry],
        audio_duration: timedelta
    ) -> List[TrackEntry]:
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
        elif not last_track.end_time:
            # Only set end time if track starts before duration ends
            if last_track.start_time < audio_duration:
                last_track.end_time = audio_duration
        
        # Check if all tracks need scaling
        if last_track.start_time > audio_duration:
            # Scale all timestamps proportionally
            scale_factor = audio_duration.total_seconds() / last_track.start_time.total_seconds()
            scale_factor = min(scale_factor, 1.0)  # Don't expand, only compress
            
            for track in tracks:
                track.start_time = timedelta(
                    seconds=track.start_time.total_seconds() * scale_factor
                )
                if track.end_time:
                    track.end_time = timedelta(
                        seconds=track.end_time.total_seconds() * scale_factor
                    )
        
        return tracks
    
    def _validate_track_durations(
        self,
        tracks: List[TrackEntry]
    ) -> List[TrackEntry]:
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
                        track.end_time = min(
                            track.start_time + self.min_track_duration,
                            max_end
                        )
                
                # Fix very long tracks
                elif duration > self.max_track_duration:
                    track.end_time = track.start_time + self.max_track_duration
        
        return tracks