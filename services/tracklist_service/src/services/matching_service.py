"""
Matching service for correlating tracklists with audio files.

This service provides algorithms for matching 1001tracklists data
with local audio files using metadata and fuzzy matching.
"""

import logging
from difflib import SequenceMatcher
from typing import Dict, Optional, Tuple
from uuid import UUID
from dataclasses import dataclass

from ..models.tracklist import Tracklist
from ..models.tracklist_models import Tracklist as ScrapedTracklist

logger = logging.getLogger(__name__)


@dataclass
class MatchingResult:
    """Result from matching a tracklist with an audio file."""

    confidence_score: float
    metadata: Optional[Dict] = None


class MatchingService:
    """Service for matching tracklists with audio files."""

    def __init__(self) -> None:
        """Initialize the matching service."""
        self.confidence_weights = {"title": 0.3, "artist": 0.25, "duration": 0.25, "date": 0.1, "event": 0.1}

    def match_tracklist_to_audio(self, scraped_tracklist: ScrapedTracklist, audio_metadata: Dict) -> Tuple[float, Dict]:
        """
        Match a scraped tracklist to audio file metadata.

        Args:
            scraped_tracklist: Scraped tracklist from 1001tracklists
            audio_metadata: Metadata from the audio file
                Expected keys: title, artist, duration_seconds, date, album

        Returns:
            Tuple of (confidence_score, match_details)
        """
        match_details = {}
        scores = []

        # Match title
        if "title" in audio_metadata and audio_metadata["title"]:
            title_score = self._fuzzy_match(
                self._normalize_title(scraped_tracklist, audio_metadata), audio_metadata["title"]
            )
            match_details["title_score"] = title_score
            scores.append(("title", title_score))

        # Match artist
        if "artist" in audio_metadata and audio_metadata["artist"]:
            artist_score = self._fuzzy_match(scraped_tracklist.dj_name, audio_metadata["artist"])
            match_details["artist_score"] = artist_score
            scores.append(("artist", artist_score))

        # Match duration
        if "duration_seconds" in audio_metadata:
            duration_score = self._match_duration(scraped_tracklist, audio_metadata["duration_seconds"])
            match_details["duration_score"] = duration_score
            scores.append(("duration", duration_score))

        # Match date if available
        if "date" in audio_metadata and scraped_tracklist.date:
            date_score = self._match_date(scraped_tracklist.date, audio_metadata["date"])
            match_details["date_score"] = date_score
            scores.append(("date", date_score))

        # Match event/album
        if "album" in audio_metadata and scraped_tracklist.event_name:
            event_score = self._fuzzy_match(scraped_tracklist.event_name, audio_metadata["album"])
            match_details["event_score"] = event_score
            scores.append(("event", event_score))

        # Calculate weighted confidence
        confidence = self._calculate_weighted_confidence(scores)
        match_details["overall_confidence"] = confidence

        return confidence, match_details

    def validate_audio_file(self, audio_file_path: str, expected_duration: Optional[int] = None) -> bool:
        """
        Validate that an audio file exists and is accessible.

        Args:
            audio_file_path: Path to the audio file
            expected_duration: Expected duration in seconds (optional)

        Returns:
            True if file is valid, False otherwise
        """
        import os

        # Check file exists
        if not os.path.exists(audio_file_path):
            logger.error(f"Audio file not found: {audio_file_path}")
            return False

        # Check file is readable
        if not os.access(audio_file_path, os.R_OK):
            logger.error(f"Audio file not readable: {audio_file_path}")
            return False

        # Check file size is reasonable (>1MB for audio)
        file_size = os.path.getsize(audio_file_path)
        if file_size < 1_000_000:  # 1MB minimum
            logger.warning(f"Audio file suspiciously small: {file_size} bytes")
            return False

        # If expected duration provided, could validate with audio library
        # For now, just log it
        if expected_duration:
            logger.debug(f"Expected duration: {expected_duration} seconds")

        return True

    def _fuzzy_match(self, str1: str, str2: str) -> float:
        """
        Perform fuzzy string matching.

        Args:
            str1: First string
            str2: Second string

        Returns:
            Similarity score between 0 and 1
        """
        if not str1 or not str2:
            return 0.0

        # Normalize strings for comparison
        str1_normalized = self._normalize_string(str1)
        str2_normalized = self._normalize_string(str2)

        # Use SequenceMatcher for fuzzy matching
        matcher = SequenceMatcher(None, str1_normalized, str2_normalized)
        score = matcher.ratio()

        # Check for substring matches (partial credit)
        if score < 0.8:
            if str1_normalized in str2_normalized or str2_normalized in str1_normalized:
                score = max(score, 0.7)

        return score

    def _normalize_string(self, s: str) -> str:
        """
        Normalize a string for matching.

        Args:
            s: String to normalize

        Returns:
            Normalized string
        """
        if not s:
            return ""

        # Convert to lowercase
        s = s.lower()

        # Remove common DJ mix indicators
        for pattern in [" @ ", " at ", " - ", " live ", " set ", " mix "]:
            s = s.replace(pattern, " ")

        # Remove special characters
        import re

        s = re.sub(r"[^\w\s]", " ", s)

        # Remove extra whitespace
        s = " ".join(s.split())

        return s

    def _normalize_title(self, scraped_tracklist: ScrapedTracklist, audio_metadata: Dict) -> str:
        """
        Create a normalized title from scraped tracklist.

        Args:
            scraped_tracklist: Scraped tracklist data
            audio_metadata: Audio file metadata

        Returns:
            Normalized title string
        """
        parts = []

        if scraped_tracklist.dj_name:
            parts.append(scraped_tracklist.dj_name)

        if scraped_tracklist.event_name:
            parts.append(scraped_tracklist.event_name)

        if scraped_tracklist.venue:
            parts.append(scraped_tracklist.venue)

        if scraped_tracklist.date:
            parts.append(str(scraped_tracklist.date))

        return " ".join(parts)

    def _match_duration(self, scraped_tracklist: ScrapedTracklist, audio_duration_seconds: int) -> float:
        """
        Match duration between tracklist and audio file.

        Args:
            scraped_tracklist: Scraped tracklist
            audio_duration_seconds: Audio file duration in seconds

        Returns:
            Duration match score between 0 and 1
        """
        # Get tracklist duration
        tracklist_duration: Optional[float] = None

        if scraped_tracklist.metadata and scraped_tracklist.metadata.duration_minutes:
            tracklist_duration = scraped_tracklist.metadata.duration_minutes * 60
        elif scraped_tracklist.tracks:
            # Estimate from last track timestamp
            last_track = scraped_tracklist.tracks[-1]
            if last_track.timestamp and hasattr(last_track.timestamp, "timestamp_ms"):
                tracklist_duration = last_track.timestamp.timestamp_ms / 1000

        if not tracklist_duration:
            return 0.5  # No duration info, neutral score

        # Calculate difference percentage
        diff_percent = abs(tracklist_duration - audio_duration_seconds) / audio_duration_seconds

        # Convert to score (0% diff = 1.0 score, 20% diff = 0.0 score)
        if diff_percent <= 0.02:  # Within 2%
            return 1.0
        elif diff_percent <= 0.05:  # Within 5%
            return 0.9
        elif diff_percent <= 0.10:  # Within 10%
            return 0.7
        elif diff_percent <= 0.20:  # Within 20%
            return 0.5
        else:
            return max(0.0, 1.0 - diff_percent)

    def _match_date(self, tracklist_date: Optional[any], audio_date: Optional[any]) -> float:
        """
        Match dates between tracklist and audio metadata.

        Args:
            tracklist_date: Date from tracklist
            audio_date: Date from audio metadata

        Returns:
            Date match score between 0 and 1
        """
        if not tracklist_date or not audio_date:
            return 0.5  # No date info, neutral score

        # Convert to comparable format
        from datetime import datetime

        if isinstance(tracklist_date, str):
            try:
                tracklist_date = datetime.fromisoformat(tracklist_date).date()
            except:
                return 0.5

        if isinstance(audio_date, str):
            try:
                audio_date = datetime.fromisoformat(audio_date).date()
            except:
                return 0.5

        # Calculate day difference
        if hasattr(tracklist_date, "date"):
            tracklist_date = tracklist_date.date()
        if hasattr(audio_date, "date"):
            audio_date = audio_date.date()

        day_diff = abs((tracklist_date - audio_date).days)

        # Score based on proximity
        if day_diff == 0:
            return 1.0
        elif day_diff <= 1:
            return 0.9
        elif day_diff <= 7:
            return 0.7
        elif day_diff <= 30:
            return 0.5
        elif day_diff <= 365:
            return 0.3
        else:
            return 0.1

    def match_tracklist_with_audio_file(self, tracklist: "Tracklist", audio_file_id: UUID) -> MatchingResult:
        """
        Match a tracklist with an audio file by its ID.

        This is a wrapper method for API endpoints that provides a simpler interface.

        Args:
            tracklist: Tracklist object to match
            audio_file_id: UUID of the audio file

        Returns:
            MatchingResult object with confidence score and metadata
        """

        # Get audio metadata (in production this would query the audio service)
        # For now, simulate metadata retrieval
        audio_metadata = self._get_audio_metadata(audio_file_id)

        # Convert Tracklist to ScrapedTracklist for matching
        scraped_tracklist = self._convert_to_scraped_tracklist(tracklist)

        # Perform matching
        confidence, details = self.match_tracklist_to_audio(scraped_tracklist, audio_metadata)

        return MatchingResult(confidence_score=confidence, metadata=audio_metadata)

    def _get_audio_metadata(self, audio_file_id: UUID) -> Dict:
        """
        Get audio file metadata.

        In production, this would query the audio service.
        For now, returns simulated metadata.
        """
        # TODO: Replace with actual audio service call
        return {
            "title": f"Audio file {audio_file_id}",
            "artist": "Unknown Artist",
            "duration_seconds": 3600,  # 1 hour default
            "date": None,
            "album": None,
        }

    def _convert_to_scraped_tracklist(self, tracklist: "Tracklist") -> ScrapedTracklist:
        """
        Convert a Tracklist to ScrapedTracklist format.
        """
        from ..models.tracklist_models import TracklistMetadata, Track, CuePoint

        # Convert tracks
        tracks = []
        for track_entry in tracklist.tracks:
            # Convert timedelta to CuePoint
            cue_point = None
            if track_entry.start_time:
                total_seconds = int(track_entry.start_time.total_seconds())
                cue_point = CuePoint(
                    track_number=track_entry.position,
                    timestamp_ms=total_seconds * 1000,
                    formatted_time=f"{total_seconds // 60:02d}:{total_seconds % 60:02d}",
                )

            track = Track(
                number=track_entry.position,
                timestamp=cue_point,
                artist=track_entry.artist,
                title=track_entry.title,
                remix=track_entry.remix,
                label=track_entry.label,
            )
            tracks.append(track)

        # Create metadata with valid fields
        metadata = TracklistMetadata(
            duration_minutes=None, play_count=None, favorite_count=None, comment_count=None, tags=[]
        )

        return ScrapedTracklist(
            url="",
            dj_name=tracklist.tracks[0].artist if tracklist.tracks else "Unknown DJ",
            date=None,
            event_name=None,
            tracks=tracks,
            metadata=metadata,
        )

    def _calculate_weighted_confidence(self, scores: list) -> float:
        """
        Calculate weighted confidence score.

        Args:
            scores: List of (category, score) tuples

        Returns:
            Weighted confidence between 0 and 1
        """
        if not scores:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0

        for category, score in scores:
            weight = self.confidence_weights.get(category, 0.1)
            weighted_sum += score * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return min(1.0, weighted_sum / total_weight)
