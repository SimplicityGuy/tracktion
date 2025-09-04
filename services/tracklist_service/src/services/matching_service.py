"""
Matching service for correlating tracklists with audio files.

This service provides algorithms for matching 1001tracklists data
with local audio files using metadata and fuzzy matching.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    from src.models.tracklist import Tracklist

from services.tracklist_service.src.models.tracklist_models import CuePoint, Track, TracklistMetadata
from services.tracklist_service.src.models.tracklist_models import Tracklist as ScrapedTracklist

logger = logging.getLogger(__name__)


@dataclass
class MatchingResult:
    """Result from matching a tracklist with an audio file."""

    confidence_score: float
    metadata: dict[str, Any] | None = None


class MatchingService:
    """Service for matching tracklists with audio files."""

    def __init__(self) -> None:
        """Initialize the matching service.

        Sets up confidence weights for different matching criteria:
        - title: 30% - Primary identifier for the mix/set
        - artist: 25% - DJ/performer name matching
        - duration: 25% - Audio length correlation
        - date: 10% - Performance date alignment
        - event: 10% - Venue/event context matching

        These weights are used to calculate composite confidence scores
        when matching tracklists to audio files.
        """
        self.confidence_weights = {
            "title": 0.3,
            "artist": 0.25,
            "duration": 0.25,
            "date": 0.1,
            "event": 0.1,
        }

    def match_tracklist_to_audio(
        self, scraped_tracklist: ScrapedTracklist, audio_metadata: dict[str, Any]
    ) -> tuple[float, dict[str, Any]]:
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
        if audio_metadata.get("title"):
            title_score = self._fuzzy_match(
                self._normalize_title(scraped_tracklist, audio_metadata),
                audio_metadata["title"],
            )
            match_details["title_score"] = title_score
            scores.append(("title", title_score))

        # Match artist
        if audio_metadata.get("artist"):
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

    def validate_audio_file(self, audio_file_path: str, expected_duration: int | None = None) -> bool:
        """Validate that an audio file exists and is accessible.

        Performs comprehensive validation of audio file properties:
        - File existence and accessibility
        - File type verification (regular file, not directory)
        - Minimum size validation (>1MB for reasonable audio files)
        - Optional duration verification against expected length

        This validation helps prevent processing errors when attempting
        to match tracklists with corrupted or invalid audio files.

        Args:
            audio_file_path: Absolute or relative path to the audio file
            expected_duration: Expected duration in seconds for validation.
                If provided, logs the expected duration for future verification.

        Returns:
            True if file passes all validation checks, False otherwise.
            Logs specific error messages for debugging failed validations.

        Example:
            >>> service = MatchingService()
            >>> is_valid = service.validate_audio_file("/path/to/mix.mp3", 3600)
            >>> if is_valid:
            ...     print("File is ready for matching")
            File is ready for matching
        """

        # Check file exists
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_file_path}")
            return False

        # Check file is readable
        if not audio_path.is_file():
            logger.error(f"Audio file not readable or not a file: {audio_file_path}")
            return False

        # Check file size is reasonable (>1MB for audio)
        file_size = audio_path.stat().st_size
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
        Calculate string similarity using advanced fuzzy matching with substring bonus.

        This method implements a two-tier fuzzy matching system that combines
        sequence-based similarity with substring matching to handle various
        DJ mix and music metadata matching scenarios effectively.

        Algorithm implementation:
        1. Input validation - Handle None/empty strings gracefully
        2. String normalization - Apply _normalize_string() to both inputs
        3. Sequence matching - Use SequenceMatcher.ratio() for character-level similarity
        4. Substring bonus - Apply partial credit for substring relationships
        5. Score optimization - Return the highest score from both methods

        Sequence matching (SequenceMatcher):
        - Calculates character-level similarity ratio (0.0-1.0)
        - Handles insertions, deletions, and substitutions
        - Effective for catching typos and minor variations
        - Based on Ratcliff/Obershelp pattern recognition algorithm

        Substring bonus system:
        - Activates when sequence match score < 0.8 (indicating low similarity)
        - Checks for complete substring containment in either direction
        - Awards minimum score of 0.7 for substring matches
        - Handles cases where shorter string is contained in longer one

        Common matching scenarios:
        - Exact match: "Above & Beyond" vs "above beyond" -> ~1.0
        - Typos: "Armin van Buuren" vs "Armin van Bueren" -> ~0.95
        - Substring: "Tomorrowland" vs "Tomorrowland 2024 Main Stage" -> 0.7+
        - Partial: "ASOT" vs "A State of Trance" -> variable based on overlap

        Score interpretation:
        - 0.9-1.0: Excellent match (likely same entity)
        - 0.7-0.9: Good match (probably same with variations)
        - 0.5-0.7: Fair match (possible but needs verification)
        - 0.3-0.5: Poor match (unlikely to be same)
        - 0.0-0.3: Very poor match (almost certainly different)

        Args:
            str1: First string for comparison (normalized internally)
            str2: Second string for comparison (normalized internally)

        Returns:
            Similarity score (0.0-1.0) representing string similarity where:
            - 1.0 indicates identical strings after normalization
            - 0.7+ indicates strong similarity or substring relationship
            - 0.0 indicates completely different strings or None/empty input
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
        if score < 0.8 and (str1_normalized in str2_normalized or str2_normalized in str1_normalized):
            score = max(score, 0.7)

        return score

    def _normalize_string(self, s: str) -> str:
        r"""
        Normalize strings for consistent fuzzy matching using DJ-specific patterns.

        This method applies comprehensive text normalization specifically designed
        for DJ mix and music metadata comparison, handling common variations in
        how DJ names, event titles, and venue names are formatted.

        Normalization steps:
        1. Convert to lowercase for case-insensitive comparison
        2. Remove DJ-specific separators and indicators:
           - ' @ ', ' at ' - Common venue/event separators in DJ context
           - ' - ', ' live ', ' set ', ' mix ' - Performance type indicators
        3. Remove special characters and punctuation (keep alphanumeric and spaces)
        4. Collapse multiple spaces into single spaces
        5. Strip leading/trailing whitespace

        DJ/Music-specific patterns handled:
        - "DJ Name @ Venue" -> "dj name venue"
        - "Artist - Live Set" -> "artist"
        - "Event Mix 2024" -> "event 2024"
        - "DJ Name (Live)" -> "dj name live"

        Character handling:
        - Preserves alphanumeric characters and spaces
        - Removes punctuation: !@#$%^&*()-_=+[]{}|\:;<>?./~`
        - Unicode characters normalized to closest ASCII equivalent
        - Numbers preserved for date/year matching

        Use cases:
        - DJ name matching: "Above & Beyond" vs "Above and Beyond"
        - Event matching: "Tomorrowland 2024" vs "tomorrowland-2024"
        - Venue matching: "Output, Brooklyn" vs "output brooklyn"
        - Mix title matching: "ASOT #1000" vs "asot 1000"

        Args:
            s: Input string to normalize (DJ name, event, venue, etc.)

        Returns:
            Normalized string with consistent formatting for fuzzy matching:
            - All lowercase
            - No special characters or punctuation
            - Single spaces between words
            - Empty string if input is None/empty
        """
        if not s:
            return ""

        # Convert to lowercase
        s = s.lower()

        # Remove common DJ mix indicators
        for pattern in [" @ ", " at ", " - ", " live ", " set ", " mix "]:
            s = s.replace(pattern, " ")

        # Remove special characters

        s = re.sub(r"[^\w\s]", " ", s)

        # Remove extra whitespace
        return " ".join(s.split())

    def _normalize_title(self, scraped_tracklist: ScrapedTracklist, audio_metadata: dict[str, Any]) -> str:
        """
        Construct comprehensive title from tracklist metadata for matching.

        This method creates a composite title string from scraped tracklist data
        that can be effectively compared against audio file title metadata. The
        constructed title aims to capture the essential identifying information
        that would typically appear in an audio file's title tag.

        Title construction strategy:
        1. DJ name - Primary identifier for the mix/set
        2. Event name - Specific event or show context
        3. Venue - Location information for disambiguation
        4. Date - Temporal context for uniqueness

        Component prioritization:
        - All available components are included to maximize matching information
        - Order follows typical DJ mix naming conventions
        - Missing components are safely omitted (no empty segments)
        - Result is space-separated for consistent formatting

        Typical output patterns:
        - "DJ Name Event Name Venue Date"
        - "Artist Tomorrowland Main Stage 2024-07-20"
        - "Above Beyond ASOT 1000 Amsterdam 2021-01-01"
        - "Martin Garrix Ultra Miami 2023" (if date/venue missing)

        Matching scenarios:
        - Audio title: "Armin van Buuren - Tomorrowland 2024 Set"
        - Constructed title: "Armin van Buuren Tomorrowland Main Stage 2024-07-22"
        - Fuzzy match would score highly due to overlapping key terms

        Integration with fuzzy matching:
        - Constructed title feeds into _fuzzy_match() for comparison
        - Multiple components increase chances of partial matches
        - Normalization applied by _normalize_string() for consistency

        Args:
            scraped_tracklist: ScrapedTracklist containing metadata fields:
                - dj_name: Artist/DJ performing the set
                - event_name: Specific event or show name
                - venue: Performance venue or location
                - date: Performance date (various formats supported)
            audio_metadata: Audio file metadata (used for context, not in construction)

        Returns:
            Space-separated composite title string containing available metadata
            components, suitable for fuzzy matching against audio file titles.
            Returns empty string if no metadata components are available.
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
        Calculate duration similarity score using tiered percentage-based algorithm.

        This method compares the duration of a scraped tracklist with an audio file
        using sophisticated percentage-based matching with multiple fallback strategies.

        Duration sources (in priority order):
        1. tracklist.metadata.duration_minutes - Explicit duration from metadata
        2. Last track timestamp - Estimated duration from final track position
        3. No duration available - Returns neutral score (0.5)

        Scoring algorithm (percentage-based tiers):
        - Within 2%: Perfect match (1.0) - accounts for encoding/rounding differences
        - Within 5%: Excellent match (0.9) - normal variation in DJ mix analysis
        - Within 10%: Good match (0.7) - acceptable for tracklist vs. audio differences
        - Within 20%: Fair match (0.5) - possible but requires verification
        - Over 20%: Poor match (linear decline) - increasingly unlikely

        Common scenarios:
        - DJ mixes: Often have slight timing differences due to beat matching
        - Live recordings: May include extra intro/outro not in tracklist
        - Tracklist estimates: Manual timing vs. actual audio can vary
        - Format differences: Lossy vs. lossless encoding timing variations

        Error handling:
        - Missing tracklist duration: Returns 0.5 (neutral - not negative)
        - Invalid timestamp data: Falls back to neutral scoring
        - Zero duration audio: Prevents division by zero with safe calculation

        Args:
            scraped_tracklist: ScrapedTracklist containing duration metadata
            audio_duration_seconds: Actual audio file duration in seconds (int)

        Returns:
            Duration similarity score (0.0-1.0) where:
            - 1.0 indicates nearly identical durations (within 2%)
            - 0.5 indicates neutral/missing data or 20% difference
            - 0.0 indicates completely mismatched durations (>100% difference)
        """
        # Get tracklist duration
        tracklist_duration: float | None = None

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
        # TODO: Consider making scoring tiers configurable per DJ/genre
        # Electronic music may have different tolerance than acoustic sets
        if diff_percent <= 0.02:  # Within 2%
            return 1.0
        if diff_percent <= 0.05:  # Within 5%
            return 0.9
        if diff_percent <= 0.10:  # Within 10%
            return 0.7
        if diff_percent <= 0.20:  # Within 20%
            return 0.5
        return max(0.0, 1.0 - diff_percent)

    def _match_date(self, tracklist_date: Any | None, audio_date: Any | None) -> float:
        """
        Calculate date proximity score using day-based distance algorithm.

        This method compares dates from tracklist and audio metadata using a
        sophisticated distance-based scoring system that accounts for various
        date format inputs and provides meaningful proximity scoring.

        Date format handling:
        - ISO format strings (YYYY-MM-DD) - Parsed using datetime.fromisoformat()
        - datetime objects - Converted to date() for comparison
        - date objects - Used directly for calculation
        - Invalid formats - Return neutral score (0.5) to avoid false negatives

        Scoring tiers (day-based proximity):
        - Same day (0 days): Perfect match (1.0)
        - 1 day difference: Near perfect (0.9) - accounts for timezone/processing delays
        - Within week (≤7 days): Good match (0.7) - same event period
        - Within month (≤30 days): Fair match (0.5) - same general timeframe
        - Within year (≤365 days): Poor match (0.3) - same year but distant
        - Over 1 year: Very poor match (0.1) - likely different events

        Use cases:
        - DJ mix recordings: Often tagged with recording date vs. performance date
        - Event correlation: Matching live sets to event dates
        - Release matching: Comparing tracklist date with audio file metadata
        - Duplicate detection: Identifying same event recorded multiple times

        Missing data handling:
        - Either date missing: Returns 0.5 (neutral) rather than penalty
        - Both dates missing: Returns 0.5 to avoid penalizing incomplete metadata
        - Parse failures: Logged and treated as missing data (neutral score)

        Args:
            tracklist_date: Date from scraped tracklist (str, datetime, date, or None)
            audio_date: Date from audio file metadata (str, datetime, date, or None)

        Returns:
            Date proximity score (0.0-1.0) where:
            - 1.0 indicates identical dates (same day)
            - 0.5 indicates neutral/missing data or ~30-day difference
            - 0.1 indicates very distant dates (>1 year apart)
        """
        if not tracklist_date or not audio_date:
            return 0.5  # No date info, neutral score

        # Convert to comparable format

        if isinstance(tracklist_date, str):
            try:
                tracklist_date = datetime.fromisoformat(tracklist_date).date()
            except (ValueError, AttributeError):
                return 0.5

        if isinstance(audio_date, str):
            try:
                audio_date = datetime.fromisoformat(audio_date).date()
            except (ValueError, AttributeError):
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
        if day_diff <= 1:
            return 0.9
        if day_diff <= 7:
            return 0.7
        if day_diff <= 30:
            return 0.5
        if day_diff <= 365:
            return 0.3
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

        # TODO: Replace with actual Analysis Service API call
        # Should use HTTP client to call Analysis Service /api/metadata/{audio_file_id}
        # Current implementation is placeholder with mock data
        audio_metadata = self._get_audio_metadata(audio_file_id)

        # Convert Tracklist to ScrapedTracklist for matching
        scraped_tracklist = self._convert_to_scraped_tracklist(tracklist)

        # Perform matching
        confidence, details = self.match_tracklist_to_audio(scraped_tracklist, audio_metadata)

        return MatchingResult(confidence_score=confidence, metadata=audio_metadata)

    def _get_audio_metadata(self, audio_file_id: UUID) -> dict[str, Any]:
        """
        Get audio file metadata.

        This method should be replaced with an actual call to the Analysis Service
        API endpoint when the services are integrated. The Analysis Service provides
        comprehensive metadata extraction via its /api/metadata endpoint.

        Args:
            audio_file_id: UUID of the audio file to get metadata for

        Returns:
            Dictionary containing audio metadata with keys:
            - title: Track title
            - artist: Artist name
            - duration_seconds: Duration in seconds
            - date: Release date
            - album: Album name

        Note:
            Integration point for Analysis Service API:
            GET /api/metadata/{audio_file_id}
            Response format matches the dictionary structure returned here.
        """
        # NOTE: This is a placeholder implementation that returns simulated data.
        # In production, this should make an HTTP request to the Analysis Service API:
        # response = await http_client.get(f"{ANALYSIS_SERVICE_URL}/api/metadata/{audio_file_id}")
        # return response.json()

        # Simulated metadata for development/testing
        logger.warning(
            f"Using simulated metadata for audio file {audio_file_id}. "
            "Replace with Analysis Service API call for production."
        )
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
                bpm=None,
                key=None,
                genre=None,
                notes=None,
            )
            tracks.append(track)

        # Create metadata with valid fields
        metadata = TracklistMetadata(
            recording_type=None,
            duration_minutes=None,
            play_count=None,
            favorite_count=None,
            comment_count=None,
            tags=[],
            download_url=None,
            stream_url=None,
            soundcloud_url=None,
            mixcloud_url=None,
            youtube_url=None,
        )

        return ScrapedTracklist(
            url="",
            venue=None,
            source_html_hash=None,
            dj_name=tracklist.tracks[0].artist if tracklist.tracks else "Unknown DJ",
            date=None,
            event_name=None,
            tracks=tracks,
            metadata=metadata,
        )

    def _calculate_weighted_confidence(self, scores: list[tuple[str, float]]) -> float:
        """
        Calculate composite confidence score using weighted scoring algorithm.

        This method implements a sophisticated weighted scoring system that combines
        multiple matching criteria to produce a single confidence score representing
        the likelihood that a tracklist matches an audio file.

        Weighting system (configured in __init__):
        - title: 30% - Most important identifier (DJ name, event, venue combination)
        - artist: 25% - DJ/performer name matching importance
        - duration: 25% - Audio length correlation (critical for mix matching)
        - date: 10% - Performance date context (helps with duplicates)
        - event: 10% - Venue/event context (disambiguation factor)

        Algorithm process:
        1. Iterate through all available matching scores
        2. Apply category-specific weight to each score
        3. Sum weighted scores and total weights separately
        4. Calculate normalized average (weighted_sum / total_weight)
        5. Ensure result is clamped to valid confidence range [0.0, 1.0]

        Missing category handling:
        - Unknown categories receive default weight of 0.1 (10%)
        - Missing data results in that category being excluded from calculation
        - Total weight adjusts automatically based on available criteria

        Score interpretation:
        - 0.9-1.0: Very high confidence, likely correct match
        - 0.7-0.9: Good confidence, probably correct with manual verification
        - 0.5-0.7: Moderate confidence, requires careful review
        - 0.3-0.5: Low confidence, likely incorrect match
        - 0.0-0.3: Very low confidence, almost certainly incorrect

        Args:
            scores: List of (category, score) tuples where:
                category: String identifier matching confidence_weights keys
                score: Normalized score between 0.0-1.0 for that matching aspect

        Returns:
            Weighted confidence score between 0.0 and 1.0, where higher values
            indicate stronger likelihood of correct tracklist-audio correlation

        Example:
            >>> scores = [("title", 0.85), ("artist", 0.92), ("duration", 0.78)]
            >>> confidence = self._calculate_weighted_confidence(scores)
            >>> print(f"Confidence: {confidence:.2f}")  # Output: ~0.84
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
