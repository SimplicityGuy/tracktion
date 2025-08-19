"""Confidence scoring system for file rename proposals."""

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """Calculates confidence scores for rename proposals."""

    def __init__(self) -> None:
        """Initialize the confidence scorer."""
        self.logger = logger

        # Weight factors for different scoring components
        self.weights = {
            "metadata_completeness": 0.25,
            "pattern_match": 0.20,
            "filename_quality": 0.15,
            "conflict_absence": 0.15,
            "source_reliability": 0.15,
            "consistency": 0.10,
        }

    def calculate_confidence(
        self,
        metadata: Dict[str, Optional[str]],
        original_filename: str,
        proposed_filename: str,
        conflicts: List[str],
        warnings: List[str],
        pattern_used: Optional[str] = None,
        source: Optional[str] = None,
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate overall confidence score for a rename proposal.

        Args:
            metadata: Metadata dictionary with values
            original_filename: Original filename
            proposed_filename: Proposed new filename
            conflicts: List of detected conflicts
            warnings: List of warnings
            pattern_used: Pattern template used (if any)
            source: Source of metadata (e.g., "id3", "filename", "cue")

        Returns:
            Tuple of (overall_score, component_scores)
        """
        component_scores = {}

        # 1. Metadata completeness score
        component_scores["metadata_completeness"] = self._score_metadata_completeness(metadata)

        # 2. Pattern match score
        component_scores["pattern_match"] = self._score_pattern_match(metadata, proposed_filename, pattern_used)

        # 3. Filename quality score
        component_scores["filename_quality"] = self._score_filename_quality(original_filename, proposed_filename)

        # 4. Conflict absence score
        component_scores["conflict_absence"] = self._score_conflict_absence(conflicts, warnings)

        # 5. Source reliability score
        component_scores["source_reliability"] = self._score_source_reliability(source)

        # 6. Consistency score
        component_scores["consistency"] = self._score_consistency(metadata, proposed_filename)

        # Calculate weighted overall score
        overall_score = 0.0
        for component, score in component_scores.items():
            weight = self.weights.get(component, 0.0)
            overall_score += score * weight

        # Ensure score is between 0 and 1
        overall_score = max(0.0, min(1.0, overall_score))

        return overall_score, component_scores

    def _score_metadata_completeness(self, metadata: Dict[str, Optional[str]]) -> float:
        """Score based on how complete the metadata is.

        Args:
            metadata: Metadata dictionary

        Returns:
            Score between 0 and 1
        """
        # Essential fields for music files
        essential_fields = ["artist", "title"]
        important_fields = ["album", "date", "track"]
        optional_fields = ["genre", "albumartist", "discnumber"]

        score = 0.0
        max_score = 0.0

        # Essential fields are worth more
        for field in essential_fields:
            max_score += 2.0
            if metadata.get(field):
                score += 2.0

        # Important fields
        for field in important_fields:
            max_score += 1.5
            if metadata.get(field):
                score += 1.5

        # Optional fields
        for field in optional_fields:
            max_score += 0.5
            if metadata.get(field):
                score += 0.5

        if max_score == 0:
            return 0.0

        return score / max_score

    def _score_pattern_match(
        self, metadata: Dict[str, Optional[str]], proposed_filename: str, pattern_used: Optional[str]
    ) -> float:
        """Score based on how well the pattern matches the metadata.

        Args:
            metadata: Metadata dictionary
            proposed_filename: Proposed filename
            pattern_used: Pattern template used

        Returns:
            Score between 0 and 1
        """
        if not pattern_used:
            # No pattern used, check basic structure
            if metadata.get("artist") and metadata.get("title"):
                # Check if both artist and title appear in filename
                artist = metadata["artist"]
                title = metadata["title"]
                if artist and title:
                    artist_lower = artist.lower()
                    title_lower = title.lower()
                    filename_lower = proposed_filename.lower()

                    if artist_lower in filename_lower and title_lower in filename_lower:
                        return 0.8
                    elif artist_lower in filename_lower or title_lower in filename_lower:
                        return 0.5
            return 0.3

        # Pattern was used - check how many placeholders were filled
        placeholders = re.findall(r"\{(\w+)(?::\d+d?)?\}", pattern_used)
        filled_count = 0
        total_count = len(placeholders)

        for placeholder in placeholders:
            if metadata.get(placeholder):
                filled_count += 1

        if total_count == 0:
            return 1.0  # No placeholders in pattern

        return filled_count / total_count

    def _score_filename_quality(self, original_filename: str, proposed_filename: str) -> float:
        """Score based on filename quality improvements.

        Args:
            original_filename: Original filename
            proposed_filename: Proposed filename

        Returns:
            Score between 0 and 1
        """
        score = 0.5  # Base score

        # Check for quality improvements
        original_lower = original_filename.lower()
        proposed_lower = proposed_filename.lower()

        # Improvement: Removes generic names
        generic_patterns = [r"^track\d+", r"^audio\d+", r"^unknown", r"^untitled", r"^recording", r"^\d+$"]

        for pattern in generic_patterns:
            if re.match(pattern, original_lower) and not re.match(pattern, proposed_lower):
                score += 0.1

        # Improvement: Better structure (contains separator)
        if " - " in proposed_filename and " - " not in original_filename:
            score += 0.1

        # Improvement: Removes excessive underscores/dots
        if original_filename.count("_") > 3 and proposed_filename.count("_") < 2:
            score += 0.05
        if original_filename.count(".") > 2 and proposed_filename.count(".") <= 1:
            score += 0.05

        # Penalty: Makes filename significantly longer
        if len(proposed_filename) > len(original_filename) * 2:
            score -= 0.1

        # Penalty: Removes too much information
        if len(proposed_filename) < len(original_filename) * 0.3:
            score -= 0.2

        return max(0.0, min(1.0, score))

    def _score_conflict_absence(self, conflicts: List[str], warnings: List[str]) -> float:
        """Score based on absence of conflicts and warnings.

        Args:
            conflicts: List of conflicts
            warnings: List of warnings

        Returns:
            Score between 0 and 1
        """
        if conflicts:
            # Conflicts severely impact confidence
            return 0.0

        # Start with perfect score
        score = 1.0

        # Deduct for each warning
        warning_penalty = 0.15
        score -= len(warnings) * warning_penalty

        return max(0.0, score)

    def _score_source_reliability(self, source: Optional[str]) -> float:
        """Score based on metadata source reliability.

        Args:
            source: Source of metadata

        Returns:
            Score between 0 and 1
        """
        if not source:
            return 0.5  # Unknown source

        # Source reliability scores
        source_scores = {
            "id3v2": 0.95,  # ID3v2 tags are most reliable
            "id3": 0.90,  # Generic ID3
            "vorbis": 0.95,  # Vorbis comments (FLAC, OGG)
            "mp4": 0.90,  # MP4/M4A tags
            "ape": 0.85,  # APE tags
            "cue": 0.80,  # CUE sheets
            "filename": 0.60,  # Parsed from filename
            "manual": 0.70,  # Manual entry
            "inferred": 0.40,  # Inferred/guessed
        }

        return source_scores.get(source.lower(), 0.5)

    def _score_consistency(self, metadata: Dict[str, Optional[str]], proposed_filename: str) -> float:
        """Score based on internal consistency of the proposal.

        Args:
            metadata: Metadata dictionary
            proposed_filename: Proposed filename

        Returns:
            Score between 0 and 1
        """
        score = 1.0  # Start with perfect consistency

        # Check artist consistency
        artist = metadata.get("artist", "")
        albumartist = metadata.get("albumartist", "")

        if artist and albumartist and artist != albumartist:
            # Different artist and album artist - slight inconsistency
            score -= 0.1

        # Check if track number makes sense
        track = metadata.get("track", "")
        if track:
            try:
                track_num = int(track.split("/")[0])
                if track_num > 100:
                    # Unusually high track number
                    score -= 0.2
            except (ValueError, IndexError):
                pass

        # Check year/date validity
        date = metadata.get("date", "")
        if date:
            try:
                year = int(date[:4])
                import datetime

                current_year = datetime.datetime.now().year
                if year < 1900 or year > current_year + 1:
                    # Invalid year
                    score -= 0.3
            except (ValueError, IndexError):
                pass

        # Check filename length
        if len(proposed_filename) > 200:
            # Very long filename
            score -= 0.2

        return max(0.0, score)

    def adjust_confidence_for_context(
        self, base_score: float, file_type: str, is_compilation: bool = False, has_multiple_artists: bool = False
    ) -> float:
        """Adjust confidence score based on context.

        Args:
            base_score: Base confidence score
            file_type: Type of file (mp3, flac, etc.)
            is_compilation: Whether this is part of a compilation
            has_multiple_artists: Whether multiple artists are involved

        Returns:
            Adjusted confidence score
        """
        adjusted_score = base_score

        # Lossless formats typically have better metadata
        lossless_formats = ["flac", "wav", "ape", "wv"]
        if file_type.lower() in lossless_formats:
            adjusted_score *= 1.05

        # Compilations are harder to name consistently
        if is_compilation:
            adjusted_score *= 0.95

        # Multiple artists make naming more complex
        if has_multiple_artists:
            adjusted_score *= 0.92

        return max(0.0, min(1.0, adjusted_score))

    def get_confidence_category(self, score: float) -> str:
        """Get human-readable confidence category.

        Args:
            score: Confidence score (0-1)

        Returns:
            Category string
        """
        if score >= 0.9:
            return "very_high"
        elif score >= 0.75:
            return "high"
        elif score >= 0.6:
            return "medium"
        elif score >= 0.4:
            return "low"
        else:
            return "very_low"

    def should_auto_approve(self, score: float, threshold: float = 0.85) -> bool:
        """Determine if proposal should be auto-approved.

        Args:
            score: Confidence score
            threshold: Minimum score for auto-approval

        Returns:
            True if should be auto-approved
        """
        return score >= threshold
