"""
Catalog search service for finding tracks in the music library.

This service provides search functionality for matching tracks in the catalog
with fuzzy matching and confidence scoring.
"""

import logging
from difflib import SequenceMatcher
from typing import Any, cast
from uuid import UUID

from services.tracklist_service.src.models.tracklist import TrackEntry
from sqlalchemy import and_, func, or_
from sqlalchemy.orm import Session

from shared.core_types.src.models import Metadata, Recording

logger = logging.getLogger(__name__)


class CatalogSearchService:
    """Service for searching and matching tracks in the catalog."""

    def __init__(self, db: Session):
        """
        Initialize catalog search service.

        Args:
            db: Database session.
        """
        self.db = db

    def search_catalog(
        self,
        query: str | None = None,
        artist: str | None = None,
        title: str | None = None,
        limit: int = 10,
    ) -> list[tuple[Recording, float]]:
        """
        Search the catalog for matching tracks.

        Args:
            query: General search query.
            artist: Artist name to search for.
            title: Track title to search for.
            limit: Maximum number of results to return.

        Returns:
            List of tuples containing (Recording, confidence_score).
        """
        if not any([query, artist, title]):
            return []

        # Build base query
        base_query = self.db.query(Recording, Metadata).join(Metadata, Recording.id == Metadata.recording_id)

        # Build search conditions
        conditions = []

        # Search in metadata
        if query:
            query_lower = query.lower()
            metadata_conditions = or_(
                func.lower(Metadata.value).contains(query_lower),
                func.lower(Recording.file_name).contains(query_lower),
            )
            conditions.append(metadata_conditions)

        if artist:
            artist_lower = artist.lower()
            artist_condition = and_(
                Metadata.key.in_(["artist", "album_artist", "performer"]),
                func.lower(Metadata.value).contains(artist_lower),
            )
            conditions.append(artist_condition)

        if title:
            title_lower = title.lower()
            title_condition = and_(
                Metadata.key == "title",
                func.lower(Metadata.value).contains(title_lower),
            )
            conditions.append(title_condition)

        if not conditions:
            return []

        # Execute query
        results = base_query.filter(or_(*conditions)).limit(limit * 3).all()

        # Group results by recording and calculate confidence scores
        recordings_map: dict[UUID, dict[str, Any]] = {}
        for recording, metadata in cast("list[tuple[Recording, Metadata]]", results):
            rec_id: UUID = recording.id
            metadata_key: str = metadata.key
            if rec_id not in recordings_map:
                recordings_map[rec_id] = {
                    "recording": recording,
                    "metadata": {},
                    "confidence": 0.0,
                }
            recordings_map[rec_id]["metadata"][metadata_key] = metadata.value

        # Calculate confidence scores
        scored_results = []
        for rec_data in recordings_map.values():
            confidence = self._calculate_confidence(
                rec_data["metadata"],
                query=query,
                artist=artist,
                title=title,
            )
            scored_results.append((rec_data["recording"], confidence))

        # Sort by confidence score
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return scored_results[:limit]

    def match_track_to_catalog(
        self,
        track: TrackEntry,
        threshold: float = 0.7,
    ) -> tuple[UUID, float] | None:
        """
        Find the best catalog match for a track entry.

        Args:
            track: Track entry to match.
            threshold: Minimum confidence threshold for a match.

        Returns:
            Tuple of (catalog_track_id, confidence) if match found, None otherwise.
        """
        # Search for matches
        results = self.search_catalog(
            artist=track.artist,
            title=track.title,
            limit=5,
        )

        if not results:
            return None

        # Get best match
        best_match, confidence = results[0]

        # Apply threshold
        if confidence >= threshold:
            match_id: UUID = best_match.id
            return (match_id, confidence)

        return None

    def fuzzy_match_tracks(
        self,
        tracks: list[TrackEntry],
        threshold: float = 0.7,
    ) -> list[TrackEntry]:
        """
        Match multiple tracks to catalog with fuzzy matching.

        Args:
            tracks: List of tracks to match.
            threshold: Minimum confidence threshold for matches.

        Returns:
            List of tracks with catalog_track_id and confidence populated.
        """
        matched_tracks = []

        for track in tracks:
            # Skip if already matched
            if track.catalog_track_id:
                matched_tracks.append(track)
                continue

            # Try to find a match
            match_result = self.match_track_to_catalog(track, threshold)

            if match_result:
                catalog_id, confidence = match_result
                track.catalog_track_id = catalog_id
                track.confidence = confidence
            else:
                # No match found, keep confidence low
                track.confidence = 0.0

            matched_tracks.append(track)

        return matched_tracks

    def _calculate_confidence(
        self,
        metadata: dict[str, Any],
        query: str | None = None,
        artist: str | None = None,
        title: str | None = None,
    ) -> float:
        """
        Calculate confidence score for a search result.

        Args:
            metadata: Recording metadata dictionary.
            query: Search query.
            artist: Artist search term.
            title: Title search term.

        Returns:
            Confidence score between 0 and 1.
        """
        scores = []

        # Artist matching
        if artist and "artist" in metadata:
            artist_score = self._fuzzy_match(artist, metadata["artist"])
            scores.append(artist_score * 0.4)  # Weight artist match at 40%
        elif artist and "album_artist" in metadata:
            artist_score = self._fuzzy_match(artist, metadata["album_artist"])
            scores.append(artist_score * 0.35)  # Slightly lower weight for album artist

        # Title matching
        if title and "title" in metadata:
            title_score = self._fuzzy_match(title, metadata["title"])
            scores.append(title_score * 0.4)  # Weight title match at 40%

        # General query matching
        if query:
            query_scores = []
            for key, value in metadata.items():
                if key in ["artist", "title", "album", "genre"]:
                    score = self._fuzzy_match(query, value)
                    query_scores.append(score)

            if query_scores:
                scores.append(max(query_scores) * 0.2)  # Weight query match at 20%

        # Additional metadata boosts
        if "bpm" in metadata:
            scores.append(0.05)  # Small boost for having BPM data
        if "key" in metadata:
            scores.append(0.05)  # Small boost for having key data
        if "genre" in metadata:
            scores.append(0.05)  # Small boost for having genre data
        if "year" in metadata or "date" in metadata:
            scores.append(0.05)  # Small boost for having date data

        # Calculate final confidence
        if not scores:
            return 0.0

        # Sum scores but cap at 1.0
        return min(sum(scores), 1.0)

    def _fuzzy_match(self, search_term: str, target: str) -> float:
        """
        Perform fuzzy string matching.

        Args:
            search_term: Term to search for.
            target: String to match against.

        Returns:
            Match score between 0 and 1.
        """
        if not search_term or not target:
            return 0.0

        # Normalize strings
        search_lower = search_term.lower().strip()
        target_lower = target.lower().strip()

        # Exact match
        if search_lower == target_lower:
            return 1.0

        # Contains match
        if search_lower in target_lower or target_lower in search_lower:
            return 0.9

        # Use SequenceMatcher for fuzzy matching
        matcher = SequenceMatcher(None, search_lower, target_lower)
        ratio = matcher.ratio()

        # Apply some common variations
        # Check for common artist name variations
        if self._check_artist_variations(search_lower, target_lower):
            ratio = max(ratio, 0.85)

        # Check for remix/edit variations
        if self._check_remix_variations(search_lower, target_lower):
            ratio = max(ratio, 0.8)

        return ratio

    def _check_artist_variations(self, search: str, target: str) -> bool:
        """
        Check for common artist name variations.

        Args:
            search: Search term.
            target: Target string.

        Returns:
            True if variation detected.
        """
        # Common variations
        variations = [
            ("feat.", "ft."),
            ("featuring", "ft."),
            ("versus", "vs"),
            ("and", "&"),
            ("presents", "pres."),
        ]

        for v1, v2 in variations:
            if v1 in search and v2 in target:
                # Replace and check again
                search_alt = search.replace(v1, v2)
                if search_alt == target:
                    return True
            elif v2 in search and v1 in target:
                search_alt = search.replace(v2, v1)
                if search_alt == target:
                    return True

        return False

    def _check_remix_variations(self, search: str, target: str) -> bool:
        """
        Check for remix/edit variations.

        Args:
            search: Search term.
            target: Target string.

        Returns:
            True if variation detected.
        """
        # Common remix indicators
        remix_terms = ["remix", "mix", "edit", "bootleg", "rework", "version", "dub"]

        # Check if one has remix info and the other doesn't
        search_has_remix = any(term in search for term in remix_terms)
        target_has_remix = any(term in target for term in remix_terms)

        if search_has_remix != target_has_remix:
            # One has remix info, check if base tracks match
            for term in remix_terms:
                search = search.replace(term, "").replace("()", "").replace("[]", "").strip()
                target = target.replace(term, "").replace("()", "").replace("[]", "").strip()

            # Check similarity of base tracks
            matcher = SequenceMatcher(None, search, target)
            if matcher.ratio() > 0.85:
                return True

        return False

    def get_catalog_track_metadata(self, track_id: UUID) -> dict[str, Any]:
        """
        Get all metadata for a catalog track.

        Args:
            track_id: Recording ID in the catalog.

        Returns:
            Dictionary of metadata key-value pairs.
        """
        metadata_items = self.db.query(Metadata).filter(Metadata.recording_id == track_id).all()

        metadata: dict[str, Any] = {}
        for item in metadata_items:
            key: str = item.key
            metadata[key] = item.value

        return metadata
