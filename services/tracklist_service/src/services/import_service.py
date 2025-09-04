"""
Import service for importing tracklists from 1001tracklists.

This service handles fetching, parsing, and caching of tracklist data
from 1001tracklists.com for import into the local catalog.
"""

import json
import logging
from datetime import timedelta
from typing import Any
from uuid import UUID

import redis

from services.tracklist_service.src.config import get_config
from services.tracklist_service.src.models.tracklist import TrackEntry, Tracklist
from services.tracklist_service.src.models.tracklist_models import Track as ScrapedTrack
from services.tracklist_service.src.models.tracklist_models import Tracklist as ScrapedTracklist
from services.tracklist_service.src.scraper.tracklist_scraper import TracklistScraper
from services.tracklist_service.src.utils.time_utils import milliseconds_to_timedelta, parse_time_string

logger = logging.getLogger(__name__)


class ImportService:
    """Service for importing tracklists from 1001tracklists."""

    def __init__(self, redis_client: redis.Redis | None = None):
        """
        Initialize the import service.

        Args:
            redis_client: Optional Redis client for caching
        """
        self.scraper = TracklistScraper()
        self.config = get_config()

        # Setup Redis client
        self.redis_client: redis.Redis | None = None
        if redis_client:
            self.redis_client = redis_client
        elif self.config.cache.enabled:
            try:
                self.redis_client = redis.Redis(
                    host=self.config.cache.redis_host,
                    port=self.config.cache.redis_port,
                    db=self.config.cache.redis_db,
                    password=self.config.cache.redis_password,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                )
                self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Redis connection failed, caching disabled: {e}")
                self.redis_client = None

        self.cache_ttl = 3600  # 1 hour cache TTL

    def fetch_tracklist_from_1001(self, url: str, force_refresh: bool = False) -> ScrapedTracklist:
        """
        Fetch tracklist data from 1001tracklists API/scraper.

        Args:
            url: The 1001tracklists URL
            force_refresh: Force re-fetch even if cached

        Returns:
            Scraped tracklist data

        Raises:
            ValueError: If URL is invalid or fetch fails
        """
        # Check cache first unless force refresh
        if not force_refresh and self.redis_client:
            cached = self._get_cached_tracklist(url)
            if cached:
                logger.info(f"Using cached tracklist for URL: {url}")
                return cached

        # Fetch from 1001tracklists using scraper
        try:
            logger.info(f"Fetching tracklist from URL: {url}")
            scraped_tracklist = self.scraper.scrape_tracklist(url)

            # Cache the result
            if self.redis_client:
                self._cache_tracklist(url, scraped_tracklist)

            return ScrapedTracklist(**scraped_tracklist) if isinstance(scraped_tracklist, dict) else scraped_tracklist
        except Exception as e:
            logger.error(f"Failed to fetch tracklist from {url}: {e}")
            raise ValueError(f"Failed to fetch tracklist: {e!s}") from e

    def transform_to_track_entries(self, scraped_tracks: list[ScrapedTrack]) -> list[TrackEntry]:
        """
        Transform scraped tracks to TrackEntry objects.

        Args:
            scraped_tracks: List of scraped Track objects

        Returns:
            List of TrackEntry objects
        """
        track_entries = []

        for track in scraped_tracks:
            # Convert timestamp to timedelta
            start_time = self._parse_timestamp(track.timestamp) if track.timestamp else timedelta(0)

            # Determine end time from next track or None
            end_time = None  # Will be calculated later based on next track

            # Create TrackEntry
            entry = TrackEntry(
                position=track.number,
                start_time=start_time,
                end_time=end_time,
                artist=track.artist,
                title=track.title,
                remix=track.remix,
                label=track.label,
                catalog_track_id=None,  # Will be set if track is found in catalog
                confidence=(0.9 if not track.is_id else 0.5),  # Lower confidence for ID'd tracks
                transition_type=None,  # Will be set from transitions if available
                bpm=None,  # Will be set from analysis if available
                key=None,  # Will be set from analysis if available
            )
            track_entries.append(entry)

        # Calculate end times based on next track start times
        for i in range(len(track_entries) - 1):
            track_entries[i].end_time = track_entries[i + 1].start_time

        # Add transition types if available
        # This would be done by analyzing the transitions from scraped data

        return track_entries

    def import_tracklist(self, url: str, audio_file_id: UUID, force_refresh: bool = False) -> Tracklist:
        """
        Import a complete tracklist from 1001tracklists.

        Args:
            url: The 1001tracklists URL
            audio_file_id: ID of the audio file to associate
            force_refresh: Force re-fetch even if cached

        Returns:
            Imported Tracklist object

        Raises:
            ValueError: If import fails
        """
        try:
            # Fetch tracklist from 1001tracklists
            scraped = self.fetch_tracklist_from_1001(url, force_refresh)

            # Transform tracks to TrackEntry objects
            track_entries = self.transform_to_track_entries(scraped.tracks)

            # Create Tracklist object
            tracklist = Tracklist(
                audio_file_id=audio_file_id,
                source="1001tracklists",
                tracks=track_entries,
                cue_file_id=None,  # Will be set after CUE generation
                confidence_score=self._calculate_confidence_score(scraped, track_entries),
                draft_version=None,
                parent_tracklist_id=None,
                default_cue_format=None,
            )

            logger.info(f"Successfully imported tracklist with {len(track_entries)} tracks")
            return tracklist

        except Exception as e:
            logger.error(f"Failed to import tracklist: {e}")
            raise ValueError(f"Import failed: {e!s}") from e

    def _parse_timestamp(self, cue_point: Any | None) -> timedelta:
        """
        Parse a CuePoint timestamp to timedelta.

        Args:
            cue_point: CuePoint object or None

        Returns:
            timedelta representing the timestamp
        """
        if not cue_point:
            return timedelta(0)

        if hasattr(cue_point, "timestamp_ms") and cue_point.timestamp_ms is not None:
            return milliseconds_to_timedelta(cue_point.timestamp_ms)

        if hasattr(cue_point, "formatted_time") and cue_point.formatted_time:
            # Use centralized parsing function
            return parse_time_string(cue_point.formatted_time)

        return timedelta(0)

    def _calculate_confidence_score(self, scraped: ScrapedTracklist, track_entries: list[TrackEntry]) -> float:
        """
        Calculate overall confidence score for the tracklist.

        Args:
            scraped: Original scraped tracklist
            track_entries: Transformed track entries

        Returns:
            Confidence score between 0 and 1
        """
        if not track_entries:
            return 0.0

        # Factors affecting confidence:
        # 1. Percentage of tracks with timestamps
        tracks_with_time = sum(1 for t in track_entries if t.start_time.total_seconds() > 0)
        time_coverage = tracks_with_time / len(track_entries) if track_entries else 0

        # 2. Percentage of non-ID tracks
        identified_tracks = sum(1 for t in scraped.tracks if not t.is_id)
        id_coverage = identified_tracks / len(scraped.tracks) if scraped.tracks else 0

        # 3. Metadata completeness
        metadata_score = 0.0
        if scraped.metadata:
            metadata_fields = [
                scraped.dj_name,
                scraped.event_name,
                scraped.date,
                scraped.metadata.duration_minutes,
            ]
            metadata_score = sum(1 for f in metadata_fields if f) / len(metadata_fields)

        # Weighted average
        confidence = time_coverage * 0.4 + id_coverage * 0.4 + metadata_score * 0.2

        return min(max(confidence, 0.0), 1.0)

    def _get_cached_tracklist(self, url: str) -> ScrapedTracklist | None:
        """
        Get cached tracklist from Redis.

        Args:
            url: The URL to use as cache key

        Returns:
            Cached tracklist or None
        """
        if not self.redis_client:
            return None

        try:
            cache_key = f"tracklist:1001:{url}"
            cached_data = self.redis_client.get(cache_key)

            if cached_data:
                data = json.loads(str(cached_data))
                return ScrapedTracklist(**data)
        except Exception as e:
            logger.warning(f"Failed to get cached tracklist: {e}")

        return None

    def _cache_tracklist(self, url: str, tracklist: ScrapedTracklist) -> None:
        """
        Cache tracklist in Redis.

        Args:
            url: The URL to use as cache key
            tracklist: Tracklist to cache
        """
        if not self.redis_client:
            return

        try:
            cache_key = f"tracklist:1001:{url}"
            data = tracklist.model_dump_json()
            self.redis_client.setex(cache_key, self.cache_ttl, data)
            logger.debug(f"Cached tracklist for {url}")
        except Exception as e:
            logger.warning(f"Failed to cache tracklist: {e}")
