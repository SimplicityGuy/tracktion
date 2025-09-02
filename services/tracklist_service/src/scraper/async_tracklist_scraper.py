"""
Async tracklist scraper for 1001tracklists.com.

Extends the async base scraper to parse complete tracklist pages including
tracks, timestamps, transitions, and metadata with circuit breaker support.
"""

import asyncio
import contextlib
import hashlib
import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

import structlog
from bs4 import BeautifulSoup, PageElement, Tag
from dateutil import parser

from services.tracklist_service.src.models.tracklist_models import (
    CuePoint,
    Track,
    Tracklist,
    TracklistMetadata,
    Transition,
    TransitionType,
)

from .async_base_scraper import AsyncScraperBase

logger = structlog.get_logger(__name__)


class AsyncTracklistScraper(AsyncScraperBase):
    """Async scraper for retrieving complete tracklist data."""

    def __init__(self) -> None:
        """Initialize the async tracklist scraper."""
        super().__init__(service_name="1001tracklists_tracklist")
        self.base_url = "https://www.1001tracklists.com"

    async def scrape_tracklist(self, url: str) -> Tracklist:
        """
        Scrape a complete tracklist from a URL asynchronously.

        Args:
            url: The 1001tracklists.com URL to scrape

        Returns:
            Tracklist object with all parsed data

        Raises:
            ValueError: If URL is invalid or page cannot be parsed
            httpx.HTTPError: If network error occurs
        """
        # Validate URL
        if not self._is_valid_url(url):
            raise ValueError(f"Invalid 1001tracklists.com URL: {url}")

        try:
            # Fetch the page
            response = await self._make_request(url)
            html = response.text
            soup = self.parse_html(html)

            # Calculate HTML hash for change detection
            html_hash = hashlib.sha256(html.encode()).hexdigest()

            # Extract basic information
            dj_name = self._extract_dj_name(soup)
            event_info = self._extract_event_info(soup)

            # Extract tracks
            tracks = self._extract_tracks(soup)

            # Extract transitions if available
            transitions = self._extract_transitions(soup, tracks)

            # Extract metadata
            metadata = self._extract_metadata(soup)

            # Create and return tracklist
            tracklist = Tracklist(
                url=url,
                dj_name=dj_name,
                event_name=event_info.get("name", ""),
                date=event_info.get("date"),
                venue=event_info.get("venue", ""),
                tracks=tracks,
                transitions=transitions,
                metadata=metadata,
                source_html_hash=html_hash,
                scraped_at=datetime.now(UTC),
            )

            logger.info(
                "Tracklist scraped successfully",
                url=url,
                dj_name=dj_name,
                track_count=len(tracks),
                transition_count=len(transitions),
            )

            return tracklist

        except Exception as e:
            logger.error(
                "Failed to scrape tracklist",
                url=url,
                error=str(e),
            )
            raise

    async def batch_scrape_tracklists(
        self,
        urls: list[str],
        max_concurrent: int = 3,
    ) -> list[Tracklist]:
        """
        Scrape multiple tracklists concurrently.

        Args:
            urls: List of tracklist URLs to scrape
            max_concurrent: Maximum concurrent scraping operations

        Returns:
            List of scraped tracklists

        Raises:
            httpx.HTTPError: If any scraping fails
        """

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _scrape_one(url: str) -> Tracklist:
            async with semaphore:
                # Rotate user agent for each request to avoid detection
                self.rotate_user_agent()
                return await self.scrape_tracklist(url)

        tasks = [_scrape_one(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        logger.info(
            "Batch tracklist scraping completed",
            total_urls=len(urls),
            successful=len(results),
        )

        return results

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is a valid 1001tracklists.com URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc in ["www.1001tracklists.com", "1001tracklists.com"]
        except Exception:
            return False

    def _extract_dj_name(self, soup: BeautifulSoup) -> str:
        """Extract DJ name from the page."""
        try:
            # Try multiple selectors
            dj_elem = soup.find("span", class_="djName") or soup.find("h1", class_="djName")
            if dj_elem:
                return str(dj_elem.get_text(strip=True))

            # Fallback to title parsing
            title_elem = soup.find("title")
            if title_elem:
                title = title_elem.get_text()
                # Extract DJ name from title format: "DJ Name @ Event - Date"
                match = re.match(r"^([^@]+)", title)
                if match:
                    return match.group(1).strip()

            return "Unknown DJ"
        except Exception as e:
            logger.warning(f"Failed to extract DJ name: {e}")
            return "Unknown DJ"

    def _extract_event_info(self, soup: BeautifulSoup) -> dict[str, Any]:
        """Extract event information from the page."""
        info = {"name": "", "date": None, "venue": ""}

        try:
            # Event name
            event_elem = soup.find("span", class_="eventName") or soup.find("h2", class_="eventName")
            if event_elem:
                info["name"] = event_elem.get_text(strip=True)

            # Event date
            date_elem = soup.find("span", class_="eventDate") or soup.find("time")
            if date_elem:
                date_text = date_elem.get_text(strip=True)
                # Try to parse the date
                with contextlib.suppress(Exception):
                    info["date"] = parser.parse(date_text).date().isoformat()

            # Event venue
            venue_elem = soup.find("span", class_="eventVenue") or soup.find("span", class_="venue")
            if venue_elem:
                info["venue"] = venue_elem.get_text(strip=True)

        except Exception as e:
            logger.warning(f"Failed to extract event info: {e}")

        return info

    def _extract_tracks(self, soup: BeautifulSoup) -> list[Track]:
        """Extract tracks from the tracklist."""
        tracks = []

        try:
            # Find track containers
            track_containers = soup.find_all("div", class_="tlpItem") or soup.find_all("div", class_="trackItem")

            for idx, container in enumerate(track_containers, 1):
                track = self._parse_track(container, idx)
                if track:
                    tracks.append(track)

        except Exception as e:
            logger.warning(f"Failed to extract tracks: {e}")

        return tracks

    def _parse_track(self, container: Tag | PageElement, position: int) -> Track | None:
        """Parse a single track from its container."""
        try:
            # Ensure container is a Tag
            if not isinstance(container, Tag):
                return None
            # Extract track title
            title_elem = container.find("span", class_="trackName") or container.find("a", class_="track")
            title = title_elem.get_text(strip=True) if title_elem else ""

            # Extract artist
            artist_elem = container.find("span", class_="artistName") or container.find("a", class_="artist")
            artist = artist_elem.get_text(strip=True) if artist_elem else ""

            # Extract time
            time_elem = container.find("span", class_="cueTime") or container.find("span", class_="time")
            time_str = time_elem.get_text(strip=True) if time_elem else None

            # Extract label
            label_elem = container.find("span", class_="labelName") or container.find("span", class_="label")
            label = label_elem.get_text(strip=True) if label_elem else None

            # Extract BPM if available
            bpm = None
            bpm_elem = container.find("span", class_="bpm")
            if bpm_elem:
                bpm_text = bpm_elem.get_text(strip=True)
                match = re.search(r"(\d+)", bpm_text)
                if match:
                    bpm = int(match.group(1))

            # Extract key if available
            key_elem = container.find("span", class_="key")
            key = key_elem.get_text(strip=True) if key_elem else None

            # Create cue points if time is available
            cue_points = []
            if time_str:
                cue_point = self._parse_time_to_cue(time_str, position)
                if cue_point:
                    cue_points.append(cue_point)

            if title:
                return Track(
                    number=position,
                    title=title,
                    artist=artist,
                    remix=None,
                    label=label,
                    timestamp=cue_points[0] if cue_points else None,
                    bpm=bpm,
                    key=key,
                    genre=None,
                    notes=None,
                )

            return None

        except Exception as e:
            logger.warning(f"Failed to parse track at position {position}: {e}")
            return None

    def _parse_time_to_cue(self, time_str: str, track_number: int) -> CuePoint | None:
        """Parse time string to CuePoint."""
        try:
            # Parse time format (HH:MM:SS or MM:SS)
            parts = time_str.split(":")
            if len(parts) == 3:
                hours, minutes, seconds = map(int, parts)
            elif len(parts) == 2:
                hours = 0
                minutes, seconds = map(int, parts)
            else:
                return None

            total_seconds = hours * 3600 + minutes * 60 + seconds
            timestamp_ms = total_seconds * 1000
            return CuePoint(track_number=track_number, timestamp_ms=timestamp_ms, formatted_time=time_str)

        except Exception:
            return None

    def _extract_transitions(self, soup: BeautifulSoup, tracks: list[Track]) -> list[Transition]:
        """Extract transition information between tracks."""
        transitions = []

        try:
            # Look for transition markers
            trans_elements = soup.find_all("span", class_="transition") or soup.find_all("div", class_="mix")

            for elem in trans_elements:
                trans_text = elem.get_text(strip=True).lower()

                # Determine transition type
                trans_type = TransitionType.UNKNOWN
                if "cut" in trans_text:
                    trans_type = TransitionType.CUT
                elif "fade" in trans_text:
                    trans_type = TransitionType.FADE
                elif "mashup" in trans_text:
                    trans_type = TransitionType.MASHUP

                # Try to determine which tracks are involved
                # This is simplified - real implementation would need better parsing
                if len(tracks) >= 2:
                    transitions.append(
                        Transition(
                            from_track=1,
                            to_track=2,
                            transition_type=trans_type,
                            timestamp_ms=None,
                            duration_ms=None,  # Would need more parsing
                            notes=None,
                        )
                    )

        except Exception as e:
            logger.warning(f"Failed to extract transitions: {e}")

        return transitions

    def _extract_metadata(self, soup: BeautifulSoup) -> TracklistMetadata:
        """Extract metadata from the page."""
        metadata = TracklistMetadata(
            recording_type=None,
            duration_minutes=None,
            play_count=None,
            favorite_count=None,
            comment_count=None,
            download_url=None,
            stream_url=None,
            soundcloud_url=None,
            mixcloud_url=None,
            youtube_url=None,
        )

        try:
            # Extract duration
            duration_elem = soup.find("span", class_="duration") or soup.find("span", class_="length")
            if duration_elem:
                duration_text = duration_elem.get_text(strip=True)
                # Try to convert duration to minutes
                try:
                    # Parse duration like "1:23:45" or "23:45"
                    parts = duration_text.split(":")
                    if len(parts) == 3:
                        hours, minutes, seconds = map(int, parts)
                        total_minutes = hours * 60 + minutes + (seconds // 60)
                    elif len(parts) == 2:
                        minutes, seconds = map(int, parts)
                        total_minutes = minutes + (seconds // 60)
                    else:
                        total_minutes = None
                    if total_minutes:
                        metadata = metadata.model_copy(update={"duration_minutes": total_minutes})
                except Exception:
                    pass

            # Extract genre (add to tags since there's no genre field)
            genre_elem = soup.find("span", class_="genre") or soup.find("a", class_="genre")
            if genre_elem:
                genre = genre_elem.get_text(strip=True)
                if genre:
                    tags = list(metadata.tags)
                    tags.append(genre)
                    metadata = metadata.model_copy(update={"tags": tags})

            # Extract play count
            plays_elem = soup.find("span", class_="plays") or soup.find("span", class_="playCount")
            if plays_elem:
                plays_text = plays_elem.get_text(strip=True)
                match = re.search(r"(\d+)", plays_text.replace(",", ""))
                if match:
                    metadata.play_count = int(match.group(1))

            # Extract upload date (no uploaded_at field in TracklistMetadata)
            # upload_elem = soup.find("time", class_="uploaded") or soup.find("span", class_="uploadDate")
            # Note: TracklistMetadata doesn't have uploaded_at field

            # Extract tags
            tag_elements = soup.find_all("a", class_="tag") or soup.find_all("span", class_="tag")
            metadata.tags = [tag.get_text(strip=True) for tag in tag_elements]

        except Exception as e:
            logger.warning(f"Failed to extract metadata: {e}")

        return metadata
